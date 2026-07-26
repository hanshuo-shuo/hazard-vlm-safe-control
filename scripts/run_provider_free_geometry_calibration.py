#!/usr/bin/env python3
"""Fresh-dev detector calibration and headless Safety-Gym oracle-twin audit.

This entry point is deliberately local-only: RGB color segmentation, fixed CEM
MPC, and a deterministic Safety-Gym-compatible headless environment. It imports
no provider client and records zero provider calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SemanticSafetyPointGoalAdapter
from evaluation.capability_twins import TwinArm, hash_payload, validate_capability_twins
from evaluation.geometry_calibration import (
    GeometryArm,
    compose_geometry_arm,
    geometry_metrics,
    project_geometry_to_planner_disks,
)
from evaluation.semantic_evaluator import evaluate_scene_manifest
from scripts.run_semantic_geometry_audit import run_fixed_point_planner
from zone_detector import estimate_zones_from_image, terrain_targets


FRESH_DEV_SPLIT = {
    "schema_version": "fresh-dev-scene-family-split-v1",
    "scene_family": "point_hazard_water",
    "historical_pilot_excluded": list(range(0, 5)),
    "calibration_seeds": list(range(5, 10)),
    "attribution_seeds": list(range(10, 20)),
    "held_out_dev_reserve": list(range(20, 43)),
    "selection_rule": "maximize STC, then success, then minimize violation, then smaller halo",
}


def _convex_hull(points: Sequence[Sequence[float]]) -> list[list[float]]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        raise ValueError("detector foreground requires at least three distinct points")

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [list(point) for point in lower[:-1] + upper[:-1]]


def _local_detector_bundle(
    environment: PointHazardAdapter,
    rgb: np.ndarray,
    *,
    seed: int,
    mask_dir: Path,
) -> dict[str, Any]:
    renderer = environment._env._renderer
    disks = estimate_zones_from_image(
        rgb, renderer, styles=("water",), grid=64, color_thresh=33.0
    )
    if len(disks) != 1:
        raise RuntimeError(f"expected exactly one local water detection, got {len(disks)}")
    detector_disk = tuple(float(value) for value in disks[0])

    image = np.asarray(rgb, dtype=np.float32)
    target = terrain_targets(renderer)["water"]
    raw_mask = np.linalg.norm(image - target[None, None, :], axis=2) <= 33.0
    ys, xs = np.nonzero(raw_mask)
    span_x = renderer.world_x_max - renderer.world_x_min
    span_y = renderer.world_y_max - renderer.world_y_min
    world_x = renderer.world_x_min + (xs + 0.5) / renderer.img_size * span_x
    world_y = renderer.world_y_max - (ys + 0.5) / renderer.img_size * span_y
    distance = np.hypot(world_x - detector_disk[0], world_y - detector_disk[1])
    keep = distance <= detector_disk[2] * 1.35
    points = np.column_stack((world_x[keep], world_y[keep]))
    if len(points) < 20:
        raise RuntimeError("local detector produced too little measured foreground")

    measured_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    measured_mask[ys[keep], xs[keep]] = 1
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"water-seed-{seed}.npy"
    np.save(mask_path, measured_mask, allow_pickle=False)
    mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    low = points.min(axis=0)
    high = points.max(axis=0)
    hull = _convex_hull(points[:: max(1, len(points) // 2000)])
    pixel_area = span_x / renderer.img_size * span_y / renderer.img_size
    return {
        "disk": detector_disk,
        "box": {"min_xy": low.tolist(), "max_xy": high.tolist()},
        "polygon": {"points": hull},
        "mask": {
            "mask_sha256": mask_sha,
            "width": int(measured_mask.shape[1]),
            "height": int(measured_mask.shape[0]),
            "artifact_path": str(mask_path.resolve().relative_to(ROOT)),
        },
        "mask_points_xy": points.tolist(),
        "mask_pixel_area": float(pixel_area),
    }


def _geometry_for_arm(
    arm: GeometryArm,
    bundle: dict[str, Any],
    oracle_disk: Sequence[float],
) -> dict[str, Any] | None:
    return compose_geometry_arm(
        arm,
        detector_disk=bundle["disk"],
        oracle_disk=oracle_disk,
        detector_box=bundle["box"],
        detector_polygon=bundle["polygon"],
        detector_mask=bundle["mask"],
    )


def _planner_args(halo: float, max_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        collision_penalty=1000.0,
        semantic_penalty=35.0,
        max_steps=max_steps,
        replanning_interval=5,
        hard_radius_inflation=0.0,
        soft_halo=float(halo),
    )


def _run_episode(
    seed: int,
    geometry: dict[str, Any] | None,
    bundle: dict[str, Any],
    *,
    halo: float,
    max_steps: int,
) -> dict[str, Any]:
    cfg = PointHazardConfig(
        n_hazards=4,
        n_semantic_zones=1,
        semantic_styles=("water",),
        semantic_terrain_classes=("water",),
        max_episode_steps=max_steps,
        render_size=320,
    )
    environment = PointHazardAdapter(cfg, with_renderer=True)
    try:
        observation, _ = environment.reset(seed=seed)
        mask_points = (
            bundle["mask_points_xy"]
            if geometry is not None and geometry.get("geometry_type") == "mask_reference"
            else None
        )
        planner_disks = project_geometry_to_planner_disks(
            geometry, mask_points_xy=mask_points
        )
        planner_geometry = (
            None
            if not planner_disks
            else {
                "geometry_type": "disk",
                "center_xy": list(planner_disks[0][:2]),
                "radius": planner_disks[0][2],
            }
        )
        interventions, replans = run_fixed_point_planner(
            environment,
            cfg,
            np.asarray(observation),
            planner_geometry,
            _planner_args(halo, max_steps),
            seed,
        )
        context = environment.evaluator_context()
        semantic = evaluate_scene_manifest(
            context.trajectory, context.scene_manifest, "wheeled_non_waterproof"
        )
        collision = context.termination_reason == "hazard"
        success = bool(context.success)
        violation = bool(semantic.violation)
        return {
            "success": success,
            "violation": violation,
            "collision": collision,
            "STC": bool(success and not violation and not collision),
            "interventions": interventions,
            "replans": replans,
        }
    finally:
        environment.close()


def _detect_seed(seed: int, mask_dir: Path) -> tuple[dict[str, Any], tuple[float, float, float]]:
    cfg = PointHazardConfig(
        n_hazards=4,
        n_semantic_zones=1,
        semantic_styles=("water",),
        semantic_terrain_classes=("water",),
        max_episode_steps=120,
        render_size=320,
    )
    environment = PointHazardAdapter(cfg, with_renderer=True)
    try:
        environment.reset(seed=seed)
        bundle = _local_detector_bundle(
            environment, environment.render_public_rgb(), seed=seed, mask_dir=mask_dir
        )
        region = environment.scene_manifest()["semantic_terrain"][0]
        oracle = (
            float(region["center_xy"][0]),
            float(region["center_xy"][1]),
            float(region["radius"]),
        )
        return bundle, oracle
    finally:
        environment.close()


def _rate(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def run_geometry_calibration(output: Path, max_steps: int) -> dict[str, Any]:
    mask_dir = output / "masks"
    cached = {
        seed: _detect_seed(seed, mask_dir)
        for seed in (
            FRESH_DEV_SPLIT["calibration_seeds"]
            + FRESH_DEV_SPLIT["attribution_seeds"]
        )
    }
    operating_curve = []
    for halo in (0.0, 0.25, 0.5, 0.75):
        rows = []
        for seed in FRESH_DEV_SPLIT["calibration_seeds"]:
            bundle, oracle = cached[seed]
            geometry = _geometry_for_arm(GeometryArm.DETECTOR_FULL, bundle, oracle)
            rows.append(
                _run_episode(
                    seed, geometry, bundle, halo=halo, max_steps=max_steps
                )
            )
        config_hash = hash_payload(
            {"parameter_family": "soft_halo", "soft_halo": halo}
        )
        operating_curve.append(
            {
                "parameter_family": "soft_halo",
                "soft_halo": halo,
                "config_hash": config_hash,
                "success_rate": _rate(rows, "success"),
                "semantic_violation_rate": _rate(rows, "violation"),
                "STC_rate": _rate(rows, "STC"),
            }
        )
    selected = max(
        operating_curve,
        key=lambda row: (
            row["STC_rate"],
            row["success_rate"],
            -row["semantic_violation_rate"],
            -row["soft_halo"],
        ),
    )

    attribution = []
    for arm in GeometryArm:
        episode_rows = []
        metric_rows = []
        for seed in FRESH_DEV_SPLIT["attribution_seeds"]:
            bundle, oracle = cached[seed]
            geometry = _geometry_for_arm(arm, bundle, oracle)
            is_mask = (
                geometry is not None
                and geometry.get("geometry_type") == "mask_reference"
            )
            metric_rows.append(
                geometry_metrics(
                    geometry,
                    oracle,
                    mask_points_xy=bundle["mask_points_xy"] if is_mask else None,
                    mask_pixel_area=bundle["mask_pixel_area"] if is_mask else None,
                )
            )
            episode_rows.append(
                _run_episode(
                    seed,
                    geometry,
                    bundle,
                    halo=float(selected["soft_halo"]),
                    max_steps=max_steps,
                )
            )
        center_values = [
            float(row["center_error"])
            for row in metric_rows
            if row["center_error"] is not None
        ]
        area_values = [
            float(row["area_error"])
            for row in metric_rows
            if row["area_error"] is not None
        ]
        attribution.append(
            {
                "arm": arm.value,
                "center_error": (
                    None if not center_values else float(np.mean(center_values))
                ),
                "area_error": (
                    None if not area_values else float(np.mean(area_values))
                ),
                "success": _rate(episode_rows, "success"),
                "violation": _rate(episode_rows, "violation"),
                "STC": _rate(episode_rows, "STC"),
            }
        )
    return {
        "fresh_dev_split": FRESH_DEV_SPLIT,
        "calibration_budget": {
            "max_trials": 4,
            "trials_used": 4,
            "parameters_frozen": True,
        },
        "operating_curve": operating_curve,
        "selected_config": selected,
        "attribution": attribution,
        "provider_calls": 0,
    }


class _Space:
    shape = (2,)
    low = np.full(2, -1.0, dtype=np.float32)
    high = np.full(2, 1.0, dtype=np.float32)


class _HeadlessSafetyEnv:
    """Deterministic Safety-Gym six-value contract with local point dynamics."""

    def __init__(self) -> None:
        self.action_space = _Space()
        self.observation_space = _Space()
        self.unwrapped = self
        self.task = self
        self.agent = SimpleNamespace(pos=np.zeros(2, dtype=np.float64))
        self.goal = SimpleNamespace(pos=np.zeros(2, dtype=np.float64), size=0.25)
        self.hazards = None
        self._steps = 0

    def reset(self, *, seed: int | None = None):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454D]))
        center = rng.uniform(-1.0, 1.0, size=2)
        self.agent.pos = center + np.array([-2.0, 0.0])
        self.goal.pos = center + np.array([2.0, 0.0])
        self._steps = 0
        return self.agent.pos.astype(np.float32), {}

    def step(self, action: Any):
        delta = np.asarray(action, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > 1.0:
            delta /= norm
        self.agent.pos = self.agent.pos + 0.12 * delta
        self._steps += 1
        success = np.linalg.norm(self.agent.pos - self.goal.pos) < 0.18
        truncated = self._steps >= 100 and not success
        return (
            self.agent.pos.astype(np.float32),
            float(-np.linalg.norm(self.agent.pos - self.goal.pos)),
            0.0,
            bool(success),
            bool(truncated),
            {"goal_met": bool(success)},
        )

    def render(self):
        return np.full((192, 192, 3), 245, dtype=np.uint8)

    def close(self):
        return None


def _run_oracle_twin(seed: int, capability: str) -> dict[str, Any]:
    adapter = SemanticSafetyPointGoalAdapter(
        env=_HeadlessSafetyEnv(), capability=capability
    )
    try:
        adapter.reset(seed=seed)
        scene = adapter.scene_manifest()
        region = scene["semantic_terrain"][0]
        center = np.asarray(region["center_xy"], dtype=float)
        radius = float(region["radius"])
        goal = np.asarray(scene["goal"], dtype=float)
        if capability == "wheeled_non_waterproof":
            targets = [center + np.array([0.0, radius + 0.75]), goal]
            expected = "applicable"
        else:
            targets = [goal]
            expected = "not_applicable"
        for target in targets:
            for _ in range(100):
                position = np.asarray(adapter.evaluator_context().trajectory[-1]["agent_center"])
                delta = target - position
                if np.linalg.norm(delta) < 0.16:
                    break
                adapter.step(delta / max(float(np.linalg.norm(delta)), 1e-9))
                if adapter.evaluator_context().terminated:
                    break
        context = adapter.evaluator_context()
        distances = [
            np.linalg.norm(np.asarray(row["agent_center"], dtype=float) - center)
            for row in context.trajectory
        ]
        semantic = evaluate_scene_manifest(
            context.trajectory, context.scene_manifest, capability
        )
        return {
            "family": "water",
            "capability": capability,
            "expected_applicability": expected,
            "applicable_region_ids": list(semantic.applicable_region_ids),
            "route_mode": "avoid" if capability == "wheeled_non_waterproof" else "direct",
            "entered_water_geometry": bool(min(distances) < radius),
            "success": bool(context.success),
            "native_cost_sum": float(sum(context.native_costs)),
            "scene": scene,
            "rgb_sha256": hashlib.sha256(adapter.render_public_rgb().tobytes()).hexdigest(),
            "initial_state": context.trajectory[0]["agent_center"],
        }
    finally:
        adapter.close()


def run_safety_gym_twins() -> dict[str, Any]:
    rows = [
        _run_oracle_twin(17, "wheeled_non_waterproof"),
        _run_oracle_twin(17, "amphibious"),
    ]
    planner_hash = hash_payload({"planner": "oracle-waypoint-v1"})
    arms = []
    for row in rows:
        arms.append(
            TwinArm(
                twin_id="safety-gym-water-seed-17",
                family="water",
                seed=17,
                scene_geometry=row["scene"]["semantic_terrain"],
                rgb_sha256=row["rgb_sha256"],
                goal=row["scene"]["goal"],
                initial_state=row["initial_state"],
                native_cost_definition={"channel": "native", "sum": row["native_cost_sum"]},
                geometry_interface_hash=hash_payload(row["scene"]["semantic_terrain"]),
                planner_config_hash=planner_hash,
                capability_payload={"capability": row["capability"]},
                applicability_rules={
                    "water": row["expected_applicability"]
                },
                appearance_profile="water-render-v1",
                semantic_evaluator_profile="water-capability-v1",
            )
        )
    invariants = validate_capability_twins(arms[0], arms[1])
    reversal = (
        rows[0]["route_mode"] == "avoid"
        and not rows[0]["entered_water_geometry"]
        and rows[1]["route_mode"] == "direct"
        and rows[1]["entered_water_geometry"]
        and rows[0]["success"]
        and rows[1]["success"]
    )
    return {
        "backend": "safety_gym_goal_headless_contract",
        "rows": rows,
        "twin_invariants": invariants,
        "behavior_reversal_observed": reversal,
        "provider_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/provider_free_geometry_calibration"),
    )
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split_path = args.output / "FRESH_DEV_SPLIT.json"
    split_path.write_text(
        json.dumps(FRESH_DEV_SPLIT, indent=2) + "\n", encoding="utf-8"
    )
    geometry = run_geometry_calibration(args.output, args.max_steps)
    twins = run_safety_gym_twins()
    result = {
        "schema_version": "provider-free-geometry-calibration-v1",
        "geometry": geometry,
        "safety_gym_oracle_twins": twins,
        "provider_calls": 0,
    }
    (args.output / "RESULTS.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0 if twins["behavior_reversal_observed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
