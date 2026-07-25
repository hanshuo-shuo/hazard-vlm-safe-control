#!/usr/bin/env python3
"""Provider-free semantic geometry audit runner.

This is the single entry point for marker-free geometry fixtures, cached
detector replay, oracle integration checks, and capability-twin smoke rows.
It never imports a provider client and rejects secret-like configuration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SemanticSafetyPointGoalAdapter
from evaluation.policy_interface import CAPABILITY_CARDS
from evaluation.semantic_evaluator import evaluate_scene_manifest
from evaluation.geometry_calibration import GeometryArm, compose_geometry_arm, disk_metrics
from evaluation.semantic_geometry import (
    CoordinateFrame,
    GeometrySource,
    ImageCoordinateTransform,
    SemanticGeometryPayload,
    adapt_regions,
    stable_sha256,
)
from mpc_expert import MPCConfig, MPCExpert


RUN_SCHEMA_VERSION = "semantic-geometry-audit-run-v1"
MANIFEST_SCHEMA_VERSION = "semantic-geometry-audit-manifest-v1"
DEV_SEEDS = frozenset(range(0, 43))
PILOT_SEEDS = frozenset(range(200, 300))
TEST_SEEDS = frozenset(range(300, 500))
SECRET_TOKENS = ("api_key", "apikey", "authorization", "bearer ", "openrouter_api_key")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def assert_no_secrets(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token.strip() in lowered for token in SECRET_TOKENS):
                raise PermissionError(f"secret-like field forbidden at {path}.{key}")
            assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in SECRET_TOKENS):
            raise PermissionError(f"secret-like value forbidden at {path}")


def git_state() -> dict[str, Any]:
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "git_sha": sha_result.stdout.strip(),
        "git_dirty": bool(status_result.stdout.strip()),
    }


def parse_seeds(tokens: Sequence[str]) -> list[int]:
    result: list[int] = []
    for token in tokens:
        if ":" in token:
            start, stop = (int(item) for item in token.split(":", 1))
            result.extend(range(start, stop))
        else:
            result.append(int(token))
    if not result or len(result) != len(set(result)):
        raise ValueError("seeds must be a non-empty unique list")
    return result


def validate_split(seeds: Sequence[int], split: str) -> None:
    allocation = {"dev": DEV_SEEDS, "pilot": PILOT_SEEDS, "test": TEST_SEEDS}[split]
    invalid = sorted(set(seeds) - allocation)
    if invalid:
        raise ValueError(f"seeds not allocated to {split}: {invalid}")


def point_config() -> PointHazardConfig:
    return PointHazardConfig(
        n_hazards=4,
        n_semantic_zones=1,
        semantic_styles=("water",),
        semantic_terrain_classes=("water",),
        max_episode_steps=80,
        render_size=320,
    )


def transform_for_point(cfg: PointHazardConfig, rgb: np.ndarray) -> ImageCoordinateTransform:
    margin = 0.4
    return ImageCoordinateTransform(
        image_width=int(rgb.shape[1]),
        image_height=int(rgb.shape[0]),
        world_x_min=-cfg.arena_half - margin,
        world_x_max=cfg.arena_half + margin,
        world_y_min=-cfg.arena_half - margin,
        world_y_max=cfg.arena_half + margin,
    )


def cached_detector_disk(cache_file: Path, seed: int) -> Mapping[str, Any] | None:
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    row = next((item for item in data.get("rows", []) if int(item["seed"]) == seed), None)
    if row is None or not row.get("detected_regions"):
        return None
    region = row["detected_regions"][0]
    center = region["center_norm1000_xy"]
    radius = float(region.get("radius_norm1000", 0.0))
    if radius <= 0:
        return None
    source_hash = row["detector_request_sha256"]
    cache_record_path = (
        ROOT
        / "results"
        / "api_cache"
        / "03_modern_perception_substitution"
        / f"{source_hash}.json"
    )
    cache_record = (
        json.loads(cache_record_path.read_text(encoding="utf-8"))
        if cache_record_path.exists()
        else None
    )
    cache_artifact_hash = (
        hashlib.sha256(cache_record_path.read_bytes()).hexdigest()
        if cache_record_path.exists()
        else None
    )
    return {
        "center": center,
        "radius": radius,
        "source_hash": source_hash,
        "cache_artifact_hash": cache_artifact_hash,
        "cache_record": cache_record,
        "cache_record_path": str(cache_record_path.relative_to(ROOT)),
    }


def geometry_payload(
    *,
    source: str,
    scene: Mapping[str, Any],
    transform: ImageCoordinateTransform,
    image_hash: str | None,
    seed: int,
    cached_detector: Path | None,
    capability_id: str,
) -> SemanticGeometryPayload:
    request_id = f"geometry-{source}-{seed}"
    scene_id = f"{scene['environment_id']}-seed-{seed}"
    records: list[dict[str, Any]] = []
    frame = CoordinateFrame.WORLD
    artifact_hash = None
    incompatible = set(CAPABILITY_CARDS[capability_id]["incompatible_terrain"])

    def applicability(semantic_class: str) -> dict[str, Any]:
        applies = semantic_class in incompatible
        return {
            "state": "applicable" if applies else "not_applicable",
            "capability_basis": [capability_id],
            "task_basis": ["avoid_incompatible_terrain"],
            "confidence": 1.0,
            "reason_code": (
                "CLASS_IN_INCOMPATIBLE_TERRAIN"
                if applies
                else "CLASS_IN_COMPATIBLE_TERRAIN"
            ),
        }

    if source == "oracle":
        for region in scene.get("semantic_terrain", []):
            records.append(
                {
                    "region_id": region["region_id"],
                    "semantic_class": region["terrain_class"],
                    "geometry_type": "disk",
                    "geometry": {
                        "center_xy": region["center_xy"],
                        "radius": region["radius"],
                    },
                    "confidence": 1.0,
                    "applicability": applicability(region["terrain_class"]),
                }
            )
    elif source == "fixture":
        records = [
            {
                "region_id": "fixture_0",
                "semantic_class": "water",
                "geometry_type": "disk",
                "geometry": {"center_xy": [0.0, 0.0], "radius": 0.5},
                "confidence": 1.0,
                "applicability": applicability("water"),
            }
        ]
    elif source == "detector":
        if cached_detector is None:
            raise ValueError("detector source requires --cached-detector-results")
        detector = cached_detector_disk(cached_detector, seed)
        if detector is not None:
            center_px = transform.provider_to_pixel(detector["center"])
            center_world = transform.pixel_to_world(center_px)
            radius_world = detector["radius"] / transform.provider_max * (
                transform.world_x_max - transform.world_x_min
            )
            records = [
                {
                    "region_id": "detector_0",
                    "semantic_class": "water",
                    "geometry_type": "disk",
                    "geometry": {"center_xy": list(center_world), "radius": radius_world},
                    # The cached pilot response did not expose a calibrated
                    # confidence.  Zero denotes unavailable, not disbelief.
                    "confidence": 0.0,
                    "applicability": applicability("water"),
                }
            ]
            artifact_hash = detector["cache_artifact_hash"]
            if artifact_hash is None:
                raise ValueError("cached detector response artifact is missing")
    elif source == "none":
        records = []
    else:
        raise ValueError("vlm_grounding is schema-reserved and requires cached structured input")
    return adapt_regions(
        source=GeometrySource(source),
        records=records,
        request_id=request_id,
        scene_id=scene_id,
        image_sha256=image_hash,
        coordinate_frame=frame,
        transform=transform,
        source_model="cached-qwen-detector" if source == "detector" else None,
        source_artifact_hash=artifact_hash,
        allow_oracle=source == "oracle",
    )


def run_fixed_point_planner(
    environment: PointHazardAdapter,
    cfg: PointHazardConfig,
    observation: np.ndarray,
    geometry: Mapping[str, Any] | None,
    args: argparse.Namespace,
    seed: int,
) -> tuple[int, int]:
    """Run the same CEM-MPC executor for every disk geometry arm."""
    expert = MPCExpert(
        SimpleNamespace(cfg=cfg),
        cfg=MPCConfig(
            horizon=6,
            n_samples=24,
            n_iters=2,
            collision_penalty=args.collision_penalty,
            soft_zone_weight=args.semantic_penalty,
        ),
        rng=np.random.default_rng(seed),
    )
    obs = np.asarray(observation, dtype=np.float32)
    semantic_disk = None
    if geometry is not None:
        if geometry.get("geometry_type") != "disk":
            raise ValueError("fixed planner execution currently supports disk arms only")
        center = geometry["center_xy"]
        semantic_disk = (float(center[0]), float(center[1]), float(geometry["radius"]))
    replans = 0
    interventions = 0
    for step in range(args.max_steps):
        if step % args.replanning_interval == 0:
            hazards = obs[6:].reshape(-1, 3)
            if semantic_disk is None:
                hard = np.empty((0, 3), dtype=np.float32)
                soft = np.empty((0, 3), dtype=np.float32)
            else:
                x, y, radius = semantic_disk
                hard = np.asarray(
                    [[x, y, radius + args.hard_radius_inflation]],
                    dtype=np.float32,
                )
                soft = np.asarray(
                    [[x, y, radius + args.soft_halo]], dtype=np.float32
                )
                interventions += 1
            expert.set_soft_zones(soft)
            expert.plan(
                obs[:2],
                obs[4:6],
                np.concatenate((hazards, hard), axis=0),
            )
            replans += 1
        action = expert.act(obs)
        obs, _reward, _cost, terminated, truncated, _info = environment.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        if terminated or truncated:
            break
    return interventions, replans


def trajectory_metrics(
    context: Any,
    evaluator: Mapping[str, Any],
    geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    points = [
        item.get("agent_center")
        for item in context.trajectory
        if item.get("agent_center") is not None
    ]
    path_length = sum(
        float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
        for left, right in zip(points, points[1:])
    )
    clearance = None
    if geometry is not None and geometry.get("geometry_type") == "disk" and points:
        center = np.asarray(geometry["center_xy"], dtype=float)
        radius = float(geometry["radius"])
        clearance = min(
            float(np.linalg.norm(np.asarray(point, dtype=float) - center) - radius)
            for point in points
        )
    physical_collision = (
        context.termination_reason == "hazard"
        or any(float(cost) > 0 for cost in context.native_costs)
    )
    timeout = context.termination_reason == "timeout"
    semantic_violation = bool(evaluator["semantic_violation"])
    return {
        "planner_minimum_clearance": clearance,
        "path_length": path_length,
        "success": bool(context.success),
        "semantic_violation": semantic_violation,
        "physical_collision": physical_collision,
        "timeout": timeout,
        "partial_horizon": not bool(context.terminated or context.truncated),
        "STC": bool(context.success and not semantic_violation and not physical_collision),
    }


def run_row(args: argparse.Namespace, seed: int, output: Path) -> tuple[dict[str, Any], bool]:
    row_id = f"{args.environment}-{args.geometry_source}-{args.capability}-seed-{seed}"
    target = output / "rows" / f"{row_id}.json"
    if args.resume and target.exists():
        return json.loads(target.read_text(encoding="utf-8")), True
    cfg = point_config()
    if args.environment == "point_hazard":
        environment = PointHazardAdapter(cfg, with_renderer=True)
    else:
        environment = SemanticSafetyPointGoalAdapter(
            capability=args.capability, render_mode=None
        )
    try:
        observation, _ = environment.reset(seed=seed)
        scene = dict(environment.scene_manifest())
        if args.environment == "point_hazard":
            rgb = environment.render_public_rgb()
            image_bytes = np.asarray(rgb, dtype=np.uint8).tobytes()
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            transform = transform_for_point(cfg, rgb)
        else:
            image_hash = None
            transform = ImageCoordinateTransform(
                320, 320, -3.0, 3.0, -3.0, 3.0
            )
        payload = geometry_payload(
            source=args.geometry_source,
            scene=scene,
            transform=transform,
            image_hash=image_hash,
            seed=seed,
            cached_detector=args.cached_detector_results,
            capability_id=args.capability,
        )
        detector_cache = (
            cached_detector_disk(args.cached_detector_results, seed)
            if args.geometry_source == "detector"
            and args.cached_detector_results is not None
            else None
        )
        capability = dict(CAPABILITY_CARDS[args.capability])
        detector_disk = None
        if args.geometry_source == "detector" and payload.regions:
            detector_geometry = payload.regions[0].geometry
            detector_disk = (
                *detector_geometry["center_xy"],
                detector_geometry["radius"],
            )
        semantic_regions = scene.get("semantic_terrain", [])
        oracle_disk = (
            (
                *semantic_regions[0]["center_xy"],
                semantic_regions[0]["radius"],
            )
            if semantic_regions
            else None
        )
        decomposition_geometry = None
        geometry_metrics = None
        if args.geometry_source == "detector":
            decomposition_geometry = compose_geometry_arm(
                GeometryArm(args.geometry_arm),
                detector_disk=detector_disk,
                oracle_disk=oracle_disk,
            )
            geometry_metrics = disk_metrics(detector_disk, oracle_disk)
        elif payload.regions:
            region_geometry = payload.regions[0].geometry
            if payload.regions[0].geometry_type.value == "disk":
                decomposition_geometry = {
                    "geometry_type": "disk",
                    "center_xy": list(region_geometry["center_xy"]),
                    "radius": float(region_geometry["radius"]),
                }
        model_visible = {
            "task_card": {
                "goal": "reach goal",
                "semantic_rule": "avoid terrain incompatible with capability",
            },
            "capability_card": capability,
            "geometry": payload.to_dict() if args.geometry_source != "oracle" else {},
        }
        if args.geometry_source == "oracle":
            # Oracle remains evaluator-side unless the condition explicitly says
            # oracle.  Its bytes are recorded separately from model-visible input.
            model_visible["oracle_condition"] = True
            model_visible["geometry"] = payload.to_dict()
        model_visible_bytes = canonical(model_visible)
        result_status = "DRY_RUN"
        trajectory: list[Any] = []
        evaluator = None
        closed_loop_metrics = None
        intervention_count = 0
        replan_count = 0
        if not args.dry_run:
            if args.environment == "point_hazard":
                intervention_count, replan_count = run_fixed_point_planner(
                    environment,
                    cfg,
                    np.asarray(observation),
                    decomposition_geometry,
                    args,
                    seed,
                )
                result_status = "FIXED_PLANNER_INFRASTRUCTURE_RUN"
            else:
                action = np.zeros(environment.action_space.shape, dtype=np.float32)
                for _ in range(args.max_steps):
                    _obs, _reward, _cost, terminated, truncated, _info = environment.step(action)
                    if terminated or truncated:
                        break
                result_status = "ZERO_ACTION_SMOKE_ONLY"
            context = environment.evaluator_context()
            trajectory = list(context.trajectory)
            evaluator = evaluate_scene_manifest(
                context.trajectory, context.scene_manifest, args.capability
            ).as_dict()
            closed_loop_metrics = trajectory_metrics(
                context, evaluator, decomposition_geometry
            )
        row = {
            "schema_version": RUN_SCHEMA_VERSION,
            "row_id": row_id,
            "timestamp": utc_now(),
            "command": [item for item in sys.argv if not any(token in item.lower() for token in SECRET_TOKENS)],
            **git_state(),
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "environment": args.environment,
            "scene_family": args.scene_family,
            "appearance_profile": args.appearance_profile,
            "scene_seed": seed,
            "split": args.split,
            "task_card": model_visible["task_card"],
            "capability_card": capability,
            "image_sha256": image_hash,
            "model_visible_payload_sha256": hashlib.sha256(model_visible_bytes).hexdigest(),
            "model_visible_payload_bytes_utf8": model_visible_bytes.decode("utf-8"),
            "geometry_payload": payload.to_dict(),
            "geometry_payload_sha256": payload.sha256,
            "geometry_decomposition": {
                "arm": args.geometry_arm,
                "geometry": decomposition_geometry,
                "metrics": geometry_metrics,
                "contains_oracle_component": "oracle" in args.geometry_arm,
                "planner_execution_status": (
                    "not_run" if args.dry_run else result_status
                ),
            },
            "applicability_source": args.applicability_source,
            "planner_config": {
                "planner": args.planner,
                "hard_radius_inflation": args.hard_radius_inflation,
                "soft_halo": args.soft_halo,
                "semantic_penalty": args.semantic_penalty,
                "collision_penalty": args.collision_penalty,
                "replanning_interval": args.replanning_interval,
                "arrival_tolerance": args.arrival_tolerance,
            },
            "config_hash": args.config_hash,
            "provider_calls": 0,
            "cached_provider_response": (
                None if detector_cache is None else detector_cache["cache_record"]
            ),
            "cached_provider_response_path": (
                None if detector_cache is None else detector_cache["cache_record_path"]
            ),
            "fallback": "none",
            "trajectory": trajectory,
            "closed_loop_metrics": closed_loop_metrics,
            "per_stage_output": {
                "recognition": {
                    "status": "cached" if detector_cache is not None else "fixture_or_none",
                },
                "applicability": {
                    "status": "capability_rule",
                    "capability": args.capability,
                },
                "grounding": {
                    "status": payload.adapter_status.value,
                    "geometry_payload_sha256": payload.sha256,
                },
                "action_proposal": {"status": "not_run"},
                "enforcement_outcome": {
                    "status": "not_run" if args.dry_run else "zero_action_smoke",
                },
            },
            "evaluator_output": evaluator,
            "intervention_count": intervention_count,
            "replan_count": replan_count,
            "token_count": (
                0
                if detector_cache is None or detector_cache["cache_record"] is None
                else detector_cache["cache_record"].get("usage", {}).get("total_tokens", 0)
            ),
            "cost_usd": (
                0.0
                if detector_cache is None or detector_cache["cache_record"] is None
                else detector_cache["cache_record"].get("usage", {}).get("cost", 0.0)
            ),
            "result_status": result_status,
            "evidence_class": "INTEGRATION_EVIDENCE",
            "paper_result": False,
        }
        assert_no_secrets(row)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return row, False
    finally:
        environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("point_hazard", "safety_gym"), default="point_hazard")
    parser.add_argument("--seeds", nargs="+", default=["0"])
    parser.add_argument("--scene-family", default="water")
    parser.add_argument("--geometry-source", choices=("none", "oracle", "fixture", "detector", "vlm_grounding"), default="fixture")
    parser.add_argument("--applicability-source", choices=("capability_rule", "fixture", "cached"), default="capability_rule")
    parser.add_argument("--capability", choices=("wheeled_non_waterproof", "amphibious"), default="wheeled_non_waterproof")
    parser.add_argument("--appearance-profile", default="water-render-v1")
    parser.add_argument("--planner", default="fixed-planner-v1")
    parser.add_argument("--split", choices=("dev", "pilot", "test"), default="dev")
    parser.add_argument("--cached-detector-results", type=Path)
    parser.add_argument(
        "--geometry-arm",
        choices=tuple(item.value for item in GeometryArm),
        default=GeometryArm.DETECTOR_FULL.value,
    )
    parser.add_argument("--output", type=Path, default=Path("results/semantic_geometry_audit"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--hard-radius-inflation", type=float, default=0.25)
    parser.add_argument("--soft-halo", type=float, default=1.0)
    parser.add_argument("--semantic-penalty", type=float, default=35.0)
    parser.add_argument("--collision-penalty", type=float, default=1000.0)
    parser.add_argument("--replanning-interval", type=int, default=5)
    parser.add_argument("--arrival-tolerance", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.seeds = parse_seeds(args.seeds)
    validate_split(args.seeds, args.split)
    if args.geometry_source == "vlm_grounding":
        raise SystemExit("vlm_grounding is reserved; provide a cached schema adapter before use")
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in {"resume"}
    }
    assert_no_secrets(config, "config")
    args.config_hash = sha(config)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sentinel = output / "TEST_RUN_SENTINEL.json"
    if args.split == "test":
        if sentinel.exists():
            raise SystemExit("REFUSED: test-run sentinel already exists; no peeking/rerun allowed")
        sentinel.write_text(
            json.dumps(
                {
                    "consumed_at": utc_now(),
                    "config_hash": args.config_hash,
                    "seeds": args.seeds,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    rows, resumed = [], 0
    failed = []
    for seed in args.seeds:
        try:
            row, was_resumed = run_row(args, seed, output)
            rows.append(row)
            resumed += int(was_resumed)
        except Exception as exc:
            failed.append({"seed": seed, "error_type": type(exc).__name__, "message": str(exc)})
    request_hashes = [row["geometry_payload_sha256"] for row in rows]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "config": config,
        "config_hash": args.config_hash,
        "row_count": len(rows),
        "failed_rows": failed,
        "resumed_rows": resumed,
        "duplicate_request_hashes": sorted(
            {item for item in request_hashes if request_hashes.count(item) > 1}
        ),
        "missing_artifact_audit": [
            seed
            for seed in args.seeds
            if not (output / "rows" / f"{args.environment}-{args.geometry_source}-{args.capability}-seed-{seed}.json").exists()
        ],
        "result_digest": sha([row["model_visible_payload_sha256"] for row in rows]),
        "provider_calls": 0,
        "test_status": "PASSED" if not failed else "FAILED",
        "evidence_class": "INTEGRATION_EVIDENCE",
    }
    assert_no_secrets(manifest)
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
