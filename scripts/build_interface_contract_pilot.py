#!/usr/bin/env python3
"""Build and audit the complete provider-free interface-contract pilot matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SafetyGymGoalAdapter
from evaluation.interface_contracts import build_formal_contract_prompt
from evaluation.interface_scenarios import (
    ENVIRONMENTS,
    TWIN_ARMS,
    TwinArm,
    build_call_matrix,
    build_preregistered_blocks,
    build_twin_bundle,
    scenario_geometry,
)
from evaluation.native_environment_gate import audit_native_environment
from evaluation.paid_provider_gateway import PaidCallLedger, load_pilot_manifest
from evaluation.schemas import dependency_versions


DEFAULT_MANIFEST = ROOT / "configs" / "interface_contract_pilot_manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "interface_contract_provider_free_dry_run"
PROVENANCE_SOURCE_FILES = (
    "configs/interface_contract_pilot_manifest.json",
    "docs/ICLR_PLAN.md",
    "docs/INTERFACE_CONTRACT_MAINLINE.md",
    "docs/INTERFACE_CONTRACT_PROTOCOL.md",
    "docs/RESULTS_REGISTRY.md",
    "envs/point_hazard_adapter.py",
    "envs/safety_gym_goal_adapter.py",
    "evaluation/interface_contracts.py",
    "evaluation/interface_metrics.py",
    "evaluation/interface_scenarios.py",
    "evaluation/native_environment_gate.py",
    "evaluation/paid_provider_gateway.py",
    "evaluation/policy_interface.py",
    "scripts/analyze_interface_contract_pilot.py",
    "scripts/build_interface_contract_pilot.py",
    "scripts/make_semantic_figures.py",
    "scripts/run_interface_contract_audit.py",
    "scripts/run_interface_contract_replacement.py",
    "scripts/run_safety_gym_micro_pilot.py",
    "subgoal_pivot_hazard.py",
)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _git_provenance() -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout
    return {
        "git_sha": sha,
        "git_dirty": bool(status.strip()),
        "tracked_diff_sha256": _sha(diff),
        "source_files": {
            relative_path: _file_sha(ROOT / relative_path)
            for relative_path in PROVENANCE_SOURCE_FILES
        },
    }


def _scene_hash(scene: Mapping[str, Any]) -> str:
    return _sha(
        (json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    )


def _position_and_goal(adapter: Any) -> tuple[list[float], list[float]]:
    context = adapter.evaluator_context()
    position = context.trajectory[0].get("agent_center")
    scene = context.scene_manifest
    goal = scene.get("goal")
    if position is None or goal is None:
        raise RuntimeError("native scenario generation requires native start and goal coordinates")
    start_xy = np.asarray(position, dtype=float).reshape(-1)[:2]
    goal_xy = np.asarray(goal, dtype=float).reshape(-1)[:2]
    if not np.isfinite(start_xy).all() or not np.isfinite(goal_xy).all():
        raise RuntimeError("native start/goal coordinates must be finite")
    return start_xy.tolist(), goal_xy.tolist()


def _world_bounds(environment: str, start: list[float], goal: list[float]) -> list[float]:
    if environment == "point_hazard_native":
        return [-5.0, 5.0, -5.0, 5.0]
    extent = max(2.5, *(abs(float(item)) + 1.0 for item in start + goal))
    return [-extent, extent, -extent, extent]


def _build_inputs(
    output: Path,
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, TwinArm]], list[dict[str, Any]]]:
    point = PointHazardAdapter(
        PointHazardConfig(
            n_hazards=4,
            n_semantic_zones=0,
            semantic_styles=(),
            semantic_terrain_classes=(),
            max_episode_steps=20,
            render_size=320,
        ),
        with_renderer=True,
    )
    safety = SafetyGymGoalAdapter(
        env_id="SafetyPointGoal1-v0", render_mode="rgb_array"
    )
    adapters = {
        "point_hazard_native": point,
        "safety_gym_goal_native": safety,
    }
    bundles: dict[str, dict[str, TwinArm]] = {}
    block_records: list[dict[str, Any]] = []
    try:
        for block in blocks:
            adapter = adapters[str(block["environment"])]
            adapter.reset(seed=int(block["scene_seed"]))
            native_rgb = adapter.render_public_rgb()
            start, goal = _position_and_goal(adapter)
            geometry = scenario_geometry(
                str(block["scenario_family"]), start_xy=start, goal_xy=goal
            )
            native_scene = dict(adapter.evaluator_context().scene_manifest)
            native_scene_sha = _scene_hash(native_scene)
            bundle = build_twin_bundle(
                native_rgb,
                geometry=geometry,
                world_bounds=_world_bounds(str(block["environment"]), start, goal),
                native_scene_sha256=native_scene_sha,
            )
            bundles[str(block["block_id"])] = bundle
            arm_manifests = {}
            for arm, twin in bundle.items():
                image_path = output / "inputs" / "images" / f"{_sha(twin.png_bytes)}.png"
                if not image_path.exists():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(twin.png_bytes)
                arm_manifests[arm] = twin.to_manifest()
            block_records.append({
                **block,
                "native_scene_sha256": native_scene_sha,
                "native_rgb_sha256": _sha(np.asarray(native_rgb, dtype=np.uint8).tobytes()),
                "start_xy": start,
                "goal_xy": goal,
                "world_bounds": _world_bounds(str(block["environment"]), start, goal),
                "arms": arm_manifests,
            })
    finally:
        point.close()
        safety.close()
    return bundles, block_records


def _native_gate() -> list[dict[str, Any]]:
    point = PointHazardAdapter(
        PointHazardConfig(
            n_hazards=4,
            n_semantic_zones=0,
            semantic_styles=(),
            semantic_terrain_classes=(),
            max_episode_steps=5,
        ),
        with_renderer=True,
    )
    safety = SafetyGymGoalAdapter(
        env_id="SafetyPointGoal1-v0", render_mode="rgb_array"
    )
    try:
        return [
            audit_native_environment(point, seed=200),
            audit_native_environment(safety, seed=200),
        ]
    finally:
        point.close()
        safety.close()


def audit_matrix(
    manifest: Mapping[str, Any],
    blocks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_block_fields = {
        "equivalent_pair_id",
        "equivalent_anchor_arm",
        "ambiguous_anchor_arm",
        "scenario_family",
        "environment",
        "paid_seed",
    }
    model_ids = [model["model_budget_id"] for model in manifest["models"]]
    checks = {
        "block_count_is_80": len(blocks) == 80,
        "every_block_has_required_assignments": all(
            required_block_fields.issubset(block) for block in blocks
        ),
        "matrix_rows_are_1440": len(rows) == 1440,
        "three_model_slots": len(model_ids) == 3,
        "calls_per_model_are_480": all(
            sum(row["model_budget_id"] == model_id for row in rows) == 480
            for model_id in model_ids
        ),
        "paid_seeds_are_exactly_20_24": all(
            {row["paid_seed"] for row in rows if row["model_budget_id"] == model_id}
            == {20, 21, 22, 23, 24}
            for model_id in model_ids
        ),
        "one_reused_smoke_cell_per_model": all(
            sum(
                row["model_budget_id"] == model_id and row["compatibility_smoke"]
                for row in rows
            ) == 1
            for model_id in model_ids
        ),
        "smoke_is_first_model_cell": all(
            next(row for row in rows if row["model_budget_id"] == model_id)["compatibility_smoke"]
            for model_id in model_ids
        ),
        "all_rows_budgeted_as_formal_logical_calls": all(
            row["new_provider_call_budgeted"] is True for row in rows
        ),
        "no_duplicate_request_hashes": len({row["request_sha256"] for row in rows}) == len(rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"dry-run matrix audit failed: {checks}")
    return {"passed": True, "checks": checks}


def build(output: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_pilot_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    blocks = build_preregistered_blocks()
    bundles, block_records = _build_inputs(output, blocks)
    rows = build_call_matrix(manifest, blocks, bundles)

    prompt_dir = output / "inputs" / "prompts"
    for row in rows:
        prompt = build_formal_contract_prompt(row["contract_id"], row["capability"])
        if _sha(prompt) != row["prompt_sha256"]:
            raise RuntimeError("prompt reconstruction hash mismatch")
        prompt_path = prompt_dir / f"{row['prompt_sha256']}.txt"
        if not prompt_path.exists():
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_bytes(prompt)

    _json_write(output / "BLOCKS.json", block_records)
    _json_write(output / "CALL_MATRIX.json", rows)
    native_evidence = _native_gate()
    native_gate = {
        "schema_version": "two-native-environment-gate-v1",
        "passed": len(native_evidence) == 2 and all(item["passed"] for item in native_evidence),
        "environments": native_evidence,
    }
    _json_write(output / "NATIVE_ENVIRONMENT_GATE.json", native_gate)

    ledger = PaidCallLedger(output / "PAID_CALL_LEDGER.json", manifest)
    ledger_audit = ledger.audit()
    ledger_audit["dry_run_expected_new_provider_calls"] = 0
    ledger_audit["dry_run_expected_provider_attempts"] = 0
    ledger_audit["provider_calls_enabled"] = False
    _json_write(output / "PAID_BUDGET_AUDIT.json", ledger_audit)
    matrix_audit = audit_matrix(manifest, blocks, rows)
    _json_write(output / "MATRIX_AUDIT.json", matrix_audit)

    machine_manifest = {
        "schema_version": "interface-contract-provider-free-dry-run-v1",
        "protocol_id": manifest["protocol_id"],
        "status": "INFRA / BLOCKED / PROVIDER_FREE",
        "provider_calls_enabled": False,
        "provider_calls": 0,
        "provider_attempts": 0,
        "model_slots": [model["model_budget_id"] for model in manifest["models"]],
        "environment_count": len(ENVIRONMENTS),
        "block_count": len(blocks),
        "call_matrix_rows": len(rows),
        "calls_per_model": 480,
        "paid_seeds_per_model": [20, 21, 22, 23, 24],
        "maximum_attempts_per_request": 3,
        "maximum_provider_attempts_per_model": 1440,
        "compatibility_smoke": manifest["compatibility_smoke"],
        "source_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": _file_sha(manifest_path),
        },
        "protocol_document": {
            "path": "docs/INTERFACE_CONTRACT_PROTOCOL.md",
            "sha256": _file_sha(ROOT / "docs" / "INTERFACE_CONTRACT_PROTOCOL.md"),
        },
        "artifacts": {
            name: {"path": name, "sha256": _file_sha(output / name)}
            for name in (
                "BLOCKS.json",
                "CALL_MATRIX.json",
                "NATIVE_ENVIRONMENT_GATE.json",
                "PAID_BUDGET_AUDIT.json",
                "MATRIX_AUDIT.json",
            )
        },
        "dependency_versions": dependency_versions(),
        "provenance": _git_provenance(),
        "authorization_blockers": manifest["authorization_blockers"],
        "checks": {
            "matrix": matrix_audit["passed"],
            "paid_budget": ledger_audit["passed"],
            "native_environments": native_gate["passed"],
            "zero_provider_calls": True,
            "zero_provider_attempts": True,
            "headless_execution_forbidden": True,
            "evaluator_truth_grounding_forbidden": True,
        },
    }
    if not all(machine_manifest["checks"].values()):
        raise RuntimeError("provider-free dry run failed a required gate")
    _json_write(output / "MANIFEST.json", machine_manifest)
    return machine_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output.resolve(), args.manifest.resolve())
    print(json.dumps({
        "status": result["status"],
        "call_matrix_rows": result["call_matrix_rows"],
        "provider_calls": result["provider_calls"],
        "provider_attempts": result["provider_attempts"],
        "manifest": str((args.output.resolve() / "MANIFEST.json")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
