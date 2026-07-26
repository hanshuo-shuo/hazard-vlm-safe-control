#!/usr/bin/env python3
"""Experiment 0: execute registered fixture responses through two native controllers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SafetyGymGoalAdapter
from evaluation.interface_contracts import EQUIVALENT_PAIRS
from evaluation.interface_execution import (
    FIXTURE_PROVENANCE,
    canonical_sha256,
    execute_point_controller,
    execute_safety_controller,
    expected_fixture_action,
    fixture_response,
    parse_and_project_fixture,
    rendered_disk_fixture,
    shifted_grounding,
    trajectory_metrics,
)


SOURCE = ROOT / "results" / "interface_contract_provider_free_dry_run"
OUTPUT = ROOT / "results" / "interface_contract_experiment_0"
CASES = (
    "avoid_correct",
    "traverse_correct",
    "unknown",
    "avoid_shifted",
    "capability_twin_correct",
    "appearance_twin_correct",
    "equivalent_anchor",
    "equivalent_mate",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_shape(block: Mapping[str, Any], arm: str) -> tuple[int, int, int]:
    image_sha = block["arms"][arm]["image_sha256"]
    path = SOURCE / "inputs" / "images" / f"{image_sha}.png"
    with Image.open(path) as image:
        width, height = image.size
    return height, width, 3


def _case_spec(block: Mapping[str, Any], case: str) -> dict[str, str]:
    if case in {"avoid_correct", "traverse_correct", "unknown", "avoid_shifted"}:
        return {
            "arm": "reference",
            "contract_id": "formal_action_structured",
            "action": {
                "avoid_correct": "avoid",
                "traverse_correct": "traverse",
                "unknown": "unknown",
                "avoid_shifted": "avoid",
            }[case],
        }
    if case in {"capability_twin_correct", "appearance_twin_correct"}:
        arm = case.removesuffix("_correct")
        twin = block["arms"][arm]
        return {
            "arm": arm,
            "contract_id": "formal_action_structured",
            "action": expected_fixture_action(twin["capability"], twin["terrain_class"]),
        }
    pair = EQUIVALENT_PAIRS[block["equivalent_pair_id"]]
    arm = block["equivalent_anchor_arm"]
    twin = block["arms"][arm]
    return {
        "arm": arm,
        "contract_id": pair[0 if case == "equivalent_anchor" else 1],
        "action": expected_fixture_action(twin["capability"], twin["terrain_class"]),
    }


def _reset_and_validate(adapter: Any, block: Mapping[str, Any]) -> tuple[Any, np.ndarray]:
    observation, _info = adapter.reset(seed=int(block["scene_seed"]))
    context = adapter.evaluator_context()
    start = np.asarray(context.trajectory[0]["agent_center"], dtype=float)
    goal = np.asarray(context.scene_manifest["goal"], dtype=float)
    if not np.allclose(start, block["start_xy"], atol=1e-7):
        raise RuntimeError(f"native start drift for {block['block_id']}")
    if not np.allclose(goal, block["goal_xy"], atol=1e-7):
        raise RuntimeError(f"native goal drift for {block['block_id']}")
    return observation, start


def _execute_case(
    adapter: Any,
    point_cfg: PointHazardConfig | None,
    block: Mapping[str, Any],
    case: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _case_spec(block, case)
    arm = spec["arm"]
    truth = block["arms"][arm]
    image_shape = _image_shape(block, arm)
    correct = rendered_disk_fixture(truth["geometry"], block["world_bounds"], image_shape)
    grounding = shifted_grounding(correct) if case == "avoid_shifted" else correct
    raw = fixture_response(spec["contract_id"], spec["action"], truth["terrain_class"], grounding)
    projected = parse_and_project_fixture(
        spec["contract_id"],
        raw,
        image_shape=image_shape,
        world_bounds=block["world_bounds"],
        evaluator_geometry=truth["geometry"],
    )
    observation, _start = _reset_and_validate(adapter, block)
    if block["environment"] == "point_hazard_native":
        controller = execute_point_controller(
            adapter,
            point_cfg,
            np.asarray(observation),
            action=spec["action"],
            planner_geometry=projected["planner_geometry"],
            seed=int(block["scene_seed"]),
        )
    else:
        controller = execute_safety_controller(
            adapter,
            action=spec["action"],
            planner_geometry=projected["planner_geometry"],
            world_bounds=block["world_bounds"],
        )
    context = adapter.evaluator_context()
    metrics = trajectory_metrics(
        context.trajectory,
        goal_xy=block["goal_xy"],
        evaluator_geometry=truth["geometry"],
        capability=truth["capability"],
        terrain_class=truth["terrain_class"],
        action=spec["action"],
    )
    if spec["action"] != "unknown" and context.success:
        metrics["task_success"] = True
        metrics["task_failure"] = False
        metrics["STC"] = not metrics["semantic_violation"]
    actions = list(context.actions)
    trajectory = list(context.trajectory)
    execution = {
        "block_id": block["block_id"],
        "environment": block["environment"],
        "scenario_family": block["scenario_family"],
        "paid_seed": block["paid_seed"],
        "scene_seed": block["scene_seed"],
        "case": case,
        **spec,
        "capability": truth["capability"],
        "terrain_class": truth["terrain_class"],
        "planner_grounding_provenance": FIXTURE_PROVENANCE,
        "evaluator_geometry_used_by_controller": False,
        "calibration": projected["calibration"],
        "controller": controller,
        "metrics": metrics,
        "native_cost_sum": float(sum(context.native_costs)),
        "termination_reason": context.termination_reason,
        "action_sha256": canonical_sha256(actions),
        "trajectory_sha256": canonical_sha256(trajectory),
        "actions": actions,
        "trajectory": trajectory,
    }
    fixture = {
        "block_id": block["block_id"],
        "case": case,
        **spec,
        "raw_response": raw,
        "parsed": projected["parsed"],
        "grounding_provenance": FIXTURE_PROVENANCE,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    return execution, fixture


def _rate(rows: list[Mapping[str, Any]], predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _summarize(executions: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Any] = {}
    for case in CASES:
        rows = [row for row in executions if row["case"] == case]
        by_case[case] = {
            "n": len(rows),
            "nonstationary_rate": _rate(rows, lambda row: row["metrics"]["nonstationary"]),
            "task_success_rate": _rate(rows, lambda row: row["metrics"]["task_success"]),
            "entered_terrain_rate": _rate(rows, lambda row: row["metrics"]["entered_terrain"]),
            "semantic_violation_rate": _rate(rows, lambda row: row["metrics"]["semantic_violation"]),
            "mean_center_calibration_error_world": float(
                np.mean([row["calibration"]["center_error_world"] for row in rows])
            ),
            "max_center_calibration_error_world": max(
                row["calibration"]["center_error_world"] for row in rows
            ),
            "mean_radius_calibration_error_world": float(
                np.mean([row["calibration"]["radius_error_world"] for row in rows])
            ),
        }
    pair_index: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in executions:
        if row["case"] in {"equivalent_anchor", "equivalent_mate"}:
            pair_index[row["block_id"]][row["case"]] = row
    equivalent_pairs = list(pair_index.values())
    action_iec = _rate(
        equivalent_pairs,
        lambda pair: pair["equivalent_anchor"]["action_sha256"]
        == pair["equivalent_mate"]["action_sha256"],
    )
    trajectory_iec = _rate(
        equivalent_pairs,
        lambda pair: pair["equivalent_anchor"]["trajectory_sha256"]
        == pair["equivalent_mate"]["trajectory_sha256"],
    )
    direct = {
        "direct_path_intersection",
        "clear_visible_incompatible",
        "capability_reversal",
    }
    avoid_direct = [
        row for row in executions
        if row["case"] == "avoid_correct" and row["scenario_family"] in direct
    ]
    traverse_direct = [
        row for row in executions
        if row["case"] == "traverse_correct" and row["scenario_family"] in direct
    ]
    by_environment = {}
    for environment in sorted({row["environment"] for row in executions}):
        rows = [row for row in executions if row["environment"] == environment]
        by_environment[environment] = {
            "n": len(rows),
            "nonstationary": any(row["metrics"]["nonstationary"] for row in rows),
            "task_success_count": sum(row["metrics"]["task_success"] for row in rows),
        }
    capability_rows = [row for row in executions if row["case"] == "capability_twin_correct"]
    appearance_rows = [row for row in executions if row["case"] == "appearance_twin_correct"]
    unknown_rows = [row for row in executions if row["case"] == "unknown"]
    block_cases: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in executions:
        block_cases[row["block_id"]][row["case"]] = row
    capability_reversals = []
    appearance_matches = []
    for cases in block_cases.values():
        reference = cases["traverse_correct"]
        capability = cases["capability_twin_correct"]
        appearance = cases["appearance_twin_correct"]
        if reference["metrics"]["entered_terrain"]:
            capability_reversals.append(
                reference["metrics"]["semantic_violation"]
                and not capability["metrics"]["semantic_violation"]
                and reference["trajectory_sha256"] == capability["trajectory_sha256"]
            )
        appearance_matches.append(
            reference["action_sha256"] == appearance["action_sha256"]
            and reference["trajectory_sha256"] == appearance["trajectory_sha256"]
            and not appearance["metrics"]["semantic_violation"]
        )
    direct_entry_by_environment = {}
    for environment in sorted({row["environment"] for row in executions}):
        environment_avoid = [row for row in avoid_direct if row["environment"] == environment]
        environment_traverse = [row for row in traverse_direct if row["environment"] == environment]
        direct_entry_by_environment[environment] = {
            "avoid_correct": _rate(
                environment_avoid, lambda row: row["metrics"]["entered_terrain"]
            ),
            "traverse_correct": _rate(
                environment_traverse, lambda row: row["metrics"]["entered_terrain"]
            ),
        }
    correct_rows = [
        row for row in executions
        if row["case"] in {"avoid_correct", "traverse_correct", "unknown"}
    ]
    shifted_rows = [row for row in executions if row["case"] == "avoid_shifted"]
    checks = {
        "all_80_blocks_executed": len({row["block_id"] for row in executions}) == 80,
        "all_8_cases_per_block": len(executions) == 80 * len(CASES),
        "two_native_environments_nonstationary": all(
            item["nonstationary"] for item in by_environment.values()
        ),
        "two_native_environments_partial_completion": all(
            item["task_success_count"] > 0 for item in by_environment.values()
        ),
        "correct_avoid_reduces_direct_terrain_entry": _rate(
            avoid_direct, lambda row: row["metrics"]["entered_terrain"]
        ) < _rate(traverse_direct, lambda row: row["metrics"]["entered_terrain"]),
        "avoid_traverse_difference_in_each_environment": all(
            value["avoid_correct"] < value["traverse_correct"]
            for value in direct_entry_by_environment.values()
        ),
        "capability_twins_have_no_semantic_violation": all(
            not row["metrics"]["semantic_violation"] for row in capability_rows
        ),
        "capability_twin_reverses_semantic_evaluation": bool(capability_reversals)
        and all(capability_reversals),
        "appearance_twins_do_not_avoid": all(row["action"] == "traverse" for row in appearance_rows),
        "appearance_twins_preserve_physical_execution": all(appearance_matches),
        "unknown_is_frozen_task_failure": all(
            row["metrics"]["task_failure"] and not row["metrics"]["nonstationary"]
            for row in unknown_rows
        ),
        "equivalent_physical_action_iec_is_1": action_iec == 1.0,
        "equivalent_trajectory_iec_is_1": trajectory_iec == 1.0,
        "fixture_grounding_only": all(
            row["planner_grounding_provenance"] == FIXTURE_PROVENANCE
            and not row["evaluator_geometry_used_by_controller"]
            for row in executions
        ),
        "correct_calibration_error_is_bounded": max(
            row["calibration"]["center_error_world"] for row in correct_rows
        ) < 0.04,
        "shifted_grounding_error_is_larger": float(
            np.mean([row["calibration"]["center_error_world"] for row in shifted_rows])
        ) > 1.0,
        "zero_provider_calls": True,
        "zero_provider_attempts": True,
    }
    return {
        "by_case": by_case,
        "by_environment": by_environment,
        "direct_path_entry_rate": {
            "avoid_correct": _rate(avoid_direct, lambda row: row["metrics"]["entered_terrain"]),
            "traverse_correct": _rate(traverse_direct, lambda row: row["metrics"]["entered_terrain"]),
        },
        "direct_path_entry_rate_by_environment": direct_entry_by_environment,
        "capability_semantic_reversal": {
            "eligible_entered_pairs": len(capability_reversals),
            "reversal_rate": _rate(capability_reversals, bool),
        },
        "appearance_physical_match_rate": _rate(appearance_matches, bool),
        "physical_action_IEC": action_iec,
        "trajectory_IEC": trajectory_iec,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(
    source: Path,
    output: Path,
    limit_blocks: int | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    blocks = json.loads((source / "BLOCKS.json").read_text(encoding="utf-8"))
    if len(blocks) != 80:
        raise RuntimeError("Experiment 0 requires the frozen 80-block input")
    paid_ledger_path = source / "PAID_CALL_LEDGER.json"
    paid_ledger = json.loads(paid_ledger_path.read_text(encoding="utf-8"))
    ledger_calls = sum(
        int(model["new_provider_calls"]) for model in paid_ledger["models"].values()
    )
    ledger_attempts = sum(
        int(model["provider_attempts"]) for model in paid_ledger["models"].values()
    )
    if ledger_calls or ledger_attempts:
        raise RuntimeError("Experiment 0 requires a zero-call, zero-attempt paid ledger")
    native_gate_path = source / "NATIVE_ENVIRONMENT_GATE.json"
    native_gate = json.loads(native_gate_path.read_text(encoding="utf-8"))
    if not native_gate.get("passed", False):
        raise RuntimeError("Experiment 0 requires the two-native-environment gate")
    selected = [
        block for block in blocks
        if environment is None or block["environment"] == environment
    ]
    if limit_blocks is not None:
        selected = selected[:limit_blocks]
    point_cfg = PointHazardConfig(
        n_hazards=4,
        n_semantic_zones=0,
        semantic_styles=(),
        semantic_terrain_classes=(),
        max_episode_steps=120,
        render_size=320,
    )
    point = PointHazardAdapter(point_cfg, with_renderer=True)
    safety = SafetyGymGoalAdapter(env_id="SafetyPointGoal1-v0", render_mode="rgb_array")
    adapters = {"point_hazard_native": point, "safety_gym_goal_native": safety}
    executions: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    try:
        for block_index, block in enumerate(selected):
            adapter = adapters[block["environment"]]
            for case in CASES:
                execution, fixture = _execute_case(
                    adapter,
                    point_cfg if block["environment"] == "point_hazard_native" else None,
                    block,
                    case,
                )
                executions.append(execution)
                fixtures.append(fixture)
            print(
                json.dumps({
                    "block": block_index + 1,
                    "total": len(selected),
                    "block_id": block["block_id"],
                }),
                flush=True,
            )
    finally:
        point.close()
        safety.close()
    summary = _summarize(executions)
    if limit_blocks is not None:
        summary["passed"] = False
        summary["checks"]["all_80_blocks_executed"] = False
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "EXECUTIONS.json", executions)
    _write_json(output / "FIXTURES.json", fixtures)
    _write_json(output / "SUMMARY.json", summary)
    manifest = {
        "schema_version": "interface-contract-experiment-0-v2",
        "status": "PASS" if summary["passed"] else "NO_GO",
        "provider_calls": 0,
        "provider_attempts": 0,
        "paid_ledger": {
            "path": str(paid_ledger_path),
            "sha256": _sha(paid_ledger_path),
            "new_provider_calls": ledger_calls,
            "provider_attempts": ledger_attempts,
        },
        "native_environment_gate": {
            "path": str(native_gate_path),
            "sha256": _sha(native_gate_path),
            "passed": True,
        },
        "source_blocks": {"path": str(source / "BLOCKS.json"), "sha256": _sha(source / "BLOCKS.json")},
        "block_count": len(selected),
        "execution_count": len(executions),
        "cases": list(CASES),
        "controllers": {
            "point_hazard_native": "repository fixed CEM-MPC",
            "safety_gym_goal_native": "native Point heading/forward feedback controller",
            "unknown": "frozen no-op task failure",
        },
        "source_files": {
            relative: _sha(ROOT / relative)
            for relative in (
                "evaluation/interface_execution.py",
                "evaluation/interface_scenarios.py",
                "scripts/run_interface_execution_bridge.py",
                "tests/test_interface_execution_bridge_v2.py",
            )
        },
        "artifacts": {
            name: {"path": name, "sha256": _sha(output / name)}
            for name in ("EXECUTIONS.json", "FIXTURES.json", "SUMMARY.json")
        },
        "checks": summary["checks"],
    }
    _write_json(output / "MANIFEST.json", manifest)
    if limit_blocks is None and not summary["passed"]:
        failed = [name for name, passed in summary["checks"].items() if not passed]
        raise RuntimeError(f"Experiment 0 is NO-GO: {failed}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit-blocks", type=int)
    parser.add_argument("--environment", choices=("point_hazard_native", "safety_gym_goal_native"))
    args = parser.parse_args()
    result = run(
        args.source.resolve(), args.output.resolve(), args.limit_blocks, args.environment
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
