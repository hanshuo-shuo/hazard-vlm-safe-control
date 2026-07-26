#!/usr/bin/env python3
"""Zero-provider native replay, CISR, and five-stage analysis for the paid scout."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from evaluation.interface_contracts import PLANNER_MAPPINGS
from evaluation.interface_execution import (
    canonical_sha256,
    execute_point_controller,
    execute_safety_controller,
    trajectory_metrics,
)
from evaluation.interface_metrics import (
    canonical_semantic_consistency,
    cisr_eq,
    cisr_map,
    grounding_consistency,
    outcome_rates,
    parse_consistency,
    physical_action_iec,
    ranking_stability_envelope,
)
from evaluation.scout_replay import (
    PROVIDER_PROVENANCE,
    formal_planner_action,
    project_grounding,
    stage_difference_summary,
    summarize_five_stage,
)


SOURCE = ROOT / "results" / "interface_contract_provider_free_dry_run"
PAID = ROOT / "results" / "interface_contract_scout_paid"
OUTPUT = ROOT / "results" / "interface_contract_scout_native_analysis"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_shape(row: Mapping[str, Any]) -> tuple[int, int, int]:
    path = SOURCE / "inputs" / "images" / f"{row['image_sha256']}.png"
    with Image.open(path) as image:
        width, height = image.size
    return height, width, 3


def reset_validate(adapter: Any, block: Mapping[str, Any]) -> np.ndarray:
    observation, _info = adapter.reset(seed=int(block["scene_seed"]))
    context = adapter.evaluator_context()
    if not np.allclose(context.trajectory[0]["agent_center"], block["start_xy"], atol=1e-7):
        raise RuntimeError(f"native start drift for {block['block_id']}")
    if not np.allclose(context.scene_manifest["goal"], block["goal_xy"], atol=1e-7):
        raise RuntimeError(f"native goal drift for {block['block_id']}")
    return np.asarray(observation)


def execute(
    row: Mapping[str, Any],
    block: Mapping[str, Any],
    adapter: Any,
    point_cfg: PointHazardConfig | None,
    *,
    mapping: str = "action_authoritative",
) -> dict[str, Any]:
    parsed = row["parsed"]
    truth = block["arms"][row["arm"]]
    projection = project_grounding(
        parsed,
        image_shape=image_shape(row),
        world_bounds=block["world_bounds"],
        evaluator_geometry=truth["geometry"],
    )
    action = formal_planner_action(parsed, row["contract_id"], mapping)
    observation = reset_validate(adapter, block)
    planner_geometry = (
        projection["planner_geometry"]
        if projection is not None
        else {
            "geometry_type": "disk",
            "center_xy": [0.0, 0.0],
            "radius": 0.0,
            "coordinate_frame": "native_world_xy",
            "provenance": "PARSE_FAILURE_NO_GROUNDING",
        }
    )
    if block["environment"] == "point_hazard_native":
        controller = execute_point_controller(
            adapter,
            point_cfg,
            observation,
            action=action,
            planner_geometry=planner_geometry,
            seed=int(block["scene_seed"]),
        )
    else:
        controller = execute_safety_controller(
            adapter,
            action=action,
            planner_geometry=planner_geometry,
            world_bounds=block["world_bounds"],
        )
    context = adapter.evaluator_context()
    metrics = trajectory_metrics(
        context.trajectory,
        goal_xy=block["goal_xy"],
        evaluator_geometry=truth["geometry"],
        capability=truth["capability"],
        terrain_class=truth["terrain_class"],
        action=action,
    )
    if action != "unknown" and context.success:
        metrics["task_success"] = True
        metrics["task_failure"] = False
        metrics["STC"] = not metrics["semantic_violation"]
    actions = list(context.actions)
    trajectory = list(context.trajectory)
    cost = float(sum(context.native_costs))
    return {
        **{key: row[key] for key in (
            "call_index", "model_budget_id", "environment", "scenario_family",
            "paid_seed", "block_id", "equivalent_pair_id", "arm", "contract_id",
            "contract_role", "capability", "terrain_class", "response_sha256",
        )},
        "parsed": parsed,
        "physical_action": action,
        "planner_mapping": mapping,
        "planner_grounding_provenance": (
            PROVIDER_PROVENANCE if projection is not None else "PARSE_FAILURE_NO_GROUNDING"
        ),
        "evaluator_geometry_used_by_controller": False,
        "evaluator_geometry": truth["geometry"],
        "calibration": projection["calibration"] if projection else None,
        "controller": controller,
        "native_cost_sum": cost,
        "collision": cost > 0,
        "termination_reason": context.termination_reason,
        "action_sha256": canonical_sha256(actions),
        "trajectory_sha256": canonical_sha256(trajectory),
        "actions": actions,
        "trajectory": trajectory,
        **metrics,
    }


def equivalent_rows(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in primary:
        if row["contract_role"] == "mate":
            side = "mate"
        elif row["contract_role"] == "anchor" and row["arm"] == next(
            item["arm"] for item in primary
            if item["block_id"] == row["block_id"]
            and item["model_budget_id"] == row["model_budget_id"]
            and item["contract_role"] == "mate"
        ):
            side = "anchor"
        else:
            continue
        result.append({
            **row,
            "comparison_id": f"{row['model_budget_id']}|{row['block_id']}",
            "pair_side": side,
        })
    return result


def complete_parse_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["comparison_id"]].append(row)
    eligible = {
        key for key, pair in grouped.items()
        if len(pair) == 2 and all(item["parsed"]["parse_status"] == "ok" for item in pair)
    }
    return [row for row in rows if row["comparison_id"] in eligible]


def metric_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parse_consistency": parse_consistency(rows),
        "canonical_semantic_consistency": canonical_semantic_consistency(rows),
        "grounding_consistency": grounding_consistency(rows),
        "physical_action_IEC": physical_action_iec(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paid", type=Path, default=PAID)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paid = args.paid.resolve()
    row_paths = sorted((paid / "rows").glob("*.json"))
    if len(row_paths) != 120:
        raise SystemExit(f"native replay requires 120 cached paid rows, found {len(row_paths)}")
    paid_result = json.loads((paid / "RESULTS.json").read_text(encoding="utf-8"))
    if paid_result.get("cached_formal_responses") != 120:
        raise SystemExit("paid result does not certify 120 cached formal responses")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in row_paths]
    blocks = {
        block["block_id"]: block
        for block in json.loads((SOURCE / "BLOCKS.json").read_text(encoding="utf-8"))
    }
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
    primary: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            block = blocks[row["block_id"]]
            primary.append(execute(
                row, block, adapters[row["environment"]],
                point_cfg if row["environment"] == "point_hazard_native" else None,
            ))
            print(json.dumps({"primary": index, "total": 120, "call_index": row["call_index"]}), flush=True)
        ambiguous_source = [row for row in rows if row["contract_role"] == "ambiguous"]
        for source_index, row in enumerate(ambiguous_source, start=1):
            block = blocks[row["block_id"]]
            for mapping in PLANNER_MAPPINGS:
                ambiguous.append(execute(
                    row, block, adapters[row["environment"]],
                    point_cfg if row["environment"] == "point_hazard_native" else None,
                    mapping=mapping,
                ))
            print(json.dumps({"ambiguous_output": source_index, "total": 20}), flush=True)
    finally:
        point.close()
        safety.close()

    equivalent = equivalent_rows(primary)
    compliant = complete_parse_pairs(equivalent)
    by_model = {}
    for model in sorted({row["model_budget_id"] for row in primary}):
        subset = [row for row in primary if row["model_budget_id"] == model]
        model_eq = [row for row in equivalent if row["model_budget_id"] == model]
        model_compliant = complete_parse_pairs(model_eq)
        by_model[model] = {
            "n": len(subset),
            "parse_rate": sum(row["parsed"]["parse_status"] == "ok" for row in subset) / len(subset),
            "all_call": metric_bundle(model_eq),
            "parse_compliant": metric_bundle(model_compliant),
        }
    analysis = {
        "schema_version": "interface-contract-scout-native-analysis-v1",
        "provider_calls": 0,
        "provider_attempts": 0,
        "primary_native_executions": len(primary),
        "ambiguous_mapping_executions": len(ambiguous),
        "all_call": {
            **metric_bundle(equivalent),
            "CISR_EQ": cisr_eq(equivalent),
        },
        "parse_compliant": {
            **metric_bundle(compliant),
            "CISR_EQ": cisr_eq(compliant),
        },
        "CISR_MAP": cisr_map(ambiguous),
        "by_model": by_model,
        "outcomes": {
            "primary": outcome_rates(primary),
            "equivalent": outcome_rates(equivalent),
            "ambiguous_mappings": outcome_rates(ambiguous),
        },
        "ranking_stability": ranking_stability_envelope(
            [{**row, "condition_id": row["contract_id"]} for row in primary]
        ),
        "stage_difference_attribution": stage_difference_summary(equivalent),
        "five_stage": summarize_five_stage(primary),
    }
    output = args.output.resolve()
    write_json(output / "PRIMARY_EXECUTIONS.json", primary)
    write_json(output / "AMBIGUOUS_MAPPING_EXECUTIONS.json", ambiguous)
    write_json(output / "ANALYSIS.json", analysis)
    manifest = {
        "schema_version": "interface-contract-scout-native-replay-v1",
        "status": "COMPLETE",
        "provider_calls": 0,
        "provider_attempts": 0,
        "paid_rows": len(rows),
        "primary_native_executions": len(primary),
        "ambiguous_mapping_executions": len(ambiguous),
        "artifacts": {
            name: {"path": name, "sha256": sha(output / name)}
            for name in (
                "PRIMARY_EXECUTIONS.json", "AMBIGUOUS_MAPPING_EXECUTIONS.json", "ANALYSIS.json"
            )
        },
    }
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
