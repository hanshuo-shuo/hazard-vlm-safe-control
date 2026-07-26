#!/usr/bin/env python3
"""Replay cached interface-contract decisions through executable controllers.

No provider is imported or called. PointHazard uses the repository's fixed MPC;
the Safety-Gym arm uses the deterministic six-value headless dynamics contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SemanticSafetyPointGoalAdapter
from evaluation.interface_contracts import CONTRACT_IDS, PLANNER_MAPPINGS, planner_action
from evaluation.semantic_evaluator import evaluate_scene_manifest
from scripts.analyze_interface_contract_results import (
    PARENT_DIR,
    REPLACEMENT_DIR,
    VALID_PARENT_MODELS,
    load_rows,
)
from scripts.run_interface_contract_audit import SEEDS, json_write
from scripts.run_interface_contract_replacement import REPLACEMENT_MODEL
from scripts.run_provider_free_geometry_calibration import _HeadlessSafetyEnv
from scripts.run_semantic_geometry_audit import run_fixed_point_planner


OUTPUT_DIR = Path("results/interface_contract_execution_bridge")
ACTIONS = ("avoid", "traverse", "unknown")


def planner_args(max_steps: int) -> argparse.Namespace:
    return argparse.Namespace(
        collision_penalty=1000.0,
        semantic_penalty=35.0,
        max_steps=max_steps,
        replanning_interval=5,
        hard_radius_inflation=0.0,
        soft_halo=0.0,
    )


def point_hazard_outcome(seed: int, capability: str, action: str) -> dict[str, Any]:
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
        observation, _ = environment.reset(seed=seed)
        if action != "unknown":
            region = environment.scene_manifest()["semantic_terrain"][0]
            geometry = None if action == "traverse" else {
                "geometry_type": "disk",
                "center_xy": region["center_xy"],
                "radius": region["radius"],
            }
            interventions, replans = run_fixed_point_planner(
                environment,
                cfg,
                np.asarray(observation),
                geometry,
                planner_args(120),
                seed,
            )
        else:
            interventions, replans = 0, 0
        context = environment.evaluator_context()
        semantic = evaluate_scene_manifest(
            context.trajectory, context.scene_manifest, capability
        )
        collision = context.termination_reason == "hazard"
        return {
            "environment": "point_hazard",
            "seed": seed,
            "capability": capability,
            "action": action,
            "success": bool(context.success),
            "semantic_violation": bool(semantic.violation),
            "collision": collision,
            "STC": bool(context.success and not semantic.violation and not collision),
            "termination_reason": context.termination_reason,
            "steps": len(context.actions),
            "interventions": interventions,
            "replans": replans,
            "native_cost_sum": float(sum(context.native_costs)),
        }
    finally:
        environment.close()


def safety_gym_headless_outcome(seed: int, capability: str, action: str) -> dict[str, Any]:
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
        targets = []
        if action == "avoid":
            targets = [center + np.array([0.0, radius + 0.75]), goal]
        elif action == "traverse":
            targets = [goal]
        elif action != "unknown":
            raise ValueError(f"unknown action: {action}")
        for target in targets:
            for _ in range(100):
                context = adapter.evaluator_context()
                if context.terminated or context.truncated:
                    break
                position = np.asarray(context.trajectory[-1]["agent_center"], dtype=float)
                delta = target - position
                norm = float(np.linalg.norm(delta))
                if norm < 0.16:
                    break
                adapter.step(delta / max(norm, 1e-12))
        context = adapter.evaluator_context()
        semantic = evaluate_scene_manifest(
            context.trajectory, context.scene_manifest, capability
        )
        collision = bool(any(cost > 0 for cost in context.native_costs))
        return {
            "environment": "safety_gym_goal",
            "seed": seed,
            "capability": capability,
            "action": action,
            "success": bool(context.success),
            "semantic_violation": bool(semantic.violation),
            "collision": collision,
            "STC": bool(context.success and not semantic.violation and not collision),
            "termination_reason": context.termination_reason,
            "steps": len(context.actions),
            "interventions": 1 if action == "avoid" else 0,
            "replans": 0,
            "native_cost_sum": float(sum(context.native_costs)),
        }
    finally:
        adapter.close()


def rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> int:
    parent = load_rows(PARENT_DIR)
    replacement = load_rows(REPLACEMENT_DIR)
    valid = [row for row in parent if row["model"] in VALID_PARENT_MODELS] + replacement
    decisions = [row for row in valid if row["task"] == "semantic_route_decision"]
    capabilities = {
        (row["environment"], int(row["seed"])): row["capability"]
        for row in decisions
    }
    lookup = {}
    executed = []
    for environment in ("point_hazard", "safety_gym_goal"):
        for seed in SEEDS:
            capability = capabilities[(environment, seed)]
            for action in ACTIONS:
                outcome = (
                    point_hazard_outcome(seed, capability, action)
                    if environment == "point_hazard"
                    else safety_gym_headless_outcome(seed, capability, action)
                )
                lookup[(environment, seed, action)] = outcome
                executed.append(outcome)
    mapped = []
    for row in decisions:
        for mapping in PLANNER_MAPPINGS:
            action = planner_action(row["parsed"], row["contract_id"], mapping)
            outcome = lookup[(row["environment"], int(row["seed"]), action)]
            mapped.append(
                {
                    "model": row["model"],
                    "environment": row["environment"],
                    "seed": row["seed"],
                    "capability": row["capability"],
                    "contract_id": row["contract_id"],
                    "planner_mapping": mapping,
                    "planner_action": action,
                    **{key: outcome[key] for key in (
                        "success", "semantic_violation", "collision", "STC",
                        "termination_reason", "steps",
                    )},
                }
            )
    summary: dict[str, Any] = {"by_contract_mapping": {}, "ambiguous_by_environment": {}}
    for contract_id in CONTRACT_IDS:
        summary["by_contract_mapping"][contract_id] = {}
        for mapping in PLANNER_MAPPINGS:
            subset = [
                row for row in mapped
                if row["contract_id"] == contract_id and row["planner_mapping"] == mapping
            ]
            summary["by_contract_mapping"][contract_id][mapping] = {
                "n": len(subset),
                "success": rate(subset, "success"),
                "semantic_violation": rate(subset, "semantic_violation"),
                "collision": rate(subset, "collision"),
                "STC": rate(subset, "STC"),
            }
    for environment in ("point_hazard", "safety_gym_goal"):
        summary["ambiguous_by_environment"][environment] = {}
        for mapping in PLANNER_MAPPINGS:
            subset = [
                row for row in mapped
                if row["environment"] == environment
                and row["contract_id"] == "ambiguous_applicability"
                and row["planner_mapping"] == mapping
            ]
            summary["ambiguous_by_environment"][environment][mapping] = {
                "n": len(subset),
                "success": rate(subset, "success"),
                "semantic_violation": rate(subset, "semantic_violation"),
                "collision": rate(subset, "collision"),
                "STC": rate(subset, "STC"),
            }
    ambiguous = summary["by_contract_mapping"]["ambiguous_applicability"]
    stc_values = [value["STC"] for value in ambiguous.values()]
    result = {
        "schema_version": "interface-contract-execution-bridge-v1",
        "status": "PROVIDER_FREE_EXECUTION_EVIDENCE",
        "formal_scale_up_allowed": False,
        "models": sorted(VALID_PARENT_MODELS | {REPLACEMENT_MODEL}),
        "seeds": list(SEEDS),
        "executors": {
            "point_hazard": "repository fixed CEM-MPC",
            "safety_gym_goal": "deterministic six-value headless dynamics",
            "unknown_policy": "no-op task failure",
        },
        "execution_lookup": executed,
        "summary": summary,
        "ambiguous_mapping_STC_range": max(stc_values) - min(stc_values),
        "mapped_rows": len(mapped),
        "provider_calls": 0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_write(OUTPUT_DIR / "RESULTS.json", result)
    (OUTPUT_DIR / "REPORT.md").write_text(
        "# Interface-contract execution bridge\n\n"
        "- Provider calls: 0\n"
        "- PointHazard: repository fixed CEM-MPC\n"
        "- Safety-Gym: deterministic headless six-value dynamics contract\n"
        "- Unknown action: no-op task failure\n"
        f"- Ambiguous-mapping executed STC range: {result['ambiguous_mapping_STC_range']:.3f}\n\n"
        "```json\n"
        + json.dumps(summary["ambiguous_by_environment"], indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
