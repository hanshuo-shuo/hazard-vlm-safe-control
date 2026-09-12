#!/usr/bin/env python3
"""Apply the fixed keep/discard rule to two same-tier development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.exp01b_r2.prepare_dev import load_contract  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _metric(result: Mapping[str, Any], split: str, metric: str) -> float:
    return float(result["aggregate"]["splits"][split]["factorized_no_cf"][metric])


def _paired_primary_bootstrap(
    baseline_path: Path,
    candidate_path: Path,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    resamples: int,
    seed: int = 20260901,
) -> dict[str, Any]:
    baseline_arrays = np.load(baseline_path.with_name("PER_SCENE_METRICS.npz"))
    candidate_arrays = np.load(candidate_path.with_name("PER_SCENE_METRICS.npz"))
    common_seeds = sorted(set(baseline["per_seed"]) & set(candidate["per_seed"]), key=int)
    reductions = []
    for model_seed in common_seeds:
        key = f"seed_{model_seed}__appearance_ood__factorized_no_cf__regret"
        if key not in baseline_arrays or key not in candidate_arrays:
            raise KeyError(f"missing paired metric array: {key}")
        left, right = baseline_arrays[key], candidate_arrays[key]
        if left.shape != right.shape:
            raise ValueError(f"paired metric shape changed for seed {model_seed}")
        reductions.append(left - right)
    per_scene = np.mean(np.stack(reductions, axis=0), axis=0)
    rng = np.random.default_rng(seed)
    sampled = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        draw = rng.integers(0, len(per_scene), size=len(per_scene))
        sampled[index] = float(per_scene[draw].mean())
    return {
        "mean_reduction": float(per_scene.mean()),
        "paired_scene_bootstrap_ci95": [
            float(np.percentile(sampled, 2.5)),
            float(np.percentile(sampled, 97.5)),
        ],
        "scene_count": len(per_scene),
        "model_seeds": [int(value) for value in common_seeds],
    }


def compare_results(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_path: Path | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    config, _ = load_contract()
    guard = config["guardrails"]
    objective = config["objective"]
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    if baseline.get("status") != "COMPLETE" or candidate.get("status") != "COMPLETE":
        reasons.append("both runs must be COMPLETE")
    if baseline.get("tier") != candidate.get("tier"):
        reasons.append("runs use different tiers")
    tier = str(candidate.get("tier"))
    if tier == "smoke":
        reasons.append("smoke runs are never eligible for keep")
    if baseline.get("dataset_audit") != candidate.get("dataset_audit"):
        reasons.append("dataset audit differs")
    if baseline.get("contract", {}).get("config_sha256") != candidate.get("contract", {}).get("config_sha256"):
        reasons.append("development contract differs")

    baseline_primary = _metric(baseline, "appearance_ood", "regret")
    candidate_primary = _metric(candidate, "appearance_ood", "regret")
    primary_reduction = baseline_primary - candidate_primary
    minimum = float(objective["minimum_quick_improvement"]) if tier == "quick" else 0.0
    checks["primary"] = {
        "baseline": baseline_primary,
        "candidate": candidate_primary,
        "reduction": primary_reduction,
        "minimum_required": minimum,
    }
    if primary_reduction <= minimum:
        reasons.append("appearance-OOD regret did not improve enough")

    scalar_guardrails = [
        (
            "appearance_false_safe",
            _metric(candidate, "appearance_ood", "false_safe") - _metric(baseline, "appearance_ood", "false_safe"),
            float(guard["false_safe_absolute_regression_max"]),
        ),
        (
            "joint_false_safe",
            _metric(candidate, "joint_ood", "false_safe") - _metric(baseline, "joint_ood", "false_safe"),
            float(guard["false_safe_absolute_regression_max"]),
        ),
        (
            "appearance_pair_ranking",
            _metric(baseline, "appearance_ood", "pair_ranking") - _metric(candidate, "appearance_ood", "pair_ranking"),
            float(guard["pair_ranking_absolute_regression_max"]),
        ),
        (
            "joint_pair_ranking",
            _metric(baseline, "joint_ood", "pair_ranking") - _metric(candidate, "joint_ood", "pair_ranking"),
            float(guard["pair_ranking_absolute_regression_max"]),
        ),
        (
            "appearance_field_miou",
            _metric(baseline, "appearance_ood", "field_miou") - _metric(candidate, "appearance_ood", "field_miou"),
            float(guard["field_miou_absolute_regression_max"]),
        ),
        (
            "joint_field_miou",
            _metric(baseline, "joint_ood", "field_miou") - _metric(candidate, "joint_ood", "field_miou"),
            float(guard["field_miou_absolute_regression_max"]),
        ),
        (
            "joint_regret",
            _metric(candidate, "joint_ood", "regret") - _metric(baseline, "joint_ood", "regret"),
            float(guard["joint_ood_regret_absolute_regression_max"]),
        ),
        (
            "iid_regret",
            _metric(candidate, "iid", "regret") - _metric(baseline, "iid", "regret"),
            float(guard["iid_regret_absolute_regression_max"]),
        ),
    ]
    checks["non_regression"] = {}
    for name, regression, maximum in scalar_guardrails:
        passed = regression <= maximum + 1e-12
        checks["non_regression"][name] = {
            "regression": regression,
            "maximum": maximum,
            "passed": passed,
        }
        if not passed:
            reasons.append(f"guardrail regression: {name}")

    baseline_shortcuts = baseline["aggregate"]["shortcuts"]
    candidate_shortcuts = candidate["aggregate"]["shortcuts"]
    shuffle_floor = max(
        float(guard["image_shuffle_pair_delta_min"]),
        float(baseline_shortcuts["min_image_shuffle_pair_delta"])
        - float(guard["image_shuffle_delta_regression_max"]),
    )
    shortcut_checks = {
        "image_shuffle": (
            float(candidate_shortcuts["min_image_shuffle_pair_delta"]),
            shuffle_floor,
        ),
        "terrain_erasure": (
            float(candidate_shortcuts["min_terrain_erasure_positive_rate"]),
            float(guard["terrain_erasure_positive_rate_min"]),
        ),
        "irrelevant_swap": (
            float(candidate_shortcuts["min_irrelevant_swap_accuracy"]),
            float(guard["irrelevant_swap_accuracy_min"]),
        ),
    }
    checks["shortcuts"] = {}
    for name, (value, floor) in shortcut_checks.items():
        passed = value + 1e-12 >= floor
        checks["shortcuts"][name] = {"value": value, "minimum": floor, "passed": passed}
        if not passed:
            reasons.append(f"shortcut guard failed: {name}")

    parameter_ratio = float(candidate["candidate_parameter_count"]) / max(
        1.0, float(baseline["candidate_parameter_count"])
    )
    runtime_ratio = float(candidate["runtime"]["elapsed_seconds"]) / max(
        1e-9, float(baseline["runtime"]["elapsed_seconds"])
    )
    checks["resources"] = {
        "parameter_ratio": parameter_ratio,
        "parameter_ratio_max": float(guard["parameter_multiplier_max"]),
        "runtime_ratio": runtime_ratio,
        "runtime_ratio_max": float(guard["runtime_multiplier_max"]),
    }
    if parameter_ratio > float(guard["parameter_multiplier_max"]):
        reasons.append("parameter budget exceeded")
    if runtime_ratio > float(guard["runtime_multiplier_max"]):
        reasons.append("runtime budget exceeded")

    seed_wins = []
    for seed in sorted(set(baseline["per_seed"]) & set(candidate["per_seed"]), key=int):
        left = float(baseline["per_seed"][seed]["appearance_ood"]["summary"]["factorized_no_cf"]["regret"])
        right = float(candidate["per_seed"][seed]["appearance_ood"]["summary"]["factorized_no_cf"]["regret"])
        seed_wins.append(right < left)
    seed_win_fraction = float(np.mean(seed_wins)) if seed_wins else 0.0
    checks["seed_win_fraction"] = seed_win_fraction
    if tier == "confirm" and seed_win_fraction + 1e-12 < float(objective["confirm_seed_win_fraction_min"]):
        reasons.append("candidate does not improve enough confirm seeds")

    bootstrap = None
    if tier == "confirm":
        if baseline_path is None or candidate_path is None:
            reasons.append("confirm comparison requires result paths for paired bootstrap")
        else:
            bootstrap = _paired_primary_bootstrap(
                baseline_path,
                candidate_path,
                baseline,
                candidate,
                resamples=int(config["tiers"]["confirm"]["bootstrap_resamples"]),
            )
            if float(bootstrap["paired_scene_bootstrap_ci95"][0]) <= 0:
                reasons.append("confirm primary bootstrap lower bound is not positive")
    checks["paired_primary_bootstrap"] = bootstrap

    return {
        "schema_version": "exp01b-r2-comparison-v1",
        "experiment_id": "EXP-01B-R2-DEV",
        "status": "KEEP" if not reasons else "DISCARD",
        "tier": tier,
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "scientific_authority": False,
        "checks": checks,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    baseline_path, candidate_path = args.baseline.resolve(), args.candidate.resolve()
    result = compare_results(
        read_json(baseline_path),
        read_json(candidate_path),
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "KEEP" else 2)
