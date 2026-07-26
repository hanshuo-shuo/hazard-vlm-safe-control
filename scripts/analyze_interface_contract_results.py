#!/usr/bin/env python3
"""Combine valid interface-contract arms without making provider calls."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.interface_contracts import (
    CANDIDATE_VARIANTS,
    CONTRACT_IDS,
    expected_action,
    summarize_interface_audit,
)
from scripts.run_interface_contract_audit import json_write, sha
from scripts.run_interface_contract_replacement import REPLACEMENT_MODEL


PARENT_DIR = Path("results/interface_contract_micro_pilot")
REPLACEMENT_DIR = Path("results/interface_contract_mistral_replacement")
OUTPUT_DIR = Path("results/interface_contract_combined_analysis")
EXECUTION_DIR = Path("results/interface_contract_execution_bridge")
INVALID_MODEL = "openai/gpt-5-mini"
VALID_PARENT_MODELS = {
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3-vl-30b-a3b-instruct",
}


def load_rows(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "rows").glob("*.json"))
    ]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "wilson_95": None}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (row["model"], row["environment"], int(row["seed"]), row["capability"])


def paired_contract_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {
        "field_order": ("action_structured", "action_structured_field_reversed"),
        "structured_vs_free_text": ("action_structured", "action_free_text"),
        "constraint_label_polarity": ("constraint_positive", "constraint_negative"),
        "compatibility_label_polarity": ("compatibility_positive", "compatibility_negative"),
        "constraint_vs_compatibility_wording": ("constraint_positive", "compatibility_positive"),
        "explicit_vs_ambiguous": ("constraint_positive", "ambiguous_applicability"),
    }
    index: dict[tuple[str, str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        index[pair_key(row)][row["contract_id"]] = row
    result = {}
    for name, (left, right) in pairs.items():
        outcomes = [
            values[left]["parsed"]["action"] == values[right]["parsed"]["action"]
            for values in index.values()
            if left in values and right in values
        ]
        result[name] = wilson(sum(outcomes), len(outcomes))
    return result


def candidate_consistency_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = {
        (row["model"], row["environment"], int(row["seed"])): row
        for row in rows
        if row["variant"] == "candidate_baseline"
    }
    result = {}
    for variant in CANDIDATE_VARIANTS:
        outcomes = []
        for row in rows:
            if row["variant"] != variant:
                continue
            reference = baseline[(row["model"], row["environment"], int(row["seed"]))]
            outcomes.append(
                row["parsed"]["physical_choice"] == reference["parsed"]["physical_choice"]
            )
        result[variant] = wilson(sum(outcomes), len(outcomes))
    return result


def validate_replacement_identity(
    parent_rows: Sequence[Mapping[str, Any]], replacement_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    reference = {}
    for row in parent_rows:
        if row["model"] not in VALID_PARENT_MODELS:
            continue
        intervention = row.get("contract_id", row.get("variant"))
        key = (row["task"], row["environment"], int(row["seed"]), intervention)
        reference.setdefault(key, (row["prompt_sha256"], row["image_sha256"]))
    comparisons = []
    for row in replacement_rows:
        intervention = row.get("contract_id", row.get("variant"))
        key = (row["task"], row["environment"], int(row["seed"]), intervention)
        comparisons.append(
            reference.get(key) == (row["prompt_sha256"], row["image_sha256"])
        )
    return {
        "replacement_rows": len(comparisons),
        "all_prompt_and_image_hashes_match_parent": bool(comparisons and all(comparisons)),
        "mismatches": len(comparisons) - sum(comparisons),
    }


def invalid_arm_diagnostic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    invalid = [row for row in rows if row["model"] == INVALID_MODEL]
    empty = sum(row["provider_record"]["raw_response"] == "None" for row in invalid)
    parse_ok = sum(row["parsed"]["parse_status"] == "ok" for row in invalid)
    truncated = len(invalid) - empty - parse_ok
    return {
        "model": INVALID_MODEL,
        "rows": len(invalid),
        "parse_ok": parse_ok,
        "empty_content": empty,
        "other_truncated_or_invalid": truncated,
        "valid_for_capability_ranking": False,
        "retained_as_interface_compatibility_failure": True,
    }


def by_model_environment(
    contract_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = {}
    models = sorted({row["model"] for row in contract_rows})
    environments = sorted({row["environment"] for row in contract_rows})
    for model in models:
        for environment in environments:
            decision = [
                row for row in contract_rows
                if row["model"] == model and row["environment"] == environment
            ]
            waypoint = [
                row for row in candidate_rows
                if row["model"] == model and row["environment"] == environment
            ]
            key = f"{model}|{environment}"
            result[key] = {
                "contract_action_accuracy": {
                    contract_id: sum(
                        row["parsed"]["parse_status"] == "ok"
                        and row["parsed"]["action"] == expected_action(row["capability"])
                        for row in decision if row["contract_id"] == contract_id
                    ) / 5
                    for contract_id in CONTRACT_IDS
                },
                "candidate_physical_consistency": candidate_consistency_counts(waypoint),
            }
    return result


def result_hash(directory: Path, name: str) -> str:
    return sha((directory / name).read_bytes())


def main() -> int:
    parent_rows = load_rows(PARENT_DIR)
    replacement_rows = load_rows(REPLACEMENT_DIR)
    if len(parent_rows) != 330 or len(replacement_rows) != 110:
        raise SystemExit("combined analysis requires complete 330-row parent and 110-row replacement")
    replacement_check = validate_replacement_identity(parent_rows, replacement_rows)
    if not replacement_check["all_prompt_and_image_hashes_match_parent"]:
        raise SystemExit("replacement prompt/image identity check failed")
    valid_rows = [row for row in parent_rows if row["model"] in VALID_PARENT_MODELS] + replacement_rows
    contract_rows = [row for row in valid_rows if row["task"] == "semantic_route_decision"]
    candidate_rows = [row for row in valid_rows if row["task"] == "candidate_waypoint_selection"]
    summary = summarize_interface_audit(contract_rows, candidate_rows)
    parent_result = json.loads((PARENT_DIR / "RESULTS.json").read_text(encoding="utf-8"))
    replacement_result = json.loads((REPLACEMENT_DIR / "RESULTS.json").read_text(encoding="utf-8"))
    execution_result = json.loads((EXECUTION_DIR / "RESULTS.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "embodied-interface-contract-combined-analysis-v1",
        "status": "COMPLETE_EXPLORATORY_ANALYSIS",
        "formal_scale_up_allowed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid_models": sorted(VALID_PARENT_MODELS | {REPLACEMENT_MODEL}),
        "invalid_arm": invalid_arm_diagnostic(parent_rows),
        "replacement_identity_check": replacement_check,
        "provider_budget": {
            "total_calls_including_invalid_arm": (
                parent_result["provider_calls_this_execution"]
                + replacement_result["provider_calls_this_execution"]
            ),
            "valid_science_rows": len(valid_rows),
            "maximum_distinct_seeds_per_model": 5,
            "seeds": [20, 21, 22, 23, 24],
            "total_spend_usd": (
                parent_result["spend_usd_this_execution"]
                + replacement_result["spend_usd_this_execution"]
            ),
        },
        "provenance": {
            "parent_protocol_sha256": result_hash(PARENT_DIR, "FROZEN_PROTOCOL.json"),
            "parent_results_sha256": result_hash(PARENT_DIR, "RESULTS.json"),
            "replacement_protocol_sha256": result_hash(REPLACEMENT_DIR, "FROZEN_PROTOCOL.json"),
            "replacement_results_sha256": result_hash(REPLACEMENT_DIR, "RESULTS.json"),
            "execution_bridge_results_sha256": result_hash(EXECUTION_DIR, "RESULTS.json"),
        },
        "summary": summary,
        "execution_bridge": {
            "executors": execution_result["executors"],
            "ambiguous_mapping_STC_range": execution_result["ambiguous_mapping_STC_range"],
            "ambiguous_by_environment": execution_result["summary"]["ambiguous_by_environment"],
            "provider_calls": execution_result["provider_calls"],
        },
        "exact_paired_counts": {
            "contracts": paired_contract_counts(contract_rows),
            "candidates": candidate_consistency_counts(candidate_rows),
        },
        "by_model_environment": by_model_environment(contract_rows, candidate_rows),
        "paper_assessment": {
            "mainline_question": (
                "Are embodied safety evaluations measuring model capability, or artifacts "
                "of the interface contract between perception, reasoning, and control?"
            ),
            "working_title": (
                "The Contract Is the Benchmark: Interface-Conditioned Safety in Embodied VLM Evaluation"
            ),
            "positive_evidence": [
                "semantically equivalent field-order and output-mode contracts change actions",
                "marker IDs and candidate serialization change physical waypoint choices",
                "plausible applicability-to-planner mappings change executed STC",
                "a model-pair ranking reversal occurs across interface conditions",
            ],
            "blocking_evidence": [
                "semantic-contract sensitivity did not pass the predeclared two-environment replication gate",
                "Mistral is stable on all contract pairs, so the effect is model-dependent rather than universal",
                "Safety-Gym still uses a headless render/dynamics contract rather than the native backend",
                "candidate-interface choices have not yet been executed in the contact/manipulation task",
                "five scene seeds are adequate only for go/no-go discovery",
            ],
            "current_decision": "PROMISING_METHOD_AND_PHENOMENON_BUT_NOT_YET_AN_ICLR_MAIN_RESULT",
        },
        "provider_calls_for_this_analysis": 0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_write(OUTPUT_DIR / "RESULTS.json", result)
    paired = result["exact_paired_counts"]["contracts"]
    candidates = result["exact_paired_counts"]["candidates"]
    mapping = execution_result["summary"]["by_contract_mapping"]["ambiguous_applicability"]
    report = f"""# Interface-conditioned embodied safety: combined five-seed assessment

- Status: exploratory go/no-go evidence; not a formal paper result
- Valid models: {', '.join(result['valid_models'])}
- Tasks: semantic route decision and candidate waypoint selection
- Environments: PointHazard and Safety-Gym headless contract render
- Seeds per model: 5 (`20–24`)
- Valid rows: {len(valid_rows)}
- Total calls including retained invalid GPT arm: {result['provider_budget']['total_calls_including_invalid_arm']}
- Total spend: ${result['provider_budget']['total_spend_usd']:.6f}

## Verdict

**Promising method and phenomenon, but the ICLR main result is not recovered yet.**

The strongest current result is not that every contract perturbation breaks every
model. It is that physically identical embodied-safety evaluations have a
measurable *interface-conditioned uncertainty envelope*: equivalent output
contracts, marker identities, candidate serialization, and downstream label
mappings can change actions, physical choices, STC, and one pairwise model
ranking. The effect is heterogeneous across model and renderer, which is itself
important but fails the preregistered universal two-environment contract gate.

## Exact paired evidence

| Intervention | Consistent / total | Rate | Wilson 95% interval |
|---|---:|---:|---:|
"""
    for name, item in paired.items():
        low, high = item["wilson_95"]
        report += f"| {name} | {item['successes']} / {item['total']} | {item['rate']:.3f} | [{low:.3f}, {high:.3f}] |\n"
    report += "\n| Waypoint intervention | Same physical choice / total | Rate | Wilson 95% interval |\n|---|---:|---:|---:|\n"
    for name, item in candidates.items():
        low, high = item["wilson_95"]
        report += f"| {name} | {item['successes']} / {item['total']} | {item['rate']:.3f} | [{low:.3f}, {high:.3f}] |\n"
    report += f"""

## Downstream planner mapping

For the same ambiguous-applicability model outputs replayed through executable
controllers, aggregate STC changes from
`{mapping['applicable_means_terrain_compatible']['STC']:.3f}` when
`applicable` is interpreted as terrain-compatible, to
`{mapping['action_authoritative']['STC']:.3f}` when the explicit action is
authoritative. The constraint interpretation reaches
`{mapping['applicable_means_constraint_applies']['STC']:.3f}`. PointHazard uses
the repository fixed CEM-MPC; the second arm uses deterministic Safety-Gym
six-value headless dynamics. `unknown` is an explicit no-op task failure.

## Ranking stability

Pairwise reversals: `{json.dumps(summary['ranking_stability']['model_pairs_with_rank_reversal'])}`.
Mistral is stable across all matched contract pairs; the observed reversal is
between Gemini Flash-Lite and Qwen3-VL. This supports a ranking-instability
claim for the evaluated pair, not a claim that every leaderboard must reverse.

## Invalid arm handling

GPT-5-mini is excluded from capability ranking: only
`{result['invalid_arm']['parse_ok']}/{result['invalid_arm']['rows']}` rows parsed,
with `{result['invalid_arm']['empty_content']}` empty contents. The raw arm is
retained as an API/request-interface compatibility failure. Its replacement
changed only model identity; all 110 prompt and image hashes match the parent
matrix.

## Required next gate

1. Replace the headless Safety-Gym render with native RGB and actual dynamics.
2. Execute candidate selections, not only route labels, through the same
   controller; retain the explicit `unknown → stop` policy.
3. Add one manipulation/contact task in a genuinely independent environment.
4. Freeze a larger formal split only after the five-seed effect survives those
   bridges; do not expand the terminated marker-only story by itself.
5. Report contract envelopes and rank intervals, not a single privileged score.
"""
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
