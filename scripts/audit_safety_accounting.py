#!/usr/bin/env python3
"""Recompute accounting from immutable old records, without any simulator/model."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c3_safe.artifacts import file_sha, finish_run, new_run, write_json, run_cli
from evaluation.outcomes import OUTCOME_SCHEMA_VERSION, episode_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/interface_contract_scout_native_analysis/PRIMARY_EXECUTIONS.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_hash = file_sha(args.source)
    rows = json.loads(args.source.read_text())
    revised = []
    for row in rows:
        # This is definition harmonization: retain the recorded goal/semantic
        # flags and original trajectories. Do not pretend to rerun native code.
        outcome = episode_outcome(success=row["task_success"], semantic_violation=row["semantic_violation"],
                                  native_costs=[row["native_cost_sum"]], termination_reason=row["termination_reason"])
        revised.append({"call_index": row["call_index"], "environment": row["environment"],
                        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                        "historical_STC": row["STC"], "semantic_safe_success": outcome.semantic_safe_success,
                        **outcome.to_dict()})
    by_env = {}
    for env in sorted({r["environment"] for r in revised}):
        group = [r for r in revised if r["environment"] == env]
        by_env[env] = {"n": len(group), "historical_STC_count": sum(r["historical_STC"] for r in group),
                       "semantic_safe_success_count": sum(r["semantic_safe_success"] for r in group),
                       "STC_count": sum(r["STC"] for r in group)}
    output = new_run(args.output)
    write_json(output / "RECOUNT.json", revised)
    summary = {"status": "POSTHOC_DEFINITION_HARMONIZATION_NOT_NEW_EXPERIMENT",
               "source": str(args.source.relative_to(ROOT)) if args.source.is_relative_to(ROOT) else str(args.source),
               "source_sha256": source_hash, "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
               "native_safety_event_definition": "positive native cost or hazard termination", "by_environment": by_env}
    finish_run(output, summary, source_paths=["scripts/audit_safety_accounting.py"])
    assert file_sha(args.source) == source_hash
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_cli(main)
