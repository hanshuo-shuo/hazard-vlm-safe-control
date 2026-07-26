#!/usr/bin/env python3
"""Provider-free analysis of cached interface-contract responses and executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def analyze(value: Mapping[str, Any]) -> dict[str, Any]:
    equivalent = list(value.get("equivalent_rows", []))
    equivalent_execution = list(value.get("equivalent_execution_rows", []))
    ambiguous_execution = list(value.get("ambiguous_execution_rows", []))
    ranking_rows = list(value.get("ranking_rows", equivalent_execution))
    return {
        "schema_version": "interface-contract-analysis-v1",
        "provider_calls": 0,
        "parse_consistency": parse_consistency(equivalent),
        "canonical_semantic_consistency": canonical_semantic_consistency(equivalent),
        "grounding_consistency": grounding_consistency(equivalent),
        "physical_action_IEC": physical_action_iec(equivalent),
        "CISR_EQ": cisr_eq(equivalent_execution) if equivalent_execution else None,
        "CISR_MAP": cisr_map(ambiguous_execution) if ambiguous_execution else None,
        "ranking_stability_envelope": (
            ranking_stability_envelope(ranking_rows) if ranking_rows else None
        ),
        "outcomes": {
            "equivalent": outcome_rates(equivalent_execution) if equivalent_execution else None,
            "ambiguous_mappings": outcome_rates(ambiguous_execution) if ambiguous_execution else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
