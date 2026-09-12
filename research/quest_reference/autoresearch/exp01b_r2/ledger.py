#!/usr/bin/env python3
"""Append one development run to the untracked autoresearch TSV ledger."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = Path(__file__).with_name("results.tsv")
HEADER = [
    "timestamp_utc",
    "run_id",
    "tier",
    "commit",
    "candidate_sha256",
    "appearance_regret",
    "worst_false_safe",
    "worst_pair_ranking",
    "worst_field_miou",
    "runtime_seconds",
    "peak_vram_mb",
    "status",
    "description",
]


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("result must be a JSON object")
    return value


def append_result(
    ledger: Path,
    result: dict[str, Any],
    *,
    status: str,
    description: str,
) -> None:
    if any(character in description for character in ("\t", "\n", "\r")):
        raise ValueError("description cannot contain tabs or newlines")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    exists = ledger.exists()
    if exists:
        with ledger.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            current = next(reader, None)
        if current != HEADER:
            raise ValueError("existing ledger header does not match contract")
    objective = result["aggregate"]["objective"]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": result["run_id"],
        "tier": result["tier"],
        "commit": result.get("git", {}).get("commit") or "unknown",
        "candidate_sha256": result["candidate_sha256"],
        "appearance_regret": f"{objective['appearance_ood_regret']:.9f}",
        "worst_false_safe": f"{objective['worst_dev_false_safe']:.9f}",
        "worst_pair_ranking": f"{objective['worst_dev_pair_ranking']:.9f}",
        "worst_field_miou": f"{objective['worst_dev_field_miou']:.9f}",
        "runtime_seconds": f"{float(result['runtime']['elapsed_seconds']):.3f}",
        "peak_vram_mb": f"{float(result['runtime']['peak_vram_mb']):.1f}",
        "status": status,
        "description": description,
    }
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--status", choices=("baseline", "keep", "discard", "crash"), required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    append_result(
        args.ledger.resolve(),
        _read_result(args.result.resolve()),
        status=args.status,
        description=args.description,
    )
    print(f"appended {args.status}: {args.ledger.resolve()}")
