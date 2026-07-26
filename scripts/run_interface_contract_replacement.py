#!/usr/bin/env python3
"""Predeclared replacement for the invalid GPT-5-mini contract-audit arm.

The original arm is retained.  This runner changes only model identity, reuses
the exact prompts/images/seeds from the frozen v1 matrix, and writes to a
separate artifact directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_interface_contract_audit import (
    MODELS,
    SEEDS,
    ProviderClient,
    analyze,
    build_matrix,
    build_scenes,
    canonical,
    execute_row,
    json_write,
    load_dotenv,
    load_saved_rows,
    public_matrix_row,
    sha,
    utc_now,
)


REPLACED_MODEL = "openai/gpt-5-mini"
REPLACEMENT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
MAX_CALLS = 110
MAX_DISTINCT_SEEDS = 5
SCHEMA_VERSION = "embodied-interface-contract-model-replacement-v1"


def frozen_replacement_protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "EXPLORATORY_REPLACEMENT_ARM_ONLY",
        "formal_scale_up_allowed": False,
        "parent_protocol": "embodied-interface-contract-micro-pilot-v1",
        "reason": (
            "GPT-5-mini produced 102/110 empty or truncated outputs because the "
            "frozen max-token request was consumed by reasoning tokens."
        ),
        "replaced_model": REPLACED_MODEL,
        "replacement_model": REPLACEMENT_MODEL,
        "only_changed_factor": "model_identity",
        "prompts_images_seeds_unchanged": True,
        "seeds": list(SEEDS),
        "maximum_distinct_seeds": MAX_DISTINCT_SEEDS,
        "maximum_calls": MAX_CALLS,
        "maximum_spend_usd": 0.5,
        "claim_boundary": "Replacement plus parent remain five-seed go/no-go evidence only.",
    }


def replacement_matrix() -> list[dict[str, Any]]:
    source = [
        row
        for row in build_matrix(build_scenes())
        if row["model"] == MODELS[0]
    ]
    rows = [{**row, "call_index": index, "model": REPLACEMENT_MODEL} for index, row in enumerate(source)]
    if len(rows) != MAX_CALLS:
        raise AssertionError(f"expected {MAX_CALLS} replacement calls")
    if {row["seed"] for row in rows} != set(SEEDS):
        raise AssertionError("replacement arm must use exactly the frozen five seeds")
    return rows


def write_report(output: Path, result: dict[str, Any]) -> None:
    summary = result.get("summary")
    if summary is None:
        return
    value = {
        "go_no_go": summary["go_no_go"],
        "paired_consistency": summary["contracts"]["paired_consistency"],
        "candidate_robustness": summary["candidates"]["by_variant"],
    }
    (output / "REPORT.md").write_text(
        "# Interface-contract replacement arm\n\n"
        f"- Generated: {utc_now()}\n"
        f"- Invalid arm retained: `{REPLACED_MODEL}`\n"
        f"- Replacement: `{REPLACEMENT_MODEL}`\n"
        "- Same prompts, images, and seeds: yes\n"
        "- Status: exploratory five-seed evidence only\n\n"
        "```json\n"
        + json.dumps(value, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/interface_contract_mistral_replacement"),
    )
    parser.add_argument("--allow-provider-requests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.allow_provider_requests:
        raise SystemExit(
            "legacy replacement provider path is permanently disabled; "
            "new paid calls must use evaluation.paid_provider_gateway"
        )
    if args.allow_provider_requests and (args.dry_run or args.analyze_only):
        raise SystemExit("provider requests cannot be combined with dry/analyze-only mode")
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be in [1,8]")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = frozen_replacement_protocol()
    json_write(args.output / "FROZEN_PROTOCOL.json", protocol)
    if args.analyze_only:
        rows = load_saved_rows(args.output)
        if len(rows) != MAX_CALLS:
            raise SystemExit(f"analyze-only requires {MAX_CALLS} rows, got {len(rows)}")
        result_path = args.output / "RESULTS.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        result.update({"status": "COMPLETE", "completed_calls": len(rows), "summary": analyze(rows)})
        json_write(result_path, result)
        write_report(args.output, result)
        return 0
    matrix = replacement_matrix()
    json_write(args.output / "CALL_MATRIX.json", [public_matrix_row(row) for row in matrix])
    if args.dry_run:
        json_write(
            args.output / "DRY_RUN.json",
            {
                "model": REPLACEMENT_MODEL,
                "calls": len(matrix),
                "seeds": list(SEEDS),
                "provider_calls": 0,
                "protocol_sha256": sha(canonical(protocol)),
            },
        )
        return 0
    load_dotenv(ROOT / ".env")
    client = ProviderClient(
        os.environ.get("OPENROUTER_API_KEY", ""),
        args.output / "cache",
        args.allow_provider_requests,
    )
    failed = []
    for offset in range(0, len(matrix), args.workers):
        batch = matrix[offset : offset + args.workers]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [(row, executor.submit(execute_row, client, row)) for row in batch]
            for row, future in futures:
                try:
                    saved = future.result()
                    json_write(args.output / "rows" / f"{row['call_index']:03d}.json", saved)
                except Exception as exc:
                    failed.append(
                        {
                            "call_index": row["call_index"],
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
        if failed:
            break
    rows = load_saved_rows(args.output)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE" if len(rows) == MAX_CALLS else "PARTIAL",
        "formal_scale_up_allowed": False,
        "protocol_sha256": sha(canonical(protocol)),
        "completed_calls": len(rows),
        "provider_calls_this_execution": client.calls,
        "cache_hits_this_execution": client.cache_hits,
        "distinct_provider_seeds_by_model": {
            model: sorted(seeds) for model, seeds in client.distinct_seeds_by_model.items()
        },
        "spend_usd_this_execution": client.spend,
        "failed": failed,
        "summary": analyze(rows) if len(rows) == MAX_CALLS else None,
    }
    json_write(args.output / "RESULTS.json", result)
    write_report(args.output, result)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
