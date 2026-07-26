#!/usr/bin/env python3
"""Build the exact provider-free 120-call interface-contract scout subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scout_authorization import (
    audit_scout_matrix,
    build_scout_matrix,
    load_scout_manifest,
)


DEFAULT_CONFIG = ROOT / "configs" / "interface_contract_scout_manifest.json"
DEFAULT_SOURCE = ROOT / "results" / "interface_contract_provider_free_dry_run"
DEFAULT_OUTPUT = ROOT / "results" / "interface_contract_scout_preflight"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(config_path: Path, source: Path, output: Path) -> dict[str, Any]:
    config = load_scout_manifest(config_path)
    source_matrix_path = source / "CALL_MATRIX.json"
    source_rows = json.loads(source_matrix_path.read_text(encoding="utf-8"))
    rows = build_scout_matrix(
        config,
        source_rows,
        prompt_dir=source / "inputs" / "prompts",
        image_dir=source / "inputs" / "images",
    )
    audit = audit_scout_matrix(config, rows)
    allowlist = {
        "schema_version": "interface-contract-scout-request-allowlist-v1",
        "status": "BLOCKED",
        "provider_calls_enabled": False,
        "request_count": len(rows),
        "by_model": {
            model["model_budget_id"]: [
                row["provider_request_sha256"]
                for row in rows
                if row["model_budget_id"] == model["model_budget_id"]
            ]
            for model in config["models"]
        },
    }
    _write_json(output / "SCOUT_CALL_MATRIX.json", rows)
    _write_json(output / "REQUEST_ALLOWLIST.json", allowlist)
    _write_json(output / "MATRIX_AUDIT.json", audit)
    result = {
        "schema_version": "interface-contract-scout-preflight-v1",
        "status": "BLOCKED / PROVIDER_FREE",
        "provider_calls_enabled": False,
        "provider_calls": 0,
        "provider_attempts": 0,
        "model_count": 2,
        "calls_per_model": 60,
        "call_matrix_rows": 120,
        "maximum_spend_usd": config["maximum_spend_usd"],
        "source": {
            "authorization_manifest": {
                "path": str(config_path.relative_to(ROOT)),
                "sha256": _sha(config_path),
            },
            "full_call_matrix": {
                "path": str(source_matrix_path.relative_to(ROOT)),
                "sha256": _sha(source_matrix_path),
            },
        },
        "artifacts": {
            name: {"path": name, "sha256": _sha(output / name)}
            for name in (
                "SCOUT_CALL_MATRIX.json", "REQUEST_ALLOWLIST.json", "MATRIX_AUDIT.json"
            )
        },
        "authorization_blockers": config["authorization_blockers"],
        "checks": {
            **audit["checks"],
            "provider_calls_are_zero": True,
            "provider_attempts_are_zero": True,
            "authorization_remains_blocked": True,
        },
    }
    _write_json(output / "MANIFEST.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.config.resolve(), args.source.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
