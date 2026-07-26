#!/usr/bin/env python3
"""Run the frozen 120-call scout through the fail-closed paid gateway."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.interface_contracts import parse_formal_contract_response
from evaluation.paid_provider_gateway import PaidCallLedger, PaidProviderGateway, ProviderRequest
from evaluation.scout_authorization import load_scout_manifest, provider_request_from_row


API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
CONFIG = ROOT / "configs" / "interface_contract_scout_manifest.json"
PREFLIGHT = ROOT / "results" / "interface_contract_scout_preflight"
SOURCE = ROOT / "results" / "interface_contract_provider_free_dry_run"
OUTPUT = ROOT / "results" / "interface_contract_scout_paid"
AUTHORIZATION_PHRASE = "I_AUTHORIZE_FROZEN_120_CALL_SCOUT"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def git_release() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout
    return {"git_sha": commit, "git_dirty": bool(status.strip())}


def pilot_like_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "interface-contract-pilot-manifest-v1",
        "protocol_id": config["protocol_id"],
        "status": "AUTHORIZED",
        "provider_calls_enabled": True,
        "paid_budget": {
            "allowed_paid_seeds": [20, 21, 22, 23, 24],
            "maximum_distinct_paid_seeds_per_model": 5,
            "maximum_new_provider_calls_per_model": 480,
            "maximum_attempts_per_request": 3,
            "maximum_retries_per_request": 2,
            "maximum_total_provider_attempts_per_model": 1440,
            "maximum_spend_usd": float(config["maximum_spend_usd"]),
        },
        "models": [
            {**dict(model), "eligibility": "ELIGIBLE"}
            for model in config["models"]
        ],
    }


def key_audit(key: str, maximum_spend: float) -> dict[str, Any]:
    response = requests.get(
        KEY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30
    )
    response.raise_for_status()
    value = response.json()["data"]
    limit = value.get("limit")
    remaining = value.get("limit_remaining")
    if limit is None or float(limit) > maximum_spend + 1e-9:
        raise PermissionError(
            f"provider-side key limit must be <= USD {maximum_spend:.2f}"
        )
    if remaining is None or float(remaining) <= 0:
        raise PermissionError("provider-side scout key has no remaining budget")
    return {
        "checked_at": utc_now(),
        "limit_usd": float(limit),
        "limit_remaining_usd": float(remaining),
        "usage_usd": float(value.get("usage") or 0.0),
        "is_free_tier": bool(value.get("is_free_tier", False)),
        "expires_at": value.get("expires_at"),
    }


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping)
        )
    return str(content or "")


class OpenRouterTransport:
    def __init__(self, key: str, rows_by_request: Mapping[str, Mapping[str, Any]]) -> None:
        self.key = key
        self.rows_by_request = rows_by_request

    def __call__(self, request: ProviderRequest, attempt: int) -> Mapping[str, Any]:
        row = self.rows_by_request[request.sha256]
        payload = {
            "model": request.provider_model,
            **dict(request.parameters),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt_bytes.decode("utf-8")},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(request.image_bytes).decode("ascii")
                        },
                    },
                ],
            }],
        }
        started = time.monotonic()
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "X-Title": "hazard-interface-contract-120-call-scout",
            },
            json=payload,
            timeout=120,
        )
        if response.status_code == 408:
            raise TimeoutError("OpenRouter HTTP 408")
        if response.status_code in {429, 500, 502, 503, 504}:
            raise RuntimeError(f"retryable OpenRouter HTTP {response.status_code}")
        response.raise_for_status()
        body = response.json()
        content = flatten_content(body["choices"][0]["message"]["content"])
        return {
            "provider_model": body.get("model"),
            "provider_endpoint": body.get("provider"),
            "expected_provider_endpoint": row["provider_endpoint_slug"],
            "raw_response": content,
            "usage": body.get("usage", {}),
            "provider_request_id": body.get("id"),
            "latency_seconds": time.monotonic() - started,
            "transport_attempt": attempt,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full", "cache-only"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--authorization", default="")
    parser.add_argument("--operator", default="user-authorized-in-chat")
    args = parser.parse_args()
    allow_provider = args.stage != "cache-only"
    if allow_provider and args.authorization != AUTHORIZATION_PHRASE:
        raise SystemExit("exact paid authorization phrase is required")

    config = load_scout_manifest(CONFIG)
    release = git_release()
    if allow_provider and release["git_dirty"]:
        raise SystemExit("paid scout requires a clean release commit")
    rows = json.loads((PREFLIGHT / "SCOUT_CALL_MATRIX.json").read_text(encoding="utf-8"))
    allowlist = json.loads((PREFLIGHT / "REQUEST_ALLOWLIST.json").read_text(encoding="utf-8"))
    prompt_dir = SOURCE / "inputs" / "prompts"
    image_dir = SOURCE / "inputs" / "images"
    requests_by_hash: dict[str, ProviderRequest] = {}
    rows_by_request: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        request = provider_request_from_row(row, prompt_dir=prompt_dir, image_dir=image_dir)
        requests_by_hash[request.sha256] = request
        rows_by_request[request.sha256] = row

    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if allow_provider and not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    key_evidence = key_audit(key, float(config["maximum_spend_usd"])) if allow_provider else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    authorization = {
        "schema_version": "interface-contract-scout-runtime-authorization-v1",
        "authorized": allow_provider,
        "authorization_source": "explicit user instruction in conversation",
        "authorized_at": utc_now(),
        "operator": args.operator,
        "stage": args.stage,
        "release": release,
        "design_manifest_sha256": sha(CONFIG),
        "call_matrix_sha256": sha(PREFLIGHT / "SCOUT_CALL_MATRIX.json"),
        "request_allowlist_sha256": sha(PREFLIGHT / "REQUEST_ALLOWLIST.json"),
        "key_audit": key_evidence,
    }
    write_json(output / "AUTHORIZATION.json", authorization)

    paid_manifest = pilot_like_manifest(config)
    ledger = PaidCallLedger(output / "PAID_CALL_LEDGER.json", paid_manifest)
    gateway = PaidProviderGateway(
        manifest=paid_manifest,
        ledger=ledger,
        cache_dir=output / "cache",
        allow_provider_requests=allow_provider,
        transport=OpenRouterTransport(key, rows_by_request) if allow_provider else None,
        authorized_request_sha256=allowlist["by_model"],
        response_validator=lambda request, response: (
            (_ for _ in ()).throw(ValueError("frozen endpoint mismatch"))
            if str(response.get("provider_endpoint", "")).lower()
            != str(rows_by_request[request.sha256]["provider_endpoint_slug"]).lower()
            else (_ for _ in ()).throw(ValueError("compatibility smoke parse failure"))
            if request.compatibility_smoke
            and parse_formal_contract_response(
                str(rows_by_request[request.sha256]["contract_id"]),
                str(response["raw_response"]),
            )["parse_status"] != "ok"
            else None
        ),
    )
    selected = (
        [row for row in rows if row["compatibility_smoke"]]
        if args.stage == "smoke"
        else rows
    )
    completed = 0
    cache_hits = 0
    failed: list[dict[str, Any]] = []
    for row in selected:
        request = requests_by_hash[row["provider_request_sha256"]]
        try:
            record = gateway.call(request)
            parsed = parse_formal_contract_response(row["contract_id"], record["raw_response"])
            saved = {
                **dict(row),
                "response_sha256": hashlib.sha256(record["raw_response"].encode()).hexdigest(),
                "parsed": parsed,
                "provider": record,
            }
            write_json(output / "rows" / f"{int(row['call_index']):03d}.json", saved)
            completed += 1
            cache_hits += record.get("call_origin") == "cache_replay"
            spend = sum(
                float(json.loads(path.read_text()).get("usage", {}).get("cost") or 0.0)
                for path in (output / "cache").glob("*.json")
            )
            if spend > float(config["maximum_spend_usd"]) + 1e-9:
                raise RuntimeError("observed spend exceeds frozen USD cap")
            print(json.dumps({
                "call_index": row["call_index"],
                "model": row["model_budget_id"],
                "stage": args.stage,
                "parse_status": parsed["parse_status"],
                "call_origin": record["call_origin"],
                "spend_usd": spend,
            }))
        except Exception as exc:
            failed.append({
                "call_index": row["call_index"],
                "model_budget_id": row["model_budget_id"],
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            break
    audit = ledger.audit()
    cache_files = list((output / "cache").glob("*.json"))
    spend = sum(
        float(json.loads(path.read_text()).get("usage", {}).get("cost") or 0.0)
        for path in cache_files
    )
    result = {
        "schema_version": "interface-contract-scout-paid-run-v1",
        "status": "COMPLETE" if not failed and completed == len(selected) else "PARTIAL",
        "stage": args.stage,
        "selected_rows": len(selected),
        "completed_rows_this_execution": completed,
        "cache_hits_this_execution": cache_hits,
        "cached_formal_responses": len(cache_files),
        "spend_usd": spend,
        "maximum_spend_usd": config["maximum_spend_usd"],
        "ledger_audit": audit,
        "failed": failed,
    }
    write_json(output / "RESULTS.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
