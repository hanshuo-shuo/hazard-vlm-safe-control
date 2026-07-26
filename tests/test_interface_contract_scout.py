"""Acceptance tests for the blocked 120-call paid-scout preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.paid_provider_gateway import PaidCallLedger, PaidProviderGateway, ProviderRequest
from evaluation.interface_contracts import parse_formal_contract_response
from evaluation.scout_authorization import load_scout_manifest, provider_request_from_row
from evaluation.scout_replay import formal_planner_action
from scripts.run_interface_contract_scout import key_audit
from scripts.replay_interface_contract_scout import exact_execution_consistency


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "interface_contract_scout_manifest.json"
RESULT = ROOT / "results" / "interface_contract_scout_preflight"


class _KeyAuditResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "data": {
                "limit": 100.0,
                "limit_remaining": 80.0,
                "usage": 20.0,
                "is_free_tier": False,
                "expires_at": None,
            }
        }


def test_main_key_requires_explicit_scout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.run_interface_contract_scout.requests.get",
        lambda *_args, **_kwargs: _KeyAuditResponse(),
    )
    with pytest.raises(PermissionError, match="provider-side key limit"):
        key_audit("secret-not-written-to-artifacts", 1.0)
    evidence = key_audit(
        "secret-not-written-to-artifacts", 1.0, allow_higher_limit_key=True
    )
    assert evidence["provider_side_cap_compliant"] is False
    assert evidence["higher_limit_key_override"] is True
    assert evidence["limit_usd"] == 100.0


def test_native_execution_consistency_compares_low_level_hashes() -> None:
    rows = [
        {
            "comparison_id": "pair",
            "pair_side": "anchor",
            "action_sha256": "same-planner-action-but-first-native-sequence",
        },
        {
            "comparison_id": "pair",
            "pair_side": "mate",
            "action_sha256": "same-planner-action-but-second-native-sequence",
        },
    ]
    metric = exact_execution_consistency(rows, "action_sha256")
    assert metric["matched_pairs"] == 1
    assert metric["consistency"] == 0.0


def test_checked_in_scout_authorization_is_resolved_but_blocked() -> None:
    manifest = load_scout_manifest(CONFIG)
    assert manifest["status"] == "BLOCKED"
    assert manifest["provider_calls_enabled"] is False
    assert manifest["maximum_spend_usd"] == 1.0
    assert manifest["request_parameters"] == {"temperature": 0, "max_tokens": 220}
    assert [model["model_budget_id"] for model in manifest["models"]] == [
        "mistral-small-3.2-24b",
        "qwen3.5-plus-02-15",
    ]


def test_scout_manifest_fails_closed_if_partially_enabled(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["provider_calls_enabled"] = True
    path = tmp_path / "scout.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid scout authorization"):
        load_scout_manifest(path)


def test_checked_in_scout_matrix_is_exact_and_provider_free() -> None:
    manifest = json.loads((RESULT / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = json.loads((RESULT / "SCOUT_CALL_MATRIX.json").read_text(encoding="utf-8"))
    allowlist = json.loads((RESULT / "REQUEST_ALLOWLIST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "BLOCKED / PROVIDER_FREE"
    assert manifest["provider_calls"] == manifest["provider_attempts"] == 0
    assert len(rows) == allowlist["request_count"] == 120
    assert {row["scenario_family"] for row in rows} == {"direct_path_intersection"}
    assert {row["paid_seed"] for row in rows} == set(range(20, 25))
    assert all(manifest["checks"].values())
    for artifact in manifest["artifacts"].values():
        path = RESULT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_every_scout_provider_request_reconstructs_to_frozen_hash() -> None:
    rows = json.loads((RESULT / "SCOUT_CALL_MATRIX.json").read_text(encoding="utf-8"))
    source = ROOT / "results" / "interface_contract_provider_free_dry_run" / "inputs"
    for row in rows:
        request = provider_request_from_row(
            row,
            prompt_dir=source / "prompts",
            image_dir=source / "images",
        )
        assert request.sha256 == row["provider_request_sha256"]


def test_ambiguous_formal_output_replays_under_both_registered_meanings() -> None:
    raw = (
        '{"terrain_class":"water","grounding":{"geometry_type":"disk",'
        '"center_norm":[0.5,0.5],"radius_norm":0.1},'
        '"applicability":"applicable","action":"avoid"}'
    )
    parsed = parse_formal_contract_response("ambiguous_applicability_v1", raw)
    assert formal_planner_action(
        parsed, "ambiguous_applicability_v1", "applicable_means_constraint_applies"
    ) == "avoid"
    assert formal_planner_action(
        parsed, "ambiguous_applicability_v1", "applicable_means_terrain_compatible"
    ) == "traverse"


def test_gateway_rejects_requests_outside_frozen_scout_allowlist(tmp_path: Path) -> None:
    scout = load_scout_manifest(CONFIG)
    pilot_like = {
        "schema_version": "interface-contract-pilot-manifest-v1",
        "protocol_id": scout["protocol_id"],
        "status": "AUTHORIZED",
        "provider_calls_enabled": True,
        "paid_budget": {
            "allowed_paid_seeds": [20, 21, 22, 23, 24],
            "maximum_distinct_paid_seeds_per_model": 5,
            "maximum_new_provider_calls_per_model": 480,
            "maximum_attempts_per_request": 3,
            "maximum_retries_per_request": 2,
            "maximum_total_provider_attempts_per_model": 1440,
            "maximum_spend_usd": 1.0,
        },
        "models": [
            {**model, "eligibility": "ELIGIBLE"} for model in scout["models"]
        ],
    }
    allowlist = json.loads((RESULT / "REQUEST_ALLOWLIST.json").read_text())
    ledger = PaidCallLedger(tmp_path / "ledger.json", pilot_like)
    gateway = PaidProviderGateway(
        manifest=pilot_like,
        ledger=ledger,
        cache_dir=tmp_path / "cache",
        allow_provider_requests=True,
        transport=lambda _request, _attempt: pytest.fail("transport must not run"),
        authorized_request_sha256=allowlist["by_model"],
    )
    model = scout["models"][0]
    request = ProviderRequest(
        request_id="not-a-frozen-scout-row",
        model_budget_id=model["model_budget_id"],
        provider=model["provider"],
        provider_model=model["provider_model"],
        model_revision=model["revision"],
        paid_seed=20,
        prompt_bytes=b"different prompt",
        image_bytes=b"different image",
        parameters={"temperature": 0, "max_tokens": 220},
    )
    with pytest.raises(PermissionError, match="outside the frozen paid-scout allowlist"):
        gateway.call(request)
    assert ledger.audit()["models"][model["model_budget_id"]]["new_provider_calls"] == 0
