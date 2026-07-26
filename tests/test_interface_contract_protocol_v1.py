"""Acceptance tests for the provider-free interface-contract pilot protocol."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SafetyGymGoalAdapter
from evaluation.interface_contracts import (
    AMBIGUOUS_CONTRACT_IDS,
    EQUIVALENT_PAIRS,
    build_formal_contract_prompt,
    parse_formal_contract_response,
)
from evaluation.interface_scenarios import (
    TWIN_ARMS,
    audit_twin_bundle,
    build_call_matrix,
    build_preregistered_blocks,
    build_twin_bundle,
)
from evaluation.native_environment_gate import audit_native_environment
from evaluation.paid_provider_gateway import (
    PaidCallLedger,
    PaidProviderGateway,
    ProviderRequest,
    load_pilot_manifest,
)
from evaluation.policy_interface import (
    GroundingProvenance,
    PlannerGrounding,
    validate_planner_grounding,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "interface_contract_pilot_manifest.json"


def _manifest(*, authorize: bool = False) -> dict:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for index, model in enumerate(value["models"]):
        model.update({
            "provider": "fixture-provider",
            "provider_model": f"fixture-model-{index}",
            "revision": "fixture-revision-v1",
            "eligibility": "ELIGIBLE",
        })
    value["provider_calls_enabled"] = authorize
    value["status"] = "AUTHORIZED" if authorize else "BLOCKED"
    return value


def _request(index: int = 0, *, seed: int = 20) -> ProviderRequest:
    return ProviderRequest(
        request_id=f"request-{index}",
        model_budget_id="mistral-small-3.2-24b",
        provider="fixture-provider",
        provider_model="fixture-model-0",
        model_revision="fixture-revision-v1",
        paid_seed=seed,
        prompt_bytes=f"prompt-{index}".encode(),
        image_bytes=f"image-{index}".encode(),
        parameters={"temperature": 0, "max_tokens": 220},
        compatibility_smoke=index == 0,
    )


def test_checked_in_manifest_is_blocked_and_freezes_all_three_budget_layers() -> None:
    manifest = load_pilot_manifest(MANIFEST_PATH)
    assert manifest["provider_calls_enabled"] is False
    assert manifest["status"] == "BLOCKED"
    assert manifest["paid_budget"] == {
        "allowed_paid_seeds": [20, 21, 22, 23, 24],
        "maximum_distinct_paid_seeds_per_model": 5,
        "maximum_new_provider_calls_per_model": 480,
        "maximum_attempts_per_request": 3,
        "maximum_retries_per_request": 2,
        "maximum_total_provider_attempts_per_model": 1440,
        "maximum_spend_usd": None,
    }
    assert manifest["compatibility_smoke"]["successful_smoke_is_formal_data"] is True
    assert manifest["compatibility_smoke"]["repeat_call_allowed"] is False


def test_contract_registry_separates_equivalent_and_ambiguous_contracts() -> None:
    assert len(EQUIVALENT_PAIRS) == 5
    assert AMBIGUOUS_CONTRACT_IDS == ("ambiguous_applicability_v1",)
    assert not set(AMBIGUOUS_CONTRACT_IDS).intersection(
        contract for pair in EQUIVALENT_PAIRS.values() for contract in pair
    )


def test_explicit_equivalent_labels_normalize_to_identical_semantics_and_grounding() -> None:
    grounding = '"grounding":{"geometry_type":"disk","center_norm":[0.4,0.6],"radius_norm":0.1}'
    cases = {
        "formal_constraint_positive": (
            '{"terrain_class":"water",' + grounding + ',"constraint_applies":"yes","action":"avoid"}'
        ),
        "formal_constraint_negative": (
            '{"terrain_class":"water",' + grounding + ',"constraint_does_not_apply":"no","action":"avoid"}'
        ),
        "formal_compatibility_positive": (
            '{"terrain_class":"water",' + grounding + ',"terrain_compatible":"no","action":"avoid"}'
        ),
        "formal_compatibility_negative": (
            '{"terrain_class":"water",' + grounding + ',"terrain_incompatible":"yes","action":"avoid"}'
        ),
    }
    parsed = [parse_formal_contract_response(contract, raw) for contract, raw in cases.items()]
    assert all(value["parse_status"] == "ok" for value in parsed)
    assert len({json.dumps(value["canonical_semantics"], sort_keys=True) for value in parsed}) == 1
    assert len({json.dumps(value["canonical_grounding"], sort_keys=True) for value in parsed}) == 1


def test_ambiguous_contract_never_guesses_safe_to_traverse() -> None:
    raw = (
        '{"terrain_class":"water","grounding":{"geometry_type":"disk",'
        '"center_norm":[0.5,0.5],"radius_norm":0.2},'
        '"applicability":"applicable","action":"avoid"}'
    )
    parsed = parse_formal_contract_response("ambiguous_applicability_v1", raw)
    assert parsed["parse_status"] == "ok"
    assert parsed["contract_class"] == "ambiguous"
    assert parsed["canonical_semantics"]["safe_to_traverse"] is None


def test_formal_planner_rejects_evaluator_truth_except_separate_oracle_upper_bound() -> None:
    sha = "a" * 64
    truth = PlannerGrounding(
        {"geometry_type": "disk", "center_xy": [0.0, 0.0], "radius": 1.0},
        GroundingProvenance.EVALUATOR_TRUTH,
        sha,
    )
    with pytest.raises(PermissionError, match="forbidden"):
        validate_planner_grounding(truth, experiment_arm="headline")
    assert validate_planner_grounding(truth, experiment_arm="oracle_upper_bound") is truth
    estimate = PlannerGrounding(
        {"geometry_type": "disk", "center_norm": [0.5, 0.5], "radius_norm": 0.1},
        GroundingProvenance.PROVIDER_ESTIMATE,
        sha,
    )
    assert validate_planner_grounding(estimate, experiment_arm="headline") is estimate


def test_every_block_has_required_assignments_and_balanced_pair_schedule() -> None:
    blocks = build_preregistered_blocks()
    required = {
        "equivalent_pair_id", "equivalent_anchor_arm", "ambiguous_anchor_arm",
        "scenario_family", "environment", "paid_seed",
    }
    assert len(blocks) == 80
    assert all(required.issubset(block) for block in blocks)
    for environment in {block["environment"] for block in blocks}:
        subset = [block for block in blocks if block["environment"] == environment]
        assert len(subset) == 40
        assert {
            pair: sum(block["equivalent_pair_id"] == pair for block in subset)
            for pair in EQUIVALENT_PAIRS
        } == {pair: 8 for pair in EQUIVALENT_PAIRS}
        assert sum(block["ambiguous_anchor_arm"] == "reference" for block in subset) == 20
        assert sum(block["ambiguous_anchor_arm"] == "visibility_twin" for block in subset) == 20


def test_strict_capability_appearance_visibility_twins() -> None:
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    bundle = build_twin_bundle(
        rgb,
        geometry={
            "geometry_type": "disk", "center_xy": [0.0, 0.0], "radius": 0.5,
            "coordinate_frame": "native_world_xy",
        },
        world_bounds=[-2, 2, -2, 2],
        native_scene_sha256="b" * 64,
    )
    audit = audit_twin_bundle(bundle)
    assert audit["passed"] is True
    assert set(bundle) == set(TWIN_ARMS)
    assert bundle["reference"].png_bytes == bundle["capability_twin"].png_bytes


def test_call_matrix_is_exactly_1440_and_smoke_is_first_reused_cell() -> None:
    manifest = _manifest()
    blocks = build_preregistered_blocks()
    base = build_twin_bundle(
        np.zeros((32, 32, 3), dtype=np.uint8),
        geometry={
            "geometry_type": "disk", "center_xy": [0.0, 0.0], "radius": 0.5,
            "coordinate_frame": "native_world_xy",
        },
        world_bounds=[-2, 2, -2, 2],
        native_scene_sha256="c" * 64,
    )
    bundles = {block["block_id"]: base for block in blocks}
    rows = build_call_matrix(manifest, blocks, bundles)
    assert len(rows) == 1440
    for model in manifest["models"]:
        subset = [row for row in rows if row["model_budget_id"] == model["model_budget_id"]]
        assert len(subset) == 480
        assert {row["paid_seed"] for row in subset} == set(range(20, 25))
        assert subset[0]["compatibility_smoke"] is True
        assert sum(row["compatibility_smoke"] for row in subset) == 1
        for field in (
            "equivalent_pair_id", "equivalent_anchor_arm", "ambiguous_anchor_arm",
            "scenario_family", "environment", "paid_seed",
        ):
            assert field in subset[0]


def test_paid_ledger_counts_failed_invalid_timeout_attempts_and_never_changes_seed(tmp_path: Path) -> None:
    manifest = _manifest(authorize=True)
    ledger = PaidCallLedger(tmp_path / "ledger.json", manifest)
    gateway = PaidProviderGateway(
        manifest=manifest,
        ledger=ledger,
        cache_dir=tmp_path / "cache",
        allow_provider_requests=True,
        transport=lambda _request, attempt: (
            (_ for _ in ()).throw(TimeoutError(f"timeout-{attempt}"))
            if attempt == 1
            else {"provider_model": "fixture-model-0", "raw_response": ""}
        ),
    )
    request = _request()
    with pytest.raises(RuntimeError, match="exhausted"):
        gateway.call(request)
    snapshot = ledger.snapshot()["models"][request.model_budget_id]
    assert snapshot["new_provider_calls"] == 1
    assert snapshot["provider_attempts"] == 3
    assert snapshot["reserved_paid_seeds"] == [20]
    assert snapshot["requests"][request.sha256]["attempts"] == 3
    with pytest.raises(PermissionError, match="attempt"):
        ledger.reserve_attempt(request.model_budget_id, request.sha256)
    with pytest.raises(PermissionError, match="20-24"):
        ledger.reserve_request(request.model_budget_id, 25, "d" * 64)


def test_successful_smoke_is_cached_formal_data_and_not_called_again(tmp_path: Path) -> None:
    manifest = _manifest(authorize=True)
    ledger = PaidCallLedger(tmp_path / "ledger.json", manifest)
    transports = []

    def transport(request: ProviderRequest, attempt: int) -> dict:
        transports.append((request.sha256, attempt))
        return {"provider_model": request.provider_model, "raw_response": "valid"}

    gateway = PaidProviderGateway(
        manifest=manifest,
        ledger=ledger,
        cache_dir=tmp_path / "cache",
        allow_provider_requests=True,
        transport=transport,
    )
    request = _request()
    first = gateway.call(request)
    second = gateway.call(request)
    assert first["new_provider_call"] is True
    assert second["new_provider_call"] is False
    assert second["call_origin"] == "cache_replay"
    assert transports == [(request.sha256, 1)]
    audit = ledger.audit()
    model = audit["models"][request.model_budget_id]
    assert model["new_provider_calls"] == 1
    assert model["provider_attempts"] == 1


def test_failed_smoke_atomically_eliminates_model_from_later_calls(tmp_path: Path) -> None:
    manifest = _manifest(authorize=True)
    ledger = PaidCallLedger(tmp_path / "ledger.json", manifest)
    gateway = PaidProviderGateway(
        manifest=manifest,
        ledger=ledger,
        cache_dir=tmp_path / "cache",
        allow_provider_requests=True,
        transport=lambda _request, _attempt: {
            "provider_model": "fixture-model-0",
            "raw_response": "invalid",
        },
        response_validator=lambda _request, _response: (_ for _ in ()).throw(
            ValueError("schema mismatch")
        ),
    )
    smoke = _request(index=0)
    with pytest.raises(RuntimeError, match="exhausted frozen attempts"):
        gateway.call(smoke)
    model = ledger.snapshot()["models"][smoke.model_budget_id]
    assert model["compatibility_smoke_status"] == "SMOKE_FAILED_ELIMINATED"
    assert model["new_provider_calls"] == 1
    assert model["provider_attempts"] == 3
    with pytest.raises(PermissionError, match="compatibility smoke has not succeeded"):
        gateway.call(_request(index=1))
    after = ledger.snapshot()["models"][smoke.model_budget_id]
    assert after["new_provider_calls"] == 1
    assert after["provider_attempts"] == 3


def test_480_call_and_1440_attempt_boundaries_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(authorize=True)
    ledger_path = tmp_path / "ledger.json"
    ledger = PaidCallLedger(ledger_path, manifest)
    snapshot = ledger.snapshot()
    model = snapshot["models"]["mistral-small-3.2-24b"]
    model["reserved_paid_seeds"] = [20, 21, 22, 23, 24]
    model["new_provider_calls"] = 480
    model["requests"] = {
        f"{index:064x}": {
            "paid_seed": 20 + index % 5,
            "attempts": 0,
            "status": "RESERVED",
            "reserved_at": "fixture",
            "events": [],
        }
        for index in range(480)
    }
    ledger_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(PermissionError, match="480"):
        ledger.reserve_request("mistral-small-3.2-24b", 20, "f" * 64)

    model["new_provider_calls"] = 1
    model["provider_attempts"] = 1440
    model["requests"] = {
        "e" * 64: {
            "paid_seed": 20,
            "attempts": 0,
            "status": "RESERVED",
            "reserved_at": "fixture",
            "events": [],
        }
    }
    ledger_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(PermissionError, match="per-model provider-attempt"):
        ledger.reserve_attempt("mistral-small-3.2-24b", "e" * 64)


def test_concurrent_duplicate_request_reservation_counts_one_logical_call(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    manifest = _manifest(authorize=True)
    ledger = PaidCallLedger(tmp_path / "ledger.json", manifest)
    request_hash = hashlib.sha256(b"same-request").hexdigest()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda _index: ledger.reserve_request(
                "mistral-small-3.2-24b", 20, request_hash
            ),
            range(32),
        ))
    assert sum(results) == 1
    audit = ledger.audit()["models"]["mistral-small-3.2-24b"]
    assert audit["new_provider_calls"] == 1
    assert audit["paid_seed_union"] == [20, 21, 22, 23, 24]


def test_ledger_snapshot_and_audit_are_read_only(tmp_path: Path) -> None:
    manifest = _manifest(authorize=True)
    ledger_path = tmp_path / "ledger.json"
    ledger = PaidCallLedger(ledger_path, manifest)
    ledger_path.write_text(
        json.dumps(ledger.snapshot(), indent=2) + "\n", encoding="utf-8"
    )
    before = ledger_path.read_bytes()

    ledger.snapshot()
    ledger.audit()

    assert ledger_path.read_bytes() == before


def test_legacy_paid_runners_are_explicitly_disabled() -> None:
    for relative in (
        "scripts/run_interface_contract_audit.py",
        "scripts/run_interface_contract_replacement.py",
        "scripts/run_safety_gym_micro_pilot.py",
        "scripts/make_semantic_figures.py",
        "subgoal_pivot_hazard.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "legacy" in source
        assert "evaluation.paid_provider_gateway" in source


def test_gateway_is_cache_only_by_default_and_unresolved_models_fail_closed(tmp_path: Path) -> None:
    checked_in = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ledger = PaidCallLedger(tmp_path / "ledger.json", checked_in)
    gateway = PaidProviderGateway(
        manifest=checked_in, ledger=ledger, cache_dir=tmp_path / "cache"
    )
    with pytest.raises(PermissionError, match="unresolved"):
        gateway.call(_request())
    resolved = _manifest()
    resolved_ledger = PaidCallLedger(tmp_path / "resolved.json", resolved)
    cache_only = PaidProviderGateway(
        manifest=resolved, ledger=resolved_ledger, cache_dir=tmp_path / "cache2"
    )
    with pytest.raises(PermissionError, match="provider calls remain disabled"):
        cache_only.call(_request())
    assert resolved_ledger.audit()["models"]["mistral-small-3.2-24b"]["new_provider_calls"] == 0


def test_real_point_hazard_passes_native_gate() -> None:
    adapter = PointHazardAdapter(
        PointHazardConfig(
            n_hazards=4, n_semantic_zones=0, semantic_styles=(),
            semantic_terrain_classes=(), max_episode_steps=3,
        ),
        with_renderer=True,
    )
    try:
        evidence = audit_native_environment(adapter, seed=200)
        assert evidence["passed"] is True
        assert all(evidence["checks"].values())
    finally:
        adapter.close()


def test_real_safety_gym_passes_native_gate() -> None:
    if importlib.util.find_spec("safety_gymnasium") is None:
        pytest.skip("Safety-Gymnasium is not installed")
    # Keep macOS OpenGL/Cocoa state out of the pytest parent. The layout
    # invariant tests use ProcessPoolExecutor later in the same suite; forking
    # after GLFW initialization can deadlock even though the native gate passed.
    code = """
import json
from envs import SafetyGymGoalAdapter
from evaluation.native_environment_gate import audit_native_environment
adapter = SafetyGymGoalAdapter(env_id='SafetyPointGoal1-v0', render_mode='rgb_array')
try:
    print(json.dumps(audit_native_environment(adapter, seed=200)))
finally:
    adapter.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence["passed"] is True
    assert all(evidence["checks"].values())


def test_checked_in_dry_run_has_complete_source_and_artifact_provenance() -> None:
    result_root = ROOT / "results" / "interface_contract_provider_free_dry_run"
    dry_manifest = json.loads((result_root / "MANIFEST.json").read_text())
    source_hashes = dry_manifest["provenance"]["source_files"]
    assert source_hashes
    assert "evaluation/paid_provider_gateway.py" in source_hashes
    assert "evaluation/native_environment_gate.py" in source_hashes
    assert "scripts/build_interface_contract_pilot.py" in source_hashes
    for relative_path, expected_sha256 in source_hashes.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_sha256
    for artifact in dry_manifest["artifacts"].values():
        actual = hashlib.sha256((result_root / artifact["path"]).read_bytes()).hexdigest()
        assert actual == artifact["sha256"]
