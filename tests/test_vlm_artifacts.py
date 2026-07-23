"""Acceptance tests for Phase-3 structured output and per-call provenance."""

from __future__ import annotations

import base64
import copy
import json
import struct
import zlib

import numpy as np
import pytest

from envs.protocol_env import EvaluatorContext
from evaluation.conditions import (
    EnforcementConfig,
    ExperimentCondition,
    FactorVector,
    PrivilegeLevel,
    Router,
    SeedSplit,
    ZoneSource,
)
from evaluation.policy_interface import build_policy_input
from evaluation.schemas import build_episode_artifact
from evaluation.vlm_artifacts import (
    ParseStatus,
    VLMCallArtifact,
    build_vlm_call_artifact,
    parse_structured_stage_output,
)


def _condition(privilege: PrivilegeLevel = PrivilegeLevel.P0) -> ExperimentCondition:
    return ExperimentCondition(
        router=Router.VLM,
        zone_source=ZoneSource.NONE,
        enforcement=EnforcementConfig(
            enforcement_id="phase3-test",
            hard_core_radius=0.0,
            soft_halo_radius=0.0,
            soft_zone_weight=0.0,
            replan_interval_steps=1,
            restart_on_target_change=True,
            arrival_radius=0.6,
            planner_id="test",
            executor_id="test",
        ),
        privilege_level=privilege,
        factor_vector=FactorVector(privilege_level=privilege),
        seed=0,
        split=SeedSplit.DEV,
    )


def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    scanline = b"\x00\xff\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanline))
        + chunk(b"IEND", b"")
    )


CANDIDATES = (
    {"candidate_id": "candidate_1", "world_xy": [1.0, 0.0], "pixel_xy": [10, 20]},
    {"candidate_id": "candidate_2", "world_xy": [0.0, 1.0], "pixel_xy": [30, 40]},
)
RAW = json.dumps(
    {
        "recognized_terrain": True,
        "unsafe_candidate_ids": ["candidate_2"],
        "selected_candidate_id": "candidate_2",
        "parse_status": "ok",
        "free_text_reason": "qualitative only",
    },
    separators=(",", ":"),
)


def _call() -> VLMCallArtifact:
    condition = _condition()
    observation = np.asarray([0, 0, 0, 0, 2.5, -1.5], dtype=np.float32)
    policy_input = build_policy_input(
        condition,
        public_observation=observation,
        public_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        public_candidate_metadata=CANDIDATES,
    )
    return build_vlm_call_artifact(
        call_id="episode-0-call-0",
        prompt="line one\n精确 prompt\n",
        input_png=_png(),
        candidate_metadata=CANDIDATES,
        policy_input=policy_input,
        model="local-mock",
        provider="offline-test",
        model_revision="fixture-v1",
        request_id="",
        temperature=0.0,
        latency_seconds=0.01,
        input_tokens=12,
        output_tokens=8,
        cost_usd=0.0,
        raw_response=RAW,
        fallback={"used": False, "mode": "hold"},
        git_sha="a" * 40,
        git_dirty=True,
        cli_config={"argv": ["pytest"], "max_new_tokens": 100},
        selected_target={"candidate_id": "candidate_2", "world_xy": [0.0, 1.0]},
        condition=condition,
        trajectory=({"step": 0, "agent_center": [0.0, 0.0]},),
    )


def test_structured_output_is_machine_judgeable() -> None:
    parsed = parse_structured_stage_output(
        RAW, candidate_ids=("candidate_1", "candidate_2")
    )
    assert parsed.parse_status is ParseStatus.OK
    assert parsed.recognized_terrain is True
    assert parsed.unsafe_candidate_ids == ("candidate_2",)
    assert parsed.selected_candidate_id == "candidate_2"
    assert parsed.free_text_reason == "qualitative only"


@pytest.mark.parametrize(
    ("raw", "status"),
    (
        ("not json", ParseStatus.JSON_ERROR),
        ('{"recognized_terrain":true,"recognized_terrain":false}', ParseStatus.JSON_ERROR),
        (
            '{"recognized_terrain":"yes","unsafe_candidate_ids":[],"selected_candidate_id":"candidate_1","parse_status":"ok"}',
            ParseStatus.SCHEMA_ERROR,
        ),
        (
            '{"recognized_terrain":true,"unsafe_candidate_ids":["candidate_9"],"selected_candidate_id":"candidate_1","parse_status":"ok"}',
            ParseStatus.SCHEMA_ERROR,
        ),
    ),
)
def test_failed_parse_exposes_no_partial_quantitative_labels(
    raw: str, status: ParseStatus
) -> None:
    parsed = parse_structured_stage_output(
        raw, candidate_ids=("candidate_1", "candidate_2")
    )
    assert parsed.parse_status is status
    assert parsed.recognized_terrain is None
    assert parsed.unsafe_candidate_ids == ()
    assert parsed.selected_candidate_id is None
    assert parsed.parse_error


def test_call_artifact_reconstructs_exact_bytes_and_reparses_offline() -> None:
    call = _call()
    restored = VLMCallArtifact.from_dict(
        json.loads(json.dumps(call.to_dict(), ensure_ascii=False))
    )
    assert restored.prompt_bytes == "line one\n精确 prompt\n".encode()
    assert restored.image_png_bytes == _png()
    assert restored.raw_response_bytes == RAW.encode()
    assert restored.audit() == {
        "prompt_hash_matches": True,
        "image_hash_matches": True,
        "policy_input_hash_matches": True,
        "condition_hash_matches": True,
        "structured_reparse_matches": True,
        "candidate_ids_unique": True,
        "authorized_tags_match_policy": True,
        "p0_has_no_authorized_privilege": True,
        "selected_target_matches_candidate": True,
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("prompt_sha256", "0" * 64),
        lambda value: value.__setitem__(
            "input_png_base64", base64.b64encode(b"not png").decode()
        ),
        lambda value: value["candidate_metadata"][0].__setitem__(
            "candidate_id", "candidate_2"
        ),
        lambda value: value["structured_parse"].__setitem__(
            "selected_candidate_id", "candidate_1"
        ),
        lambda value: value["selected_target"].__setitem__(
            "world_xy", [99.0, 99.0]
        ),
    ),
)
def test_tampered_call_artifact_is_rejected(mutate) -> None:
    value = copy.deepcopy(_call().to_dict())
    mutate(value)
    with pytest.raises(ValueError):
        VLMCallArtifact.from_dict(value)


def test_secret_config_and_eval_only_authorization_are_rejected() -> None:
    value = copy.deepcopy(_call().to_dict())
    value["cli_config"]["api_key"] = "must-not-log"
    with pytest.raises(PermissionError, match="secret field"):
        VLMCallArtifact.from_dict(value)

    value = copy.deepcopy(_call().to_dict())
    value["authorized_information_tags"]["public_rgb"] = "EVAL_ONLY"
    with pytest.raises(PermissionError, match="EVAL_ONLY"):
        VLMCallArtifact.from_dict(value)


def test_episode_artifact_embeds_only_audited_matching_calls(monkeypatch) -> None:
    call = _call()
    monkeypatch.setattr(
        "evaluation.schemas.git_provenance", lambda: (call.git_sha, call.git_dirty)
    )
    context = EvaluatorContext(
        environment_backend="unit",
        environment_id="PointHazard-v0",
        environment_version="test",
        scene_manifest={"scene_id": "fixture"},
        trajectory=({"step": 0, "agent_center": [0.0, 0.0]},),
        actions=(),
        rewards=(),
        native_costs=(),
        success=False,
        truncated=True,
        termination_reason="timeout",
    )
    artifact = build_episode_artifact(
        context,
        seed=0,
        initial_observation=np.zeros(6, dtype=np.float32),
        protocol_version="1.2.2",
        condition=_condition(),
        vlm_calls=(call.to_dict(),),
    )
    assert artifact.vlm_calls == [call.to_dict()]

    mismatched = copy.deepcopy(call.to_dict())
    mismatched["git_sha"] = "b" * 40
    with pytest.raises(ValueError, match="code state"):
        build_episode_artifact(
            context,
            seed=0,
            initial_observation=np.zeros(6, dtype=np.float32),
            protocol_version="1.2.2",
            condition=_condition(),
            vlm_calls=(mismatched,),
        )
