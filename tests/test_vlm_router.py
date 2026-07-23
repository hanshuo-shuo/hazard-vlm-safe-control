"""Offline acceptance tests for the public VLM request adapter."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter
from evaluation.conditions import (
    EnforcementConfig,
    ExperimentCondition,
    FactorVector,
    PrivilegeLevel,
    Router,
    SeedSplit,
    ZoneSource,
)
from evaluation.vlm_router import (
    STRUCTURED_PROMPT_VERSION,
    generate_candidate_metadata,
    prepare_vlm_request,
    run_offline_fixture_decision,
)


def _condition(
    *,
    capability: str = "wheeled_non_waterproof",
    privilege: PrivilegeLevel = PrivilegeLevel.P0,
) -> ExperimentCondition:
    return ExperimentCondition(
        router=Router.VLM,
        zone_source=ZoneSource.NONE,
        enforcement=EnforcementConfig(
            enforcement_id="fixture",
            hard_core_radius=0.0,
            soft_halo_radius=0.0,
            soft_zone_weight=0.0,
            replan_interval_steps=1,
            restart_on_target_change=True,
            arrival_radius=0.6,
            planner_id="fixture",
            executor_id="fixture",
        ),
        privilege_level=privilege,
        factor_vector=FactorVector(
            capability=capability,
            privilege_level=privilege,
        ),
        seed=0,
        split=SeedSplit.DEV,
    )


def _observation() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 0.0, 3.0, 2.0], dtype=np.float32)


def _rgb() -> np.ndarray:
    image = np.full((120, 120, 3), 245, dtype=np.uint8)
    image[58:63, 58:63] = (40, 90, 200)
    return image


def test_frozen_candidate_order_geometry_and_clipping() -> None:
    candidates = generate_candidate_metadata(
        (4.8, 4.8), image_shape=(120, 120), arena_half=5.0
    )
    assert tuple(item["candidate_id"] for item in candidates) == tuple(range(1, 9))
    assert candidates[0]["world_xy"] == (4.9, 4.8)
    assert candidates[2]["world_xy"] == pytest.approx((4.8, 4.9))
    assert candidates[4]["world_xy"] == pytest.approx((2.3, 4.8))
    assert len(candidates) == 8


def test_request_is_deterministic_and_does_not_mutate_public_rgb() -> None:
    rgb = _rgb()
    before = rgb.copy()
    first = prepare_vlm_request(
        _condition(),
        public_observation=_observation(),
        public_rgb=rgb,
        arena_half=5.0,
    )
    second = prepare_vlm_request(
        _condition(),
        public_observation=_observation(),
        public_rgb=rgb,
        arena_half=5.0,
    )
    assert np.array_equal(rgb, before)
    assert first.input_png == second.input_png
    assert first.prompt_bytes == second.prompt_bytes
    assert first.policy_input.sha256 == second.policy_input.sha256
    assert first.input_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_real_point_hazard_public_frame_prepares_offline_request() -> None:
    config = PointHazardConfig(
        n_hazards=1,
        n_semantic_zones=1,
        max_episode_steps=5,
        max_layout_resamples=10,
        render_size=120,
    )
    environment = PointHazardAdapter(config, with_renderer=True)
    try:
        observation, _public_info = environment.reset(seed=0)
        request = prepare_vlm_request(
            _condition(),
            public_observation=observation,
            public_rgb=environment.render_public_rgb(),
            arena_half=config.arena_half,
        )
    finally:
        environment.close()
    assert request.input_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(request.candidate_metadata) == 8
    assert request.condition_sha256 == _condition().condition_sha256


def test_capability_twin_changes_prompt_card_but_not_scene_projection() -> None:
    wheeled = prepare_vlm_request(
        _condition(capability="wheeled_non_waterproof"),
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
    )
    amphibious = prepare_vlm_request(
        _condition(capability="amphibious"),
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
    )
    assert wheeled.input_png == amphibious.input_png
    assert wheeled.candidate_metadata == amphibious.candidate_metadata
    assert wheeled.prompt_bytes != amphibious.prompt_bytes
    assert b"semantic_zones" not in wheeled.prompt_bytes
    assert b"terrain_class" not in wheeled.prompt_bytes
    assert wheeled.prompt_version == STRUCTURED_PROMPT_VERSION


def test_offline_fixture_vertical_slice_selects_and_audits_target() -> None:
    condition = _condition()
    request = prepare_vlm_request(
        condition,
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
    )
    raw = json.dumps(
        {
            "recognized_terrain": True,
            "unsafe_candidate_ids": [3],
            "selected_candidate_id": 1,
            "parse_status": "ok",
        },
        separators=(",", ":"),
    )
    decision = run_offline_fixture_decision(
        request,
        raw_response=raw,
        condition=condition,
        call_id="fixture-episode-0-call-0",
        git_sha="a" * 40,
        git_dirty=True,
        trajectory=({"step": 0, "agent_center": [0.0, 0.0]},),
    )
    assert decision.candidate_id == 1
    assert decision.world_xy == pytest.approx((2.5, 0.0))
    assert decision.call_artifact.cost_usd == 0.0
    assert decision.call_artifact.provider == "offline-fixture"
    assert decision.call_artifact.audit()["selected_target_matches_candidate"]


def test_invalid_fixture_response_and_p0_truth_fail_closed() -> None:
    condition = _condition()
    request = prepare_vlm_request(
        condition,
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
    )
    with pytest.raises(ValueError, match="did not pass structured parsing"):
        run_offline_fixture_decision(
            request,
            raw_response='{"choice":1,"reason":"legacy schema"}',
            condition=condition,
            call_id="invalid",
            git_sha="a" * 40,
            git_dirty=True,
            trajectory=(),
        )

    with pytest.raises(PermissionError, match="cannot receive scene truth"):
        prepare_vlm_request(
            _condition(),
            public_observation=_observation(),
            public_rgb=_rgb(),
            arena_half=5.0,
            evaluator_semantic_terrain=(
                {
                    "region_id": "zone_0",
                    "center_xy": [1.0, 0.0],
                    "radius": 0.5,
                    "terrain_class": "water",
                },
            ),
        )


def test_p0_p4_prompt_ladder_is_cumulative_and_keeps_image_fixed() -> None:
    terrain = (
        {
            "region_id": "zone_2",
            "center_xy": [1.25, 0.0],
            "radius": 0.5,
            "terrain_class": "water",
        },
        {
            "region_id": "zone_10",
            "center_xy": [-1.0, 1.0],
            "radius": 0.25,
            "terrain_class": "mud",
        },
    )
    requests = {
        level: prepare_vlm_request(
            _condition(privilege=level),
            public_observation=_observation(),
            public_rgb=_rgb(),
            arena_half=5.0,
            evaluator_semantic_terrain=() if level is PrivilegeLevel.P0 else terrain,
        )
        for level in PrivilegeLevel
    }
    assert len({request.input_png for request in requests.values()}) == 1
    assert requests[PrivilegeLevel.P0].policy_input.permission_audit()[
        "authorized_privilege_tag_count"
    ] == 0
    assert all(
        requests[level].policy_input.permission_audit()[
            "authorized_privilege_tag_count"
        ]
        == 1
        for level in (
            PrivilegeLevel.P1,
            PrivilegeLevel.P2,
            PrivilegeLevel.P3,
            PrivilegeLevel.P4,
        )
    )
    assert len(
        {
            tuple(
                (
                    item["candidate_id"],
                    tuple(item["world_xy"]),
                    tuple(item["pixel_xy"]),
                )
                for item in request.candidate_metadata
            )
            for request in requests.values()
        }
    ) == 1
    prompts = {
        level: request.prompt_bytes.decode("utf-8")
        for level, request in requests.items()
    }
    assert {
        level.value: hashlib.sha256(request.prompt_bytes).hexdigest()
        for level, request in requests.items()
    } == {
        "P0": "eddd585c575b3291eb1d53c8e8ed5399807b30dabee87878033b956c2bd6465d",
        "P1": "1459811a856c56fb37ecb2033875497c60a551a97d2cc5715626b655d6ffab99",
        "P2": "3fd3cf76ae1529411ed79a1f8e23bb75c2d80515fd48d3b708968e53609bf5d4",
        "P3": "cdbd2a465b4b8c5a231364eba31f09fdb4ca824f5d1bd6dbfaae75df74bb249d",
        "P4": "3747780df0a6c28e3966d5dd84b0b4aea0430e1fe7711fcc45b04ea5d031e0b2",
    }
    assert "SCENE TERRAIN CLASSES" not in prompts[PrivilegeLevel.P0]
    assert '"classes":["mud","water"]' in prompts[PrivilegeLevel.P1]
    assert "SCENE TERRAIN REGIONS" not in prompts[PrivilegeLevel.P1]
    assert "SCENE TERRAIN REGIONS" in prompts[PrivilegeLevel.P2]
    assert prompts[PrivilegeLevel.P2].index("zone_2") < prompts[PrivilegeLevel.P2].index("zone_10")
    assert "CANDIDATE TERRAIN LABELS" not in prompts[PrivilegeLevel.P2]
    assert "CANDIDATE TERRAIN LABELS" in prompts[PrivilegeLevel.P3]
    assert '"candidate_id":1,"label":"unsafe"' in prompts[PrivilegeLevel.P3]
    assert "CANDIDATE TERRAIN CLEARANCE" not in prompts[PrivilegeLevel.P3]
    assert "CANDIDATE TERRAIN CLEARANCE" in prompts[PrivilegeLevel.P4]
    assert '"candidate_id":1,"clearance":-0.500,"rank":' in prompts[PrivilegeLevel.P4]


def test_amphibious_privilege_labels_change_without_image_or_regions() -> None:
    terrain = (
        {
            "region_id": "zone_0",
            "center_xy": [1.25, 0.0],
            "radius": 0.5,
            "terrain_class": "water",
        },
    )
    wheeled = prepare_vlm_request(
        _condition(privilege=PrivilegeLevel.P3),
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
        evaluator_semantic_terrain=terrain,
    )
    amphibious = prepare_vlm_request(
        _condition(capability="amphibious", privilege=PrivilegeLevel.P3),
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
        evaluator_semantic_terrain=terrain,
    )
    assert wheeled.input_png == amphibious.input_png
    assert wheeled.candidate_metadata == amphibious.candidate_metadata
    assert '"candidate_id":1,"label":"unsafe"' in wheeled.prompt_bytes.decode()
    assert '"candidate_id":1,"label":"safe"' in amphibious.prompt_bytes.decode()
    assert '"candidate_id":1,"clearance"' not in amphibious.prompt_bytes.decode()


def test_p4_without_incompatible_regions_uses_inf_and_shared_rank() -> None:
    request = prepare_vlm_request(
        _condition(capability="amphibious", privilege=PrivilegeLevel.P4),
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
        evaluator_semantic_terrain=(
            {
                "region_id": "zone_0",
                "center_xy": [1.25, 0.0],
                "radius": 0.5,
                "terrain_class": "water",
            },
        ),
    )
    prompt = request.prompt_bytes.decode()
    assert prompt.count('"clearance":"INF","rank":1,"safety_score":"INF"') == 8


@pytest.mark.parametrize(
    ("terrain", "message"),
    (
        (
            (
                {
                    "region_id": "zone_01",
                    "center_xy": [0.0, 0.0],
                    "radius": 0.5,
                    "terrain_class": "water",
                },
            ),
            "invalid region_id",
        ),
        (
            (
                {
                    "region_id": "zone_0",
                    "center_xy": [0.0, 0.0],
                    "radius": 0.5,
                    "terrain_class": "unknown",
                },
            ),
            "unknown terrain class",
        ),
        (
            (
                {
                    "region_id": "zone_0",
                    "center_xy": [0.0, 0.0],
                    "radius": 0.5,
                    "terrain_class": "water",
                },
                {
                    "region_id": "zone_0",
                    "center_xy": [1.0, 0.0],
                    "radius": 0.5,
                    "terrain_class": "mud",
                },
            ),
            "duplicate region_id",
        ),
    ),
)
def test_privilege_projection_rejects_noncanonical_scene_truth(
    terrain, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_vlm_request(
            _condition(privilege=PrivilegeLevel.P4),
            public_observation=_observation(),
            public_rgb=_rgb(),
            arena_half=5.0,
            evaluator_semantic_terrain=terrain,
        )


def test_prepared_request_is_bound_to_exact_condition() -> None:
    condition = _condition()
    request = prepare_vlm_request(
        condition,
        public_observation=_observation(),
        public_rgb=_rgb(),
        arena_half=5.0,
    )
    other = ExperimentCondition(
        router=Router.VLM,
        zone_source=ZoneSource.NONE,
        enforcement=condition.enforcement,
        privilege_level=condition.privilege_level,
        factor_vector=condition.factor_vector,
        seed=1,
        split=SeedSplit.DEV,
    )
    with pytest.raises(ValueError, match="does not match decision condition"):
        run_offline_fixture_decision(
            request,
            raw_response=json.dumps(
                {
                    "recognized_terrain": False,
                    "unsafe_candidate_ids": [],
                    "selected_candidate_id": 1,
                    "parse_status": "ok",
                }
            ),
            condition=other,
            call_id="condition-drift",
            git_sha="a" * 40,
            git_dirty=True,
            trajectory=(),
        )
