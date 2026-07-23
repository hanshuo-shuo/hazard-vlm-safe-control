"""Direct/replay × none/oracle vertical-slice acceptance tests."""

from __future__ import annotations

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
from evaluation.harness import (
    CostMapPayload,
    ReplaySource,
    TargetRecord,
    build_cost_map,
    reconstruct_replay_audit,
    replay_diagnostics,
    run_point_hazard_episode,
)


def _enforcement() -> EnforcementConfig:
    cfg = _config()
    return EnforcementConfig(
        enforcement_id="mpc-shared-soft-v1",
        hard_core_radius=0.25,
        soft_halo_radius=1.0,
        soft_zone_weight=35.0,
        replan_interval_steps=5,
        restart_on_target_change=True,
        arrival_radius=0.600,
        planner_id="cem-mpc-v1",
        executor_id="point-mass-v1",
        planner_parameters={
            "horizon": 6,
            "population": 24,
            "iterations": 2,
            "safety_margin": 0.15,
        },
        executor_parameters={
            "arena_half": cfg.arena_half,
            "dt": cfg.dt,
            "drag": cfg.drag,
            "force_scale": cfg.force_scale,
            "max_speed": cfg.max_speed,
            "agent_radius": cfg.agent_radius,
            "goal_radius": cfg.goal_radius,
            "max_episode_steps": cfg.max_episode_steps,
            "action_clip": 1.0,
        },
    )


def _condition(router: Router, zone_source: ZoneSource, *, seed: int = 0) -> ExperimentCondition:
    return ExperimentCondition(
        router=router,
        zone_source=zone_source,
        enforcement=_enforcement(),
        privilege_level=PrivilegeLevel.P0,
        factor_vector=FactorVector(),
        seed=seed,
        split=SeedSplit.DEV,
    )


def _vlm_condition(
    privilege: PrivilegeLevel,
    *,
    capability: str = "wheeled_non_waterproof",
    seed: int = 0,
) -> ExperimentCondition:
    return ExperimentCondition(
        router=Router.VLM,
        zone_source=ZoneSource.NONE,
        enforcement=_enforcement(),
        privilege_level=privilege,
        factor_vector=FactorVector(
            appearance="water-render-v1",
            capability=capability,
            privilege_level=privilege,
        ),
        seed=seed,
        split=SeedSplit.DEV,
    )


def _config() -> PointHazardConfig:
    return PointHazardConfig(
        n_hazards=2,
        n_semantic_zones=1,
        max_episode_steps=30,
        max_layout_resamples=10,
    )


def _registered_config(*, style: str = "water") -> PointHazardConfig:
    return PointHazardConfig(
        n_hazards=2,
        n_semantic_zones=1,
        semantic_styles=(style,),
        semantic_terrain_classes=("water",),
        max_episode_steps=30,
        max_layout_resamples=10,
        render_size=120,
    )


def _source_from_result(result, seed: int) -> ReplaySource:
    trajectory = tuple(
        tuple(item["agent_center"]) for item in result.artifact.trajectory
    )
    start = trajectory[0]
    target = TargetRecord(
        decision_index=0,
        world_xy=start,
        target_kind="generated_subgoal",
        selected_candidate_id=7,
        decision_step=0,
        source_position=start,
        candidates=((7, start),),
    )
    return ReplaySource(
        run_id="source-direct-none-seed-0",
        seed=seed,
        source_condition=result.artifact.condition,
        goal_center=tuple(result.artifact.scene_manifest["goal"]),
        targets=(target,),
        trajectory=trajectory,
        termination_step=len(trajectory) - 1,
        final_position=trajectory[-1],
        outcome=result.artifact.termination_reason,
    )


def test_cost_map_source_is_explicit_and_tainted() -> None:
    none_map = build_cost_map(_condition(Router.DIRECT, ZoneSource.NONE))
    assert none_map == CostMapPayload(source=ZoneSource.NONE)

    oracle_map = build_cost_map(
        _condition(Router.DIRECT, ZoneSource.ORACLE),
        oracle_zones=((1.0, 2.0, 0.9),),
    )
    assert oracle_map.provenance == "PRIVILEGED"
    assert oracle_map.hard_zones == ((1.0, 2.0, 0.25),)
    assert oracle_map.soft_zones == ((1.0, 2.0, 1.0),)

    with pytest.raises(ValueError, match="none cost map must be empty"):
        CostMapPayload(source=ZoneSource.NONE, hard_zones=((0.0, 0.0, 1.0),))


def test_replay_source_rejects_candidate_identity_drift() -> None:
    with pytest.raises(ValueError, match="uniquely match"):
        TargetRecord(
            decision_index=0,
            world_xy=(1.0, 0.0),
            target_kind="generated_subgoal",
            selected_candidate_id=3,
            decision_step=2,
            source_position=(0.0, 0.0),
            candidates=((3, (1.1, 0.0)),),
        )


def test_replay_diagnostics_use_absorbing_endpoints() -> None:
    source = ReplaySource(
        run_id="unit",
        seed=0,
        source_condition=_condition(Router.DIRECT, ZoneSource.NONE).to_dict(),
        goal_center=(3.0, 0.0),
        targets=(),
        trajectory=((0.0, 0.0), (1.0, 0.0)),
        termination_step=1,
        final_position=(1.0, 0.0),
        outcome="goal",
    )
    diagnostics = replay_diagnostics(
        source,
        ((0.0, 0.0), (0.05, 0.0), (2.1, 0.0)),
        (),
    )
    assert diagnostics["first_divergence_step"] == 1
    assert diagnostics["endpoint_distance"] == pytest.approx(1.1)
    assert diagnostics["large_replay_divergence"] is True


def test_direct_replay_none_oracle_vertical_slice_is_auditable() -> None:
    cfg = _config()
    direct_results = {}
    for zone_source in (ZoneSource.NONE, ZoneSource.ORACLE):
        environment = PointHazardAdapter(cfg, with_renderer=False)
        try:
            direct_results[zone_source] = run_point_hazard_episode(
                environment,
                cfg,
                _condition(Router.DIRECT, zone_source),
            )
        finally:
            environment.close()

    replay_source = _source_from_result(direct_results[ZoneSource.NONE], seed=0)
    replay_source = ReplaySource.from_dict(
        json.loads(json.dumps(replay_source.to_dict()))
    )
    replay_results = {}
    for zone_source in (ZoneSource.NONE, ZoneSource.ORACLE):
        environment = PointHazardAdapter(cfg, with_renderer=False)
        try:
            replay_results[zone_source] = run_point_hazard_episode(
                environment,
                cfg,
                _condition(Router.REPLAY, zone_source),
                replay_source=replay_source,
            )
        finally:
            environment.close()

    all_results = [*direct_results.values(), *replay_results.values()]
    scene_manifests = [
        json.dumps(result.artifact.scene_manifest, sort_keys=True)
        for result in all_results
    ]
    assert len(set(scene_manifests)) == 1
    assert len({result.artifact.enforcement for result in all_results}) == 1
    assert all(result.artifact.stc_audit is not None for result in all_results)
    assert all(
        result.artifact.safe_task_completion == result.artifact.stc_audit["STC"]
        for result in all_results
    )
    assert all(
        result.artifact.policy_input_audit["call_count"]
        == len(result.artifact.trajectory) - 1
        for result in all_results
    )
    assert all(
        result.artifact.policy_input_audit["forbidden_field_count"] == 0
        and result.artifact.policy_input_audit["eval_only_tag_count"] == 0
        and result.artifact.policy_input_audit["authorized_privilege_tag_count"] == 0
        for result in all_results
    )

    for result in replay_results.values():
        assert result.replay_diagnostics is not None
        assert result.replay_diagnostics["recorded_targets_reached"] == 1
        assert result.replay_diagnostics["first_unreached_target_index"] is None
        assert result.replay_diagnostics["switch_events"][0]["switch_step"] == 0
        assert result.planner_events[0]["target_identity"] == "true_goal"
        assert result.target_sequence == [replay_source.targets[0].to_dict()]
        serialized = result.artifact.to_dict()
        assert serialized["target_sequence"] == result.target_sequence
        assert serialized["replay_diagnostics"] == result.replay_diagnostics
        assert serialized["condition_sha256"] == result.artifact.condition_sha256
        assert reconstruct_replay_audit(result.artifact, replay_source) == (
            result.replay_diagnostics
        )


def test_harness_rejects_non_frozen_replay_switch_semantics() -> None:
    enforcement = _enforcement().to_dict()
    enforcement["arrival_radius"] = 0.35
    condition = ExperimentCondition(
        router=Router.DIRECT,
        zone_source=ZoneSource.NONE,
        enforcement=EnforcementConfig.from_dict(enforcement),
        privilege_level=PrivilegeLevel.P0,
        factor_vector=FactorVector(),
        seed=0,
        split=SeedSplit.DEV,
    )
    environment = PointHazardAdapter(_config(), with_renderer=False)
    try:
        with pytest.raises(ValueError, match="arrival_radius=0.600"):
            run_point_hazard_episode(environment, _config(), condition)
    finally:
        environment.close()


def test_harness_rejects_executor_config_drift() -> None:
    cfg = _config()
    drifted_cfg = PointHazardConfig(
        n_hazards=cfg.n_hazards,
        n_semantic_zones=cfg.n_semantic_zones,
        max_episode_steps=cfg.max_episode_steps,
        dt=0.2,
    )
    environment = PointHazardAdapter(drifted_cfg, with_renderer=False)
    try:
        with pytest.raises(ValueError, match="dt does not match environment"):
            run_point_hazard_episode(
                environment,
                drifted_cfg,
                _condition(Router.DIRECT, ZoneSource.NONE),
            )
    finally:
        environment.close()


def test_semantic_terrain_truth_is_registered_independently_of_appearance() -> None:
    manifests = {}
    frames = {}
    for style in ("water", "grass"):
        cfg = _registered_config(style=style)
        environment = PointHazardAdapter(cfg, with_renderer=True)
        try:
            environment.reset(seed=0)
            manifests[style] = environment.scene_manifest()
            frames[style] = environment.render_public_rgb()
        finally:
            environment.close()

    water = manifests["water"]["semantic_terrain"][0]
    grass = manifests["grass"]["semantic_terrain"][0]
    assert water["terrain_class"] == grass["terrain_class"] == "water"
    assert water["region_id"] == grass["region_id"] == "zone_0"
    assert water["center_xy"] == grass["center_xy"]
    assert water["radius"] == grass["radius"]
    assert water["appearance_profile"] == "water-render-v1"
    assert grass["appearance_profile"] == "grass-render-v1"
    assert not np.array_equal(frames["water"], frames["grass"])


@pytest.mark.parametrize("privilege", tuple(PrivilegeLevel))
def test_registered_p0_p4_offline_vlm_harness_is_auditable(
    privilege: PrivilegeLevel,
) -> None:
    cfg = _registered_config()
    condition = _vlm_condition(privilege)
    requests = []

    def responder(request, decision_index):
        requests.append(request)
        selected = request.candidate_metadata[decision_index % 8]["candidate_id"]
        return json.dumps(
            {
                "recognized_terrain": privilege is not PrivilegeLevel.P0,
                "unsafe_candidate_ids": [],
                "selected_candidate_id": selected,
                "parse_status": "ok",
            },
            separators=(",", ":"),
        )

    environment = PointHazardAdapter(cfg, with_renderer=True)
    try:
        result = run_point_hazard_episode(
            environment,
            cfg,
            condition,
            offline_vlm_responder=responder,
        )
    finally:
        environment.close()

    assert requests
    assert len(result.artifact.vlm_calls) == len(requests)
    assert len(result.target_sequence) == len(requests)
    assert result.artifact.target_sequence == result.target_sequence
    assert result.artifact.policy_input_audit["call_count"] == len(requests)
    expected_privilege_tags = 0 if privilege is PrivilegeLevel.P0 else len(requests)
    assert (
        result.artifact.policy_input_audit["authorized_privilege_tag_count"]
        == expected_privilege_tags
    )
    assert result.artifact.policy_input_audit["forbidden_field_count"] == 0
    assert result.artifact.policy_input_audit["eval_only_tag_count"] == 0
    assert all(call["provider"] == "offline-fixture" for call in result.artifact.vlm_calls)
    assert all(call["cost_usd"] == 0.0 for call in result.artifact.vlm_calls)
    assert result.artifact.scene_manifest["semantic_terrain"][0][
        "terrain_class"
    ] == "water"


def test_registered_semantic_evaluator_respects_capability_twin() -> None:
    cfg = _registered_config()
    results = {}
    for capability in ("wheeled_non_waterproof", "amphibious"):
        environment = PointHazardAdapter(cfg, with_renderer=True)
        try:
            results[capability] = run_point_hazard_episode(
                environment,
                cfg,
                _vlm_condition(PrivilegeLevel.P0, capability=capability),
                offline_vlm_responder=lambda request, _index: json.dumps(
                    {
                        "recognized_terrain": True,
                        "unsafe_candidate_ids": [],
                        "selected_candidate_id": 1,
                        "parse_status": "ok",
                    }
                ),
            )
        finally:
            environment.close()

    assert (
        results["wheeled_non_waterproof"].artifact.scene_manifest
        == results["amphibious"].artifact.scene_manifest
    )
    assert results["amphibious"].artifact.semantic_violation is False


def test_harness_rejects_scene_appearance_factor_drift() -> None:
    cfg = _registered_config(style="grass")
    environment = PointHazardAdapter(cfg, with_renderer=True)
    try:
        with pytest.raises(ValueError, match="appearance profiles do not match"):
            run_point_hazard_episode(
                environment,
                cfg,
                _vlm_condition(PrivilegeLevel.P0),
                offline_vlm_responder=lambda _request, _index: "{}",
            )
    finally:
        environment.close()
