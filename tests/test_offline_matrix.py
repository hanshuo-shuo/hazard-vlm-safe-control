"""Acceptance tests for the frozen provider-free condition matrix."""

from __future__ import annotations

import json

import pytest

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter
from evaluation.conditions import (
    EnforcementConfig,
    PrivilegeLevel,
    Router,
    ZoneSource,
)
from evaluation.harness import (
    ReplaySource,
    TargetRecord,
    run_point_hazard_episode,
)
from evaluation.offline_matrix import build_offline_gate_matrix


def _config() -> PointHazardConfig:
    return PointHazardConfig(
        n_hazards=2,
        n_semantic_zones=1,
        semantic_styles=("water",),
        semantic_terrain_classes=("water",),
        max_episode_steps=20,
        max_layout_resamples=10,
        render_size=96,
    )


def _enforcement() -> EnforcementConfig:
    cfg = _config()
    return EnforcementConfig(
        enforcement_id="offline-gate-mpc-v1",
        hard_core_radius=0.25,
        soft_halo_radius=1.0,
        soft_zone_weight=35.0,
        replan_interval_steps=5,
        restart_on_target_change=True,
        arrival_radius=0.600,
        planner_id="cem-mpc-v1",
        executor_id="point-mass-v1",
        planner_parameters={
            "horizon": 4,
            "population": 16,
            "iterations": 1,
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


def _fixture_response(request, decision_index):
    selected_id = request.candidate_metadata[decision_index % 8]["candidate_id"]
    return json.dumps(
        {
            "recognized_terrain": True,
            "unsafe_candidate_ids": [],
            "selected_candidate_id": selected_id,
            "parse_status": "ok",
        },
        separators=(",", ":"),
    )


def _run(condition):
    cfg = _config()
    environment = PointHazardAdapter(cfg, with_renderer=condition.router is Router.VLM)
    try:
        return run_point_hazard_episode(
            environment,
            cfg,
            condition,
            offline_vlm_responder=(
                _fixture_response if condition.router is Router.VLM else None
            ),
        )
    finally:
        environment.close()


def _replay_source(result) -> ReplaySource:
    trajectory = tuple(
        tuple(item["agent_center"]) for item in result.artifact.trajectory
    )
    start = trajectory[0]
    target = TargetRecord(
        decision_index=0,
        world_xy=start,
        target_kind="generated_subgoal",
        selected_candidate_id=1,
        decision_step=0,
        source_position=start,
        candidates=((1, start),),
    )
    return ReplaySource(
        run_id="offline-matrix-direct-none-source",
        seed=result.artifact.seed,
        source_condition=result.artifact.condition,
        goal_center=tuple(result.artifact.scene_manifest["goal"]),
        targets=(target,),
        trajectory=trajectory,
        termination_step=len(trajectory) - 1,
        final_position=trajectory[-1],
        outcome=result.artifact.termination_reason,
    )


def test_matrix_freezes_exact_factor_axes_and_canonical_hash() -> None:
    matrix = build_offline_gate_matrix(_enforcement())
    assert len(matrix.entries) == 14
    assert len({entry.entry_id for entry in matrix.entries}) == 14
    assert len(
        {entry.condition.condition_sha256 for entry in matrix.entries}
    ) == 14
    assert matrix.to_dict()["provider_calls_enabled"] is False
    assert all(not entry.provider_calls_enabled for entry in matrix.entries)
    assert matrix.canonical_json() == build_offline_gate_matrix(
        _enforcement()
    ).canonical_json()
    assert matrix.sha256 == build_offline_gate_matrix(_enforcement()).sha256

    shared = [
        entry
        for entry in matrix.entries
        if entry.family == "shared-router-zone-source"
    ]
    assert {
        (entry.condition.router, entry.condition.zone_source)
        for entry in shared
    } == {
        (Router.DIRECT, ZoneSource.NONE),
        (Router.DIRECT, ZoneSource.ORACLE),
        (Router.REPLAY, ZoneSource.NONE),
        (Router.REPLAY, ZoneSource.ORACLE),
    }
    vlm = [
        entry for entry in matrix.entries
        if entry.condition.router is Router.VLM
    ]
    assert {
        entry.condition.privilege_level for entry in vlm
    } == set(PrivilegeLevel)
    assert {entry.condition.zone_source for entry in vlm} == {ZoneSource.NONE}
    assert {
        entry.condition.factor_vector.capability for entry in vlm
    } == {"wheeled_non_waterproof", "amphibious"}


def test_all_fourteen_offline_conditions_execute_without_provider_calls() -> None:
    matrix = build_offline_gate_matrix(_enforcement())
    direct_entries = [
        entry
        for entry in matrix.entries
        if entry.condition.router is Router.DIRECT
    ]
    direct_results = {
        entry.condition.zone_source: _run(entry.condition)
        for entry in direct_entries
    }
    source = _replay_source(direct_results[ZoneSource.NONE])

    artifacts = [result.artifact for result in direct_results.values()]
    for entry in matrix.entries:
        condition = entry.condition
        if condition.router is Router.DIRECT:
            continue
        cfg = _config()
        environment = PointHazardAdapter(
            cfg, with_renderer=condition.router is Router.VLM
        )
        try:
            result = run_point_hazard_episode(
                environment,
                cfg,
                condition,
                replay_source=source if condition.router is Router.REPLAY else None,
                offline_vlm_responder=(
                    _fixture_response if condition.router is Router.VLM else None
                ),
            )
        finally:
            environment.close()
        artifacts.append(result.artifact)

    assert len(artifacts) == 14
    assert len({artifact.condition_sha256 for artifact in artifacts}) == 14
    assert all(
        call["provider"] == "offline-fixture"
        and call["cost_usd"] == 0.0
        and call["cli_config"]["provider_calls_enabled"] is False
        for artifact in artifacts
        for call in artifact.vlm_calls
    )
    assert all(
        not artifact.vlm_calls
        for artifact in artifacts
        if artifact.router != Router.VLM.value
    )


def test_matrix_rejects_provider_enablement() -> None:
    entry = build_offline_gate_matrix(_enforcement()).entries[0]
    with pytest.raises(ValueError, match="cannot enable provider calls"):
        type(entry)(
            entry_id=entry.entry_id,
            family=entry.family,
            condition=entry.condition,
            provider_calls_enabled=True,
        )
