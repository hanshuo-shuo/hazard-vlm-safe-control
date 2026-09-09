"""Unified direct/replay PointHazard vertical slice.

The runner owns evaluator-only data.  The controller boundary receives only a
public observation, an absolute target, and an explicit cost-map payload.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from env_pointhazard import PointHazardConfig
from envs.protocol_env import ProtocolEnvironment, jsonable
from evaluation.conditions import (
    ExperimentCondition,
    PrivilegeLevel,
    Router,
    ZoneSource,
)
from evaluation.schemas import EpisodeArtifact, build_episode_artifact, git_provenance
from evaluation.policy_interface import (
    DirectTargetPolicy,
    ReplayTargetPolicy,
    build_policy_input,
)
from evaluation.outcomes import context_outcome
from evaluation.semantic_evaluator import evaluate_scene_manifest
from evaluation.vlm_router import (
    PreparedVLMRequest,
    prepare_vlm_request,
    run_offline_fixture_decision,
)
from mpc_expert import MPCConfig, MPCExpert


REPLAY_ARRIVAL_RADIUS = 0.600
DIVERGENCE_THRESHOLD = 0.100
LARGE_DIVERGENCE_THRESHOLD = 1.000
EXECUTOR_CONFIG_FIELDS = (
    "arena_half",
    "dt",
    "drag",
    "force_scale",
    "max_speed",
    "agent_radius",
    "goal_radius",
    "max_episode_steps",
)
_INDEXED_APPEARANCE_SEQUENCE = (
    "water-render-v1",
    "mud-render-v1",
    "grass-render-v1",
)


def _xy(value: Any, field_name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be a finite world_xy pair")
    return float(array[0]), float(array[1])


def _disks(value: Any, field_name: str) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be finite Nx3 disks")
    if np.any(array[:, 2] < 0.0):
        raise ValueError(f"{field_name} radii must be non-negative")
    return tuple(tuple(float(item) for item in row) for row in array)


def distance64(a: Sequence[float], b: Sequence[float]) -> float:
    """Protocol-v1 distance over materialized binary64 coordinates."""
    ax, ay = _xy(a, "a")
    bx, by = _xy(b, "b")
    dx = float(ax - bx)
    dy = float(ay - by)
    d2 = float(float(dx * dx) + float(dy * dy))
    return math.sqrt(d2)


def _validate_executor_config(
    env_config: PointHazardConfig,
    condition: ExperimentCondition,
) -> None:
    parameters = dict(condition.enforcement.executor_parameters)
    expected = set(EXECUTOR_CONFIG_FIELDS) | {"action_clip"}
    if set(parameters) != expected:
        raise ValueError(f"executor_parameters must contain exactly {sorted(expected)}")
    for field_name in EXECUTOR_CONFIG_FIELDS:
        if parameters[field_name] != getattr(env_config, field_name):
            raise ValueError(f"executor parameter {field_name} does not match environment")
    if parameters["action_clip"] != 1.0:
        raise ValueError("PointHazard action_clip must be 1.0")


def _validate_semantic_scene_factor(
    condition: ExperimentCondition,
    semantic_terrain: Sequence[Mapping[str, Any]],
) -> None:
    """Bind the appearance factor to registered renderer profiles."""
    if not semantic_terrain:
        return
    profiles = tuple(str(item["appearance_profile"]) for item in semantic_terrain)
    appearance = condition.factor_vector.appearance
    if appearance == "water-mud-grass-indexed-render-v1":
        expected = tuple(
            _INDEXED_APPEARANCE_SEQUENCE[
                index % len(_INDEXED_APPEARANCE_SEQUENCE)
            ]
            for index in range(len(profiles))
        )
    else:
        expected = (appearance,) * len(profiles)
    if profiles != expected:
        raise ValueError(
            "scene appearance profiles do not match condition factor: "
            f"expected {expected}, got {profiles}"
        )


@dataclass(frozen=True)
class CostMapPayload:
    """Policy-facing keep-out geometry with explicit source taint."""

    source: ZoneSource
    hard_zones: tuple[tuple[float, float, float], ...] = ()
    soft_zones: tuple[tuple[float, float, float], ...] = ()
    provenance: str = "DERIVED_PUBLIC"

    def __post_init__(self) -> None:
        source = ZoneSource(self.source)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "hard_zones", _disks(self.hard_zones, "hard_zones"))
        object.__setattr__(self, "soft_zones", _disks(self.soft_zones, "soft_zones"))
        expected = "DERIVED_PUBLIC" if source is ZoneSource.NONE else "PRIVILEGED"
        if source not in {ZoneSource.NONE, ZoneSource.ORACLE}:
            raise ValueError("vertical slice supports only none and oracle cost maps")
        if self.provenance != expected:
            raise ValueError(f"{source.value} cost map must have {expected} provenance")
        if source is ZoneSource.NONE and (self.hard_zones or self.soft_zones):
            raise ValueError("none cost map must be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "hard_zones": [list(row) for row in self.hard_zones],
            "soft_zones": [list(row) for row in self.soft_zones],
            "provenance": self.provenance,
        }


def build_cost_map(
    condition: ExperimentCondition,
    *,
    oracle_zones: Sequence[Sequence[float]] = (),
) -> CostMapPayload:
    """Project evaluator truth into the arm's explicit controller payload."""
    if condition.zone_source is ZoneSource.NONE:
        return CostMapPayload(source=ZoneSource.NONE)
    if condition.zone_source is not ZoneSource.ORACLE:
        raise ValueError("vertical slice supports only none and oracle zone sources")
    zones = _disks(oracle_zones, "oracle_zones")
    enforcement = condition.enforcement
    hard = tuple(
        (x, y, enforcement.hard_core_radius)
        for x, y, _source_radius in zones
        if enforcement.hard_core_radius > 0.0
    )
    soft = tuple(
        (x, y, enforcement.soft_halo_radius)
        for x, y, _source_radius in zones
        if enforcement.soft_halo_radius > 0.0
    )
    return CostMapPayload(
        source=ZoneSource.ORACLE,
        hard_zones=hard,
        soft_zones=soft,
        provenance="PRIVILEGED",
    )


@dataclass(frozen=True)
class TargetRecord:
    decision_index: int
    world_xy: tuple[float, float]
    target_kind: str
    selected_candidate_id: int | None
    decision_step: int
    source_position: tuple[float, float]
    candidates: tuple[tuple[int, tuple[float, float]], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
        ):
            raise ValueError("decision_index must be a non-negative integer")
        if (
            isinstance(self.decision_step, bool)
            or not isinstance(self.decision_step, int)
            or self.decision_step < 0
        ):
            raise ValueError("decision_step must be a non-negative integer")
        object.__setattr__(self, "world_xy", _xy(self.world_xy, "world_xy"))
        object.__setattr__(
            self, "source_position", _xy(self.source_position, "source_position")
        )
        if self.target_kind not in {"generated_subgoal", "true_goal"}:
            raise ValueError("target_kind must be generated_subgoal or true_goal")
        normalized_candidates = tuple(
            (int(candidate_id), _xy(candidate_xy, "candidate world_xy"))
            for candidate_id, candidate_xy in self.candidates
        )
        if any(
            isinstance(candidate_id, bool)
            or not isinstance(candidate_id, int)
            or candidate_id < 0
            for candidate_id, _candidate_xy in self.candidates
        ):
            raise ValueError("candidate ids must be non-negative integers")
        object.__setattr__(self, "candidates", normalized_candidates)
        if self.target_kind == "true_goal":
            if self.selected_candidate_id is not None:
                raise ValueError("true_goal selected_candidate_id must be null")
        else:
            if isinstance(self.selected_candidate_id, bool) or not isinstance(
                self.selected_candidate_id, int
            ):
                raise ValueError("generated subgoal requires an integer candidate id")
            matches = [
                xy for candidate_id, xy in normalized_candidates
                if candidate_id == self.selected_candidate_id
            ]
            if len(matches) != 1 or matches[0] != self.world_xy:
                raise ValueError("selected candidate must uniquely match stored world_xy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "world_xy": list(self.world_xy),
            "target_kind": self.target_kind,
            "selected_candidate_id": self.selected_candidate_id,
            "decision_step": self.decision_step,
            "source_position": list(self.source_position),
            "candidates": [
                {"candidate_id": candidate_id, "world_xy": list(world_xy)}
                for candidate_id, world_xy in self.candidates
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetRecord":
        expected = {
            "decision_index",
            "world_xy",
            "target_kind",
            "selected_candidate_id",
            "decision_step",
            "source_position",
            "candidates",
        }
        if set(value) != expected:
            raise ValueError(f"target record must contain exactly {sorted(expected)}")
        candidates = tuple(
            (item["candidate_id"], item["world_xy"]) for item in value["candidates"]
        )
        return cls(
            decision_index=value["decision_index"],
            world_xy=value["world_xy"],
            target_kind=value["target_kind"],
            selected_candidate_id=value["selected_candidate_id"],
            decision_step=value["decision_step"],
            source_position=value["source_position"],
            candidates=candidates,
        )


@dataclass(frozen=True)
class ReplaySource:
    run_id: str
    seed: int
    source_condition: Mapping[str, Any]
    goal_center: tuple[float, float]
    targets: tuple[TargetRecord, ...]
    trajectory: tuple[tuple[float, float], ...]
    termination_step: int
    final_position: tuple[float, float]
    outcome: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        try:
            detached_condition = json.loads(
                json.dumps(
                    dict(self.source_condition),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("source_condition must contain finite JSON values") from exc
        required_condition = {"router", "factor_vector", "enforcement", "seed"}
        if not required_condition.issubset(detached_condition):
            raise ValueError(
                f"source_condition must contain {sorted(required_condition)}"
            )
        if detached_condition["seed"] != self.seed:
            raise ValueError("source condition seed must match replay source seed")
        object.__setattr__(self, "source_condition", detached_condition)
        object.__setattr__(self, "goal_center", _xy(self.goal_center, "goal_center"))
        ordered = tuple(sorted(self.targets, key=lambda item: item.decision_index))
        if tuple(item.decision_index for item in ordered) != tuple(range(len(ordered))):
            raise ValueError("replay decision indices must be exactly 0..K-1")
        if any(item.target_kind != "generated_subgoal" for item in ordered):
            raise ValueError("recorded replay targets must be generated subgoals")
        object.__setattr__(self, "targets", ordered)
        trajectory = tuple(_xy(item, "trajectory position") for item in self.trajectory)
        if not trajectory:
            raise ValueError("source trajectory must include state 0")
        if (
            isinstance(self.termination_step, bool)
            or not isinstance(self.termination_step, int)
            or self.termination_step < 0
        ):
            raise ValueError("termination_step must be a non-negative integer")
        if self.termination_step != len(trajectory) - 1:
            raise ValueError("termination_step must identify the final trajectory state")
        object.__setattr__(self, "trajectory", trajectory)
        final_position = _xy(self.final_position, "final_position")
        if final_position != trajectory[-1]:
            raise ValueError("final_position must match the terminal trajectory state")
        object.__setattr__(self, "final_position", final_position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "source_condition": dict(self.source_condition),
            "goal_center": list(self.goal_center),
            "targets": [item.to_dict() for item in self.targets],
            "trajectory": [list(item) for item in self.trajectory],
            "termination_step": self.termination_step,
            "final_position": list(self.final_position),
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplaySource":
        expected = {
            "run_id",
            "seed",
            "source_condition",
            "goal_center",
            "targets",
            "trajectory",
            "termination_step",
            "final_position",
            "outcome",
        }
        if set(value) != expected:
            raise ValueError(f"replay source must contain exactly {sorted(expected)}")
        return cls(
            run_id=value["run_id"],
            seed=value["seed"],
            source_condition=value["source_condition"],
            goal_center=value["goal_center"],
            targets=tuple(TargetRecord.from_dict(item) for item in value["targets"]),
            trajectory=tuple(value["trajectory"]),
            termination_step=value["termination_step"],
            final_position=value["final_position"],
            outcome=value["outcome"],
        )


class PlanToController(Protocol):
    def plan_to(
        self,
        observation: np.ndarray,
        target: Sequence[float],
        cost_map: CostMapPayload,
    ) -> bool:
        ...

    def act(self, observation: np.ndarray) -> np.ndarray:
        ...


class MPCPlanToController:
    """MPC adapter implementing the single shared enforcement boundary."""

    def __init__(
        self,
        env_config: PointHazardConfig,
        condition: ExperimentCondition,
        *,
        rng_seed: int,
    ) -> None:
        params = dict(condition.enforcement.planner_parameters)
        aliases = {"population": "n_samples", "iterations": "n_iters"}
        params = {aliases.get(key, key): value for key, value in params.items()}
        allowed = set(MPCConfig.__dataclass_fields__)
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported MPC planner parameters: {sorted(unknown)}")
        params["soft_zone_weight"] = condition.enforcement.soft_zone_weight
        self._expert = MPCExpert(
            SimpleNamespace(cfg=env_config),
            cfg=MPCConfig(**params),
            rng=np.random.default_rng(rng_seed),
        )
        self.plan_calls: list[dict[str, Any]] = []

    def plan_to(
        self,
        observation: np.ndarray,
        target: Sequence[float],
        cost_map: CostMapPayload,
    ) -> bool:
        obs = np.asarray(observation, dtype=np.float32)
        hazards = obs[6:].reshape(-1, 3)
        hard_zones = np.asarray(cost_map.hard_zones, dtype=np.float32).reshape(-1, 3)
        avoid = np.concatenate((hazards, hard_zones), axis=0)
        self._expert.set_soft_zones(
            np.asarray(cost_map.soft_zones, dtype=np.float32).reshape(-1, 3)
        )
        ok = self._expert.plan(obs[:2], np.asarray(target, dtype=np.float32), avoid)
        self.plan_calls.append(
            {"target": list(_xy(target, "target")), "cost_map": cost_map.to_dict(), "ok": ok}
        )
        return ok

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._expert.act(np.asarray(observation, dtype=np.float32))


@dataclass
class HarnessResult:
    artifact: EpisodeArtifact
    target_sequence: list[dict[str, Any]]
    planner_events: list[dict[str, Any]]
    replay_diagnostics: dict[str, Any] | None
    cost_map: dict[str, Any]


def replay_diagnostics(
    source: ReplaySource,
    replay_trajectory: Sequence[Sequence[float]],
    switch_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    replay = tuple(_xy(item, "replay trajectory position") for item in replay_trajectory)
    if not replay:
        raise ValueError("replay trajectory must include state 0")
    horizon = max(len(source.trajectory), len(replay))
    divergences = [
        distance64(
            source.trajectory[min(index, len(source.trajectory) - 1)],
            replay[min(index, len(replay) - 1)],
        )
        for index in range(horizon)
    ]
    first = next(
        (index for index, value in enumerate(divergences) if value > DIVERGENCE_THRESHOLD),
        None,
    )
    reached = len(switch_events)
    return {
        "switch_events": [dict(event) for event in switch_events],
        "recorded_targets_reached": reached,
        "first_unreached_target_index": reached if reached < len(source.targets) else None,
        "first_divergence_step": first,
        "endpoint_distance": divergences[-1],
        "max_position_divergence": max(divergences),
        "large_replay_divergence": max(divergences) > LARGE_DIVERGENCE_THRESHOLD,
    }


def reconstruct_replay_audit(
    artifact: EpisodeArtifact,
    source: ReplaySource,
) -> dict[str, Any]:
    """Recompute the saved replay projection using only the two artifacts."""
    saved_targets = [item.to_dict() for item in source.targets]
    if artifact.target_sequence != saved_targets:
        raise ValueError("artifact target sequence does not match replay source")
    if artifact.replay_diagnostics is None:
        raise ValueError("artifact has no replay diagnostics")
    switches = artifact.replay_diagnostics.get("switch_events")
    if not isinstance(switches, list):
        raise ValueError("artifact replay diagnostics have no switch_events list")
    rebuilt = replay_diagnostics(
        source,
        [item["agent_center"] for item in artifact.trajectory],
        switches,
    )
    if jsonable(rebuilt) != jsonable(artifact.replay_diagnostics):
        raise ValueError("saved replay diagnostics do not reconstruct")
    return rebuilt


def run_point_hazard_episode(
    environment: ProtocolEnvironment,
    env_config: PointHazardConfig,
    condition: ExperimentCondition,
    *,
    replay_source: ReplaySource | None = None,
    controller: PlanToController | None = None,
    offline_vlm_responder: (
        Callable[[PreparedVLMRequest, int], str | bytes] | None
    ) = None,
) -> HarnessResult:
    """Run one direct/replay/VLM offline episode and return its audit record."""
    if condition.router not in {Router.DIRECT, Router.REPLAY, Router.VLM}:
        raise ValueError("unsupported router")
    if condition.zone_source not in {ZoneSource.NONE, ZoneSource.ORACLE}:
        raise ValueError("vertical slice supports only none and oracle zone sources")
    if condition.router is Router.VLM and condition.zone_source is not ZoneSource.NONE:
        raise ValueError("offline VLM router requires zone_source=none")
    if condition.enforcement.arrival_radius != REPLAY_ARRIVAL_RADIUS:
        raise ValueError("protocol v1 requires arrival_radius=0.600")
    if not condition.enforcement.restart_on_target_change:
        raise ValueError("protocol v1 requires restart_on_target_change=true")
    if condition.router is Router.REPLAY and replay_source is None:
        raise ValueError("replay router requires a ReplaySource")
    if condition.router is Router.DIRECT and replay_source is not None:
        raise ValueError("direct router does not accept a ReplaySource")
    if condition.router is Router.VLM and replay_source is not None:
        raise ValueError("VLM router does not accept a ReplaySource")
    if condition.router is Router.VLM and offline_vlm_responder is None:
        raise ValueError("VLM router requires an offline_vlm_responder")
    if condition.router is not Router.VLM and offline_vlm_responder is not None:
        raise ValueError("offline_vlm_responder is valid only for router=vlm")
    if replay_source is not None and replay_source.seed != condition.seed:
        raise ValueError("replay source seed must match condition seed")
    _validate_executor_config(env_config, condition)

    observation, public_reset_info = environment.reset(seed=condition.seed)
    initial_observation = np.asarray(observation, dtype=np.float32).copy()
    true_goal = _xy(public_reset_info["goal"], "goal")
    if replay_source is not None and replay_source.goal_center != true_goal:
        raise ValueError("replay source goal must match the paired scene goal")
    manifest = environment.scene_manifest()
    semantic_terrain = manifest.get("semantic_terrain", ())
    _validate_semantic_scene_factor(condition, semantic_terrain)
    if condition.router is Router.VLM and not semantic_terrain:
        raise ValueError("VLM harness requires registered semantic terrain")
    oracle_zones = manifest.get("legacy_semantic_zones", ())
    oracle_disks = [
        [*item["center_xy"], item["radius"]]
        for item in oracle_zones
    ]
    cost_map = build_cost_map(condition, oracle_zones=oracle_disks)
    executor = controller or MPCPlanToController(
        env_config, condition, rng_seed=condition.seed
    )

    records = () if replay_source is None else replay_source.targets
    target_policy = (
        DirectTargetPolicy() if replay_source is None else ReplayTargetPolicy()
    )
    target_index = 0
    switch_events: list[dict[str, Any]] = []
    planner_events: list[dict[str, Any]] = []
    target_sequence = [item.to_dict() for item in records]
    step = 0
    terminated = truncated = False
    policy_input_hashes: list[str] = []
    permission_audits: list[dict[str, Any]] = []
    vlm_calls = []
    vlm_target: tuple[float, float] | None = None
    vlm_identity: list[Any] | None = None
    code_sha, code_dirty = git_provenance()

    while not (terminated or truncated):
        position = _xy(np.asarray(observation)[:2], "agent position")
        changed = False
        if condition.router is Router.REPLAY and target_index < len(records):
            error = distance64(position, records[target_index].world_xy)
            if error <= condition.enforcement.arrival_radius:
                switch_events.append(
                    {
                        "target_index": target_index,
                        "switch_step": step,
                        "switch_position": list(position),
                        "arrival_error": error,
                    }
                )
                target_index += 1
                changed = True

        identity: str | list[Any]
        candidate_metadata: list[dict[str, Any]] = []
        if condition.router is Router.VLM:
            needs_decision = (
                vlm_target is None
                or distance64(position, vlm_target)
                <= condition.enforcement.arrival_radius
            )
            if needs_decision:
                decision_index = len(vlm_calls)
                request = prepare_vlm_request(
                    condition,
                    public_observation=np.asarray(observation),
                    public_rgb=environment.render_public_rgb(),
                    arena_half=env_config.arena_half,
                    evaluator_semantic_terrain=(
                        ()
                        if condition.privilege_level is PrivilegeLevel.P0
                        else semantic_terrain
                    ),
                )
                raw_response = offline_vlm_responder(request, decision_index)
                decision = run_offline_fixture_decision(
                    request,
                    raw_response=raw_response,
                    condition=condition,
                    call_id=(
                        f"point-hazard-seed-{condition.seed}-decision-{decision_index}"
                    ),
                    git_sha=code_sha,
                    git_dirty=code_dirty,
                    trajectory=environment.evaluator_context().trajectory,
                )
                vlm_target = decision.world_xy
                vlm_identity = ["vlm_candidate", decision_index, decision.candidate_id]
                candidates = tuple(
                    (
                        int(item["candidate_id"]),
                        tuple(float(value) for value in item["world_xy"]),
                    )
                    for item in request.candidate_metadata
                )
                record = TargetRecord(
                    decision_index=decision_index,
                    world_xy=decision.world_xy,
                    target_kind="generated_subgoal",
                    selected_candidate_id=decision.candidate_id,
                    decision_step=step,
                    source_position=position,
                    candidates=candidates,
                )
                target_sequence.append(record.to_dict())
                vlm_calls.append(decision.call_artifact)
                policy_input_hashes.append(request.policy_input.sha256)
                permission_audits.append(request.policy_input.permission_audit())
                changed = decision_index > 0
            assert vlm_target is not None and vlm_identity is not None
            identity = vlm_identity
            target = vlm_target
        elif target_index < len(records):
            identity = ["recorded", target_index]
            candidate_metadata = [
                {
                    "target_identity": identity,
                    "world_xy": list(records[target_index].world_xy),
                }
            ]
        else:
            identity = "true_goal"
        if condition.router is not Router.VLM:
            policy_input = build_policy_input(
                condition,
                public_observation=np.asarray(observation),
                public_candidate_metadata=candidate_metadata,
            )
            target = target_policy.select_target(policy_input)
            policy_input_hashes.append(policy_input.sha256)
            permission_audits.append(policy_input.permission_audit())

        due_interval = (
            step > 0
            and not changed
            and step % condition.enforcement.replan_interval_steps == 0
        )
        if step == 0 or changed or due_interval:
            ok = executor.plan_to(np.asarray(observation), target, cost_map)
            planner_events.append(
                {
                    "step": step,
                    "target_identity": identity,
                    "target_world_xy": list(target),
                    "reason": "initialize" if step == 0 else (
                        "target_change" if changed else "interval"
                    ),
                    "plan_ok": bool(ok),
                }
            )

        action = executor.act(np.asarray(observation))
        observation, _reward, _cost, terminated, truncated, _info = environment.step(action)
        step += 1

    context = environment.evaluator_context()
    if semantic_terrain:
        semantic_evaluation = evaluate_scene_manifest(
            context.trajectory,
            context.scene_manifest,
            condition.factor_vector.capability,
        )
        context = replace(
            context,
            semantic_violations=semantic_evaluation.per_step_violation,
        )
    diagnostics = None
    if replay_source is not None:
        diagnostics = replay_diagnostics(
            replay_source,
            [item["agent_center"] for item in context.trajectory],
            switch_events,
        )
    stc_audit = context_outcome(context).to_dict()
    policy_input_audit = {
        "call_count": len(policy_input_hashes),
        "input_sha256": policy_input_hashes,
        "forbidden_field_count": sum(
            item["forbidden_field_count"] for item in permission_audits
        ),
        "eval_only_tag_count": sum(
            item["eval_only_tag_count"] for item in permission_audits
        ),
        "authorized_privilege_tag_count": sum(
            item["authorized_privilege_tag_count"] for item in permission_audits
        ),
    }
    artifact = build_episode_artifact(
        context,
        seed=condition.seed,
        initial_observation=initial_observation,
        protocol_version=condition.factor_vector.protocol_version,
        condition=condition,
        target_sequence=target_sequence,
        planner_events=planner_events,
        replay_diagnostics=diagnostics,
        cost_map=cost_map.to_dict(),
        stc_audit=stc_audit,
        policy_input_audit=policy_input_audit,
        vlm_calls=vlm_calls,
    )
    return HarnessResult(
        artifact=artifact,
        target_sequence=target_sequence,
        planner_events=planner_events,
        replay_diagnostics=diagnostics,
        cost_map=cost_map.to_dict(),
    )
