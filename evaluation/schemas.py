"""Replay-ready episode artifact schema and provenance helpers."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from envs.protocol_env import EvaluatorContext, jsonable
from evaluation.conditions import ExperimentCondition
from evaluation.vlm_artifacts import VLMCallArtifact
from evaluation.outcomes import OUTCOME_SCHEMA_VERSION, context_outcome, native_cost_violation as has_native_cost


ARTIFACT_SCHEMA_VERSION = "safety-accounting-episode-v1"
SAFETY_GYM_PROTOCOL_VERSION = "safety-gym-goal-draft-v0"


def _summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _summary(item) for key, item in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": __import__("hashlib").sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return {"length": len(value), "items": [_summary(item) for item in value[:8]]}
    if np.isscalar(value):
        return {"type": type(value).__name__}
    return {"type": type(value).__name__}


def dependency_versions() -> dict[str, str]:
    """Record reproducibility-critical versions without importing optional VLMs."""
    result = {"python": platform.python_version()}
    for name, label in (
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("mujoco", "mujoco"),
        ("safety-gymnasium", "safety_gymnasium"),
    ):
        try:
            result[label] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[label] = "not-installed"
    return result


def git_provenance() -> tuple[str, bool]:
    """Return the current SHA and dirty flag; failures are explicit metadata."""
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to record git provenance for episode artifact") from exc
    return sha_result.stdout.strip(), bool(status_result.stdout.strip())


@dataclass
class EpisodeArtifact:
    """One independent episode record shared by both supported backends."""

    protocol_version: str
    environment_backend: str
    environment_id: str
    environment_version: str
    seed: int
    initial_observation_summary: dict[str, Any]
    actions: list[Any]
    rewards: list[float]
    native_costs: list[float]
    terminated: bool
    truncated: bool
    termination_reason: str
    trajectory: list[dict[str, Any]]
    scene_manifest: dict[str, Any]
    git_sha: str
    git_dirty: bool
    dependency_versions: dict[str, str]
    semantic_violation: bool = False
    semantic_violation_steps: list[int] = field(default_factory=list)
    native_cost_violation: bool = False
    physical_collision: bool = False
    timeout: bool = False
    safe_task_completion: bool = False
    success: bool = False
    native_cost_total: float = 0.0
    router: str = "direct_goal"
    zone_source: str = "none"
    enforcement: str = "none"
    condition: dict[str, Any] | None = None
    condition_sha256: str | None = None
    target_sequence: list[dict[str, Any]] = field(default_factory=list)
    planner_events: list[dict[str, Any]] = field(default_factory=list)
    replay_diagnostics: dict[str, Any] | None = None
    cost_map: dict[str, Any] | None = None
    stc_audit: dict[str, Any] | None = None
    policy_input_audit: dict[str, Any] | None = None
    vlm_calls: list[dict[str, Any]] = field(default_factory=list)
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = {
            "artifact_schema_version": self.artifact_schema_version,
            "protocol_version": self.protocol_version,
            "environment_backend": self.environment_backend,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "seed": self.seed,
            "initial_observation_summary": self.initial_observation_summary,
            "actions": self.actions,
            "rewards": self.rewards,
            "native_costs": self.native_costs,
            "native_cost_total": self.native_cost_total,
            "semantic_violation": self.semantic_violation,
            "semantic_violation_steps": self.semantic_violation_steps,
            "native_cost_violation": self.native_cost_violation,
            "physical_collision": self.physical_collision,
            "timeout": self.timeout,
            "safe_task_completion": self.safe_task_completion,
            "semantic_safe_success": self.success and not self.semantic_violation,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "trajectory": self.trajectory,
            "scene_manifest": self.scene_manifest,
            "router": self.router,
            "zone_source": self.zone_source,
            "enforcement": self.enforcement,
            "condition": self.condition,
            "condition_sha256": self.condition_sha256,
            "target_sequence": self.target_sequence,
            "planner_events": self.planner_events,
            "replay_diagnostics": self.replay_diagnostics,
            "cost_map": self.cost_map,
            "stc_audit": self.stc_audit,
            "policy_input_audit": self.policy_input_audit,
            "vlm_calls": self.vlm_calls,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "dependency_versions": self.dependency_versions,
        }
        return jsonable(data)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False) + "\n"

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target


def build_episode_artifact(
    context: EvaluatorContext,
    *,
    seed: int,
    initial_observation: Any,
    protocol_version: str = SAFETY_GYM_PROTOCOL_VERSION,
    router: str = "direct_goal",
    zone_source: str = "none",
    enforcement: str = "none",
    condition: ExperimentCondition | None = None,
    target_sequence: Sequence[Mapping[str, Any]] = (),
    planner_events: Sequence[Mapping[str, Any]] = (),
    replay_diagnostics: Mapping[str, Any] | None = None,
    cost_map: Mapping[str, Any] | None = None,
    stc_audit: Mapping[str, Any] | None = None,
    policy_input_audit: Mapping[str, Any] | None = None,
    vlm_calls: Sequence[VLMCallArtifact | Mapping[str, Any]] = (),
) -> EpisodeArtifact:
    if condition is not None:
        if condition.seed != int(seed):
            raise ValueError("artifact seed must match condition seed")
        if condition.factor_vector.protocol_version != protocol_version:
            raise ValueError("artifact protocol_version must match condition factor vector")
        router = condition.router.value
        zone_source = condition.zone_source.value
        enforcement = condition.enforcement.enforcement_id
    git_sha, git_dirty = git_provenance()
    violation_steps = [
        index + 1 for index, violated in enumerate(context.semantic_violations) if violated
    ]
    semantic_violation = bool(violation_steps)
    outcome = context_outcome(context)
    native_cost_violation = has_native_cost(context.native_costs)
    physical_collision = outcome.physical_collision
    timeout = outcome.timeout
    computed_stc = outcome.safe_task_completion
    if stc_audit is not None:
        expected = {
            "reached_goal": bool(context.success),
            "physical_collision": physical_collision,
            "applicable_semantic_violation": semantic_violation,
            "timeout": timeout,
            "STC": computed_stc,
        }
        if dict(stc_audit) != expected:
            raise ValueError("stc_audit does not match evaluator context")
    audited_calls: list[dict[str, Any]] = []
    for call in vlm_calls:
        if condition is None:
            raise ValueError("VLM call artifacts require an experiment condition")
        record = call if isinstance(call, VLMCallArtifact) else VLMCallArtifact.from_dict(call)
        if record.condition_sha256 != condition.condition_sha256:
            raise ValueError("VLM call condition does not match episode condition")
        if (record.git_sha, record.git_dirty) != (git_sha, git_dirty):
            raise ValueError("VLM call code state does not match episode code state")
        record.audit()
        audited_calls.append(record.to_dict())
    return EpisodeArtifact(
        protocol_version=protocol_version,
        environment_backend=context.environment_backend,
        environment_id=context.environment_id,
        environment_version=context.environment_version,
        seed=int(seed),
        initial_observation_summary=_summary(initial_observation),
        actions=list(context.actions),
        rewards=list(context.rewards),
        native_costs=list(context.native_costs),
        native_cost_total=float(sum(context.native_costs)),
        semantic_violation=semantic_violation,
        semantic_violation_steps=violation_steps,
        native_cost_violation=native_cost_violation,
        physical_collision=physical_collision,
        timeout=timeout,
        safe_task_completion=computed_stc,
        success=context.success,
        terminated=context.terminated,
        truncated=context.truncated,
        termination_reason=context.termination_reason,
        trajectory=[dict(item) for item in context.trajectory],
        scene_manifest=dict(context.scene_manifest),
        git_sha=git_sha,
        git_dirty=git_dirty,
        dependency_versions=dependency_versions(),
        router=router,
        zone_source=zone_source,
        enforcement=enforcement,
        condition=None if condition is None else condition.to_dict(),
        condition_sha256=None if condition is None else condition.condition_sha256,
        target_sequence=[dict(item) for item in target_sequence],
        planner_events=[dict(item) for item in planner_events],
        replay_diagnostics=None if replay_diagnostics is None else dict(replay_diagnostics),
        cost_map=None if cost_map is None else dict(cost_map),
        stc_audit=None if stc_audit is None else dict(stc_audit),
        policy_input_audit=(
            None if policy_input_audit is None else dict(policy_input_audit)
        ),
        vlm_calls=audited_calls,
    )
