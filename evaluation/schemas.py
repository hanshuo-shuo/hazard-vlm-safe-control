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
    safe_task_completion: bool = False
    success: bool = False
    native_cost_total: float = 0.0
    router: str = "direct_goal"
    zone_source: str = "none"
    enforcement: str = "none"
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
            "safe_task_completion": self.safe_task_completion,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "trajectory": self.trajectory,
            "scene_manifest": self.scene_manifest,
            "router": self.router,
            "zone_source": self.zone_source,
            "enforcement": self.enforcement,
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
) -> EpisodeArtifact:
    git_sha, git_dirty = git_provenance()
    violation_steps = [
        index + 1 for index, violated in enumerate(context.semantic_violations) if violated
    ]
    semantic_violation = bool(violation_steps)
    native_cost_violation = any(float(cost) > 0.0 for cost in context.native_costs)
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
        safe_task_completion=bool(
            context.success and not native_cost_violation and not semantic_violation
        ),
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
    )
