"""Minimal policy-facing environment protocol.

The protocol deliberately does not expose the wrapped simulator object.  A
runner may use :meth:`evaluator_context` and :meth:`scene_manifest` when it is
building an artifact, but those values are evaluator-only and must never be
passed to a policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


def copy_public_value(value: Any) -> Any:
    """Copy an observation without retaining references into a simulator."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {str(key): copy_public_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(copy_public_value(item) for item in value)
    if isinstance(value, list):
        return [copy_public_value(item) for item in value]
    return deepcopy(value)


def jsonable(value: Any) -> Any:
    """Convert ordinary numpy/dataclass values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class EvaluatorContext:
    """Simulator truth and trajectory state reserved for evaluation/artifacts."""

    environment_backend: str
    environment_id: str
    environment_version: str
    scene_manifest: Mapping[str, Any]
    trajectory: tuple[Mapping[str, Any], ...]
    actions: tuple[Any, ...]
    rewards: tuple[float, ...]
    native_costs: tuple[float, ...]
    semantic_violations: tuple[bool, ...] = ()
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible evaluator-only projection."""
        return jsonable({
            "environment_backend": self.environment_backend,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "scene_manifest": dict(self.scene_manifest),
            "trajectory": list(self.trajectory),
            "actions": list(self.actions),
            "rewards": list(self.rewards),
            "native_costs": list(self.native_costs),
            "semantic_violations": list(self.semantic_violations),
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
        })


@runtime_checkable
class ProtocolEnvironment(Protocol):
    """Common interface used by policy runners for all supported backends."""

    backend: str
    environment_id: str
    environment_version: str
    action_space: Any
    observation_space: Any

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        ...

    def step(self, action: Any) -> tuple[Any, float, float, bool, bool, dict[str, Any]]:
        ...

    def public_observation(self) -> Any:
        ...

    def render_public_rgb(self) -> np.ndarray:
        ...

    def evaluator_context(self) -> EvaluatorContext:
        ...

    def scene_manifest(self) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...
