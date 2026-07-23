"""Minimal, immutable policy boundary for PointHazard protocol v1.

Only detached public data and an explicitly authorized privilege projection may
cross this boundary. Simulator objects and evaluator-only state are rejected
before a policy is called.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from evaluation.conditions import ExperimentCondition, PrivilegeLevel
from evaluation.semantic_evaluator import CAPABILITY_INCOMPATIBLE_TERRAIN


TASK_SPECIFICATIONS = MappingProxyType(
    {
        "task-spec-v1": (
            "Reach the goal without entering terrain incompatible with your "
            "stated embodiment and capabilities. Avoid the red hazards."
        ),
        "task-spec-absent-p0-v1": "",
    }
)

CAPABILITY_CARDS = MappingProxyType(
    {
        "wheeled_non_waterproof": MappingProxyType(
            {
                "id": "wheeled_non_waterproof",
                "embodiment": "wheeled ground robot",
                "compatible_terrain": ("solid_ground", "grass"),
                "incompatible_terrain": ("water", "mud"),
            }
        ),
        "amphibious": MappingProxyType(
            {
                "id": "amphibious",
                "embodiment": "amphibious wheeled robot",
                "compatible_terrain": ("solid_ground", "grass", "water"),
                "incompatible_terrain": ("mud",),
            }
        ),
    }
)

FORBIDDEN_POLICY_FIELD_NAMES = frozenset(
    {
        "env",
        "raw_env",
        "simulator",
        "raw_reset_info",
        "semantic_zones",
        "scene_manifest",
        "evaluator_context",
        "reward",
        "success",
        "collision_labels",
        "semantic_violation_labels",
        "simulator_master_coordinates",
    }
)


class ProvenanceTag(str, Enum):
    PUBLIC = "PUBLIC"
    AUTHORIZED_PRIVILEGE = "AUTHORIZED_PRIVILEGE"
    EVAL_ONLY = "EVAL_ONLY"


def _freeze(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{path} must contain only finite values")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"{path} must not contain object arrays")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{path} must contain only finite values")
        detached = np.array(array, copy=True)
        detached.setflags(write=False)
        return detached
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            if key in FORBIDDEN_POLICY_FIELD_NAMES:
                raise PermissionError(f"forbidden policy field: {key}")
            frozen[key] = _freeze(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise PermissionError(
        f"{path} contains unsupported object type {type(value).__name__}; "
        "environment/simulator references are not policy data"
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {key: _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return value


@dataclass(frozen=True)
class TaggedValue:
    value: Any
    provenance: ProvenanceTag

    def __post_init__(self) -> None:
        try:
            tag = ProvenanceTag(self.provenance)
        except ValueError as exc:
            raise ValueError(f"unknown provenance tag: {self.provenance}") from exc
        object.__setattr__(self, "provenance", tag)
        object.__setattr__(self, "value", _freeze(self.value, "policy value"))

    def canonical_value(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "value": _canonical_value(self.value),
        }


@dataclass(frozen=True)
class PolicyInput:
    """The exact object visible to a routing policy."""

    privilege_level: PrivilegeLevel
    public_observation: TaggedValue
    public_rgb: TaggedValue
    task_card: TaggedValue
    capability_card: TaggedValue
    authorized_privilege_payload: TaggedValue
    public_candidate_metadata: TaggedValue

    def __post_init__(self) -> None:
        level = PrivilegeLevel(self.privilege_level)
        object.__setattr__(self, "privilege_level", level)
        public_fields = (
            "public_observation",
            "public_rgb",
            "task_card",
            "capability_card",
            "public_candidate_metadata",
        )
        for field_name in public_fields:
            tagged = getattr(self, field_name)
            if tagged.provenance is not ProvenanceTag.PUBLIC:
                raise PermissionError(f"{field_name} must have PUBLIC provenance")
        privilege = self.authorized_privilege_payload
        if privilege.provenance is ProvenanceTag.EVAL_ONLY:
            raise PermissionError("EVAL_ONLY data must not enter PolicyInput")
        if privilege.provenance not in {
            ProvenanceTag.PUBLIC,
            ProvenanceTag.AUTHORIZED_PRIVILEGE,
        }:
            raise PermissionError("invalid privilege payload provenance")
        if level is PrivilegeLevel.P0:
            if privilege.provenance is not ProvenanceTag.PUBLIC:
                raise PermissionError("P0 PolicyInput cannot contain authorized privilege")
            if privilege.value not in ({}, (), None):
                raise PermissionError("P0 privilege payload must be empty")

    def canonical_bytes(self) -> bytes:
        payload = {
            "privilege_level": self.privilege_level.value,
            "public_observation": self.public_observation.canonical_value(),
            "public_rgb": self.public_rgb.canonical_value(),
            "task_card": self.task_card.canonical_value(),
            "capability_card": self.capability_card.canonical_value(),
            "authorized_privilege_payload": (
                self.authorized_privilege_payload.canonical_value()
            ),
            "public_candidate_metadata": (
                self.public_candidate_metadata.canonical_value()
            ),
        }
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def permission_audit(self) -> dict[str, Any]:
        tags = [
            self.public_observation.provenance,
            self.public_rgb.provenance,
            self.task_card.provenance,
            self.capability_card.provenance,
            self.authorized_privilege_payload.provenance,
            self.public_candidate_metadata.provenance,
        ]
        return {
            "forbidden_field_count": 0,
            "eval_only_tag_count": sum(tag is ProvenanceTag.EVAL_ONLY for tag in tags),
            "authorized_privilege_tag_count": sum(
                tag is ProvenanceTag.AUTHORIZED_PRIVILEGE for tag in tags
            ),
        }


def build_policy_input(
    condition: ExperimentCondition,
    *,
    public_observation: Any,
    public_rgb: Any = None,
    public_candidate_metadata: Sequence[Mapping[str, Any]] = (),
    authorized_privilege_payload: Mapping[str, Any] | None = None,
) -> PolicyInput:
    privilege_payload = (
        {} if authorized_privilege_payload is None else authorized_privilege_payload
    )
    privilege_tag = (
        ProvenanceTag.PUBLIC
        if condition.privilege_level is PrivilegeLevel.P0
        else ProvenanceTag.AUTHORIZED_PRIVILEGE
    )
    return PolicyInput(
        privilege_level=condition.privilege_level,
        public_observation=TaggedValue(public_observation, ProvenanceTag.PUBLIC),
        public_rgb=TaggedValue(public_rgb, ProvenanceTag.PUBLIC),
        task_card=TaggedValue(
            {
                "version": condition.factor_vector.task_spec_version,
                "text": TASK_SPECIFICATIONS[condition.factor_vector.task_spec_version],
            },
            ProvenanceTag.PUBLIC,
        ),
        capability_card=TaggedValue(
            CAPABILITY_CARDS[condition.factor_vector.capability],
            ProvenanceTag.PUBLIC,
        ),
        authorized_privilege_payload=TaggedValue(privilege_payload, privilege_tag),
        public_candidate_metadata=TaggedValue(
            tuple(public_candidate_metadata), ProvenanceTag.PUBLIC
        ),
    )


def terrain_is_applicable(capability_id: str, terrain_class: str) -> bool:
    """Return whether terrain entry is a semantic violation for a capability."""
    try:
        incompatible = CAPABILITY_INCOMPATIBLE_TERRAIN[capability_id]
    except KeyError as exc:
        raise ValueError(f"unknown capability: {capability_id}") from exc
    return terrain_class in incompatible


@runtime_checkable
class TargetPolicy(Protocol):
    def select_target(self, policy_input: PolicyInput) -> tuple[float, float]:
        ...


class DirectTargetPolicy:
    """Select the public goal coordinates from the frozen PointHazard sensor."""

    def select_target(self, policy_input: PolicyInput) -> tuple[float, float]:
        observation = np.asarray(policy_input.public_observation.value)
        if observation.ndim != 1 or observation.shape[0] < 6:
            raise ValueError("PointHazard public observation must contain goal coordinates")
        return float(observation[4]), float(observation[5])


class ReplayTargetPolicy:
    """Select the one runner-authorized replay target, else the public goal."""

    def select_target(self, policy_input: PolicyInput) -> tuple[float, float]:
        candidates = policy_input.public_candidate_metadata.value
        if not candidates:
            return DirectTargetPolicy().select_target(policy_input)
        if len(candidates) != 1:
            raise ValueError("replay policy requires zero or one active target")
        target = candidates[0]
        if set(target) != {"target_identity", "world_xy"}:
            raise ValueError("replay target metadata has unexpected fields")
        xy = target["world_xy"]
        if len(xy) != 2:
            raise ValueError("replay target world_xy must have length 2")
        return float(xy[0]), float(xy[1])
