"""Frozen, serializable experiment-condition contract for PointHazard.

The condition identifies experiment controls, not simulator truth.  All fields
in this module are evaluator/run metadata and must not be passed wholesale to a
policy.  Policy-facing projections are introduced separately in Phase 2.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


CONDITION_SCHEMA_VERSION = "point-hazard-condition-v1"


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Router(_StringEnum):
    DIRECT = "direct"
    VLM = "vlm"
    REPLAY = "replay"


class ZoneSource(_StringEnum):
    NONE = "none"
    ORACLE = "oracle"
    DETECTOR = "detector"
    VLM = "vlm"


class PrivilegeLevel(_StringEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class SeedSplit(_StringEnum):
    DEV = "dev"
    PILOT = "pilot"
    FORMAL = "formal"


SEED_RANGES = {
    SeedSplit.DEV: range(0, 50),
    SeedSplit.PILOT: range(200, 300),
    SeedSplit.FORMAL: range(300, 500),
}
BURNED_DEV_SEEDS = frozenset(range(43, 48))

REGISTERED_APPEARANCES = frozenset(
    {
        "water-render-v1",
        "mud-render-v1",
        "grass-render-v1",
        "water-mud-grass-indexed-render-v1",
        "restricted-render-dev-v0",
    }
)
REGISTERED_CAPABILITIES = frozenset(
    {"wheeled_non_waterproof", "amphibious"}
)

FIELD_PROVENANCE = MappingProxyType(
    {
        "router": "EVAL_ONLY",
        "zone_source": "EVAL_ONLY",
        "enforcement": "EVAL_ONLY",
        "privilege_level": "EVAL_ONLY",
        "factor_vector": "EVAL_ONLY",
        "seed": "EVAL_ONLY",
        "split": "EVAL_ONLY",
    }
)


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _finite_nonnegative(value: float, field_name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        comparator = "positive" if positive else "non-negative"
        raise ValueError(f"{field_name} must be a finite {comparator} number")
    return number


def _frozen_json_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        detached = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain finite JSON values") from exc
    return _freeze_json(detached)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class FactorVector:
    """The exact seven-key factor vector frozen by protocol 1.2.2."""

    appearance: str = "water-render-v1"
    task_spec_version: str = "task-spec-v1"
    capability: str = "wheeled_non_waterproof"
    privilege_level: PrivilegeLevel = PrivilegeLevel.P0
    annotation_scheme: str = "subgoal-ring-8-r2.5-v1"
    evaluator_version: str = "point-center-discrete-v1.2.1"
    protocol_version: str = "1.2.2"

    def __post_init__(self) -> None:
        try:
            level = PrivilegeLevel(self.privilege_level)
        except ValueError as exc:
            raise ValueError("privilege_level must be P0, P1, P2, P3, or P4") from exc
        object.__setattr__(self, "privilege_level", level)
        if self.appearance not in REGISTERED_APPEARANCES:
            raise ValueError(f"unregistered appearance: {self.appearance}")
        if self.capability not in REGISTERED_CAPABILITIES:
            raise ValueError(f"unregistered capability: {self.capability}")
        if self.task_spec_version not in {"task-spec-v1", "task-spec-absent-p0-v1"}:
            raise ValueError(f"unregistered task_spec_version: {self.task_spec_version}")
        if self.task_spec_version == "task-spec-absent-p0-v1" and level is not PrivilegeLevel.P0:
            raise ValueError("task-spec-absent-p0-v1 is permitted only at P0")
        if self.annotation_scheme != "subgoal-ring-8-r2.5-v1":
            raise ValueError(f"unregistered annotation_scheme: {self.annotation_scheme}")
        if self.evaluator_version != "point-center-discrete-v1.2.1":
            raise ValueError(f"unregistered evaluator_version: {self.evaluator_version}")
        if self.protocol_version != "1.2.2":
            raise ValueError(f"unregistered protocol_version: {self.protocol_version}")

    def to_dict(self) -> dict[str, str]:
        return {
            "appearance": self.appearance,
            "task_spec_version": self.task_spec_version,
            "capability": self.capability,
            "privilege_level": self.privilege_level.value,
            "annotation_scheme": self.annotation_scheme,
            "evaluator_version": self.evaluator_version,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FactorVector":
        expected = {
            "appearance",
            "task_spec_version",
            "capability",
            "privilege_level",
            "annotation_scheme",
            "evaluator_version",
            "protocol_version",
        }
        if set(value) != expected:
            raise ValueError(f"factor_vector must contain exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class EnforcementConfig:
    """All low-level safety/cost-map controls that a comparison must hold fixed."""

    enforcement_id: str
    hard_core_radius: float
    soft_halo_radius: float
    soft_zone_weight: float
    replan_interval_steps: int
    restart_on_target_change: bool
    arrival_radius: float
    planner_id: str
    executor_id: str
    planner_parameters: Mapping[str, Any] = field(default_factory=dict)
    executor_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.enforcement_id, "enforcement_id")
        _require_nonempty(self.planner_id, "planner_id")
        _require_nonempty(self.executor_id, "executor_id")
        object.__setattr__(
            self, "hard_core_radius", _finite_nonnegative(self.hard_core_radius, "hard_core_radius")
        )
        object.__setattr__(
            self, "soft_halo_radius", _finite_nonnegative(self.soft_halo_radius, "soft_halo_radius")
        )
        object.__setattr__(
            self, "soft_zone_weight", _finite_nonnegative(self.soft_zone_weight, "soft_zone_weight")
        )
        object.__setattr__(
            self,
            "arrival_radius",
            _finite_nonnegative(self.arrival_radius, "arrival_radius", positive=True),
        )
        if (
            isinstance(self.replan_interval_steps, bool)
            or not isinstance(self.replan_interval_steps, int)
            or self.replan_interval_steps < 1
        ):
            raise ValueError("replan_interval_steps must be a positive integer")
        if not isinstance(self.restart_on_target_change, bool):
            raise ValueError("restart_on_target_change must be boolean")
        object.__setattr__(
            self,
            "planner_parameters",
            _frozen_json_mapping(self.planner_parameters, "planner_parameters"),
        )
        object.__setattr__(
            self,
            "executor_parameters",
            _frozen_json_mapping(self.executor_parameters, "executor_parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enforcement_id": self.enforcement_id,
            "hard_core_radius": self.hard_core_radius,
            "soft_halo_radius": self.soft_halo_radius,
            "soft_zone_weight": self.soft_zone_weight,
            "replan_interval_steps": self.replan_interval_steps,
            "restart_on_target_change": self.restart_on_target_change,
            "arrival_radius": self.arrival_radius,
            "planner_id": self.planner_id,
            "executor_id": self.executor_id,
            "planner_parameters": _thaw_json(self.planner_parameters),
            "executor_parameters": _thaw_json(self.executor_parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnforcementConfig":
        expected = {
            "enforcement_id",
            "hard_core_radius",
            "soft_halo_radius",
            "soft_zone_weight",
            "replan_interval_steps",
            "restart_on_target_change",
            "arrival_radius",
            "planner_id",
            "executor_id",
            "planner_parameters",
            "executor_parameters",
        }
        if set(value) != expected:
            raise ValueError(f"enforcement must contain exactly {sorted(expected)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class ExperimentCondition:
    """Complete auditable condition identity for one episode."""

    router: Router
    zone_source: ZoneSource
    enforcement: EnforcementConfig
    privilege_level: PrivilegeLevel
    factor_vector: FactorVector
    seed: int
    split: SeedSplit
    schema_version: str = CONDITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            router = Router(self.router)
            zone_source = ZoneSource(self.zone_source)
            privilege_level = PrivilegeLevel(self.privilege_level)
            split = SeedSplit(self.split)
        except ValueError as exc:
            raise ValueError(f"invalid condition enum: {exc}") from exc
        object.__setattr__(self, "router", router)
        object.__setattr__(self, "zone_source", zone_source)
        object.__setattr__(self, "privilege_level", privilege_level)
        object.__setattr__(self, "split", split)
        if self.schema_version != CONDITION_SCHEMA_VERSION:
            raise ValueError(f"unsupported condition schema: {self.schema_version}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed not in SEED_RANGES[split] or (
            split is SeedSplit.DEV and self.seed in BURNED_DEV_SEEDS
        ):
            raise ValueError(f"seed {self.seed} is not allocated to split {split.value}")
        if privilege_level is not self.factor_vector.privilege_level:
            raise ValueError("top-level privilege_level must match factor_vector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "router": self.router.value,
            "zone_source": self.zone_source.value,
            "enforcement": self.enforcement.to_dict(),
            "privilege_level": self.privilege_level.value,
            "factor_vector": self.factor_vector.to_dict(),
            "seed": self.seed,
            "split": self.split.value,
            "field_provenance": dict(FIELD_PROVENANCE),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    @property
    def condition_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentCondition":
        expected = {
            "schema_version",
            "router",
            "zone_source",
            "enforcement",
            "privilege_level",
            "factor_vector",
            "seed",
            "split",
            "field_provenance",
        }
        if set(value) != expected:
            raise ValueError(f"condition must contain exactly {sorted(expected)}")
        if dict(value["field_provenance"]) != dict(FIELD_PROVENANCE):
            raise ValueError("condition field_provenance does not match the frozen registry")
        return cls(
            schema_version=value["schema_version"],
            router=value["router"],
            zone_source=value["zone_source"],
            enforcement=EnforcementConfig.from_dict(value["enforcement"]),
            privilege_level=value["privilege_level"],
            factor_vector=FactorVector.from_dict(value["factor_vector"]),
            seed=value["seed"],
            split=value["split"],
        )
