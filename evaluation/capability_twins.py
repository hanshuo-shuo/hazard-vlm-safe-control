"""Capability and appearance twin contracts for provider-free evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


TWIN_SCHEMA_VERSION = "semantic-capability-twin-v1"

CAPABILITY_FAMILIES = {
    "water": {
        "incompatible": "wheeled_non_waterproof",
        "compatible": "amphibious",
        "status": "implemented",
    },
    "mud_rough": {
        "incompatible": "ordinary_wheeled",
        "compatible": "tracked_all_terrain",
        "status": "protocol_fixture",
    },
    "clearance_footprint": {
        "incompatible": "large_robot",
        "compatible": "compact_robot",
        "status": "protocol_only_physical_feasibility_TBD",
    },
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


@dataclass(frozen=True)
class TwinArm:
    twin_id: str
    family: str
    seed: int
    scene_geometry: Mapping[str, Any]
    rgb_sha256: str
    goal: Any
    initial_state: Any
    native_cost_definition: Any
    geometry_interface_hash: str
    planner_config_hash: str
    capability_payload: Mapping[str, Any]
    applicability_rules: Mapping[str, Any]
    appearance_profile: str
    semantic_evaluator_profile: str
    schema_version: str = TWIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TWIN_SCHEMA_VERSION:
            raise ValueError("unsupported twin schema")
        if len(self.rgb_sha256) != 64:
            raise ValueError("rgb_sha256 must be SHA-256")


def validate_capability_twins(left: TwinArm, right: TwinArm) -> dict[str, bool]:
    """Capability twins may differ only in capability/applicability payloads."""
    checks = {
        "same_twin_id": left.twin_id == right.twin_id,
        "same_family": left.family == right.family,
        "same_seed": left.seed == right.seed,
        "identical_geometry": _canonical(left.scene_geometry) == _canonical(right.scene_geometry),
        "identical_rendering": left.rgb_sha256 == right.rgb_sha256,
        "identical_goal": _canonical(left.goal) == _canonical(right.goal),
        "identical_initial_state": _canonical(left.initial_state) == _canonical(right.initial_state),
        "identical_native_cost": _canonical(left.native_cost_definition) == _canonical(right.native_cost_definition),
        "identical_geometry_interface": left.geometry_interface_hash == right.geometry_interface_hash,
        "identical_planner": left.planner_config_hash == right.planner_config_hash,
        "appearance_unchanged": left.appearance_profile == right.appearance_profile,
        "capability_differs": _canonical(left.capability_payload) != _canonical(right.capability_payload),
        "applicability_differs": _canonical(left.applicability_rules) != _canonical(right.applicability_rules),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid capability twin: {[key for key, ok in checks.items() if not ok]}")
    return checks


def validate_appearance_twins(
    left: TwinArm,
    right: TwinArm,
    *,
    allow_semantic_change: bool,
) -> dict[str, bool]:
    checks = {
        "same_geometry": _canonical(left.scene_geometry) == _canonical(right.scene_geometry),
        "same_capability": _canonical(left.capability_payload) == _canonical(right.capability_payload),
        "same_planner": left.planner_config_hash == right.planner_config_hash,
        "appearance_differs": left.rgb_sha256 != right.rgb_sha256,
        "semantic_change_matches_declaration": (
            (left.semantic_evaluator_profile != right.semantic_evaluator_profile)
            == allow_semantic_change
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid appearance twin: {[key for key, ok in checks.items() if not ok]}")
    return checks


def hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
