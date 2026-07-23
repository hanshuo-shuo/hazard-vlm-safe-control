"""Tests for the frozen experiment-condition contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from evaluation.conditions import (
    CONDITION_SCHEMA_VERSION,
    EnforcementConfig,
    ExperimentCondition,
    FactorVector,
    PrivilegeLevel,
    Router,
    SeedSplit,
    ZoneSource,
)


def _enforcement() -> EnforcementConfig:
    return EnforcementConfig(
        enforcement_id="mpc-soft-v1",
        hard_core_radius=0.2,
        soft_halo_radius=0.6,
        soft_zone_weight=40.0,
        replan_interval_steps=5,
        restart_on_target_change=True,
        arrival_radius=0.35,
        planner_id="cem-mpc-v1",
        executor_id="point-mass-v1",
        planner_parameters={"horizon": 20, "population": 256, "schedule": [1, 2]},
        executor_parameters={"action_clip": 1.0},
    )


def _condition(**overrides: object) -> ExperimentCondition:
    values = {
        "router": Router.DIRECT,
        "zone_source": ZoneSource.NONE,
        "enforcement": _enforcement(),
        "privilege_level": PrivilegeLevel.P0,
        "factor_vector": FactorVector(),
        "seed": 0,
        "split": SeedSplit.DEV,
    }
    values.update(overrides)
    return ExperimentCondition(**values)


def test_condition_round_trip_is_byte_stable() -> None:
    condition = _condition()
    restored = ExperimentCondition.from_dict(json.loads(condition.canonical_json()))

    assert restored == condition
    assert restored.canonical_json() == condition.canonical_json()
    assert restored.condition_sha256 == condition.condition_sha256
    assert restored.to_dict()["schema_version"] == CONDITION_SCHEMA_VERSION


def test_condition_requires_exact_factor_and_provenance_members() -> None:
    serialized = _condition().to_dict()
    serialized["factor_vector"]["unexpected"] = "leak"
    with pytest.raises(ValueError, match="factor_vector must contain exactly"):
        ExperimentCondition.from_dict(serialized)

    serialized = _condition().to_dict()
    serialized["field_provenance"]["router"] = "PUBLIC"
    with pytest.raises(ValueError, match="field_provenance"):
        ExperimentCondition.from_dict(serialized)


def test_privilege_and_seed_split_cross_constraints() -> None:
    with pytest.raises(ValueError, match="must match factor_vector"):
        _condition(privilege_level=PrivilegeLevel.P2)
    with pytest.raises(ValueError, match="not allocated"):
        _condition(seed=43)
    with pytest.raises(ValueError, match="not allocated"):
        _condition(seed=200, split=SeedSplit.DEV)

    pilot = _condition(seed=200, split=SeedSplit.PILOT)
    formal = _condition(seed=300, split=SeedSplit.FORMAL)
    assert pilot.split is SeedSplit.PILOT
    assert formal.split is SeedSplit.FORMAL


def test_factor_vector_enforces_protocol_1_2_2_domains() -> None:
    with pytest.raises(ValueError, match="permitted only at P0"):
        FactorVector(
            task_spec_version="task-spec-absent-p0-v1",
            privilege_level=PrivilegeLevel.P1,
        )
    with pytest.raises(ValueError, match="unregistered appearance"):
        FactorVector(appearance="unknown-render")
    with pytest.raises(ValueError, match="unregistered protocol_version"):
        FactorVector(protocol_version="1.2.1")


def test_enforcement_is_explicit_finite_and_frozen() -> None:
    enforcement = _enforcement()
    assert enforcement.to_dict()["planner_parameters"]["horizon"] == 20
    with pytest.raises(TypeError):
        enforcement.planner_parameters["horizon"] = 1
    with pytest.raises(TypeError):
        enforcement.planner_parameters["schedule"][0] = 9
    with pytest.raises(FrozenInstanceError):
        enforcement.arrival_radius = 1.0
    with pytest.raises(ValueError, match="finite non-negative"):
        EnforcementConfig(
            enforcement_id="bad",
            hard_core_radius=0.0,
            soft_halo_radius=0.0,
            soft_zone_weight=float("nan"),
            replan_interval_steps=1,
            restart_on_target_change=False,
            arrival_radius=0.1,
            planner_id="planner",
            executor_id="executor",
        )
    with pytest.raises(ValueError, match="positive integer"):
        EnforcementConfig(
            enforcement_id="bad-interval",
            hard_core_radius=0.0,
            soft_halo_radius=0.0,
            soft_zone_weight=0.0,
            replan_interval_steps=1.5,
            restart_on_target_change=False,
            arrival_radius=0.1,
            planner_id="planner",
            executor_id="executor",
        )


@pytest.mark.parametrize("router", list(Router))
@pytest.mark.parametrize("zone_source", list(ZoneSource))
def test_registered_router_zone_source_product_serializes(
    router: Router, zone_source: ZoneSource
) -> None:
    condition = _condition(router=router, zone_source=zone_source)
    assert condition.to_dict()["router"] == router.value
    assert condition.to_dict()["zone_source"] == zone_source.value
