"""Static and dynamic acceptance tests for the minimal policy boundary."""

from __future__ import annotations

from dataclasses import fields
import inspect

import numpy as np
import pytest

from evaluation.conditions import (
    EnforcementConfig,
    ExperimentCondition,
    FactorVector,
    PrivilegeLevel,
    Router,
    SeedSplit,
    ZoneSource,
)
from evaluation.policy_interface import (
    FORBIDDEN_POLICY_FIELD_NAMES,
    DirectTargetPolicy,
    PolicyInput,
    ProvenanceTag,
    ReplayTargetPolicy,
    TaggedValue,
    build_policy_input,
    terrain_is_applicable,
)


def _condition(
    *,
    capability: str = "wheeled_non_waterproof",
    privilege: PrivilegeLevel = PrivilegeLevel.P0,
) -> ExperimentCondition:
    return ExperimentCondition(
        router=Router.DIRECT,
        zone_source=ZoneSource.NONE,
        enforcement=EnforcementConfig(
            enforcement_id="permission-test",
            hard_core_radius=0.0,
            soft_halo_radius=0.0,
            soft_zone_weight=0.0,
            replan_interval_steps=1,
            restart_on_target_change=True,
            arrival_radius=0.6,
            planner_id="test",
            executor_id="test",
        ),
        privilege_level=privilege,
        factor_vector=FactorVector(capability=capability, privilege_level=privilege),
        seed=0,
        split=SeedSplit.DEV,
    )


def _observation() -> np.ndarray:
    return np.asarray([0, 0, 0, 0, 2.5, -1.5, 1, 1, 0.3], dtype=np.float32)


def test_static_policy_schema_and_signatures_have_no_raw_environment() -> None:
    schema_names = {field.name for field in fields(PolicyInput)}
    assert not schema_names.intersection(FORBIDDEN_POLICY_FIELD_NAMES)
    for policy_type in (DirectTargetPolicy, ReplayTargetPolicy):
        signature = inspect.signature(policy_type.select_target)
        assert tuple(signature.parameters) == ("self", "policy_input")
        assert not set(vars(policy_type())).intersection({"env", "_env", "simulator"})


def test_policy_input_is_detached_immutable_and_policy_sees_only_allowed_fields() -> None:
    observation = _observation()
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    policy_input = build_policy_input(
        _condition(),
        public_observation=observation,
        public_rgb=rgb,
    )
    observation[:] = 99
    rgb[:] = 0

    assert DirectTargetPolicy().select_target(policy_input) == (2.5, -1.5)
    assert policy_input.public_observation.value.flags.writeable is False
    assert policy_input.public_rgb.value.flags.writeable is False
    with pytest.raises(ValueError):
        policy_input.public_observation.value[0] = 1
    assert policy_input.permission_audit() == {
        "forbidden_field_count": 0,
        "eval_only_tag_count": 0,
        "authorized_privilege_tag_count": 0,
    }


def test_forbidden_fields_objects_and_eval_only_tags_fail_fast() -> None:
    with pytest.raises(PermissionError, match="forbidden policy field"):
        TaggedValue({"scene_manifest": {}}, ProvenanceTag.PUBLIC)
    with pytest.raises(PermissionError, match="unsupported object type"):
        TaggedValue(object(), ProvenanceTag.PUBLIC)

    kwargs = dict(
        privilege_level=PrivilegeLevel.P0,
        public_observation=TaggedValue(_observation(), ProvenanceTag.PUBLIC),
        public_rgb=TaggedValue(None, ProvenanceTag.PUBLIC),
        task_card=TaggedValue({}, ProvenanceTag.PUBLIC),
        capability_card=TaggedValue({}, ProvenanceTag.PUBLIC),
        public_candidate_metadata=TaggedValue((), ProvenanceTag.PUBLIC),
    )
    with pytest.raises(PermissionError, match="EVAL_ONLY"):
        PolicyInput(
            **kwargs,
            authorized_privilege_payload=TaggedValue({}, ProvenanceTag.EVAL_ONLY),
        )
    with pytest.raises(PermissionError, match="P0"):
        PolicyInput(
            **kwargs,
            authorized_privilege_payload=TaggedValue(
                {"terrain": "water"}, ProvenanceTag.AUTHORIZED_PRIVILEGE
            ),
        )


def test_evaluator_truth_mutation_cannot_change_policy_input_bytes() -> None:
    observation = _observation()
    first = build_policy_input(_condition(), public_observation=observation)
    evaluator_truth = {
        "semantic_zones": [[1.0, 2.0, 0.8]],
        "success": False,
    }
    evaluator_truth["semantic_zones"][0][0] = -99.0
    evaluator_truth["success"] = True
    second = build_policy_input(_condition(), public_observation=observation)
    assert first.canonical_bytes() == second.canonical_bytes()


def test_replay_policy_receives_only_explicit_public_target_metadata() -> None:
    policy_input = build_policy_input(
        _condition(),
        public_observation=_observation(),
        public_candidate_metadata=(
            {"target_identity": ["recorded", 0], "world_xy": [1.25, 0.5]},
        ),
    )
    assert ReplayTargetPolicy().select_target(policy_input) == (1.25, 0.5)


def test_capability_twin_changes_only_card_and_evaluator_applicability() -> None:
    observation = _observation()
    rgb = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    candidates = ({"candidate_id": 1, "world_xy": [1.0, 0.0]},)
    wheeled = build_policy_input(
        _condition(capability="wheeled_non_waterproof"),
        public_observation=observation,
        public_rgb=rgb,
        public_candidate_metadata=candidates,
    )
    amphibious = build_policy_input(
        _condition(capability="amphibious"),
        public_observation=observation,
        public_rgb=rgb,
        public_candidate_metadata=candidates,
    )

    assert np.array_equal(
        wheeled.public_observation.value, amphibious.public_observation.value
    )
    assert np.array_equal(wheeled.public_rgb.value, amphibious.public_rgb.value)
    assert (
        wheeled.public_candidate_metadata.canonical_value()
        == amphibious.public_candidate_metadata.canonical_value()
    )
    assert (
        wheeled.capability_card.canonical_value()
        != amphibious.capability_card.canonical_value()
    )
    assert terrain_is_applicable("wheeled_non_waterproof", "water") is True
    assert terrain_is_applicable("amphibious", "water") is False
