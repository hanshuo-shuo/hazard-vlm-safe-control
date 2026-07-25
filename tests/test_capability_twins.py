from dataclasses import replace

import pytest

from evaluation.capability_twins import TwinArm, hash_payload, validate_appearance_twins, validate_capability_twins


def arm():
    return TwinArm(
        twin_id="water-0",
        family="water",
        seed=0,
        scene_geometry={"water": [0, 0, 1]},
        rgb_sha256="a" * 64,
        goal=[1, 1],
        initial_state=[-1, -1],
        native_cost_definition={"hazards": "native"},
        geometry_interface_hash=hash_payload({"geometry": 1}),
        planner_config_hash=hash_payload({"planner": 1}),
        capability_payload={"id": "wheeled_non_waterproof"},
        applicability_rules={"water": True},
        appearance_profile="water-render-v1",
        semantic_evaluator_profile="water-unsafe",
    )


def test_capability_twin_changes_only_capability_and_applicability():
    left = arm()
    right = replace(
        left,
        capability_payload={"id": "amphibious"},
        applicability_rules={"water": False},
    )
    assert all(validate_capability_twins(left, right).values())


def test_capability_twin_render_change_rejected():
    left = arm()
    right = replace(
        left,
        rgb_sha256="b" * 64,
        capability_payload={"id": "amphibious"},
        applicability_rules={"water": False},
    )
    with pytest.raises(ValueError, match="identical_rendering"):
        validate_capability_twins(left, right)


def test_appearance_twin_changes_designated_fields_only():
    left = arm()
    right = replace(
        left,
        rgb_sha256="b" * 64,
        appearance_profile="blue-carpet-render-v1",
        semantic_evaluator_profile="safe-surface",
    )
    assert all(validate_appearance_twins(left, right, allow_semantic_change=True).values())
