"""Acceptance tests for Experiment 0's provider-free executable bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.interface_contracts import EQUIVALENT_PAIRS, parse_formal_contract_response
from evaluation.interface_execution import (
    FIXTURE_PROVENANCE,
    fixture_response,
    parse_and_project_fixture,
    rendered_disk_fixture,
    shifted_grounding,
    trajectory_metrics,
)
from evaluation.interface_scenarios import SCENARIO_FAMILIES, scenario_geometry


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "interface_contract_experiment_0"


def test_native_scaled_scenario_geometry_keeps_both_endpoints_outside() -> None:
    start = np.asarray([0.98, -0.60])
    goal = np.asarray([-0.47, -0.64])
    for family in SCENARIO_FAMILIES:
        geometry = scenario_geometry(family, start_xy=start, goal_xy=goal)
        center = np.asarray(geometry["center_xy"])
        radius = float(geometry["radius"])
        assert np.linalg.norm(start - center) > radius
        assert np.linalg.norm(goal - center) > radius


@pytest.mark.parametrize("action", ["avoid", "traverse", "unknown"])
def test_equivalent_fixture_pairs_normalize_identically(action: str) -> None:
    grounding = {"center_norm": [0.25, 0.40], "radius_norm": 0.10}
    for left, right in EQUIVALENT_PAIRS.values():
        parsed = [
            parse_formal_contract_response(
                contract,
                fixture_response(contract, action, "water", grounding),
            )
            for contract in (left, right)
        ]
        assert all(item["parse_status"] == "ok" for item in parsed)
        assert parsed[0]["canonical_semantics"] == parsed[1]["canonical_semantics"]
        assert parsed[0]["canonical_grounding"] == parsed[1]["canonical_grounding"]


def test_normalized_pixel_world_calibration_and_shift_error_are_explicit() -> None:
    truth = {
        "geometry_type": "disk",
        "center_xy": [0.42, -0.61],
        "radius": 0.26,
    }
    bounds = [-2.5, 2.5, -2.5, 2.5]
    shape = (256, 256, 3)
    correct = rendered_disk_fixture(truth, bounds, shape)
    shifted = shifted_grounding(correct)
    results = []
    for grounding in (correct, shifted):
        raw = fixture_response("formal_action_structured", "avoid", "water", grounding)
        results.append(
            parse_and_project_fixture(
                "formal_action_structured",
                raw,
                image_shape=shape,
                world_bounds=bounds,
                evaluator_geometry=truth,
            )
        )
    correct_error = results[0]["calibration"]["center_error_world"]
    shifted_error = results[1]["calibration"]["center_error_world"]
    assert 0.0 <= correct_error < 0.02
    assert shifted_error > 1.0
    assert results[0]["planner_geometry"]["provenance"] == FIXTURE_PROVENANCE


def test_unknown_is_noop_task_failure() -> None:
    result = trajectory_metrics(
        [{"agent_center": [0.0, 0.0]}],
        goal_xy=[1.0, 0.0],
        evaluator_geometry={"center_xy": [0.5, 0.5], "radius": 0.2},
        capability="wheeled_non_waterproof",
        terrain_class="water",
        action="unknown",
    )
    assert result["task_failure"] is True
    assert result["task_success"] is False
    assert result["nonstationary"] is False


def test_checked_experiment_0_artifacts_are_complete_and_provider_free() -> None:
    manifest = json.loads((RESULT / "MANIFEST.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULT / "SUMMARY.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["block_count"] == 80
    assert manifest["execution_count"] == 640
    assert manifest["provider_calls"] == 0
    assert manifest["provider_attempts"] == 0
    assert summary["physical_action_IEC"] == 1.0
    assert summary["trajectory_IEC"] == 1.0
    assert all(summary["checks"].values())
    for artifact in manifest["artifacts"].values():
        path = RESULT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

