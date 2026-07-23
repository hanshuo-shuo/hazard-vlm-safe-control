"""Gate tests for the optional Safety-Gymnasium Goal vertical slice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from envs import PointHazardAdapter, SafetyGymGoalAdapter, SemanticSafetyPointGoalAdapter
from evaluation.conditions import EnforcementConfig, ExperimentCondition, FactorVector
from evaluation.schemas import build_episode_artifact
from evaluation.semantic_evaluator import capability_twin_labels, evaluate_scene_manifest


class _FakeSpace:
    shape = (2,)
    low = np.full(2, -1.0, dtype=np.float32)
    high = np.full(2, 1.0, dtype=np.float32)


class _FakeSafetyEnv:
    """Six-value deterministic stand-in used for adapter boundary tests."""

    def __init__(self) -> None:
        self.action_space = _FakeSpace()
        self.observation_space = _FakeSpace()
        self.unwrapped = self
        self.goal = np.array([1.0, 0.0], dtype=np.float64)
        self.hazards = type("Hazards", (), {"pos": np.array([[0.5, 0.5]], dtype=np.float64), "size": np.array([0.2])})()
        self._state = np.zeros(2, dtype=np.float32)
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self._state = np.array([float(seed or 0), 0.0], dtype=np.float32)
        return self._state.copy(), {"semantic_zones": [[99, 99, 1]], "secret": "must not leak"}

    def step(self, action):
        self._state = self._state + np.asarray(action, dtype=np.float32)
        return (
            self._state.copy(),
            1.5,
            0.25,
            False,
            True,
            {"semantic_zones": [[99, 99, 1]], "goal_met": False},
        )

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_optional_adapter_smoke_and_permission_boundary() -> None:
    adapter = SafetyGymGoalAdapter(env=_FakeSafetyEnv())
    observation, reset_info = adapter.reset(seed=7)
    assert np.array_equal(observation, np.array([7.0, 0.0], dtype=np.float32))
    assert "semantic_zones" not in reset_info
    step = adapter.step(np.array([0.1, 0.0], dtype=np.float32))
    assert len(step) == 6
    assert step[2] == pytest.approx(0.25)
    assert step[5] == {}
    assert adapter.render_public_rgb().shape == (8, 8, 3)

    # A dummy policy receives only the copied public observation, never the
    # adapter, raw info, unwrapped simulator, or evaluator context.
    policy_input = adapter.public_observation()
    assert isinstance(policy_input, np.ndarray)
    assert not hasattr(policy_input, "semantic_zones")
    assert "semantic_zones" not in json.dumps(policy_input.tolist())
    adapter.close()
    assert adapter._env.closed is True


def test_deterministic_seed_and_different_seed() -> None:
    left = SafetyGymGoalAdapter(env=_FakeSafetyEnv())
    right = SafetyGymGoalAdapter(env=_FakeSafetyEnv())
    left_obs, _ = left.reset(seed=11)
    right_obs, _ = right.reset(seed=11)
    assert np.array_equal(left_obs, right_obs)
    left_step = left.step(np.zeros(2, dtype=np.float32))
    right_step = right.step(np.zeros(2, dtype=np.float32))
    assert np.array_equal(left_step[0], right_step[0])
    assert left_step[1:5] == right_step[1:5]

    other = SafetyGymGoalAdapter(env=_FakeSafetyEnv())
    other_obs, _ = other.reset(seed=12)
    assert not np.array_equal(left_obs, other_obs)
    left.close()
    right.close()
    other.close()


def test_capability_twin_and_native_cost_separation() -> None:
    scene = {
        "semantic_terrain": [
            {
                "region_id": "zone_0",
                "center_xy": [0.0, 0.0],
                "radius": 0.5,
                "terrain_class": "water",
                "appearance_profile": "water-render-v1",
            }
        ]
    }
    trajectory = [
        {"state_index": 0, "agent_center": [2.0, 0.0]},
        {"state_index": 1, "agent_center": [0.0, 0.0]},
    ]
    twins = capability_twin_labels(trajectory, scene)
    assert twins["wheeled_non_waterproof"].violation is True
    assert twins["amphibious"].violation is False

    # Semantic terrain does not synthesize or rewrite native physical cost.
    wheeled = evaluate_scene_manifest(trajectory, scene, "wheeled_non_waterproof")
    native_costs = [0.0]
    assert wheeled.violation is True
    assert native_costs == [0.0]
    assert wheeled.violation != bool(native_costs[0])


def test_semantic_adapter_capability_twins_share_scene_and_render() -> None:
    wheeled = SemanticSafetyPointGoalAdapter(
        env=_FakeSafetyEnv(), capability="wheeled_non_waterproof"
    )
    amphibious = SemanticSafetyPointGoalAdapter(
        env=_FakeSafetyEnv(), capability="amphibious"
    )
    try:
        wheeled.reset(seed=23)
        amphibious.reset(seed=23)
        assert wheeled.scene_manifest()["semantic_terrain"] == amphibious.scene_manifest()["semantic_terrain"]
        assert np.array_equal(wheeled.public_observation(), amphibious.public_observation())
        assert np.array_equal(wheeled.render_public_rgb(), amphibious.render_public_rgb())
        assert wheeled.capability != amphibious.capability
        assert wheeled.step(np.zeros(2, dtype=np.float32))[2] == pytest.approx(0.25)
        assert amphibious.step(np.zeros(2, dtype=np.float32))[2] == pytest.approx(0.25)
    finally:
        wheeled.close()
        amphibious.close()


def test_artifact_is_json_serializable_and_fields_are_separate(tmp_path: Path) -> None:
    adapter = PointHazardAdapter()
    initial, _ = adapter.reset(seed=0)
    adapter.step(np.zeros(2, dtype=np.float32))
    condition = ExperimentCondition(
        router="direct",
        zone_source="none",
        enforcement=EnforcementConfig(
            enforcement_id="none-v1",
            hard_core_radius=0.0,
            soft_halo_radius=0.0,
            soft_zone_weight=0.0,
            replan_interval_steps=1,
            restart_on_target_change=False,
            arrival_radius=0.35,
            planner_id="direct-goal-v1",
            executor_id="point-mass-v1",
        ),
        privilege_level="P0",
        factor_vector=FactorVector(),
        seed=0,
        split="dev",
    )
    artifact = build_episode_artifact(
        adapter.evaluator_context(),
        seed=0,
        initial_observation=initial,
        protocol_version="1.2.1",
        condition=condition,
    )
    path = artifact.write(tmp_path / "episode.json")
    loaded = json.loads(path.read_text())
    assert loaded["native_costs"] == [0.0]
    assert loaded["semantic_violation"] is False
    assert loaded["native_cost_violation"] is False
    assert loaded["safe_task_completion"] is False
    assert loaded["git_sha"]
    assert "python" in loaded["dependency_versions"]
    assert "semantic_violation" in loaded and "native_costs" in loaded
    assert loaded["condition"]["router"] == "direct"
    assert loaded["condition"]["factor_vector"] == FactorVector().to_dict()
    assert loaded["condition_sha256"] == condition.condition_sha256
    assert loaded["enforcement"] == "none-v1"
    adapter.close()


def test_real_safety_gym_adapter_smoke_if_installed() -> None:
    pytest.importorskip("safety_gymnasium")
    adapter = SafetyGymGoalAdapter()
    try:
        obs, _ = adapter.reset(seed=0)
        assert obs is not None
        scene = adapter.scene_manifest()
        assert scene["goal"] is not None
        assert scene["physical_hazards"]
        assert all(record["radius"] is not None for record in scene["physical_hazards"])
        action = np.zeros(adapter.action_space.shape, dtype=np.float32)
        stepped = adapter.step(action)
        assert len(stepped) == 6
        assert np.asarray(adapter.render_public_rgb()).ndim == 3
    finally:
        adapter.close()
