"""Behavioral checks for the new accounting and spatial-exposure boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from c3_safe.costs import Capability, Rules, relation_cost, semantic_contact
from c3_safe.geometry import GroundProjection, intersects_disk, measure_exposure, oracle_field
from envs.safety_gym_goal_adapter import SemanticSafetyPointGoalAdapter
from evaluation.interface_execution import trajectory_metrics
from evaluation.outcomes import context_outcome, episode_outcome
from evaluation.semantic_evaluator import evaluate_scene_manifest


def water(center=(0., 0.), radius=.5):
    return {"region_id": "water_0", "terrain_class": "water", "center_xy": center, "radius": radius}


def test_native_safety_event_cannot_be_counted_as_full_stc():
    metrics = trajectory_metrics(
        [{"agent_center": [-1., 1.]}, {"agent_center": [1., 1.]}],
        goal_xy=[1., 1.], evaluator_geometry=water(), capability="wheeled_non_waterproof",
        terrain_class="water", action="traverse", native_costs=[0., 2.], native_success=True,
    )
    assert metrics["semantic_safe_success"] is True
    assert metrics["collision"] is True
    assert metrics["STC"] is False


def test_nonterminal_native_cost_and_later_timeout_are_valid_outcomes():
    outcome = episode_outcome(success=False, semantic_violation=False,
                              native_costs=[1., 0.], termination_reason="timeout")
    assert outcome.timeout and outcome.physical_collision and not outcome.safe_task_completion


def test_native_goal_on_time_limit_step_has_explicit_goal_outcome():
    from envs.safety_gym_goal_adapter import _termination_reason
    reason = _termination_reason(False, True, {"goal_met":True})
    assert reason == "goal"
    assert episode_outcome(success=True,semantic_violation=False,
                           native_costs=[0.],termination_reason=reason).safe_task_completion


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.])
def test_invalid_native_cost_is_not_silently_safe(cost):
    with pytest.raises(ValueError):
        episode_outcome(success=True, semantic_violation=False,
                        native_costs=[cost], termination_reason="goal")


def test_motion_detects_crossing_and_grazing_with_endpoints_outside():
    assert intersects_disk([[-1., 0.], [1., 0.]], .1, [0., 0.], .25)
    assert intersects_disk([[.6, 0.], [.6, 0.]], .2, [0., 0.], .5)
    assert not intersects_disk([[-1., .8], [1., .8]], .2, [0., 0.], .5)


def test_cross_bypass_visual_and_capability_rule_twins():
    projection = GroundProjection.orthographic(256, 256, (-2, 2, -2, 2))
    regions = [water()]
    field = oracle_field(regions, projection)
    crossing = [[-1., 0.], [1., 0.]]
    bypass = [[-1., 0.], [-1., 1.], [1., 1.], [1., 0.]]
    cross, _ = measure_exposure(field, crossing, .15, projection)
    avoid, _ = measure_exposure(field, bypass, .15, projection)
    shifted, _ = measure_exposure(oracle_field([water((0., 1.))], projection), crossing, .15, projection)
    assert cross.validity == avoid.validity == shifted.validity == "valid"
    assert cross.values[0] > .9 and avoid.values[0] == shifted.values[0] == 0
    assert relation_cost(cross, Capability(), Rules())["capability_cost"] > 0
    assert relation_cost(cross, Capability(1., 0.), Rules())["semantic_cost"] == 0
    assert relation_cost(cross, Capability(1., 0.), Rules(avoid_water=1.))["rule_cost"] > 0
    assert semantic_contact(crossing, .15, regions, Capability(), Rules())
    assert not semantic_contact(bypass, .15, regions, Capability(), Rules())


def test_rule_off_does_not_disable_incompatibility_and_fragile_is_a_rule():
    cost = relation_cost([1., 0., 0.], Capability(), Rules())
    assert cost["capability_cost"] == 1 and cost["rule_cost"] == 0
    assert relation_cost([0., 0., 1.], Capability(), Rules())["semantic_cost"] == 0
    assert relation_cost([0., 0., 1.], Capability(), Rules(protect_fragile=1.))["rule_cost"] == 1


def test_occlusion_out_of_view_and_empty_projection_are_unknown():
    p = GroundProjection.orthographic(64, 64, (-2, 2, -2, 2))
    f = oracle_field([water()], p)
    for motion, radius, visibility in [
        ([[1.9, 0.], [2.1, 0.]], .2, None),
        ([[0., 0.], [.1, 0.]], .2, np.zeros((64, 64), dtype=bool)),
        ([[0., 0.], [0., 0.]], .00001, None),
    ]:
        exposure, _ = measure_exposure(f, motion, radius, p, visibility=visibility)
        assert exposure.validity == "unknown_projection" and exposure.values is None
        assert relation_cost(exposure, Capability(), Rules())["semantic_cost"] is None


def test_camera_projection_matches_independent_pinhole_equations():
    p = GroundProjection.mujoco_camera(320, 240, [0, 0, 5], np.eye(3), 60)
    xyz = np.array([[0., 0.], [.8, -.4], [-1., 1.]])
    uv, valid = p.project(xyz)
    focal = 120 / np.tan(np.pi / 6)
    expected = np.column_stack((160 + focal * xyz[:, 0] / 5, 120 - focal * xyz[:, 1] / 5))
    np.testing.assert_allclose(uv, expected)
    assert valid.all()


def test_swept_geometry_matches_dense_independent_motion_sampling():
    rng = np.random.default_rng(701)
    samples = np.linspace(0, 1, 4001)
    for _ in range(100):
        path = rng.uniform(-2, 2, (2, 2))
        center = rng.uniform(-1, 1, 2)
        r, body = rng.uniform(.1, .5, 2)
        dense = path[0] + samples[:, None] * (path[1] - path[0])
        expected = np.linalg.norm(dense - center, axis=1).min() <= r + body
        assert intersects_disk(path, body, center, r) == expected


class PositionedStandIn:
    def __init__(self):
        self.action_space = self.observation_space = SimpleNamespace(shape=(2,))
        self.unwrapped = self
        self.agent_pos = np.zeros(2)
        self.goal = np.array([2., 2.])

    def reset(self, *, seed=None):
        self.agent_pos[:] = [-2., -2.]
        return self.agent_pos.copy(), {}

    def step(self, action):
        self.agent_pos = np.asarray(action, dtype=float)
        return self.agent_pos.copy(), 1., 0., False, False, {}

    def render(self):
        return np.zeros((128, 128, 3), dtype=np.uint8)

    def close(self):
        pass


def test_semantic_adapter_records_violation_at_step_without_changing_native_cost():
    p = GroundProjection.orthographic(128, 128, (-3, 3, -3, 3))
    adapter = SemanticSafetyPointGoalAdapter(env=PositionedStandIn(), robot_radius=.2, projection=p)
    adapter.reset(seed=23)
    region = adapter.scene_manifest()["semantic_terrain"][0]
    result = adapter.step(region["center_xy"])
    ctx = adapter.evaluator_context()
    detached = evaluate_scene_manifest(ctx.trajectory, ctx.scene_manifest, adapter.capability)
    assert ctx.semantic_violations == (True,) == detached.per_step_violation
    assert ctx.native_costs == (0.,) and result[2] == 0. and result[5] == {}
    assert not context_outcome(ctx).safe_task_completion
    rgb = adapter.render_public_rgb()
    mask = oracle_field([region], p)[..., 0] > 0
    np.testing.assert_array_equal(np.any(rgb > 0, axis=-1), mask)
