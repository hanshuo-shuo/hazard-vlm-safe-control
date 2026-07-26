"""Hard acceptance gate for native RGB, dynamics, outcomes, and coordinates."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


REQUIRED_NATIVE_ATTRIBUTES = (
    "native_rgb_source",
    "native_dynamics_source",
    "native_reward_source",
    "native_cost_source",
    "native_termination_source",
    "native_coordinate_frame",
)


def audit_native_environment(
    adapter: Any,
    *,
    seed: int,
    formal_grounding_provenance: str = "PROVIDER_ESTIMATE",
) -> dict[str, Any]:
    """Execute real reset/render/step and fail if any native requirement is absent."""
    missing = [name for name in REQUIRED_NATIVE_ATTRIBUTES if not getattr(adapter, name, None)]
    if missing:
        raise RuntimeError(f"native environment metadata is incomplete: {missing}")
    if bool(getattr(adapter, "headless_execution", True)):
        raise PermissionError("headless execution is forbidden by the formal native gate")
    if formal_grounding_provenance == "EVALUATOR_TRUTH":
        raise PermissionError("evaluator-truth grounding is forbidden in the main experiment")

    initial, reset_info = adapter.reset(seed=seed)
    rgb = np.asarray(adapter.render_public_rgb())
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError("native RGB gate requires uint8 HxWx3 output")
    action = np.zeros(adapter.action_space.shape, dtype=np.float32)
    result = adapter.step(action)
    if not isinstance(result, tuple) or len(result) != 6:
        raise RuntimeError("native step gate requires the six-value protocol API")
    _obs, reward, cost, terminated, truncated, returned_info = result
    if isinstance(reward, bool) or not np.isfinite(float(reward)):
        raise RuntimeError("native reward must be finite")
    if isinstance(cost, bool) or not np.isfinite(float(cost)):
        raise RuntimeError("native cost must be finite")
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
        raise RuntimeError("native termination flags must be boolean")
    step_count = 1
    last_reward = float(reward)
    last_cost = float(cost)
    while not bool(terminated or truncated) and step_count < 2000:
        result = adapter.step(action)
        _obs, next_reward, next_cost, terminated, truncated, returned_info = result
        if not np.isfinite(float(next_reward)) or not np.isfinite(float(next_cost)):
            raise RuntimeError("native reward/cost became non-finite")
        last_reward = float(next_reward)
        last_cost = float(next_cost)
        step_count += 1
    if not bool(terminated or truncated):
        raise RuntimeError("native environment did not emit termination/truncation within 2000 steps")
    context = adapter.evaluator_context()
    positions = [item.get("agent_center") for item in context.trajectory]
    if len(positions) < 2 or any(position is None for position in positions):
        raise RuntimeError("native-coordinate trajectory must include reset and post-step positions")
    if any(
        np.asarray(position, dtype=float).shape != (2,)
        or not np.isfinite(np.asarray(position, dtype=float)).all()
        for position in positions
    ):
        raise RuntimeError("native-coordinate trajectory contains invalid positions")
    if len(context.rewards) != step_count or len(context.native_costs) != step_count:
        raise RuntimeError("native reward/cost channels were not retained in evaluator context")
    if float(context.rewards[0]) != float(reward) or float(context.native_costs[0]) != float(cost):
        raise RuntimeError("evaluator context rewrote native reward or cost")
    if float(context.rewards[-1]) != last_reward or float(context.native_costs[-1]) != last_cost:
        raise RuntimeError("evaluator context rewrote final native reward or cost")
    if context.terminated != bool(terminated) or context.truncated != bool(truncated):
        raise RuntimeError("evaluator context rewrote native termination")
    checks = {
        "native_rgb": True,
        "native_dynamics_step_api": True,
        "native_reward": True,
        "native_cost": True,
        "native_termination": True,
        "native_coordinate_trajectory": True,
        "headless_execution_forbidden": True,
        "evaluator_truth_grounding_forbidden": True,
        "reset_info_contains_no_semantic_truth": "semantic_zones" not in reset_info,
        "step_info_contains_no_semantic_truth": "semantic_zones" not in returned_info,
    }
    if not all(checks.values()):
        raise RuntimeError(f"native environment acceptance failed: {checks}")
    return {
        "schema_version": "native-environment-gate-v1",
        "passed": True,
        "environment_backend": context.environment_backend,
        "environment_id": context.environment_id,
        "environment_version": context.environment_version,
        "seed": seed,
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "rgb_shape": list(rgb.shape),
        "native_sources": {
            name: getattr(adapter, name) for name in REQUIRED_NATIVE_ATTRIBUTES
        },
        "reward": float(reward),
        "cost": float(cost),
        "reward_total": float(sum(context.rewards)),
        "cost_total": float(sum(context.native_costs)),
        "native_steps": step_count,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "trajectory": [list(map(float, position)) for position in positions],
        "formal_grounding_provenance": formal_grounding_provenance,
        "checks": checks,
    }


def audit_native_evidence(evidence: Mapping[str, Any]) -> None:
    if evidence.get("schema_version") != "native-environment-gate-v1":
        raise ValueError("unsupported native environment evidence")
    if evidence.get("passed") is not True or not all(evidence.get("checks", {}).values()):
        raise PermissionError("native environment evidence does not pass every hard gate")
