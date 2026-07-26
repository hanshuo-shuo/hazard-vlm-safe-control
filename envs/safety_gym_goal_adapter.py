"""Optional Safety-Gymnasium Goal adapters.

The dependency is imported lazily so PointHazard remains usable on machines
without MuJoCo/Safety-Gymnasium.  No simulator object is returned by this
module's policy-facing methods.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import replace
from typing import Any, Callable, Mapping

import numpy as np

from envs.protocol_env import EvaluatorContext, copy_public_value, jsonable


def _package_version(package: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _require_safety_gymnasium() -> Any:
    try:
        import safety_gymnasium
    except ImportError as exc:
        raise RuntimeError(
            "Safety-Gymnasium is optional and is not installed. Install it with "
            "`python -m pip install -r requirements-safety-gym.txt`, then rerun "
            "the SafetyPointGoal1-v0 smoke command."
        ) from exc
    return safety_gymnasium


class SafetyGymGoalAdapter:
    """Adapter for the native ``SafetyPointGoal1-v0`` six-value API."""

    backend = "safety_gym_goal"
    native_rgb_source = "Safety-Gymnasium.render(rgb_array)"
    native_dynamics_source = "Safety-Gymnasium.step"
    native_reward_source = "Safety-Gymnasium.step.reward"
    native_cost_source = "Safety-Gymnasium.step.cost"
    native_termination_source = "Safety-Gymnasium.step.terminated_truncated"
    native_coordinate_frame = "Safety-Gymnasium-world-xy"

    def __init__(
        self,
        env_id: str = "SafetyPointGoal1-v0",
        *,
        render_mode: str = "rgb_array",
        env: Any | None = None,
        env_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._native_runtime = env is None
        if env is not None:
            self._env = env
        else:
            factory = env_factory
            if factory is None:
                safety_gymnasium = _require_safety_gymnasium()
                factory = safety_gymnasium.make
            self._env = factory(env_id, render_mode=render_mode)
        self.environment_id = env_id
        self.environment_version = _package_version("safety-gymnasium", "unknown")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._obs: Any = None
        self._scene: dict[str, Any] = {}
        self._trajectory: list[dict[str, Any]] = []
        self._actions: list[Any] = []
        self._rewards: list[float] = []
        self._native_costs: list[float] = []
        self._semantic_violations: list[bool] = []
        self._success = False
        self._terminated = False
        self._truncated = False
        self._termination_reason = "unknown"
        self._seed: int | None = None

    @property
    def headless_execution(self) -> bool:
        # Injected stand-ins are accepted for unit tests but can never pass the
        # formal native-environment gate.
        return not self._native_runtime

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self._seed = seed
        result = self._env.reset(seed=seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Safety-Gymnasium reset() must return (observation, info)")
        obs, _raw_info = result
        self._obs = copy_public_value(obs)
        self._trajectory = [{"state_index": 0, "agent_center": self._agent_position()}]
        self._actions = []
        self._rewards = []
        self._native_costs = []
        self._semantic_violations = []
        self._success = False
        self._terminated = False
        self._truncated = False
        self._termination_reason = "unknown"
        self._scene = self._build_scene_manifest()
        # Raw Safety-Gymnasium info can contain implementation-specific state.
        # Keep it out of the policy-facing interface; the runner has the typed
        # evaluator context for artifact construction instead.
        return self.public_observation(), {}

    def step(self, action: Any) -> tuple[Any, float, float, bool, bool, dict[str, Any]]:
        result = self._env.step(action)
        if not isinstance(result, tuple) or len(result) != 6:
            raise RuntimeError(
                "Safety-Gymnasium step() must return "
                "(obs, reward, cost, terminated, truncated, info)"
            )
        obs, reward, cost, terminated, truncated, _raw_info = result
        self._obs = copy_public_value(obs)
        native_cost = _scalar_cost(cost)
        self._actions.append(copy_public_value(action))
        self._rewards.append(float(reward))
        self._native_costs.append(native_cost)
        self._semantic_violations.append(False)
        self._trajectory.append({"state_index": len(self._trajectory), "agent_center": self._agent_position()})
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._termination_reason = _termination_reason(bool(terminated), bool(truncated), _raw_info)
        self._success = bool(
            _raw_info.get("goal_met", _raw_info.get("goal_success", _raw_info.get("success", False)))
        )
        return self.public_observation(), float(reward), native_cost, bool(terminated), bool(truncated), {}

    def public_observation(self) -> Any:
        if self._obs is None:
            raise RuntimeError("reset() must be called before public_observation()")
        return copy_public_value(self._obs)

    def render_public_rgb(self) -> np.ndarray:
        frame = self._env.render()
        if frame is None:
            raise RuntimeError("Safety-Gymnasium render() returned no RGB frame")
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise RuntimeError(f"expected HxWx3 RGB frame, got shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb.copy()

    def scene_manifest(self) -> Mapping[str, Any]:
        return jsonable(copy_public_value(self._scene))

    def evaluator_context(self) -> EvaluatorContext:
        return EvaluatorContext(
            environment_backend=self.backend,
            environment_id=self.environment_id,
            environment_version=self.environment_version,
            scene_manifest=self.scene_manifest(),
            trajectory=tuple(jsonable(item) for item in self._trajectory),
            actions=tuple(jsonable(item) for item in self._actions),
            rewards=tuple(self._rewards),
            native_costs=tuple(self._native_costs),
            semantic_violations=tuple(self._semantic_violations),
            success=self._success,
            terminated=self._terminated,
            truncated=self._truncated,
            termination_reason=self._termination_reason,
        )

    def _build_scene_manifest(self) -> dict[str, Any]:
        unwrapped = getattr(self._env, "unwrapped", self._env)
        task = getattr(unwrapped, "task", unwrapped)
        goal_owner = getattr(task, "goal", None)
        hazards_owner = getattr(task, "hazards", None)
        goal = _find_vector(goal_owner, ("pos", "position"))
        if goal is None:
            goal = _find_vector(task, ("goal", "goal_pos"))
        hazards = _find_records(hazards_owner, id_prefix="hazard")
        task_steps = _positive_int(
            getattr(task, "num_steps", None),
            getattr(self._env, "_max_episode_steps", None),
        )
        timestep = _finite_scalar(getattr(getattr(task, "model", None), "opt", None), "timestep")
        goal_radius = _finite_scalar(goal_owner, "size")
        return {
            "environment_backend": self.backend,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "seed": self._seed,
            "initial_observation_summary": _observation_summary(self._obs),
            "goal": goal,
            "goal_radius": goal_radius,
            "agent_radius": 0.0,
            "max_episode_steps": task_steps,
            "dt": timestep,
            "physical_hazards": hazards,
            "semantic_terrain": [],
            "native_coordinate_frame": self.native_coordinate_frame,
            "native_runtime": self._native_runtime,
        }

    def _agent_position(self) -> list[float] | None:
        unwrapped = getattr(self._env, "unwrapped", self._env)
        task = getattr(unwrapped, "task", unwrapped)
        agent = getattr(task, "agent", None)
        position = _find_vector(agent, ("pos", "position"))
        if position is not None:
            return position
        position = _find_vector(task, ("agent_pos", "robot_pos", "pos"))
        if position is not None:
            return position
        for owner_name in ("agent", "robot"):
            owner = getattr(unwrapped, owner_name, None)
            position = _find_vector(owner, ("pos", "position"))
            if position is not None:
                return position
        return None

    def close(self) -> None:
        self._env.close()


class SemanticSafetyPointGoalAdapter(SafetyGymGoalAdapter):
    """Internal water-terrain variant layered on the native Goal backend.

    The water patch is a visual-only overlay.  It is not inserted into the
    MuJoCo model and never changes Safety-Gymnasium's native cost channel.
    Geometry is generated from the scene seed and lives only in the evaluator
    manifest/context.
    """

    def __init__(self, *args: Any, capability: str = "wheeled_non_waterproof", **kwargs: Any) -> None:
        if capability not in {"wheeled_non_waterproof", "amphibious"}:
            raise ValueError("capability must be wheeled_non_waterproof or amphibious")
        super().__init__(*args, **kwargs)
        self.capability = capability
        self._water_region: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        result = super().reset(seed=seed)
        if seed is None:
            raise ValueError("semantic Safety-Gymnasium reset requires an explicit scene seed")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454D]))
        center = rng.uniform(-1.0, 1.0, size=2).astype(np.float64)
        radius = float(rng.uniform(0.45, 0.65))
        self._water_region = {
            "region_id": "zone_0",
            "center_xy": center.tolist(),
            "radius": radius,
            "terrain_class": "water",
            "appearance_profile": "water-render-v1",
        }
        self._scene["semantic_terrain"] = [self._water_region]
        return result

    def render_public_rgb(self) -> np.ndarray:
        frame = super().render_public_rgb()
        if self._water_region is None:
            return frame
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        cx_world, cy_world = self._water_region["center_xy"]
        scale = min(width, height) / 6.0
        cx = int(width / 2 + cx_world * scale)
        cy = int(height / 2 - cy_world * scale)
        radius = max(2, int(self._water_region["radius"] * scale))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(60, 170, 210, 105), outline=(30, 105, 150, 230), width=3)
        for offset in (-0.35, 0.0, 0.35):
            y = cy + int(offset * radius)
            draw.arc((cx - int(0.75 * radius), y - int(0.15 * radius), cx + int(0.75 * radius), y + int(0.15 * radius)), 0, 180, fill=(240, 250, 255, 220), width=2)
        return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)

    def evaluator_context(self) -> EvaluatorContext:
        context = super().evaluator_context()
        scene = dict(context.scene_manifest)
        scene["semantic_terrain"] = [self._water_region] if self._water_region else []
        return replace(context, scene_manifest=scene)


def _scalar_cost(cost: Any) -> float:
    value = np.asarray(cost)
    if value.size != 1:
        raise RuntimeError(f"native Safety-Gymnasium cost must be scalar, got shape {value.shape}")
    return float(value.reshape(()))


def _termination_reason(terminated: bool, truncated: bool, info: Mapping[str, Any]) -> str:
    if terminated:
        return "terminated"
    if truncated:
        return "timeout"
    return "running"


def _observation_summary(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return {str(key): _observation_summary(value) for key, value in sorted(obs.items())}
    if isinstance(obs, np.ndarray):
        return {"shape": list(obs.shape), "dtype": str(obs.dtype), "sha256": hashlib.sha256(obs.tobytes()).hexdigest()}
    if np.isscalar(obs):
        return {"type": type(obs).__name__}
    return {"type": type(obs).__name__}


def _find_vector(owner: Any, names: tuple[str, ...]) -> list[float] | None:
    if owner is None:
        return None
    for name in names:
        value = getattr(owner, name, None)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size >= 2 and np.isfinite(array[:2]).all():
            return array[:2].tolist()
    return None


def _find_records(owner: Any, name: str | None = None, *, id_prefix: str = "hazard") -> list[dict[str, Any]]:
    records = getattr(owner, name, None) if name is not None and owner is not None else owner
    if records is None:
        return []
    if hasattr(records, "pos"):
        raw_positions = np.asarray(records.pos, dtype=np.float64)
        if raw_positions.ndim == 1:
            if raw_positions.size < 2:
                return []
            positions = raw_positions.reshape(1, -1)[:, :2]
        elif raw_positions.ndim >= 2 and raw_positions.shape[-1] >= 2:
            positions = raw_positions.reshape(-1, raw_positions.shape[-1])[:, :2]
        else:
            return []
        radii = np.asarray(getattr(records, "size", np.zeros(len(positions))), dtype=np.float64).reshape(-1)
        if radii.size == 1 and len(positions) > 1:
            radii = np.repeat(radii, len(positions))
        return [
            {
                f"{id_prefix}_id": f"{id_prefix}_{i}",
                "center_xy": pos.tolist(),
                "radius": float(radii[i]) if i < len(radii) else None,
            }
            for i, pos in enumerate(positions)
        ]
    return []


def _finite_scalar(owner: Any, name: str) -> float | None:
    value = getattr(owner, name, None) if owner is not None else None
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1 or not np.isfinite(array.reshape(-1)[0]):
        return None
    return float(array.reshape(-1)[0])


def _positive_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if integer > 0 and float(value) == integer:
            return integer
    return None
