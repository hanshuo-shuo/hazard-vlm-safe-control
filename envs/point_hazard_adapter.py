"""Protocol adapter for the existing PointHazard environment.

PointHazard predates Gymnasium's cost-returning API, so this adapter supplies a
zero native-cost channel and filters the raw ``info`` dictionary.  Semantic
geometry remains available only through the evaluator context/scene manifest.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from env_pointhazard import PointHazardConfig, PointHazardEnv, make_env
from envs.protocol_env import EvaluatorContext, copy_public_value, jsonable


def _version() -> str:
    return "repository-point-hazard"


class PointHazardAdapter:
    """Expose PointHazard through the permission-bounded protocol interface."""

    backend = "point_hazard"
    environment_id = "PointHazard-v1"
    environment_version = _version()

    def __init__(
        self,
        cfg: PointHazardConfig | None = None,
        *,
        seed: int | None = None,
        with_renderer: bool = True,
    ) -> None:
        self._env = make_env(cfg=cfg, seed=seed, with_renderer=with_renderer)
        self.cfg = self._env.cfg
        self.action_space = _make_box(self._env.action_low, self._env.action_high)
        obs_low = np.full(self._env.obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(self._env.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = _make_box(obs_low, obs_high)
        self._obs: np.ndarray | None = None
        self._scene: dict[str, Any] = {}
        self._trajectory: list[dict[str, Any]] = []
        self._actions: list[Any] = []
        self._rewards: list[float] = []
        self._native_costs: list[float] = []
        self._semantic_violations: list[bool] = []
        self._seed: int | None = seed
        self._success = False
        self._terminated = False
        self._truncated = False
        self._termination_reason = "unknown"

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._seed = seed
        obs, raw_info = self._env.reset(seed=seed)
        self._obs = np.asarray(obs, dtype=np.float32).copy()
        self._trajectory = [{"state_index": 0, "agent_center": self._env.pos.copy()}]
        self._actions = []
        self._rewards = []
        self._native_costs = []
        self._semantic_violations = []
        self._success = False
        self._terminated = False
        self._truncated = False
        self._termination_reason = "unknown"
        self._scene = self._build_scene_manifest(raw_info)
        # Only public sensor duplicates are returned.  In particular, semantic
        # zones, layout validity and outcome labels are not reset info.
        public_info = {
            "goal": np.asarray(raw_info["goal"], dtype=np.float32).copy(),
            "hazards": np.asarray(raw_info["hazards"], dtype=np.float32).copy(),
            "start": np.asarray(raw_info["start"], dtype=np.float32).copy(),
        }
        return self.public_observation(), public_info

    def step(self, action: Any) -> tuple[np.ndarray, float, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, raw_info = self._env.step(action)
        self._obs = np.asarray(obs, dtype=np.float32).copy()
        native_cost = 0.0
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        self._rewards.append(float(reward))
        self._native_costs.append(native_cost)
        # Legacy PointHazard semantic zones are not part of this backend's
        # frozen evaluator; do not turn their internal label into a native
        # Safety-Gym semantic result.
        self._semantic_violations.append(False)
        self._trajectory.append({
            "state_index": len(self._trajectory),
            "agent_center": self._env.pos.copy(),
        })
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._termination_reason = str(raw_info.get("termination_reason", "unknown"))
        self._success = bool(raw_info.get("goal_success", False))
        # The returned info is for the runner, not a policy.  It carries no
        # simulator geometry and no semantic labels.
        return self.public_observation(), float(reward), native_cost, bool(terminated), bool(truncated), {
            "native_cost": native_cost,
        }

    def public_observation(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("reset() must be called before public_observation()")
        return copy_public_value(self._obs)

    def render_public_rgb(self) -> np.ndarray:
        frame = self._env.render(info_text=None)
        if frame is None:
            raise RuntimeError("PointHazard renderer returned no RGB frame")
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise RuntimeError(f"expected HxWx3 RGB frame, got shape {rgb.shape}")
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

    def _build_scene_manifest(self, raw_info: Mapping[str, Any]) -> dict[str, Any]:
        legacy_zones = [
            {
                "region_id": f"zone_{i}",
                "center_xy": row[:2],
                "radius": row[2],
            }
            for i, row in enumerate(np.asarray(raw_info.get("semantic_zones", [])))
        ]
        return {
            "environment_backend": self.backend,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "seed_scene": self._seed,
            "goal": self._env.goal.copy(),
            "start": self._env.pos.copy(),
            "physical_hazards": [
                {"hazard_id": f"hazard_{i}", "center_xy": row[:2], "radius": row[2]}
                for i, row in enumerate(self._env.hazards)
            ],
            # PointHazard's legacy zones have no frozen terrain-class registry;
            # retain their geometry for audit without pretending they are the
            # new Safety-Gymnasium water terrain.
            "semantic_terrain": [],
            "legacy_semantic_zones": legacy_zones,
            "layout_valid": bool(raw_info.get("layout_valid", False)),
            "placement_attempts": int(raw_info.get("placement_attempts", 0)),
            "resample_count": int(raw_info.get("resample_count", 0)),
            "observation_summary": _observation_summary(self._obs),
        }

    def close(self) -> None:
        self._env.close()


def _observation_summary(obs: np.ndarray | None) -> dict[str, Any]:
    if obs is None:
        return {}
    arr = np.asarray(obs)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _make_box(low: np.ndarray, high: np.ndarray) -> Any:
    try:
        from gymnasium.spaces import Box
    except ImportError:
        return _ArraySpace(np.asarray(low), np.asarray(high))
    return Box(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32), dtype=np.float32)


class _ArraySpace:
    """Small fallback used only when Gymnasium is not installed."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low
        self.high = high
        self.shape = low.shape

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)
