"""
Gymnasium wrapper around PointHazardEnv for Stable-Baselines3 compatibility.

SB3 requires:
  - gymnasium.Env subclass
  - observation_space / action_space as gymnasium.spaces objects
  - reset() returns (obs, info)
  - step() returns (obs, reward, terminated, truncated, info)

Usage:
    from pointmaze_copilot.env_hazard_gym import make_gym_env
    env = make_gym_env(seed=42)
    # now compatible with SB3's SAC, PPO, etc.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pointmaze_copilot.env_pointhazard import PointHazardConfig, PointHazardEnv
from pointmaze_copilot.hazard_renderer import HazardRenderer


class PointHazardGymEnv(gym.Env):
    """Gymnasium-compatible wrapper for PointHazardEnv."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        cfg: PointHazardConfig | None = None,
        render_mode: str | None = "rgb_array",
    ):
        super().__init__()
        self.cfg = cfg or PointHazardConfig()
        self._env = PointHazardEnv(cfg=self.cfg)

        # Attach renderer
        self._env.attach_renderer(HazardRenderer.from_env(self._env))

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._env.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._env.act_dim,), dtype=np.float32,
        )
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # Expose internals for expert / data collection
    @property
    def inner(self) -> PointHazardEnv:
        return self._env


def make_gym_env(
    cfg: PointHazardConfig | None = None,
    seed: int | None = None,
) -> PointHazardGymEnv:
    env = PointHazardGymEnv(cfg=cfg)
    if seed is not None:
        env.reset(seed=seed)
    return env
