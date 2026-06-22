"""
PointPushHazard-v0: point agent pushes a square box to a goal while avoiding hazards.

This is a lightweight Safe-Gymnasium-style PointPush task implemented with
NumPy physics, matching the local PointHazardEnv style in this repository:

  - Open square arena with bouncing boundaries.
  - Point agent is a controlled circular body.
  - Box is an axis-aligned square slider with mass, drag, and contact impulses.
  - Red circular hazards terminate the episode if touched by either body.
  - Success is defined by the box center entering the green goal radius.
  - No MuJoCo/OpenGL dependency; rendering is handled by PIL.

Observation layout (default n_hazards=8 -> 34 dims):
    [ 0] agent_x
    [ 1] agent_y
    [ 2] agent_vx
    [ 3] agent_vy
    [ 4] box_x
    [ 5] box_y
    [ 6] box_vx
    [ 7] box_vy
    [ 8] goal_x
    [ 9] goal_y
    [10] h0_x       hazard 0 center x
    [11] h0_y       hazard 0 center y
    [12] h0_r       hazard 0 radius
    ...             (n_hazards * 3 entries)

Hazards are sorted by the minimum edge clearance to either the agent or the
box, so the first hazard entries are the most safety-relevant for the full
push system.

Action: [fx, fy] in [-1, 1]^2.  Mapped to agent force = force_scale * action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PointPushHazardConfig:
    # Arena geometry
    arena_half: float = 5.0

    # Integration and dynamics
    dt: float = 0.1
    physics_substeps: int = 4
    force_scale: float = 5.0
    agent_drag: float = 0.5
    box_drag: float = 0.8
    max_agent_speed: float = 4.0
    max_box_speed: float = 3.0
    agent_mass: float = 1.0
    box_mass: float = 2.0
    contact_restitution: float = 0.0
    contact_friction: float = 0.4
    contact_slop: float = 0.001
    contact_percent: float = 0.85
    contact_iterations: int = 2
    wall_bounce: float = 0.5

    # Body sizes
    agent_radius: float = 0.3
    box_half_size: float = 0.35
    goal_radius: float = 0.55

    # Hazards
    n_hazards: int = 8
    hazard_radius_min: float = 0.2
    hazard_radius_max: float = 0.5

    # Spawn separation
    min_hazard_pair_sep: float = 0.35
    min_agent_start_clearance: float = 0.8
    min_box_start_clearance: float = 0.8
    min_goal_clearance: float = 0.8
    min_agent_box_gap: float = 0.8
    min_box_goal_dist: float = 3.0
    start_agent_behind_box: bool = True
    start_push_gap: float = 0.22
    start_lateral_noise: float = 0.20
    require_box_goal_corridor: bool = True
    box_goal_corridor_clearance: float = 0.65
    agent_push_corridor_clearance: float = 0.45

    # Episode
    max_episode_steps: int = 350

    # Reward
    step_penalty: float = -0.01
    box_progress_reward_scale: float = 1.0
    hazard_penalty: float = -50.0
    goal_reward: float = 100.0

    # Sampling and rendering
    max_rejection_tries: int = 3000
    render_size: int = 480

    @property
    def box_radius(self) -> float:
        """Conservative radius of the rendered square box."""
        return float(self.box_half_size) * math.sqrt(2.0)


class PointPushHazardEnv:
    """
    Minimal gym-like point-push environment.

    Methods:
        reset(seed=None) -> (obs, info)
        step(action)     -> (obs, reward, terminated, truncated, info)
        render()         -> RGB numpy array if a renderer is attached
        close()          -> None
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: PointPushHazardConfig | None = None, seed: int | None = None):
        self.cfg = cfg or PointPushHazardConfig()
        self.rng = np.random.default_rng(seed)

        self.act_dim = 2
        self.cond_dim = 8
        self.obs_dim = 4 + 4 + 2 + 3 * int(self.cfg.n_hazards)
        self.action_low = -np.ones(2, dtype=np.float32)
        self.action_high = np.ones(2, dtype=np.float32)

        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.box_pos = np.zeros(2, dtype=np.float32)
        self.box_vel = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.hazards = np.zeros((self.cfg.n_hazards, 3), dtype=np.float32)
        self.t = 0
        self._agent_trail: list[np.ndarray] = []
        self._box_trail: list[np.ndarray] = []
        self._renderer = None

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_xy(self, margin: float) -> np.ndarray:
        lo = -self.cfg.arena_half + float(margin)
        hi = +self.cfg.arena_half - float(margin)
        return self.rng.uniform(lo, hi, size=2).astype(np.float32)

    def _clear_of_all_hazards_circle(
        self,
        p: np.ndarray,
        radius: float,
        *,
        extra: float = 0.0,
    ) -> bool:
        p = np.asarray(p, dtype=np.float32).reshape(2)
        for hx, hy, hr in self.hazards:
            hxy = np.array([hx, hy], dtype=np.float32)
            d = float(np.linalg.norm(p - hxy))
            if d < (float(radius) + float(hr) + float(extra)):
                return False
        return True

    def _box_clear_of_all_hazards(self, p: np.ndarray, *, extra: float = 0.0) -> bool:
        p = np.asarray(p, dtype=np.float32).reshape(2)
        half = float(self.cfg.box_half_size)
        for hx, hy, hr in self.hazards:
            center = np.array([hx, hy], dtype=np.float32)
            if self._square_circle_clearance(p, half, center, float(hr)) < float(extra):
                return False
        return True

    def _box_segment_clear_of_all_hazards(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        extra: float = 0.0,
    ) -> bool:
        start = np.asarray(start, dtype=np.float32).reshape(2)
        end = np.asarray(end, dtype=np.float32).reshape(2)
        dist = float(np.linalg.norm(end - start))
        n = max(2, int(math.ceil(dist / 0.15)))
        for i in range(n + 1):
            alpha = float(i) / float(n)
            p = (1.0 - alpha) * start + alpha * end
            if not self._box_clear_of_all_hazards(p.astype(np.float32), extra=extra):
                return False
        return True

    def _circle_segment_clear_of_all_hazards(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        radius: float,
        extra: float = 0.0,
    ) -> bool:
        start = np.asarray(start, dtype=np.float32).reshape(2)
        end = np.asarray(end, dtype=np.float32).reshape(2)
        dist = float(np.linalg.norm(end - start))
        n = max(2, int(math.ceil(dist / 0.15)))
        for i in range(n + 1):
            alpha = float(i) / float(n)
            p = (1.0 - alpha) * start + alpha * end
            if not self._clear_of_all_hazards_circle(
                p.astype(np.float32), radius, extra=extra
            ):
                return False
        return True

    def _agent_box_overlaps_at(self, agent_pos: np.ndarray, box_pos: np.ndarray) -> bool:
        _normal, penetration = self._agent_box_contact(agent_pos, box_pos)
        return bool(penetration > 0.0)

    def _initial_push_pose_for(self, box: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        cfg = self.cfg
        push_vec = np.asarray(goal, dtype=np.float32) - np.asarray(box, dtype=np.float32)
        push_norm = float(np.linalg.norm(push_vec))
        if push_norm <= 1e-6:
            return None
        push_dir = push_vec / push_norm
        base_dist = (
            float(cfg.box_half_size)
            + float(cfg.agent_radius)
            + float(cfg.start_push_gap)
            + 0.05
        )
        pose = np.asarray(box, dtype=np.float32) - push_dir * base_dist
        limit = float(cfg.arena_half) - float(cfg.agent_radius) - 0.05
        if np.any(pose < -limit) or np.any(pose > limit):
            return None
        return pose.astype(np.float32)

    def _sample_layout(self) -> None:
        cfg = self.cfg

        hazards: list[np.ndarray] = []
        tries = 0
        while len(hazards) < int(cfg.n_hazards):
            tries += 1
            if tries > int(cfg.max_rejection_tries):
                raise RuntimeError(
                    f"Failed to sample {cfg.n_hazards} hazards; relax separation."
                )
            r = float(self.rng.uniform(cfg.hazard_radius_min, cfg.hazard_radius_max))
            xy = self._sample_xy(margin=r + 0.1)
            ok = True
            for hx, hy, hr in hazards:
                d = float(np.linalg.norm(xy - np.array([hx, hy], dtype=np.float32)))
                if d < (r + float(hr) + float(cfg.min_hazard_pair_sep)):
                    ok = False
                    break
            if ok:
                hazards.append(np.array([xy[0], xy[1], r], dtype=np.float32))
        self.hazards = np.stack(hazards, axis=0)

        for _ in range(int(cfg.max_rejection_tries)):
            goal = self._sample_xy(margin=max(cfg.goal_radius, cfg.box_radius) + 0.1)
            if self._clear_of_all_hazards_circle(
                goal, cfg.box_radius, extra=cfg.min_goal_clearance
            ):
                break
        else:
            raise RuntimeError("Failed to sample valid box goal.")
        self.goal = goal

        box = None
        box_corridor_phases = [
            float(cfg.box_goal_corridor_clearance),
            max(0.45, float(cfg.box_goal_corridor_clearance) * 0.75),
            max(0.30, float(cfg.box_goal_corridor_clearance) * 0.55),
            0.15,
        ]
        agent_corridor_phases = [
            float(cfg.agent_push_corridor_clearance),
            max(0.30, float(cfg.agent_push_corridor_clearance) * 0.75),
            max(0.20, float(cfg.agent_push_corridor_clearance) * 0.55),
            0.10,
        ]
        tries_per_phase = max(1, int(cfg.max_rejection_tries))
        for box_extra, agent_extra in zip(box_corridor_phases, agent_corridor_phases):
            for _ in range(tries_per_phase):
                candidate = self._sample_xy(margin=cfg.box_half_size + 0.1)
                if not self._box_clear_of_all_hazards(candidate, extra=cfg.min_box_start_clearance):
                    continue
                if float(np.linalg.norm(candidate - self.goal)) < float(cfg.min_box_goal_dist):
                    continue
                if bool(cfg.require_box_goal_corridor) and not self._box_segment_clear_of_all_hazards(
                    candidate,
                    self.goal,
                    extra=box_extra,
                ):
                    continue
                push_pose = self._initial_push_pose_for(candidate, self.goal)
                if push_pose is None:
                    continue
                if not self._clear_of_all_hazards_circle(
                    push_pose, cfg.agent_radius, extra=cfg.min_agent_start_clearance
                ):
                    continue
                push_vec = self.goal - candidate
                push_dir = push_vec / max(1e-6, float(np.linalg.norm(push_vec)))
                final_push_pose = self.goal - push_dir * float(np.linalg.norm(candidate - push_pose))
                if not self._circle_segment_clear_of_all_hazards(
                    push_pose,
                    final_push_pose,
                    radius=cfg.agent_radius,
                    extra=agent_extra,
                ):
                    continue
                box = candidate
                break
            if box is not None:
                break

        if box is None:
            raise RuntimeError("Failed to sample valid box start.")
        self.box_pos = box
        self.box_vel = np.zeros(2, dtype=np.float32)

        start = None
        if bool(cfg.start_agent_behind_box):
            push_vec = self.goal - self.box_pos
            push_norm = float(np.linalg.norm(push_vec))
            if push_norm > 1e-6:
                push_dir = push_vec / push_norm
                lateral = np.array([-push_dir[1], push_dir[0]], dtype=np.float32)
                base_dist = (
                    float(cfg.box_half_size)
                    + float(cfg.agent_radius)
                    + float(cfg.start_push_gap)
                )
                for _ in range(int(cfg.max_rejection_tries)):
                    lateral_offset = float(self.rng.normal(0.0, cfg.start_lateral_noise))
                    dist_jitter = float(self.rng.uniform(-0.05, 0.20))
                    candidate = (
                        self.box_pos
                        - push_dir * (base_dist + dist_jitter)
                        + lateral * lateral_offset
                    ).astype(np.float32)
                    limit = float(cfg.arena_half) - float(cfg.agent_radius)
                    if np.any(candidate < -limit) or np.any(candidate > limit):
                        continue
                    if not self._clear_of_all_hazards_circle(
                        candidate, cfg.agent_radius, extra=cfg.min_agent_start_clearance
                    ):
                        continue
                    if self._agent_box_overlaps_at(candidate, self.box_pos):
                        continue
                    start = candidate
                    break

        if start is None:
            for _ in range(int(cfg.max_rejection_tries)):
                candidate = self._sample_xy(margin=cfg.agent_radius + 0.1)
                if not self._clear_of_all_hazards_circle(
                    candidate, cfg.agent_radius, extra=cfg.min_agent_start_clearance
                ):
                    continue
                if self._agent_box_overlaps_at(candidate, self.box_pos):
                    continue
                if float(np.linalg.norm(candidate - self.box_pos)) < (
                    cfg.agent_radius + cfg.box_radius + cfg.min_agent_box_gap
                ):
                    continue
                start = candidate
                break

        if start is None:
            raise RuntimeError("Failed to sample valid agent start.")
        self.pos = start
        self.vel = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _square_circle_clearance(
        square_xy: np.ndarray,
        half_size: float,
        circle_xy: np.ndarray,
        circle_radius: float,
    ) -> float:
        square_xy = np.asarray(square_xy, dtype=np.float32).reshape(2)
        circle_xy = np.asarray(circle_xy, dtype=np.float32).reshape(2)
        delta = circle_xy - square_xy
        closest_delta = np.clip(delta, -float(half_size), float(half_size))
        outside_delta = delta - closest_delta
        return float(np.linalg.norm(outside_delta) - float(circle_radius))

    def _agent_box_contact(
        self,
        agent_pos: np.ndarray,
        box_pos: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        agent_pos = np.asarray(agent_pos, dtype=np.float32).reshape(2)
        box_pos = np.asarray(box_pos, dtype=np.float32).reshape(2)
        rel = agent_pos - box_pos
        half = float(self.cfg.box_half_size)
        closest_rel = np.clip(rel, -half, half)
        closest = box_pos + closest_rel
        delta = agent_pos - closest
        dist = float(np.linalg.norm(delta))

        if dist > 1e-8:
            penetration = float(self.cfg.agent_radius) - dist
            if penetration <= 0.0:
                return np.zeros(2, dtype=np.float32), float(penetration)
            normal = delta / dist
            return normal.astype(np.float32), float(penetration)

        # Degenerate case: the agent center is inside the square projection.
        side_gaps = np.array(
            [
                half - rel[0],
                half + rel[0],
                half - rel[1],
                half + rel[1],
            ],
            dtype=np.float32,
        )
        side = int(np.argmin(side_gaps))
        if side == 0:
            normal = np.array([1.0, 0.0], dtype=np.float32)
        elif side == 1:
            normal = np.array([-1.0, 0.0], dtype=np.float32)
        elif side == 2:
            normal = np.array([0.0, 1.0], dtype=np.float32)
        else:
            normal = np.array([0.0, -1.0], dtype=np.float32)
        penetration = float(self.cfg.agent_radius) + float(side_gaps[side])
        return normal, penetration

    def _clip_speed(self, vel: np.ndarray, max_speed: float) -> np.ndarray:
        speed = float(np.linalg.norm(vel))
        if speed > float(max_speed):
            vel = vel * (float(max_speed) / speed)
        return vel.astype(np.float32)

    def _handle_agent_wall(self) -> None:
        limit = float(self.cfg.arena_half) - float(self.cfg.agent_radius)
        for i in range(2):
            if self.pos[i] < -limit:
                self.pos[i] = -limit
                self.vel[i] = -self.vel[i] * float(self.cfg.wall_bounce)
            elif self.pos[i] > limit:
                self.pos[i] = limit
                self.vel[i] = -self.vel[i] * float(self.cfg.wall_bounce)

    def _handle_box_wall(self) -> None:
        limit = float(self.cfg.arena_half) - float(self.cfg.box_half_size)
        for i in range(2):
            if self.box_pos[i] < -limit:
                self.box_pos[i] = -limit
                self.box_vel[i] = -self.box_vel[i] * float(self.cfg.wall_bounce)
            elif self.box_pos[i] > limit:
                self.box_pos[i] = limit
                self.box_vel[i] = -self.box_vel[i] * float(self.cfg.wall_bounce)

    def _resolve_agent_box_collision(self) -> bool:
        normal, penetration = self._agent_box_contact(self.pos, self.box_pos)
        if penetration <= float(self.cfg.contact_slop) or float(np.linalg.norm(normal)) < 1e-8:
            return False

        inv_agent = 1.0 / max(1e-6, float(self.cfg.agent_mass))
        inv_box = 1.0 / max(1e-6, float(self.cfg.box_mass))
        inv_sum = inv_agent + inv_box

        correction_mag = (
            max(float(penetration) - float(self.cfg.contact_slop), 0.0)
            / inv_sum
            * float(self.cfg.contact_percent)
        )
        correction = correction_mag * normal
        self.pos = (self.pos + correction * inv_agent).astype(np.float32)
        self.box_pos = (self.box_pos - correction * inv_box).astype(np.float32)

        rel_vel = self.vel - self.box_vel
        vel_along_normal = float(np.dot(rel_vel, normal))
        if vel_along_normal < 0.0:
            impulse_mag = (
                -(1.0 + float(self.cfg.contact_restitution)) * vel_along_normal / inv_sum
            )
            impulse = impulse_mag * normal
            self.vel = (self.vel + impulse * inv_agent).astype(np.float32)
            self.box_vel = (self.box_vel - impulse * inv_box).astype(np.float32)

            tangent = rel_vel - vel_along_normal * normal
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1e-8:
                tangent = tangent / tangent_norm
                jt = -float(np.dot(rel_vel, tangent)) / inv_sum
                max_friction = float(self.cfg.contact_friction) * impulse_mag
                jt = float(np.clip(jt, -max_friction, max_friction))
                friction_impulse = jt * tangent
                self.vel = (self.vel + friction_impulse * inv_agent).astype(np.float32)
                self.box_vel = (self.box_vel - friction_impulse * inv_box).astype(np.float32)

        self.vel = self._clip_speed(self.vel, self.cfg.max_agent_speed)
        self.box_vel = self._clip_speed(self.box_vel, self.cfg.max_box_speed)
        return True

    # ------------------------------------------------------------------
    # Obs / collision
    # ------------------------------------------------------------------

    def _agent_hazard_clearances(self) -> np.ndarray:
        if self.hazards.size == 0:
            return np.empty(0, dtype=np.float32)
        center_d = np.linalg.norm(self.hazards[:, :2] - self.pos[None, :], axis=1)
        return (center_d - self.hazards[:, 2] - float(self.cfg.agent_radius)).astype(np.float32)

    def _box_hazard_clearances(self) -> np.ndarray:
        vals = [
            self._square_circle_clearance(
                self.box_pos, self.cfg.box_half_size, h[:2], float(h[2])
            )
            for h in self.hazards
        ]
        return np.asarray(vals, dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        if self.hazards.size > 0:
            agent_clear = self._agent_hazard_clearances()
            box_clear = self._box_hazard_clearances()
            order = np.argsort(np.minimum(agent_clear, box_clear))
            hazards_sorted = self.hazards[order]
        else:
            hazards_sorted = self.hazards

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[0:2] = self.pos
        obs[2:4] = self.vel
        obs[4:6] = self.box_pos
        obs[6:8] = self.box_vel
        obs[8:10] = self.goal
        obs[10:] = hazards_sorted.reshape(-1)
        return obs

    def _agent_hazard_collision(self) -> bool:
        return bool(np.any(self._agent_hazard_clearances() < 0.0))

    def _box_hazard_collision(self) -> bool:
        return bool(np.any(self._box_hazard_clearances() < 0.0))

    def _goal_reached(self) -> bool:
        return float(np.linalg.norm(self.box_pos - self.goal)) < float(self.cfg.goal_radius)

    def _info(self, *, contact: bool = False) -> dict[str, Any]:
        agent_clear = self._agent_hazard_clearances()
        box_clear = self._box_hazard_clearances()
        min_agent_clear = float(np.min(agent_clear)) if agent_clear.size else float("inf")
        min_box_clear = float(np.min(box_clear)) if box_clear.size else float("inf")
        agent_hazard = bool(min_agent_clear < 0.0)
        box_hazard = bool(min_box_clear < 0.0)
        return {
            "agent_pos": self.pos.copy(),
            "agent_vel": self.vel.copy(),
            "box_pos": self.box_pos.copy(),
            "box_vel": self.box_vel.copy(),
            "goal": self.goal.copy(),
            "hazards": self.hazards.copy(),
            "dist_box_to_goal": float(np.linalg.norm(self.box_pos - self.goal)),
            "dist_agent_to_box": float(np.linalg.norm(self.pos - self.box_pos)),
            "min_agent_hazard_clearance": min_agent_clear,
            "min_box_hazard_clearance": min_box_clear,
            "agent_box_contact": bool(contact),
            "agent_hazard_hit": agent_hazard,
            "box_hazard_hit": box_hazard,
            "hazard_hit": bool(agent_hazard or box_hazard),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample_layout()
        self.t = 0
        self._agent_trail = [self.pos.copy()]
        self._box_trail = [self.box_pos.copy()]
        info = self._info(contact=False)
        info["start"] = self.pos.copy()
        info["box_start"] = self.box_pos.copy()
        return self._build_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action[:2], -1.0, 1.0)
        force = float(self.cfg.force_scale) * action

        prev_box_goal_dist = float(np.linalg.norm(self.box_pos - self.goal))
        contact = False
        n_sub = max(1, int(self.cfg.physics_substeps))
        sub_dt = float(self.cfg.dt) / float(n_sub)

        for _ in range(n_sub):
            self.vel = self.vel + (force - float(self.cfg.agent_drag) * self.vel) * sub_dt
            self.box_vel = self.box_vel + (-float(self.cfg.box_drag) * self.box_vel) * sub_dt
            self.vel = self._clip_speed(self.vel, self.cfg.max_agent_speed)
            self.box_vel = self._clip_speed(self.box_vel, self.cfg.max_box_speed)

            self.pos = (self.pos + self.vel * sub_dt).astype(np.float32)
            self.box_pos = (self.box_pos + self.box_vel * sub_dt).astype(np.float32)

            self._handle_agent_wall()
            self._handle_box_wall()
            for _iter in range(max(1, int(self.cfg.contact_iterations))):
                contact = self._resolve_agent_box_collision() or contact
                self._handle_agent_wall()
                self._handle_box_wall()

            if self._agent_hazard_collision() or self._box_hazard_collision():
                break

        self.t += 1
        self._agent_trail.append(self.pos.copy())
        self._box_trail.append(self.box_pos.copy())

        new_box_goal_dist = float(np.linalg.norm(self.box_pos - self.goal))
        progress = prev_box_goal_dist - new_box_goal_dist
        reward = float(self.cfg.step_penalty) + float(self.cfg.box_progress_reward_scale) * progress

        terminated = False
        truncated = False
        info = self._info(contact=contact)

        if info["agent_hazard_hit"]:
            reward += float(self.cfg.hazard_penalty)
            terminated = True
            info["termination_reason"] = "agent_hazard"
        elif info["box_hazard_hit"]:
            reward += float(self.cfg.hazard_penalty)
            terminated = True
            info["termination_reason"] = "box_hazard"
        elif self._goal_reached():
            reward += float(self.cfg.goal_reward)
            terminated = True
            info["termination_reason"] = "goal"

        if self.t >= int(self.cfg.max_episode_steps) and not terminated:
            truncated = True
            info["termination_reason"] = "timeout"

        info["goal_success"] = bool(terminated and info.get("termination_reason") == "goal")
        info["box_goal_progress"] = float(progress)
        return self._build_obs(), float(reward), bool(terminated), bool(truncated), info

    def attach_renderer(self, renderer) -> None:
        self._renderer = renderer

    def render(self, info_text: str | None = None) -> np.ndarray | None:
        if self._renderer is None:
            return None
        return self._renderer.render(
            agent_xy=self.pos,
            box_xy=self.box_pos,
            goal_xy=self.goal,
            hazards=self.hazards,
            agent_vel_xy=self.vel,
            box_vel_xy=self.box_vel,
            agent_trail=self._agent_trail,
            box_trail=self._box_trail,
            info_text=info_text,
        )

    def close(self) -> None:
        return None


def make_env(
    cfg: PointPushHazardConfig | None = None,
    seed: int | None = None,
    with_renderer: bool = True,
) -> PointPushHazardEnv:
    env = PointPushHazardEnv(cfg=cfg, seed=seed)
    if with_renderer:
        try:
            from pointpush_hazard_renderer import PointPushHazardRenderer
        except ImportError:
            from hazard.pointpush_hazard_renderer import PointPushHazardRenderer

        env.attach_renderer(PointPushHazardRenderer.from_env(env))
    return env
