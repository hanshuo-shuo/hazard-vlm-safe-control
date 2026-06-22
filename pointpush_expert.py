"""
God-view expert for PointPushHazardEnv.

The expert is intentionally privileged and used for environment validation,
not as the learned policy:
  1. Plan a grid A* path for the box center to the goal, with hazards inflated
     by the box radius.
  2. Move the point agent to a push pose behind the box for the next box-path
     waypoint.
  3. Push along the planned box direction while keeping lateral alignment.

It is a practical physics sanity checker: if the environment contact,
terminations, or rewards are wrong, this controller tends to expose it quickly.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from env_pointpushhazard import PointPushHazardConfig, PointPushHazardEnv


@dataclass
class PointPushExpertConfig:
    grid_res: int = 90
    safety_margin: float = 0.22
    agent_safety_margin: float = 0.18
    waypoint_advance_dist: float = 0.45
    box_replan_every: int = 8
    agent_replan_every: int = 4
    push_pose_slack: float = 0.16
    push_pose_reached_dist: float = 0.18
    push_speed: float = 1.8
    approach_speed: float = 2.1
    kp: float = 3.2
    lateral_gain: float = 1.5
    drag_feedforward: bool = True
    prefer_direct_corridor: bool = True


def parse_pointpush_obs(
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    agent_pos = obs[0:2].astype(np.float32)
    agent_vel = obs[2:4].astype(np.float32)
    box_pos = obs[4:6].astype(np.float32)
    box_vel = obs[6:8].astype(np.float32)
    goal = obs[8:10].astype(np.float32)
    hazards = obs[10 : 10 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
    return agent_pos, agent_vel, box_pos, box_vel, goal, hazards


def min_agent_hazard_clearance(
    agent_pos: np.ndarray,
    hazards: np.ndarray,
    cfg: PointPushHazardConfig,
) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - agent_pos[None, :], axis=1)
    return float(np.min(center_d - hazards[:, 2] - float(cfg.agent_radius)))


def min_box_hazard_clearance(
    box_pos: np.ndarray,
    hazards: np.ndarray,
    cfg: PointPushHazardConfig,
) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    half = float(cfg.box_half_size)
    vals = []
    for hx, hy, hr in hazards:
        delta = np.array([hx, hy], dtype=np.float32) - box_pos
        closest = np.clip(delta, -half, half)
        outside = delta - closest
        vals.append(float(np.linalg.norm(outside) - float(hr)))
    return float(np.min(vals))


class PointPushExpert:
    NEIGHBORS_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, env: PointPushHazardEnv, cfg: PointPushExpertConfig | None = None):
        self.env = env
        self.cfg = cfg or PointPushExpertConfig()
        self.path: list[np.ndarray] | None = None
        self.path_idx = 0
        self.agent_path: list[np.ndarray] | None = None
        self.agent_path_idx = 0
        self.last_plan_ok = False
        self.last_agent_plan_ok = False
        self.last_push_pose = np.zeros(2, dtype=np.float32)
        self.last_push_dir = np.array([1.0, 0.0], dtype=np.float32)
        self._last_plan_t = -10_000
        self._last_agent_plan_t = -10_000

    # ------------------------------------------------------------------
    # Grid planning
    # ------------------------------------------------------------------

    def _world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        a = float(self.env.cfg.arena_half)
        gr = int(self.cfg.grid_res)
        u = (float(xy[0]) + a) / (2.0 * a)
        v = (float(xy[1]) + a) / (2.0 * a)
        c = int(np.clip(u * gr, 0, gr - 1))
        r = int(np.clip(v * gr, 0, gr - 1))
        return r, c

    def _cell_to_world(self, r: int, c: int) -> np.ndarray:
        a = float(self.env.cfg.arena_half)
        gr = int(self.cfg.grid_res)
        x = -a + (c + 0.5) * (2.0 * a / gr)
        y = -a + (r + 0.5) * (2.0 * a / gr)
        return np.array([x, y], dtype=np.float32)

    def _build_occupancy(
        self,
        hazards: np.ndarray,
        *,
        body_radius: float,
        safety_margin: float,
    ) -> np.ndarray:
        cfg = self.env.cfg
        a = float(cfg.arena_half)
        gr = int(self.cfg.grid_res)
        cell = 2.0 * a / float(gr)
        coords = -a + (np.arange(gr, dtype=np.float32) + 0.5) * cell
        xx, yy = np.meshgrid(coords, coords)

        blocked = np.zeros((gr, gr), dtype=bool)
        wall_limit = a - float(body_radius)
        blocked |= (np.abs(xx) > wall_limit) | (np.abs(yy) > wall_limit)

        inflate = float(body_radius) + float(safety_margin)
        for hx, hy, hr in np.asarray(hazards, dtype=np.float32).reshape(-1, 3):
            d2 = (xx - float(hx)) ** 2 + (yy - float(hy)) ** 2
            blocked |= d2 <= (float(hr) + inflate) ** 2
        return blocked

    def _build_agent_occupancy(self, hazards: np.ndarray, box_pos: np.ndarray) -> np.ndarray:
        blocked = self._build_occupancy(
            hazards,
            body_radius=float(self.env.cfg.agent_radius),
            safety_margin=float(self.cfg.agent_safety_margin),
        )

        cfg = self.env.cfg
        a = float(cfg.arena_half)
        gr = int(self.cfg.grid_res)
        cell = 2.0 * a / float(gr)
        coords = -a + (np.arange(gr, dtype=np.float32) + 0.5) * cell
        xx, yy = np.meshgrid(coords, coords)
        box_xy = np.asarray(box_pos, dtype=np.float32).reshape(2)
        half = float(cfg.box_half_size)
        dx = np.abs(xx - float(box_xy[0])) - half
        dy = np.abs(yy - float(box_xy[1])) - half
        outside_x = np.maximum(dx, 0.0)
        outside_y = np.maximum(dy, 0.0)
        outside_dist = np.sqrt(outside_x * outside_x + outside_y * outside_y)
        inside = (dx <= 0.0) & (dy <= 0.0)
        inflated = float(cfg.agent_radius) + 0.05
        blocked |= inside | (outside_dist <= inflated)
        return blocked

    def _nearest_free(self, rc: tuple[int, int], blocked: np.ndarray) -> tuple[int, int] | None:
        if not blocked[rc]:
            return rc
        gr = blocked.shape[0]
        sr, sc = rc
        for radius in range(1, gr):
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < gr and 0 <= nc < gr and not blocked[nr, nc]:
                        return nr, nc
            for dc in range(-radius + 1, radius):
                for dr in (-radius, radius):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < gr and 0 <= nc < gr and not blocked[nr, nc]:
                        return nr, nc
        return None

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: np.ndarray,
    ) -> list[tuple[int, int]] | None:
        if blocked[start] or blocked[goal]:
            return None

        def h(rc: tuple[int, int]) -> float:
            return float(np.hypot(rc[0] - goal[0], rc[1] - goal[1]))

        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (h(start), 0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        gscore: dict[tuple[int, int], float] = {start: 0.0}
        gr = blocked.shape[0]

        while open_heap:
            _f, g, rc = heapq.heappop(open_heap)
            if rc == goal:
                cells = []
                cur: tuple[int, int] | None = rc
                while cur is not None:
                    cells.append(cur)
                    cur = came_from[cur]
                cells.reverse()
                return cells
            if g > gscore.get(rc, float("inf")):
                continue

            for dr, dc in self.NEIGHBORS_8:
                nr, nc = rc[0] + dr, rc[1] + dc
                if not (0 <= nr < gr and 0 <= nc < gr):
                    continue
                if blocked[nr, nc]:
                    continue
                step = float(np.hypot(dr, dc))
                ng = g + step
                key = (nr, nc)
                if ng < gscore.get(key, float("inf")):
                    gscore[key] = ng
                    came_from[key] = rc
                    heapq.heappush(open_heap, (ng + h(key), ng, key))
        return None

    @staticmethod
    def _smooth_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells[:]
        out = [cells[0]]
        last_dir: tuple[int, int] | None = None
        for i in range(1, len(cells)):
            prev = cells[i - 1]
            cur = cells[i]
            d = (cur[0] - prev[0], cur[1] - prev[1])
            if d != last_dir:
                if i > 1:
                    out.append(prev)
                last_dir = d
        out.append(cells[-1])
        return out

    def plan_box_path(self, box_pos: np.ndarray, goal: np.ndarray, hazards: np.ndarray) -> bool:
        blocked = self._build_occupancy(
            hazards,
            body_radius=float(self.env.cfg.box_radius),
            safety_margin=float(self.cfg.safety_margin),
        )
        start = self._nearest_free(self._world_to_cell(box_pos), blocked)
        end = self._nearest_free(self._world_to_cell(goal), blocked)
        if start is None or end is None:
            self.path = None
            self.path_idx = 0
            self.last_plan_ok = False
            return False
        cells = self._astar(start, end, blocked)
        if cells is None:
            self.path = None
            self.path_idx = 0
            self.last_plan_ok = False
            return False
        cells = self._smooth_cells(cells)
        path = [self._cell_to_world(r, c) for r, c in cells]
        path[0] = np.asarray(box_pos, dtype=np.float32)
        path[-1] = np.asarray(goal, dtype=np.float32)
        self.path = path
        self.path_idx = 1 if len(path) > 1 else 0
        self.last_plan_ok = True
        self._last_plan_t = int(self.env.t)
        return True

    def plan_agent_path(
        self,
        agent_pos: np.ndarray,
        target: np.ndarray,
        hazards: np.ndarray,
        box_pos: np.ndarray,
    ) -> bool:
        blocked = self._build_agent_occupancy(hazards, box_pos)
        start = self._nearest_free(self._world_to_cell(agent_pos), blocked)
        end = self._nearest_free(self._world_to_cell(target), blocked)
        if start is None or end is None:
            self.agent_path = None
            self.agent_path_idx = 0
            self.last_agent_plan_ok = False
            return False
        cells = self._astar(start, end, blocked)
        if cells is None:
            self.agent_path = None
            self.agent_path_idx = 0
            self.last_agent_plan_ok = False
            return False
        cells = self._smooth_cells(cells)
        path = [self._cell_to_world(r, c) for r, c in cells]
        path[0] = np.asarray(agent_pos, dtype=np.float32)
        path[-1] = np.asarray(target, dtype=np.float32)
        self.agent_path = path
        self.agent_path_idx = 1 if len(path) > 1 else 0
        self.last_agent_plan_ok = True
        self._last_agent_plan_t = int(self.env.t)
        return True

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _pd_to_velocity(self, desired_vel: np.ndarray, current_vel: np.ndarray) -> np.ndarray:
        force = float(self.cfg.kp) * (desired_vel - current_vel)
        if self.cfg.drag_feedforward:
            force = force + float(self.env.cfg.agent_drag) * desired_vel
        action = force / float(self.env.cfg.force_scale)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _hazard_avoidance_velocity(self, pos: np.ndarray, hazards: np.ndarray) -> np.ndarray:
        avoid = np.zeros(2, dtype=np.float32)
        danger_dist = 0.75
        for hx, hy, hr in np.asarray(hazards, dtype=np.float32).reshape(-1, 3):
            away = pos - np.array([hx, hy], dtype=np.float32)
            dist = float(np.linalg.norm(away))
            if dist < 1e-6:
                continue
            clearance = dist - float(hr) - float(self.env.cfg.agent_radius)
            if clearance < danger_dist:
                strength = (danger_dist - clearance) / danger_dist
                avoid += away / dist * strength * 1.8
        return avoid.astype(np.float32)

    def _desired_velocity_to_target(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        target: np.ndarray,
        *,
        speed: float,
    ) -> np.ndarray:
        delta = np.asarray(target, dtype=np.float32) - np.asarray(pos, dtype=np.float32)
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            desired_vel = np.zeros(2, dtype=np.float32)
        else:
            desired_vel = delta / dist * min(float(speed), dist / max(1e-6, self.env.cfg.dt))
        return self._pd_to_velocity(desired_vel.astype(np.float32), vel)

    def _agent_path_target(self, agent_pos: np.ndarray, final_target: np.ndarray) -> np.ndarray:
        if self.agent_path is None or len(self.agent_path) == 0:
            return np.asarray(final_target, dtype=np.float32)
        while self.agent_path_idx < len(self.agent_path) - 1:
            if float(np.linalg.norm(self.agent_path[self.agent_path_idx] - agent_pos)) < 0.30:
                self.agent_path_idx += 1
            else:
                break
        return self.agent_path[self.agent_path_idx].astype(np.float32)

    def _path_direction(self, box_pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if (
            bool(self.cfg.prefer_direct_corridor)
            and bool(getattr(self.env.cfg, "require_box_goal_corridor", False))
        ):
            delta = np.asarray(goal, dtype=np.float32) - np.asarray(box_pos, dtype=np.float32)
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                return self.last_push_dir.copy()
            return (delta / norm).astype(np.float32)

        if self.path is None or len(self.path) <= 1:
            delta = np.asarray(goal, dtype=np.float32) - np.asarray(box_pos, dtype=np.float32)
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                return self.last_push_dir.copy()
            return (delta / norm).astype(np.float32)

        while self.path_idx < len(self.path) - 1:
            if float(np.linalg.norm(self.path[self.path_idx] - box_pos)) < self.cfg.waypoint_advance_dist:
                self.path_idx += 1
            else:
                break
        target = self.path[self.path_idx]
        delta = target - box_pos
        if self.path_idx == len(self.path) - 1 and float(np.linalg.norm(delta)) < 0.35:
            delta = np.asarray(goal, dtype=np.float32) - box_pos
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return self.last_push_dir.copy()
        return (delta / norm).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        agent_pos, agent_vel, box_pos, box_vel, goal, hazards = parse_pointpush_obs(obs, self.env.cfg)
        should_replan = (
            self.path is None
            or not self.last_plan_ok
            or int(self.env.t) - self._last_plan_t >= int(self.cfg.box_replan_every)
        )
        if should_replan:
            self.plan_box_path(box_pos, goal, hazards)

        push_dir = self._path_direction(box_pos, goal)
        self.last_push_dir = push_dir.copy()
        push_dist = (
            float(self.env.cfg.box_half_size)
            + float(self.env.cfg.agent_radius)
            + float(self.cfg.push_pose_slack)
        )
        push_pose = box_pos - push_dir * push_dist
        arena_limit = float(self.env.cfg.arena_half) - float(self.env.cfg.agent_radius)
        push_pose = np.clip(push_pose, -arena_limit, arena_limit).astype(np.float32)
        self.last_push_pose = push_pose.copy()

        to_push_pose = push_pose - agent_pos
        dist_to_push_pose = float(np.linalg.norm(to_push_pose))
        agent_to_box = box_pos - agent_pos
        behind_score = float(np.dot(agent_to_box, push_dir))
        lateral = agent_to_box - behind_score * push_dir
        lateral_err = float(np.linalg.norm(lateral))
        in_push_slot = (
            dist_to_push_pose < float(self.cfg.push_pose_reached_dist)
            or (behind_score > 0.35 and lateral_err < 0.22)
        )

        if not in_push_slot:
            should_plan_agent = (
                self.agent_path is None
                or not self.last_agent_plan_ok
                or int(self.env.t) - self._last_agent_plan_t >= int(self.cfg.agent_replan_every)
                or (
                    self.agent_path is not None
                    and float(np.linalg.norm(self.agent_path[-1] - push_pose)) > 0.35
                )
            )
            if should_plan_agent:
                self.plan_agent_path(agent_pos, push_pose, hazards, box_pos)
            waypoint = self._agent_path_target(agent_pos, push_pose)
            action = self._desired_velocity_to_target(
                agent_pos,
                agent_vel,
                waypoint,
                speed=float(self.cfg.approach_speed),
            )
            avoid = self._hazard_avoidance_velocity(agent_pos, hazards)
            if float(np.linalg.norm(avoid)) > 1e-6:
                delta = waypoint - agent_pos
                dist = float(np.linalg.norm(delta))
                desired = avoid
                if dist > 1e-6:
                    desired = desired + delta / dist * float(self.cfg.approach_speed)
                action = self._pd_to_velocity(desired.astype(np.float32), agent_vel)
            return action

        self.agent_path = None
        self.agent_path_idx = 0

        box_clear = min_box_hazard_clearance(box_pos, hazards, self.env.cfg)
        push_speed = float(self.cfg.push_speed)
        if box_clear < 0.35:
            push_speed *= max(0.65, box_clear / 0.35)

        agent_clear = min_agent_hazard_clearance(agent_pos, hazards, self.env.cfg)
        if agent_clear < 0.35:
            push_speed *= max(0.55, agent_clear / 0.35)

        # If the agent drifted into a side/front contact, return to the push
        # pose instead of continuing to apply force in a bad direction.
        if behind_score < 0.20 or lateral_err > 0.45:
            return self._desired_velocity_to_target(
                agent_pos,
                agent_vel,
                push_pose,
                speed=float(self.cfg.approach_speed),
            )

        lateral_correction = np.zeros(2, dtype=np.float32)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm > 1e-6:
            lateral_correction = lateral * float(self.cfg.lateral_gain)

        desired_vel = (
            push_dir * push_speed
            + lateral_correction
            + 0.25 * box_vel
            + self._hazard_avoidance_velocity(agent_pos, hazards)
        ).astype(np.float32)
        return self._pd_to_velocity(desired_vel, agent_vel)
