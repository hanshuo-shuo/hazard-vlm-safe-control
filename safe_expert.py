"""
Safe expert for PointHazardEnv.

A* on a grid of the arena, with hazards inflated by (agent_radius +
safety_margin), then a PD-with-feedforward tracking controller that converts
the planned waypoint sequence into bounded force commands.

Provably safe: any path returned by A* keeps the agent's center at distance
> hazard_radius from every hazard center, so the agent body (radius
agent_radius) never overlaps a hazard.

Public API:
    expert = SafeExpert(env, cfg=SafeExpertConfig())
    ok = expert.plan(start_xy, goal_xy, hazards)         # explicit re-plan
    ok = expert.reset_from_obs(obs)                       # plan from env obs
    a  = expert.act(obs)                                  # PD step → action
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SafeExpertConfig:
    grid_res: int = 100             # arena discretized as grid_res × grid_res
    safety_margin: float = 0.15    # extra clearance ON TOP OF agent_radius
    desired_speed: float = 2.5      # m/s along the planned path
    speed_noise_std: float = 0.4    # per-episode Gaussian noise on desired_speed
    lookahead_dist: float = 0.6     # PD pure-pursuit lookahead in world units
    waypoint_advance_dist: float = 0.3  # advance index when within this distance
    kp: float = 3.0                 # PD gain on velocity error
    use_drag_feedforward: bool = True   # cancel env drag at steady state
    hazard_slowdown_dist: float = 1.2   # start slowing when within this dist of hazard edge
    hazard_min_speed_factor: float = 0.4  # min speed = desired_speed * factor near hazard


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SafeExpert:
    """A*-planned, PD-tracked safe expert for PointHazardEnv."""

    NEIGHBORS_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, env, cfg: SafeExpertConfig | None = None):
        self.env = env
        self.cfg = cfg or SafeExpertConfig()

        a = self.env.cfg.arena_half
        gr = self.cfg.grid_res
        self._cell_size = 2.0 * a / gr

        # Cached path state
        self.path: list[np.ndarray] | None = None  # world (x, y) waypoints
        self.path_idx: int = 0
        self.last_plan_ok: bool = False
        self._episode_speed: float = cfg.desired_speed if cfg else 2.5
        # Own RNG (not the global np.random) so per-episode speed noise is
        # reproducible: callers can reseed self.rng from the episode seed. Using
        # the global RNG here would make runs non-reproducible and pollute it.
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Grid / world conversions
    # ------------------------------------------------------------------

    def _world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        a = self.env.cfg.arena_half
        gr = self.cfg.grid_res
        u = (float(xy[0]) + a) / (2.0 * a)
        v = (float(xy[1]) + a) / (2.0 * a)
        c = int(np.clip(u * gr, 0, gr - 1))
        r = int(np.clip(v * gr, 0, gr - 1))
        return (r, c)

    def _cell_to_world(self, r: int, c: int) -> np.ndarray:
        a = self.env.cfg.arena_half
        gr = self.cfg.grid_res
        x = -a + (c + 0.5) * (2.0 * a / gr)
        y = -a + (r + 0.5) * (2.0 * a / gr)
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------

    def _build_occupancy(self, hazards: np.ndarray) -> np.ndarray:
        """Return (gr, gr) bool array where True = blocked (hazard)."""
        a = self.env.cfg.arena_half
        gr = self.cfg.grid_res
        clearance = self.env.cfg.agent_radius + self.cfg.safety_margin

        cs = -a + (np.arange(gr) + 0.5) * self._cell_size
        XX, YY = np.meshgrid(cs, cs)  # XX[r, c] = x at column c, etc.

        blocked = np.zeros((gr, gr), dtype=bool)
        for hx, hy, hr in np.asarray(hazards):
            d2 = (XX - float(hx)) ** 2 + (YY - float(hy)) ** 2
            blocked |= d2 < (float(hr) + clearance) ** 2
        return blocked

    # ------------------------------------------------------------------
    # A*
    # ------------------------------------------------------------------

    def _astar(
        self,
        start_rc: tuple[int, int],
        goal_rc: tuple[int, int],
        blocked: np.ndarray,
    ) -> list[tuple[int, int]] | None:
        gr = self.cfg.grid_res
        if blocked[start_rc] or blocked[goal_rc]:
            return None

        def h(rc: tuple[int, int]) -> float:
            return float(np.hypot(rc[0] - goal_rc[0], rc[1] - goal_rc[1]))

        open_heap: list = []
        # heap entries: (f, g, rc)
        heapq.heappush(open_heap, (h(start_rc), 0.0, start_rc))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_rc: None}
        gscore: dict[tuple[int, int], float] = {start_rc: 0.0}

        while open_heap:
            f, g, rc = heapq.heappop(open_heap)
            if rc == goal_rc:
                # Reconstruct
                path: list[tuple[int, int]] = []
                cur: tuple[int, int] | None = rc
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
                return path
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
                if ng < gscore.get((nr, nc), float("inf")):
                    gscore[(nr, nc)] = ng
                    came_from[(nr, nc)] = rc
                    heapq.heappush(open_heap, (ng + h((nr, nc)), ng, (nr, nc)))
        return None

    # ------------------------------------------------------------------
    # Path post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_path(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Drop interior cells whose direction equals the previous one."""
        if len(cells) <= 2:
            return cells[:]
        out = [cells[0]]
        for i in range(1, len(cells) - 1):
            prev = cells[i - 1]
            cur = cells[i]
            nxt = cells[i + 1]
            d1 = (cur[0] - prev[0], cur[1] - prev[1])
            d2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            if d1 != d2:
                out.append(cur)
        out.append(cells[-1])
        return out

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        hazards: np.ndarray,
    ) -> bool:
        blocked = self._build_occupancy(hazards)
        start_rc = self._world_to_cell(np.asarray(start_xy, dtype=np.float32))
        goal_rc = self._world_to_cell(np.asarray(goal_xy, dtype=np.float32))
        cells = self._astar(start_rc, goal_rc, blocked)
        if cells is None:
            self.path = None
            self.path_idx = 0
            self.last_plan_ok = False
            return False

        cells = self._smooth_path(cells)
        wps = [self._cell_to_world(r, c) for (r, c) in cells]
        wps[-1] = np.asarray(goal_xy, dtype=np.float32)  # exact goal at the end
        self.path = wps
        self.path_idx = 0
        self.last_plan_ok = True
        # Sample a new speed for this episode (own RNG, reproducible)
        self._episode_speed = max(
            0.5,
            self.cfg.desired_speed + float(self.rng.standard_normal()) * self.cfg.speed_noise_std,
        )
        return True

    def reset_from_obs(self, obs: np.ndarray) -> bool:
        # PointHazardEnv emits ABSOLUTE coordinates: obs[4:6] is the goal center
        # and obs[6:] is (x, y, r) per hazard (see env_pointhazard._build_obs).
        n = self.env.cfg.n_hazards
        pos = obs[0:2].astype(np.float32)
        start = pos.copy()
        goal = obs[4:6].astype(np.float32)
        hazards = obs[6 : 6 + 3 * n].reshape(n, 3).astype(np.float32)
        return self.plan(start, goal, hazards)

    # ------------------------------------------------------------------
    # PD tracking control
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> np.ndarray:
        """One PD-tracking step. Returns force command in [-1, 1]^2."""
        if self.path is None or len(self.path) == 0:
            return np.zeros(2, dtype=np.float32)

        pos = obs[0:2].astype(np.float32)
        vel = obs[2:4].astype(np.float32)

        # Advance index past waypoints we already passed.
        while self.path_idx < len(self.path) - 1:
            d = float(np.linalg.norm(self.path[self.path_idx] - pos))
            if d < self.cfg.waypoint_advance_dist:
                self.path_idx += 1
            else:
                break

        # Pure-pursuit lookahead point: first waypoint at distance >= lookahead.
        target = self.path[-1]
        for i in range(self.path_idx, len(self.path)):
            wp = self.path[i]
            if float(np.linalg.norm(wp - pos)) >= self.cfg.lookahead_dist:
                target = wp
                break

        # Desired velocity (slow down near final goal)
        delta = target - pos
        d = float(np.linalg.norm(delta))
        if d < 1e-6:
            desired_vel = np.zeros(2, dtype=np.float32)
        else:
            speed = self._episode_speed
            # Slow when within ~goal_radius of the final goal.
            final_d = float(np.linalg.norm(self.path[-1] - pos))
            if final_d < 1.0:
                speed *= max(0.3, final_d / 1.0)
            # Slow when close to any hazard edge.
            n_haz = self.env.cfg.n_hazards
            haz_rel = obs[6 : 6 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
            haz_edge_dists = np.linalg.norm(haz_rel[:, :2], axis=1) - haz_rel[:, 2]
            min_haz_d = float(np.min(haz_edge_dists))
            if min_haz_d < self.cfg.hazard_slowdown_dist:
                t = max(0.0, min_haz_d / self.cfg.hazard_slowdown_dist)
                haz_factor = (
                    self.cfg.hazard_min_speed_factor
                    + (1.0 - self.cfg.hazard_min_speed_factor) * t
                )
                speed *= haz_factor
            desired_vel = (delta / d) * speed

        # PD with optional drag feedforward.
        err = desired_vel - vel
        force = self.cfg.kp * err
        if self.cfg.use_drag_feedforward:
            force = force + self.env.cfg.drag * desired_vel

        action = force / float(self.env.cfg.force_scale)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
