"""
PointHazard-v0: open arena point-mass with random hazards and a random goal.

Key differences from the old PointMaze setup:
  - No maze walls.  Open square arena with bouncing boundary.
  - Hazards (lava circles) are sampled fresh every reset.
  - Single goal per episode, also sampled fresh every reset.
  - Hazard contact -> terminate with large penalty.
  - Pure numpy point-mass dynamics (no MuJoCo dependency for the env itself).
  - Custom PIL renderer (no OpenGL).

Observation layout (default n_hazards=8 -> 30 dims):
    [ 0] x          agent x
    [ 1] y          agent y
    [ 2] vx         agent vx
    [ 3] vy         agent vy
    [ 4] gx         goal x
    [ 5] gy         goal y
    [ 6] h0_x       hazard 0 center x       <- begin hazard block
    [ 7] h0_y       hazard 0 center y
    [ 8] h0_r       hazard 0 radius
    ...                                      (n_hazards * 3 entries)

Hazards are sorted by distance to agent in the obs vector at every step
(closest first), so a downstream policy/diffusion can take a fixed prefix of
"k nearest hazards" without re-sorting.

Action: [fx, fy] in [-1, 1]^2.  Mapped to a 2D force = force_scale * action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PointHazardConfig:
    # Arena geometry (square, centered at origin)
    arena_half: float = 5.0           # arena spans [-arena_half, +arena_half]^2

    # Dynamics
    dt: float = 0.1                   # integration timestep
    drag: float = 0.5                 # linear drag coefficient
    force_scale: float = 5.0          # action 1.0 -> force 5.0 (arena/sec^2)
    max_speed: float = 4.0            # |v| clipped to this

    # Sizes
    agent_radius: float = 0.3         # agent body radius for collision
    goal_radius: float = 0.5          # success threshold (||p - g|| < this)

    # Hazards
    n_hazards: int = 8                # number of hazards per episode (fixed)
    hazard_radius_min: float = 0.2
    hazard_radius_max: float = 0.5

    # Spawn separation (rejection sampling).  All distances are *center-to-
    # center*, so the safe separation between objects is roughly
    # min_separation_extra plus the relevant radii.
    min_hazard_pair_sep: float = 1.4    # min center distance between two hazards
    min_start_clearance: float = 0.8    # extra clearance from start to nearest hazard edge
    min_goal_clearance: float = 0.8     # extra clearance from goal to nearest hazard edge
    min_start_goal_dist: float = 4.0    # ensure start and goal are far apart

    # Semantic keep-out zones (Path-B "main-result" task).  These are off-limits
    # regions that are deliberately UNLIKE the geometric hazards:
    #   - they NEVER terminate the episode (entering one is a soft violation),
    #   - they are NOT written into the obs vector,
    # so a purely geometric controller is blind to them.  Only a controller that
    # is explicitly *told* the zone (the hand-coded "oracle" upper bound) or a
    # VLM that *sees* the zone drawn in the rendered image can route around it.
    # That asymmetry is the whole point: avoiding the zone needs a semantic/
    # language constraint that an A*/MPC cost cannot express without a human
    # first labelling the region.  Disabled when n_semantic_zones == 0.
    n_semantic_zones: int = 0
    semantic_radius_min: float = 0.8
    semantic_radius_max: float = 1.2
    semantic_on_corridor: bool = True   # place zones straddling the start->goal line
    semantic_corridor_t_min: float = 0.4  # fractional position along start->goal
    semantic_corridor_t_max: float = 0.6
    semantic_step_penalty: float = 0.0  # optional soft reward cost per step inside a zone
    # Heterogeneous appearances: a pool of per-zone render styles (e.g.
    # ("water","mud","grass")) assigned round-robin by zone index. Empty = all zones
    # use the renderer's single default style (homogeneous, backward-compatible).
    # When set with >1 zone, zones also SPREAD along the corridor (per-zone t-band)
    # so multiple distinct terrains line the route. Only consulted when non-empty,
    # so single-zone homogeneous runs keep their exact RNG sequence.
    semantic_styles: tuple[str, ...] = ()

    # Episode
    max_episode_steps: int = 300

    # Reward
    step_penalty: float = -0.01
    hazard_penalty: float = -50.0
    goal_reward: float = 100.0

    # Sampling safety
    max_rejection_tries: int = 2000

    # Rendering (read by external renderer)
    render_size: int = 480


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PointHazardEnv:
    """
    Minimal gym-like point-mass environment with random hazards and a random goal.

    Methods:
        reset(seed=None) -> (obs, info)
        step(action)     -> (obs, reward, terminated, truncated, info)
        render()         -> RGB numpy array (uses HazardRenderer if attached)
        close()          -> None
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: PointHazardConfig | None = None, seed: int | None = None):
        self.cfg = cfg or PointHazardConfig()
        self.rng = np.random.default_rng(seed)

        # Spaces (gym-style attributes; we don't depend on gymnasium).
        self.act_dim = 2
        self.cond_dim = 4  # [x, y, vx, vy] for diffusion conditioning
        self.obs_dim = 4 + 2 + 3 * self.cfg.n_hazards

        self.action_low = -np.ones(2, dtype=np.float32)
        self.action_high = np.ones(2, dtype=np.float32)

        # Episode state (set in reset)
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.hazards = np.zeros((self.cfg.n_hazards, 3), dtype=np.float32)  # (x, y, r)
        self.semantic_zones = np.zeros((0, 3), dtype=np.float32)  # (x, y, r) off-limits
        self.semantic_zone_styles = None  # per-zone render styles, or None = uniform
        self._semantic_steps = 0  # steps spent inside any semantic zone this episode
        self.t = 0
        self._trail: list[np.ndarray] = []

        # External renderer attached lazily by callers.
        self._renderer = None

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_xy(self, margin: float) -> np.ndarray:
        lo = -self.cfg.arena_half + margin
        hi = +self.cfg.arena_half - margin
        return self.rng.uniform(lo, hi, size=2).astype(np.float32)

    def _sample_layout(self) -> None:
        """Sample hazards, start, goal with rejection on minimum-distance constraints."""
        cfg = self.cfg

        # 1. Sample hazards with center-to-center separation.
        hazards: list[np.ndarray] = []
        tries = 0
        while len(hazards) < cfg.n_hazards:
            tries += 1
            if tries > cfg.max_rejection_tries:
                raise RuntimeError(
                    f"Failed to sample {cfg.n_hazards} non-overlapping hazards in "
                    f"{cfg.max_rejection_tries} tries; relax min_hazard_pair_sep "
                    f"or shrink hazards."
                )
            r = float(self.rng.uniform(cfg.hazard_radius_min, cfg.hazard_radius_max))
            xy = self._sample_xy(margin=r + 0.1)  # keep hazard fully inside arena
            ok = True
            for hx, hy, hr in hazards:
                d = float(np.linalg.norm(xy - np.array([hx, hy], dtype=np.float32)))
                # require separation that exceeds both radii by min_hazard_pair_sep
                if d < (r + hr + cfg.min_hazard_pair_sep - 1.0):
                    ok = False
                    break
            if ok:
                hazards.append(np.array([xy[0], xy[1], r], dtype=np.float32))
        self.hazards = np.stack(hazards, axis=0)

        # 2. Sample start position outside any hazard (with clearance).
        for _ in range(cfg.max_rejection_tries):
            start = self._sample_xy(margin=cfg.agent_radius + 0.1)
            if self._clear_of_all_hazards(start, cfg.agent_radius + cfg.min_start_clearance):
                break
        else:
            raise RuntimeError("Failed to sample valid start position.")
        self.pos = start
        self.vel = np.zeros(2, dtype=np.float32)

        # 3. Sample goal position outside any hazard AND far from start.
        for _ in range(cfg.max_rejection_tries):
            goal = self._sample_xy(margin=cfg.goal_radius + 0.1)
            if not self._clear_of_all_hazards(goal, cfg.goal_radius + cfg.min_goal_clearance):
                continue
            if float(np.linalg.norm(goal - self.pos)) < cfg.min_start_goal_dist:
                continue
            break
        else:
            raise RuntimeError("Failed to sample valid goal position.")
        self.goal = goal

        # 4. Sample semantic keep-out zones (after start+goal so they can be
        #    placed on the corridor between them — see _sample_semantic_zones).
        self.semantic_zones = self._sample_semantic_zones()

    def _sample_semantic_zones(self) -> np.ndarray:
        """Sample off-limits zones, by default straddling the start->goal line.

        Placing a zone so the straight start->goal segment passes through it
        guarantees a geometry-only controller (which heads roughly straight at
        the goal) will cut through it — making the semantic constraint actually
        *bite*.  Zones are rejected if they would swallow the start/goal or
        overlap a hard hazard or another zone, so there is always room to detour
        around them.
        """
        cfg = self.cfg
        if cfg.n_semantic_zones <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        seg = self.goal - self.pos
        seg_len = float(np.linalg.norm(seg))
        seg_dir = seg / (seg_len + 1e-9)
        perp = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float32)

        n = cfg.n_semantic_zones
        zones: list[np.ndarray] = []
        for zi in range(n):
            placed = False
            for _try in range(cfg.max_rejection_tries):
                r = float(self.rng.uniform(cfg.semantic_radius_min, cfg.semantic_radius_max))
                if cfg.semantic_on_corridor:
                    if n > 1:
                        # Spread zones along the route: zone zi gets its own t-band
                        # tiling [0.18, 0.82], so multiple terrains line the path
                        # rather than piling at the midpoint (and rejecting).
                        lo = 0.18 + 0.64 * zi / n
                        hi = 0.18 + 0.64 * (zi + 1) / n
                        t = float(self.rng.uniform(lo, hi))
                    else:
                        t = float(self.rng.uniform(cfg.semantic_corridor_t_min,
                                                   cfg.semantic_corridor_t_max))
                    # lateral offset < r keeps the segment intersecting the zone
                    lateral = float(self.rng.uniform(-0.4, 0.4)) * r
                    center = self.pos + seg_dir * (t * seg_len) + perp * lateral
                else:
                    center = self._sample_xy(margin=r + 0.1)
                lim = cfg.arena_half - r - 0.1
                center = np.clip(center, -lim, lim).astype(np.float32)

                # Keep start and goal outside the zone (need clear endpoints).
                if float(np.linalg.norm(center - self.pos)) < r + cfg.agent_radius + 0.3:
                    continue
                if float(np.linalg.norm(center - self.goal)) < r + cfg.goal_radius + 0.3:
                    continue
                # Leave a gap to hard hazards so an avoider can slip past.
                ok = True
                for hx, hy, hr in self.hazards:
                    if float(np.linalg.norm(center - np.array([hx, hy], dtype=np.float32))) < r + hr + 0.3:
                        ok = False
                        break
                if ok:
                    for zx, zy, zr in zones:
                        if float(np.linalg.norm(center - np.array([zx, zy], dtype=np.float32))) < r + zr + 0.3:
                            ok = False
                            break
                if ok:
                    zones.append(np.array([center[0], center[1], r], dtype=np.float32))
                    placed = True
                    break
            if not placed:
                # Fallback: drop a small zone at this zone's OWN corridor position
                # (spread by zi) so failed placements don't pile into duplicates.
                if cfg.semantic_on_corridor and n > 1:
                    tf = 0.18 + 0.64 * (zi + 0.5) / n
                else:
                    tf = 0.5
                c = self.pos + seg_dir * (tf * seg_len)
                zones.append(np.array([c[0], c[1], cfg.semantic_radius_min], dtype=np.float32))
        # Per-zone appearance styles (parallel to `zones`). Assigned by index from
        # the pool — no RNG draw, so single-zone homogeneous runs are unaffected.
        # None when no pool is set => renderer uses its single default style.
        pool = cfg.semantic_styles
        self.semantic_zone_styles = (
            [pool[i % len(pool)] for i in range(len(zones))] if pool else None
        )
        return np.stack(zones, axis=0)

    def _in_semantic_zone(self, p: np.ndarray) -> bool:
        """True if the agent's *center* lies inside any semantic keep-out zone."""
        for zx, zy, zr in self.semantic_zones:
            if float(np.linalg.norm(p - np.array([zx, zy], dtype=np.float32))) < float(zr):
                return True
        return False

    def _clear_of_all_hazards(self, p: np.ndarray, body_radius: float) -> bool:
        """True if a body of radius `body_radius` centered at p touches no hazard."""
        for hx, hy, hr in self.hazards:
            d = float(np.linalg.norm(p - np.array([hx, hy], dtype=np.float32)))
            if d < (body_radius + hr):
                return False
        return True

    # ------------------------------------------------------------------
    # Obs / collision
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """[x, y, vx, vy, gx, gy, h0_x, h0_y, h0_r, ...] sorted by distance to agent."""
        # Sort hazards by distance to current agent position (closest first).
        d = np.linalg.norm(self.hazards[:, :2] - self.pos[None, :], axis=1)
        order = np.argsort(d)
        haz_sorted = self.hazards[order]  # (n_hazards, 3)

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[0:2] = self.pos
        obs[2:4] = self.vel
        obs[4:6] = self.goal
        obs[6:] = haz_sorted.reshape(-1)
        return obs

    def _hazard_collision(self) -> bool:
        for hx, hy, hr in self.hazards:
            d = float(np.linalg.norm(self.pos - np.array([hx, hy], dtype=np.float32)))
            if d < (self.cfg.agent_radius + float(hr)):
                return True
        return False

    def _goal_reached(self) -> bool:
        return float(np.linalg.norm(self.pos - self.goal)) < self.cfg.goal_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample_layout()
        self.t = 0
        self._semantic_steps = 0
        self._trail = [self.pos.copy()]
        info = {
            "goal": self.goal.copy(),
            "hazards": self.hazards.copy(),
            "semantic_zones": self.semantic_zones.copy(),
            "start": self.pos.copy(),
        }
        return self._build_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        force = self.cfg.force_scale * action

        # Semi-implicit Euler integration with linear drag.
        self.vel = self.vel + (force - self.cfg.drag * self.vel) * self.cfg.dt
        speed = float(np.linalg.norm(self.vel))
        if speed > self.cfg.max_speed:
            self.vel = self.vel * (self.cfg.max_speed / speed)
        new_pos = self.pos + self.vel * self.cfg.dt

        # Bounce off arena walls (reflect velocity, clamp position).
        a = self.cfg.arena_half - self.cfg.agent_radius
        for i in range(2):
            if new_pos[i] < -a:
                new_pos[i] = -a
                self.vel[i] = -self.vel[i] * 0.5
            elif new_pos[i] > a:
                new_pos[i] = a
                self.vel[i] = -self.vel[i] * 0.5
        self.pos = new_pos.astype(np.float32)
        self._trail.append(self.pos.copy())
        self.t += 1

        # Reward + termination.
        reward = float(self.cfg.step_penalty)
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        # Semantic keep-out zones: a soft violation (optional reward cost), never
        # a termination — the geometric controller can plough straight through.
        in_zone = self._in_semantic_zone(self.pos)
        if in_zone:
            self._semantic_steps += 1
            reward += float(self.cfg.semantic_step_penalty)

        if self._hazard_collision():
            reward += float(self.cfg.hazard_penalty)
            terminated = True
            info["termination_reason"] = "hazard"
        elif self._goal_reached():
            reward += float(self.cfg.goal_reward)
            terminated = True
            info["termination_reason"] = "goal"

        if self.t >= self.cfg.max_episode_steps and not terminated:
            truncated = True
            info["termination_reason"] = "timeout"

        info["dist_to_goal"] = float(np.linalg.norm(self.pos - self.goal))
        info["goal_success"] = bool(terminated and info.get("termination_reason") == "goal")
        info["hazard_hit"] = bool(terminated and info.get("termination_reason") == "hazard")
        info["in_semantic_zone"] = bool(in_zone)
        info["semantic_steps"] = int(self._semantic_steps)
        info["goal"] = self.goal.copy()
        info["hazards"] = self.hazards.copy()
        info["semantic_zones"] = self.semantic_zones.copy()

        return self._build_obs(), reward, terminated, truncated, info

    def attach_renderer(self, renderer) -> None:
        """Attach an external renderer (HazardRenderer)."""
        self._renderer = renderer

    def render(self, info_text: str | None = None) -> np.ndarray | None:
        if self._renderer is None:
            return None
        return self._renderer.render(
            agent_xy=self.pos,
            goal_xy=self.goal,
            hazards=self.hazards,
            vel_xy=self.vel,
            trail=self._trail,
            info_text=info_text,
            semantic_zones=self.semantic_zones,
            semantic_styles=self.semantic_zone_styles,
        )

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_env(
    cfg: PointHazardConfig | None = None,
    seed: int | None = None,
    with_renderer: bool = True,
) -> PointHazardEnv:
    env = PointHazardEnv(cfg=cfg, seed=seed)
    if with_renderer:
        # Lazy import so the env file is importable without PIL.
        # NOTE: the renderer lives at the repo root (`hazard_renderer.py`), not
        # inside the `hazard/` data directory — importing `hazard.hazard_renderer`
        # silently breaks because `hazard/` holds only weights/logs/gifs.
        from hazard_renderer import HazardRenderer
        env.attach_renderer(HazardRenderer.from_env(env))
    return env