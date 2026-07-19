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
    # center*.  `min_hazard_pair_sep` is the required extra gap between the
    # hazard edges, so the center-distance threshold is r_i + r_j + this value.
    min_hazard_pair_sep: float = 1.4    # extra edge-to-edge gap between hazards
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
    max_layout_resamples: int = 100     # complete layout retries after placement failure

    # Rendering (read by external renderer)
    render_size: int = 480


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def zone_layout_valid(
    hazards: np.ndarray,
    zones: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    cfg: PointHazardConfig,
) -> tuple[bool, list[str]]:
    """Validate the complete sampled layout, including semantic zones.

    This is intentionally independent of ``PointHazardEnv`` so callers and
    tests can validate a recorded scene without recreating the environment.
    The checks mirror the rejection predicates used by the sampler.  In
    particular, no caller may treat an unverified fallback zone as valid.
    """
    reasons: list[str] = []
    hazards = np.asarray(hazards, dtype=np.float32)
    zones = np.asarray(zones, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    tol = 1e-6

    if hazards.ndim != 2 or hazards.shape != (int(cfg.n_hazards), 3):
        reasons.append(f"hazards_shape={hazards.shape}")
    if zones.ndim != 2 or zones.shape != (max(0, int(cfg.n_semantic_zones)), 3):
        reasons.append(f"zones_shape={zones.shape}")
    if start.shape != (2,):
        reasons.append(f"start_shape={start.shape}")
    if goal.shape != (2,):
        reasons.append(f"goal_shape={goal.shape}")
    if reasons:
        return False, reasons

    if not np.isfinite(hazards).all():
        reasons.append("hazards_nonfinite")
    if not np.isfinite(zones).all():
        reasons.append("zones_nonfinite")
    if not np.isfinite(start).all():
        reasons.append("start_nonfinite")
    if not np.isfinite(goal).all():
        reasons.append("goal_nonfinite")

    def check_inside(objects: np.ndarray, label: str) -> None:
        for i, (x, y, r) in enumerate(objects):
            if r < 0.0:
                reasons.append(f"{label}[{i}]_negative_radius")
            if label == "hazard" and not (
                float(cfg.hazard_radius_min) - tol
                <= float(r)
                <= float(cfg.hazard_radius_max) + tol
            ):
                reasons.append(f"{label}[{i}]_radius_range")
            if label == "zone" and not (
                float(cfg.semantic_radius_min) - tol
                <= float(r)
                <= float(cfg.semantic_radius_max) + tol
            ):
                reasons.append(f"{label}[{i}]_radius_range")
            if abs(float(x)) + float(r) > float(cfg.arena_half) - 0.1 + tol:
                reasons.append(f"{label}[{i}]_outside_arena")

    check_inside(hazards, "hazard")
    check_inside(zones, "zone")
    if np.isfinite(start).all() and (
        np.any(np.abs(start) > float(cfg.arena_half) - float(cfg.agent_radius) - 0.1 + tol)
    ):
        reasons.append("start_outside_arena")
    if np.isfinite(goal).all() and (
        np.any(np.abs(goal) > float(cfg.arena_half) - float(cfg.goal_radius) - 0.1 + tol)
    ):
        reasons.append("goal_outside_arena")

    for i in range(len(hazards)):
        hx, hy, hr = hazards[i]
        for j in range(i):
            jx, jy, jr = hazards[j]
            d = float(np.linalg.norm(np.array([hx - jx, hy - jy], dtype=np.float32)))
            if d < float(hr + jr + cfg.min_hazard_pair_sep) - tol:
                reasons.append(f"hazard[{j},{i}]_pair_sep")
        if float(np.linalg.norm(start - hazards[i, :2])) < float(
            cfg.agent_radius + cfg.min_start_clearance + hr
        ) - tol:
            reasons.append(f"start_hazard[{i}]_clearance")
        if float(np.linalg.norm(goal - hazards[i, :2])) < float(
            cfg.goal_radius + cfg.min_goal_clearance + hr
        ) - tol:
            reasons.append(f"goal_hazard[{i}]_clearance")

    for i, (zx, zy, zr) in enumerate(zones):
        zxy = np.array([zx, zy], dtype=np.float32)
        if float(np.linalg.norm(zxy - start)) < float(
            zr + cfg.agent_radius + 0.3
        ) - tol:
            reasons.append(f"zone[{i}]_contains_start")
        if float(np.linalg.norm(zxy - goal)) < float(
            zr + cfg.goal_radius + 0.3
        ) - tol:
            reasons.append(f"zone[{i}]_contains_goal")
        for j, (hx, hy, hr) in enumerate(hazards):
            if float(np.linalg.norm(zxy - np.array([hx, hy], dtype=np.float32))) < float(
                zr + hr + 0.3
            ) - tol:
                reasons.append(f"zone[{i}]_hazard[{j}]_overlap")
        for j in range(i):
            jxy = zones[j, :2]
            if float(np.linalg.norm(zxy - jxy)) < float(zr + zones[j, 2] + 0.3) - tol:
                reasons.append(f"zone[{j},{i}]_overlap")

    if float(np.linalg.norm(goal - start)) < float(cfg.min_start_goal_dist) - tol:
        reasons.append("start_goal_distance")

    return not reasons, reasons


class _LayoutSamplingFailure(RuntimeError):
    """Internal signal to discard one complete candidate layout."""

    def __init__(self, message: str, placement_attempts: int = 0):
        super().__init__(message)
        self.placement_attempts = int(placement_attempts)


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
        self.layout_valid = False
        self.placement_attempts = 0
        self.resample_count = 0
        self._semantic_steps = 0  # steps spent inside any semantic zone this episode
        self.t = 0
        self._trail: list[np.ndarray] = []

        # External renderer attached lazily by callers.
        self._renderer = None

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_xy(self, margin: float, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        lo = -self.cfg.arena_half + margin
        hi = +self.cfg.arena_half - margin
        return rng.uniform(lo, hi, size=2).astype(np.float32)

    def _sample_hazards(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one hard-hazard layout, or fail without a partial layout."""
        cfg = self.cfg
        hazards: list[np.ndarray] = []
        tries = 0
        while len(hazards) < cfg.n_hazards:
            tries += 1
            if tries > cfg.max_rejection_tries:
                raise _LayoutSamplingFailure(
                    f"Failed to sample {cfg.n_hazards} non-overlapping hazards in "
                    f"{cfg.max_rejection_tries} tries; relax min_hazard_pair_sep "
                    f"or shrink hazards."
                )
            r = float(rng.uniform(cfg.hazard_radius_min, cfg.hazard_radius_max))
            xy = self._sample_xy(margin=r + 0.1, rng=rng)  # keep hazard fully inside arena
            ok = True
            for hx, hy, hr in hazards:
                d = float(np.linalg.norm(xy - np.array([hx, hy], dtype=np.float32)))
                # min_hazard_pair_sep is an edge-to-edge gap, so the center
                # distance must exceed both radii plus that configured gap.
                if d < (r + hr + cfg.min_hazard_pair_sep):
                    ok = False
                    break
            if ok:
                hazards.append(np.array([xy[0], xy[1], r], dtype=np.float32))
        return np.stack(hazards, axis=0) if hazards else np.zeros((0, 3), dtype=np.float32)

    def _sample_start_goal(
        self, rng: np.random.Generator, hazards: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample valid start and goal positions for one hazard candidate."""
        cfg = self.cfg
        for _ in range(cfg.max_rejection_tries):
            start = self._sample_xy(margin=cfg.agent_radius + 0.1, rng=rng)
            if self._clear_of_all_hazards(
                start, cfg.agent_radius + cfg.min_start_clearance, hazards
            ):
                break
        else:
            raise _LayoutSamplingFailure("Failed to sample valid start position.")

        for _ in range(cfg.max_rejection_tries):
            goal = self._sample_xy(margin=cfg.goal_radius + 0.1, rng=rng)
            if not self._clear_of_all_hazards(
                goal, cfg.goal_radius + cfg.min_goal_clearance, hazards
            ):
                continue
            if float(np.linalg.norm(goal - start)) < cfg.min_start_goal_dist:
                continue
            break
        else:
            raise _LayoutSamplingFailure("Failed to sample valid goal position.")
        return start, goal

    def _sample_layout(
        self,
        rng: np.random.Generator | None = None,
        *,
        allow_zone_fallback: bool = False,
    ) -> dict[str, Any]:
        """Sample one complete layout; a failed zone placement is atomic."""
        rng = self.rng if rng is None else rng
        hazards = self._sample_hazards(rng)
        start, goal = self._sample_start_goal(rng, hazards)
        zones, styles, placement_attempts = self._sample_semantic_zones(
            rng,
            hazards,
            start,
            goal,
            allow_fallback=allow_zone_fallback,
        )
        valid, reasons = zone_layout_valid(hazards, zones, start, goal, self.cfg)
        if not valid:
            raise _LayoutSamplingFailure(
                "Sampled layout failed validation: " + "; ".join(reasons),
                placement_attempts=placement_attempts,
            )

        # Commit state only after the complete candidate has passed validation.
        self.hazards = hazards
        self.pos = start
        self.goal = goal
        self.vel = np.zeros(2, dtype=np.float32)
        self.semantic_zones = zones
        self.semantic_zone_styles = styles
        return {"placement_attempts": placement_attempts}

    def _sample_semantic_zones(
        self,
        rng: np.random.Generator,
        hazards: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        allow_fallback: bool = False,
    ) -> tuple[np.ndarray, list[str] | None, int]:
        """Sample off-limits zones, by default straddling the start->goal line.

        Placing a zone so the straight start->goal segment passes through it
        makes it likely that a geometry-only controller (which heads roughly
        straight at the goal) will cut through it — making the semantic
        constraint actually *bite*. Zones are rejected if they would swallow the start/goal or
        overlap a hard hazard or another zone, so there is always room to detour
        around them.  There is deliberately no unchecked fallback: failure
        discards the whole candidate layout in ``reset``.
        """
        cfg = self.cfg
        if cfg.n_semantic_zones <= 0:
            return np.zeros((0, 3), dtype=np.float32), None, 0

        seg = goal - start
        seg_len = float(np.linalg.norm(seg))
        seg_dir = seg / (seg_len + 1e-9)
        perp = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float32)

        n = cfg.n_semantic_zones
        zones: list[np.ndarray] = []
        placement_attempts = 0
        for zi in range(n):
            placed = False
            for _try in range(cfg.max_rejection_tries):
                placement_attempts += 1
                r = float(rng.uniform(cfg.semantic_radius_min, cfg.semantic_radius_max))
                if cfg.semantic_on_corridor:
                    if n > 1:
                        # Spread zones along the route: zone zi gets its own t-band
                        # tiling [0.18, 0.82], so multiple terrains line the path
                        # rather than piling at the midpoint (and rejecting).
                        lo = 0.18 + 0.64 * zi / n
                        hi = 0.18 + 0.64 * (zi + 1) / n
                        t = float(rng.uniform(lo, hi))
                    else:
                        t = float(rng.uniform(cfg.semantic_corridor_t_min,
                                              cfg.semantic_corridor_t_max))
                    # lateral offset < r keeps the segment intersecting the zone
                    lateral = float(rng.uniform(-0.4, 0.4)) * r
                    center = start + seg_dir * (t * seg_len) + perp * lateral
                else:
                    center = self._sample_xy(margin=r + 0.1, rng=rng)
                lim = cfg.arena_half - r - 0.1
                center = np.clip(center, -lim, lim).astype(np.float32)

                # Keep start and goal outside the zone (need clear endpoints).
                if float(np.linalg.norm(center - start)) < r + cfg.agent_radius + 0.3:
                    continue
                if float(np.linalg.norm(center - goal)) < r + cfg.goal_radius + 0.3:
                    continue
                # Leave a gap to hard hazards so an avoider can slip past.
                ok = True
                for hx, hy, hr in hazards:
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
                if not allow_fallback:
                    raise _LayoutSamplingFailure(
                        f"Failed to place semantic zone {zi} in "
                        f"{cfg.max_rejection_tries} tries",
                        placement_attempts=placement_attempts,
                    )
                # The corridor is a preferred placement policy, not an
                # invariant of the evaluator.  A hard hazard can occasionally
                # block one narrow t-band for an otherwise valid layout.  In
                # that case use a checked secondary sampler before discarding
                # the complete layout.  This branch is enabled only after all
                # complete-layout resamples have been exhausted, preserving
                # the existing golden resample sequence.
                return self._fallback_semantic_zones(
                    rng,
                    hazards,
                    start,
                    goal,
                    placement_attempts,
                    failed_zone=zi,
                )
        # Per-zone appearance styles (parallel to `zones`). Assigned by index from
        # the pool — no RNG draw, so single-zone homogeneous runs are unaffected.
        # None when no pool is set => renderer uses its single default style.
        pool = cfg.semantic_styles
        styles = [pool[i % len(pool)] for i in range(len(zones))] if pool else None
        return np.stack(zones, axis=0), styles, placement_attempts

    def _fallback_semantic_zones(
        self,
        rng: np.random.Generator,
        hazards: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        placement_attempts: int,
        *,
        failed_zone: int,
    ) -> tuple[np.ndarray, list[str] | None, int]:
        """Find a valid zone layout after a preferred corridor band is blocked.

        This is not an unchecked fallback: every joint candidate is validated
        by ``zone_layout_valid`` before it is returned.  The first phase keeps
        zones in their assigned corridor bands but allows a wider lateral
        offset; the second phase relaxes only the presentation preference and
        samples the open arena.  The evaluator geometry and all rejection
        predicates remain unchanged.
        """
        cfg = self.cfg
        n = int(cfg.n_semantic_zones)
        seg = goal - start
        seg_len = float(np.linalg.norm(seg))
        seg_dir = seg / (seg_len + 1e-9)
        perp = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float32)
        phases = (True, False) if cfg.semantic_on_corridor else (False,)

        for keep_corridor in phases:
            for _try in range(cfg.max_rejection_tries):
                placement_attempts += 1
                candidates: list[np.ndarray] = []
                for zi in range(n):
                    r = float(rng.uniform(cfg.semantic_radius_min, cfg.semantic_radius_max))
                    if keep_corridor:
                        if n > 1:
                            lo = 0.18 + 0.64 * zi / n
                            hi = 0.18 + 0.64 * (zi + 1) / n
                            t = float(rng.uniform(lo, hi))
                        else:
                            t = float(rng.uniform(
                                cfg.semantic_corridor_t_min,
                                cfg.semantic_corridor_t_max,
                            ))
                        # Widen only the lateral search.  The route remains the
                        # primary axis, while blocked strips can be bypassed.
                        lateral = float(rng.uniform(-1.5, 1.5)) * r
                        center = start + seg_dir * (t * seg_len) + perp * lateral
                    else:
                        center = self._sample_xy(margin=r + 0.1, rng=rng)
                    lim = cfg.arena_half - r - 0.1
                    center = np.clip(center, -lim, lim).astype(np.float32)
                    candidates.append(np.array([center[0], center[1], r], dtype=np.float32))

                zones = np.stack(candidates, axis=0)
                valid, _reasons = zone_layout_valid(hazards, zones, start, goal, cfg)
                if valid:
                    pool = cfg.semantic_styles
                    styles = [pool[i % len(pool)] for i in range(n)] if pool else None
                    return zones, styles, placement_attempts

        raise _LayoutSamplingFailure(
            f"Failed to place semantic zone {failed_zone} after preferred and "
            f"checked fallback sampling",
            placement_attempts=placement_attempts,
        )

    def _in_semantic_zone(self, p: np.ndarray) -> bool:
        """True if the agent's *center* lies inside any semantic keep-out zone."""
        for zx, zy, zr in self.semantic_zones:
            if float(np.linalg.norm(p - np.array([zx, zy], dtype=np.float32))) < float(zr):
                return True
        return False

    def _clear_of_all_hazards(
        self,
        p: np.ndarray,
        body_radius: float,
        hazards: np.ndarray | None = None,
    ) -> bool:
        """True if a body of radius `body_radius` centered at p touches no hazard."""
        hazards = self.hazards if hazards is None else hazards
        for hx, hy, hr in hazards:
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
        # Each complete layout attempt gets its own child stream.  With an
        # explicit seed this makes retries deterministic and prevents a failed
        # candidate from shifting the random stream used by later candidates.
        if seed is not None:
            root_sequence = np.random.SeedSequence(seed)
        else:
            root_seed = int(self.rng.integers(0, 2**63 - 1, dtype=np.uint64))
            root_sequence = np.random.SeedSequence(root_seed)
        n_resample_slots = max(0, int(self.cfg.max_layout_resamples))
        child_sequences = root_sequence.spawn(n_resample_slots + 1)

        last_failure: _LayoutSamplingFailure | None = None
        placement_attempts = 0
        for resample_count, child_sequence in enumerate(child_sequences):
            try:
                metadata = self._sample_layout(
                    np.random.default_rng(child_sequence),
                    allow_zone_fallback=(resample_count == n_resample_slots),
                )
                placement_attempts += int(metadata["placement_attempts"])
                break
            except _LayoutSamplingFailure as exc:
                last_failure = exc
                placement_attempts += int(exc.placement_attempts)
        else:
            assert last_failure is not None
            raise RuntimeError(
                f"Failed to sample a valid layout after {len(child_sequences)} attempts "
                f"({n_resample_slots} resamples): {last_failure}"
            ) from last_failure

        self.layout_valid = True
        self.placement_attempts = placement_attempts
        self.resample_count = resample_count
        valid, reasons = zone_layout_valid(
            self.hazards, self.semantic_zones, self.pos, self.goal, self.cfg
        )
        self.layout_valid = bool(valid)
        assert self.layout_valid, "reset produced an invalid layout: " + "; ".join(reasons)
        self.t = 0
        self._semantic_steps = 0
        self._trail = [self.pos.copy()]
        info = {
            "goal": self.goal.copy(),
            "hazards": self.hazards.copy(),
            "semantic_zones": self.semantic_zones.copy(),
            "start": self.pos.copy(),
            "layout_valid": self.layout_valid,
            "placement_attempts": self.placement_attempts,
            "resample_count": self.resample_count,
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
        info["layout_valid"] = self.layout_valid
        info["placement_attempts"] = self.placement_attempts
        info["resample_count"] = self.resample_count
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
