"""
Sampling-based MPC controller for PointHazardEnv.

Unlike `safe_expert.py` (A* on a coarse grid + PD pure-pursuit, which ignores
the agent's momentum and so can overshoot into a hazard), this controller rolls
candidate action sequences through the *exact* env dynamics — semi-implicit
Euler with linear drag, force scaling, speed clipping, and wall bounce — and
scores candidate rollouts by predicted hazard and soft-zone costs. It is a
safety-oriented sampling MPC controller (empirical): sampled rollout behavior
is reported as an observed metric, not as a formal safety property.

Optimizer: cross-entropy method (CEM) with a warm-started nominal sequence.

Drop-in for SafeExpert — same public API:
    ctrl = MPCExpert(env, cfg=MPCConfig())
    ok = ctrl.plan(start_xy, goal_xy, hazards)   # set target + avoid set
    ok = ctrl.reset_from_obs(obs)                 # set target from env obs
    a  = ctrl.act(obs)                            # one receding-horizon step
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig:
    horizon: int = 20               # rollout length (steps)
    n_samples: int = 256            # action sequences sampled per CEM iteration
    n_iters: int = 3                # CEM refinement iterations
    elite_frac: float = 0.1         # top fraction kept to refit the sampling dist
    init_std: float = 0.6           # initial per-step action std
    min_std: float = 0.05           # floor on the refit std (keeps exploration)
    safety_margin: float = 0.15     # inflate hazard radius by this when scoring

    # Cost weights
    goal_weight: float = 1.0        # per-step distance-to-goal
    terminal_weight: float = 8.0    # final distance-to-goal
    clearance_weight: float = 4.0   # penalty for getting within safety_margin
    effort_weight: float = 0.01     # control magnitude penalty
    collision_penalty: float = 1e4  # one-off penalty when a rollout hits a hazard
    soft_zone_weight: float = 30.0  # per-step penalty for being inside a SOFT zone
                                    # (a VLM-perceived keep-out: strongly avoided
                                    # but never forbidden, so it cannot livelock)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MPCExpert:
    """CEM-MPC over the exact PointHazard dynamics."""

    def __init__(self, env, cfg: MPCConfig | None = None, *, rng: np.random.Generator | None = None):
        self.env = env
        self.cfg = cfg or MPCConfig()
        self.rng = rng or np.random.default_rng()

        self.goal = np.zeros(2, dtype=np.float32)
        self.hazards = np.zeros((0, 3), dtype=np.float32)
        # Soft keep-out zones (x, y, r): strongly penalized but NOT hard-rejected.
        # Used to feed an uncertain VLM-perceived zone to the planner without the
        # livelock a mis-placed hard obstacle can cause. Empty for plain/oracle use.
        self.soft_zones = np.zeros((0, 3), dtype=np.float32)
        self._nominal = np.zeros((self.cfg.horizon, 2), dtype=np.float32)
        self.last_plan_ok = False

    def set_soft_zones(self, zones: np.ndarray) -> None:
        """Set soft keep-out disks (x, y, r). They add a per-step cost inside the
        disk but never the hard collision penalty, so the planner skirts them when
        it can yet always retains a feasible path to the target."""
        z = np.asarray(zones, dtype=np.float32).reshape(-1, 3) if len(zones) else np.zeros((0, 3), np.float32)
        self.soft_zones = z

    # ------------------------------------------------------------------
    # Target setup (mirrors SafeExpert's API)
    # ------------------------------------------------------------------

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray, hazards: np.ndarray) -> bool:
        # MPC is reactive; "planning" just sets the target and avoid set. The
        # start is implicit in act(obs). Returns True unless the goal sits
        # inside a hazard (then there is no safe terminal state).
        self.goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)
        self.hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
        self._nominal[:] = 0.0
        self.last_plan_ok = self._goal_is_reachable()
        return self.last_plan_ok

    def reset_from_obs(self, obs: np.ndarray) -> bool:
        # PointHazardEnv obs is absolute (see env_pointhazard._build_obs).
        n = self.env.cfg.n_hazards
        goal = obs[4:6].astype(np.float32)
        hazards = obs[6 : 6 + 3 * n].reshape(n, 3).astype(np.float32)
        return self.plan(obs[0:2].astype(np.float32), goal, hazards)

    def _goal_is_reachable(self) -> bool:
        if self.hazards.size == 0:
            return True
        d = np.linalg.norm(self.hazards[:, :2] - self.goal[None, :], axis=1)
        return bool(np.all(d - self.hazards[:, 2] - self.env.cfg.agent_radius > -1e-6))

    # ------------------------------------------------------------------
    # Vectorized dynamics rollout (matches env_pointhazard.step exactly)
    # ------------------------------------------------------------------

    def _rollout_costs(
        self, pos0: np.ndarray, vel0: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Roll K action sequences and return their costs.

        actions: (K, H, 2) in [-1, 1].  Returns (K,) cost array.
        """
        c = self.env.cfg
        cfg = self.cfg
        K = actions.shape[0]

        pos = np.tile(pos0.astype(np.float32), (K, 1))
        vel = np.tile(vel0.astype(np.float32), (K, 1))
        cost = np.zeros(K, dtype=np.float32)
        alive = np.ones(K, dtype=bool)  # False once a rollout has hit a hazard

        wall = c.arena_half - c.agent_radius
        haz_xy = self.hazards[:, :2]
        haz_r = self.hazards[:, 2] + c.agent_radius + cfg.safety_margin  # inflated

        for t in range(actions.shape[1]):
            force = c.force_scale * actions[:, t, :]
            vel = vel + (force - c.drag * vel) * c.dt
            speed = np.linalg.norm(vel, axis=1, keepdims=True)
            too_fast = (speed > c.max_speed).ravel()
            if np.any(too_fast):
                vel[too_fast] = vel[too_fast] * (c.max_speed / speed[too_fast])
            pos = pos + vel * c.dt

            # Wall bounce (reflect + clamp), per dimension.
            for d in range(2):
                lo = pos[:, d] < -wall
                hi = pos[:, d] > wall
                pos[lo, d] = -wall
                vel[lo, d] = -vel[lo, d] * 0.5
                pos[hi, d] = wall
                vel[hi, d] = -vel[hi, d] * 0.5

            # Clearance to nearest (inflated) hazard.
            if haz_xy.shape[0] > 0:
                diff = pos[:, None, :] - haz_xy[None, :, :]      # (K, M, 2)
                dist = np.linalg.norm(diff, axis=2)              # (K, M)
                clearance = np.min(dist - haz_r[None, :], axis=1)  # (K,)
                hit = (clearance < 0.0) & alive
                # Penalize earlier collisions more, so a cornered controller
                # buys time (stalls/evades) and lets the receding horizon escape
                # rather than diving toward a goal-adjacent hazard.
                cost[hit] += cfg.collision_penalty * (actions.shape[1] - t)
                alive &= ~hit
                # Shaping: discourage hugging the inflated hazard boundary.
                shaping = np.maximum(0.0, cfg.safety_margin - np.maximum(clearance, 0.0))
                cost[alive] += cfg.clearance_weight * shaping[alive]

            # Soft keep-out zones: per-step penalty for being inside, scaled by how
            # deep, but never the hard collision penalty (so a path always exists).
            if self.soft_zones.shape[0] > 0:
                sz_xy = self.soft_zones[:, :2]
                sz_r = self.soft_zones[:, 2]
                sdiff = pos[:, None, :] - sz_xy[None, :, :]      # (K, M, 2)
                sdist = np.linalg.norm(sdiff, axis=2)            # (K, M)
                inside = np.maximum(0.0, sz_r[None, :] - sdist)  # depth past edge
                cost[alive] += cfg.soft_zone_weight * np.max(inside, axis=1)[alive]

            d_goal = np.linalg.norm(pos - self.goal[None, :], axis=1)
            cost[alive] += cfg.goal_weight * d_goal[alive]
            cost[alive] += cfg.effort_weight * np.sum(actions[:, t, :] ** 2, axis=1)[alive]

        # Terminal goal cost (only meaningful for still-alive rollouts).
        d_goal_final = np.linalg.norm(pos - self.goal[None, :], axis=1)
        cost[alive] += cfg.terminal_weight * d_goal_final[alive]
        return cost

    # ------------------------------------------------------------------
    # Receding-horizon step
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pos = obs[0:2].astype(np.float32)
        vel = obs[2:4].astype(np.float32)

        mean = self._nominal.copy()              # (H, 2) warm start
        std = np.full((cfg.horizon, 2), cfg.init_std, dtype=np.float32)
        n_elite = max(1, int(cfg.elite_frac * cfg.n_samples))

        best_seq = mean
        for _ in range(cfg.n_iters):
            noise = self.rng.standard_normal((cfg.n_samples, cfg.horizon, 2)).astype(np.float32)
            actions = np.clip(mean[None] + std[None] * noise, -1.0, 1.0)
            costs = self._rollout_costs(pos, vel, actions)
            elite_idx = np.argsort(costs)[:n_elite]
            elites = actions[elite_idx]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), cfg.min_std)
            best_seq = actions[int(np.argmin(costs))]

        # Commit the first action; warm-start the next step with the shifted plan.
        action = best_seq[0].copy()
        self._nominal[:-1] = best_seq[1:]
        self._nominal[-1] = 0.0
        return np.clip(action, -1.0, 1.0).astype(np.float32)
