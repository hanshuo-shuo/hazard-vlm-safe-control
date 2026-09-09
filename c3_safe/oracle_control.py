"""One shared global router + MPC for blind and privileged oracle diagnostics."""
from types import SimpleNamespace

import numpy as np

from c3_safe.geometry import intersects_disk
from mpc_expert import MPCConfig, MPCExpert
from safe_expert import SafeExpert, SafeExpertConfig


class FootprintGrid(SafeExpert):
    def _build_occupancy(self, hazards):
        blocked = super()._build_occupancy(hazards)
        centers = -self.env.cfg.arena_half + (np.arange(self.cfg.grid_res) + .5) * self._cell_size
        boundary = np.abs(centers) > self.env.cfg.arena_half - self.env.cfg.agent_radius
        return blocked | boundary[:, None] | boundary[None, :]


class RoutedMPC:
    """Global waypoints prevent local horizon traps; both arms use this code.

    The supplied disk list is the sole information-source intervention. It must
    contain native physical disks for the blind arm and explicitly privileged
    active terrain disks for the oracle arm. This is an empirical controller,
    not a learned policy or a formal safety shield.
    """
    def __init__(self, cfg, mpc_cfg: MPCConfig, observation, disks, *, seed: int,
                 grid_res: int = 100, route_margin: float = .25):
        model = SimpleNamespace(cfg=cfg)
        self.disks = np.asarray(disks, dtype=np.float32).reshape(-1,3)
        self.radius = cfg.agent_radius + mpc_cfg.safety_margin
        router = FootprintGrid(model, SafeExpertConfig(grid_res=grid_res,
                              safety_margin=route_margin, speed_noise_std=0.))
        if not router.plan(observation[:2],observation[4:6],self.disks):
            raise RuntimeError("global diagnostic planner found no feasible route")
        self.path = router.path
        self.controller = MPCExpert(model,cfg=mpc_cfg,rng=np.random.default_rng(seed))
        self.target_index = None
        self.path_index = 0

    def clear(self, start, end):
        return not any(intersects_disk([start,end],self.radius, disk[:2],disk[2]) for disk in self.disks)

    def act(self, observation):
        position = observation[:2]
        while self.path_index < len(self.path)-1 and np.linalg.norm(self.path[self.path_index]-position) < .3:
            self.path_index += 1
        target = self.path_index
        for index in range(len(self.path)-1,self.path_index-1,-1):
            if self.clear(position,self.path[index]):
                target = index
                break
        if target != self.target_index:
            self.controller.plan(position,self.path[target],self.disks)
            self.target_index = target
        return self.controller.act(observation)
