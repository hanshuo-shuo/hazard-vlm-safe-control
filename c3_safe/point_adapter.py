"""New semantic accounting atop unchanged native PointHazard dynamics."""
from __future__ import annotations

from dataclasses import asdict

from envs.point_hazard_adapter import PointHazardAdapter
from c3_safe.costs import Capability, Rules, semantic_contact
from c3_safe.geometry import GroundProjection


class C3PointHazardAdapter(PointHazardAdapter):
    def __init__(self, *args, capability: Capability = Capability(), rules: Rules = Rules(), **kwargs):
        super().__init__(*args, **kwargs)
        self.capability = capability
        self.rules = rules

    def reset(self, *, seed=None):
        result = super().reset(seed=seed)
        if self.cfg.n_semantic_zones and not self.cfg.semantic_terrain_classes:
            raise ValueError("C3 scenes require explicit terrain classes, independently of appearance")
        self._scene.update({
            "agent_radius": self.cfg.agent_radius,
            "semantic_metric_version": "swept-disk-v1",
            "footprint_source": "PointHazardConfig.agent_radius",
            "capability": asdict(self.capability), "rules": asdict(self.rules),
            "capability_vector": self.capability.vector, "rule_vector": self.rules.vector,
        })
        if self._env._renderer is not None:
            projection = self.ground_projection()
            self._scene["camera_calibration"] = projection.to_dict()
            self._scene["camera_calibration_hash"] = projection.sha256
        return result

    def ground_projection(self):
        renderer = self._env._renderer
        if renderer is None:
            raise RuntimeError("a renderer is required for spatial supervision")
        return GroundProjection.orthographic(renderer.img_size, renderer.img_size,
            (renderer.world_x_min, renderer.world_x_max, renderer.world_y_min, renderer.world_y_max))

    def step(self, action):
        before = self._trajectory[-1]["agent_center"]
        result = super().step(action)
        after = self._trajectory[-1]["agent_center"]
        self._semantic_violations[-1] = semantic_contact(
            [before, after], self.cfg.agent_radius, self._scene["semantic_terrain"], self.capability, self.rules,
        )
        return result
