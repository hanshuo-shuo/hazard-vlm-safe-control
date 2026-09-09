"""Separate environmental incompatibility and normative task rules."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from c3_safe.geometry import TERRAIN_CHANNEL, Exposure, intersects_disk


def _unit_vector(values, size: int, label: str):
    x = np.asarray(values, dtype=float)
    if x.shape != (size,) or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError(f"{label} must have {size} finite values in [0,1]")
    return x


@dataclass(frozen=True)
class Capability:
    waterproof: float = 0.
    rough_mobility: float = 0.

    def __post_init__(self):
        _unit_vector(self.vector, 2, "capability")

    @property
    def vector(self):
        return (self.waterproof, self.rough_mobility)


@dataclass(frozen=True)
class Rules:
    avoid_water: float = 0.
    avoid_mud: float = 0.
    protect_fragile: float = 0.

    def __post_init__(self):
        _unit_vector(self.vector, 3, "rules")

    @property
    def vector(self):
        return (self.avoid_water, self.avoid_mud, self.protect_fragile)


def relation_cost(exposure: Exposure | Sequence[float], capability: Capability, rules: Rules,
                  *, capability_weights=(1., 1.), rule_weights=(1., 1., 1.)) -> dict:
    if isinstance(exposure, Exposure):
        if exposure.validity != "valid":
            return {"validity": "unknown_projection", "capability_cost": None,
                    "rule_cost": None, "semantic_cost": None}
        exposure = exposure.values
    e = _unit_vector(exposure, 3, "exposure")
    w, v = np.asarray(capability_weights, dtype=float), np.asarray(rule_weights, dtype=float)
    if w.shape != (2,) or v.shape != (3,) or not np.isfinite(w).all() or not np.isfinite(v).all() or (w < 0).any() or (v < 0).any():
        raise ValueError("cost weights must be finite and nonnegative")
    cap = float(np.sum(w * np.maximum(e[:2] - capability.vector, 0)))
    rule = float(np.sum(v * rules.vector * e))
    return {"validity": "valid", "capability_cost": cap, "rule_cost": rule,
            "semantic_cost": -math.expm1(-(cap + rule))}


def active_region(region: dict, capability: Capability, rules: Rules) -> bool:
    terrain = region["terrain_class"]
    if terrain == "solid_ground":
        return False
    if terrain not in TERRAIN_CHANNEL:
        raise ValueError(f"unregistered terrain: {terrain}")
    channel = TERRAIN_CHANNEL[terrain]
    return bool((channel < 2 and capability.vector[channel] < 1.) or rules.vector[channel] > 0.)


def semantic_contact(motion, robot_radius: float, regions, capability: Capability, rules: Rules) -> bool:
    """Independent binary safety truth; do not threshold the soft q95 cost."""
    return any(active_region(r, capability, rules) and
               intersects_disk(motion, robot_radius, r["center_xy"], r["radius"])
               for r in regions)
