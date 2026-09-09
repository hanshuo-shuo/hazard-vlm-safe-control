"""Spatial fields and swept circular footprints in a calibrated ground plane.

Pixel coordinates refer to pixel edges; samples are at (column+.5, row+.5).
All motion points and radii use world units. No function consults a simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np


CHANNELS = ("water_ingress_demand", "low_traction_demand", "fragile_surface_property")
TERRAIN_CHANNEL = {"water": 0, "mud": 1, "grass": 2, "fragile": 2}


def finite_points(value: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("motion must be a nonempty finite Nx2 array")
    return points


def distance_to_motion(points: np.ndarray, motion: Sequence[Sequence[float]]) -> np.ndarray:
    """Exact distance to a polyline, including stationary transitions."""
    query = np.asarray(points, dtype=np.float64)
    if query.shape[-1] != 2 or not np.isfinite(query).all():
        raise ValueError("query points must be finite with final dimension 2")
    path = finite_points(motion)
    distance = np.linalg.norm(query - path[0], axis=-1)
    for left, right in zip(path, path[1:]):
        delta = right - left
        length2 = float(delta @ delta)
        fraction = np.clip(np.sum((query - left) * delta, axis=-1) / max(length2, 1e-30), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(query - (left + fraction[..., None] * delta), axis=-1))
    return distance


def intersects_disk(motion: Sequence[Sequence[float]], robot_radius: float,
                    center: Sequence[float], radius: float) -> bool:
    if not np.isfinite([robot_radius, radius]).all() or robot_radius < 0 or radius <= 0:
        raise ValueError("radii must be finite; robot radius >= 0 and terrain radius > 0")
    return bool(distance_to_motion(np.asarray(center, dtype=float), motion) <= radius + robot_radius)


@dataclass(frozen=True)
class GroundProjection:
    """A ground-plane homography and its image dimensions.

    ``matrix`` maps world [x,y,1] to homogeneous image [u,v,depth].
    Positive depth is required. A projection alone does not certify visibility;
    native callers supply a depth-tested visibility mask separately.
    """
    width: int
    height: int
    matrix: tuple[tuple[float, ...], ...]
    source: str

    def __post_init__(self):
        h = np.asarray(self.matrix, dtype=float)
        if self.width < 2 or self.height < 2 or h.shape != (3, 3) or not np.isfinite(h).all():
            raise ValueError("invalid projection")
        if abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("ground projection is singular")

    @classmethod
    def orthographic(cls, width: int, height: int, bounds: Sequence[float]):
        xmin, xmax, ymin, ymax = map(float, bounds)
        if not np.isfinite(bounds).all() or xmax <= xmin or ymax <= ymin:
            raise ValueError("invalid world bounds")
        sx, sy = width / (xmax - xmin), height / (ymax - ymin)
        return cls(width, height, ((sx, 0., -sx*xmin), (0., -sy, sy*ymax), (0., 0., 1.)), "orthographic-world-v1")

    @classmethod
    def mujoco_camera(cls, width: int, height: int, position, rotation, fovy: float, ground_z: float = 0.):
        r = np.asarray(rotation, dtype=float).reshape(3, 3).T
        pos = np.asarray(position, dtype=float)
        if not 0 < float(fovy) < 180 or pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("invalid camera")
        plane = np.column_stack((r[:, :2], r[:, 2] * ground_z - r @ pos))
        f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        intrinsics = np.array([[f, 0, -width/2], [0, -f, -height/2], [0, 0, -1.]])
        h = intrinsics @ plane
        return cls(width, height, tuple(map(tuple, h.tolist())), "mujoco-calibrated-ground-v1")

    def project(self, world) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(world, dtype=float)
        if p.shape[-1] != 2 or not np.isfinite(p).all():
            raise ValueError("invalid world coordinates")
        homogeneous = np.concatenate((p, np.ones(p.shape[:-1] + (1,))), axis=-1) @ np.asarray(self.matrix).T
        depth = homogeneous[..., 2]
        uv = homogeneous[..., :2] / np.where(np.abs(depth) > 1e-12, depth, np.nan)[..., None]
        valid = (depth > 1e-12) & (uv[..., 0] >= 0) & (uv[..., 0] < self.width) & (uv[..., 1] >= 0) & (uv[..., 1] < self.height)
        return uv, valid

    def world_grid(self) -> tuple[np.ndarray, np.ndarray]:
        y, x = np.mgrid[:self.height, :self.width]
        image = np.stack((x + .5, y + .5, np.ones_like(x)), axis=-1)
        homogeneous = image @ np.linalg.inv(np.asarray(self.matrix)).T
        scale = homogeneous[..., 2]
        world = homogeneous[..., :2] / np.where(np.abs(scale) > 1e-12, scale, np.nan)[..., None]
        valid = (scale > 1e-12) & np.isfinite(world).all(axis=-1)
        return world, valid

    def to_dict(self):
        return {"width": self.width, "height": self.height, "matrix": self.matrix, "source": self.source}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def oracle_field(regions, projection: GroundProjection) -> np.ndarray:
    """Simulator/fixture-only upper-bound supervision, never a VLM label."""
    grid, valid = projection.world_grid()
    field = np.zeros((projection.height, projection.width, len(CHANNELS)), dtype=np.float32)
    for region in regions:
        terrain = region["terrain_class"]
        if terrain == "solid_ground":
            continue
        if terrain not in TERRAIN_CHANNEL:
            raise ValueError(f"unregistered terrain: {terrain}")
        radius = float(region["radius"])
        center = np.asarray(region["center_xy"], dtype=float)
        if radius <= 0 or not np.isfinite(radius) or center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("invalid terrain geometry")
        inside = valid & (np.linalg.norm(grid - center, axis=-1) <= radius)
        field[..., TERRAIN_CHANNEL[terrain]][inside] = 1.
    return field


@dataclass(frozen=True)
class Exposure:
    validity: str
    values: tuple[float, ...] | None
    maximum: tuple[float, ...] | None
    mean: tuple[float, ...] | None
    footprint_pixels: int
    quantile: float

    def to_dict(self):
        return dict(vars(self))


def measure_exposure(field: np.ndarray, motion, robot_radius: float,
                     projection: GroundProjection, *, quantile: float = .95,
                     visibility: np.ndarray | None = None) -> tuple[Exposure, np.ndarray]:
    """Pool a spatial field over a swept disk; invalid projection is explicit.

    Quantile exposure is a soft severity signal. It is NOT the binary contact
    truth: a grazing overlap can have q95=0 even when maximum=1. Binary safety
    labels must use ``intersects_disk`` or the independent simulator evaluator.
    """
    path = finite_points(motion)
    values = np.asarray(field, dtype=float)
    if values.shape != (projection.height, projection.width, len(CHANNELS)):
        raise ValueError("field dimensions do not match projection and channels")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("field values must be finite in [0,1]")
    if not np.isfinite(robot_radius) or robot_radius <= 0 or not 0 <= quantile <= 1:
        raise ValueError("positive robot radius and quantile in [0,1] required")
    grid, valid = projection.world_grid()
    # Invalid camera rays carry no usable ground coordinate.
    mask = valid & (distance_to_motion(np.where(valid[..., None], grid, 0.), path) <= robot_radius)
    # Project the bounding square of every swept disk. This conservative check
    # also covers the intervening capsules, which are convex for each segment.
    corners = path[:, None, :] + robot_radius * np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    _, corners_valid = projection.project(corners)
    visible = valid if visibility is None else np.asarray(visibility, dtype=bool) & valid
    if visible.shape != mask.shape:
        raise ValueError("visibility mask shape mismatch")
    count = int(mask.sum())
    if not corners_valid.all() or count == 0 or (mask & ~visible).any():
        return Exposure("unknown_projection", None, None, None, count, quantile), mask
    samples = values[mask]
    return Exposure("valid", tuple(np.quantile(samples, quantile, axis=0)),
                    tuple(samples.max(axis=0)), tuple(samples.mean(axis=0)), count, quantile), mask
