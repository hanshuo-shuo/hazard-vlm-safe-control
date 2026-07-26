"""Offline detector geometry decomposition and calibration discipline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class GeometryArm(str, Enum):
    DETECTOR_FULL = "detector_full_geometry"
    DETECTOR_CENTER_RADIUS = "detector_center_detector_radius"
    DETECTOR_CENTER_ORACLE_RADIUS = "detector_center_oracle_radius"
    ORACLE_CENTER_DETECTOR_RADIUS = "oracle_center_detector_radius"
    ORACLE_CENTER_RADIUS = "oracle_center_oracle_radius"
    DETECTOR_BOX = "detector_bounding_box"
    DETECTOR_POLYGON = "detector_polygon_convex_hull"
    DETECTOR_MASK = "detector_mask"
    BLIND = "blind"
    ORACLE = "oracle"


@dataclass(frozen=True)
class CalibrationBudget:
    split: str
    max_trials: int
    trials_used: int
    parameters_frozen: bool = False
    test_run_consumed: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"dev", "pilot", "test"}:
            raise ValueError("split must be dev, pilot, or test")
        if self.max_trials < 1 or not 0 <= self.trials_used <= self.max_trials:
            raise ValueError("invalid calibration budget")
        if self.split == "test" and not self.parameters_frozen:
            raise PermissionError("test split requires frozen parameters")
        if self.split == "test" and self.test_run_consumed:
            raise PermissionError("test-run sentinel has already been consumed")


def compose_geometry_arm(
    arm: GeometryArm,
    *,
    detector_disk: Sequence[float] | None,
    oracle_disk: Sequence[float] | None,
    detector_box: Mapping[str, Any] | None = None,
    detector_polygon: Mapping[str, Any] | None = None,
    detector_mask: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compose one decomposition arm without manufacturing missing geometry."""
    arm = GeometryArm(arm)
    if arm is GeometryArm.BLIND:
        return None
    if arm in {GeometryArm.ORACLE, GeometryArm.ORACLE_CENTER_RADIUS}:
        if oracle_disk is None:
            raise ValueError("oracle disk is required")
        return {"geometry_type": "disk", "center_xy": list(oracle_disk[:2]), "radius": float(oracle_disk[2])}
    if arm in {GeometryArm.DETECTOR_FULL, GeometryArm.DETECTOR_CENTER_RADIUS}:
        if detector_disk is None:
            return None
        return {"geometry_type": "disk", "center_xy": list(detector_disk[:2]), "radius": float(detector_disk[2])}
    if arm is GeometryArm.DETECTOR_CENTER_ORACLE_RADIUS:
        if detector_disk is None or oracle_disk is None:
            return None
        return {"geometry_type": "disk", "center_xy": list(detector_disk[:2]), "radius": float(oracle_disk[2])}
    if arm is GeometryArm.ORACLE_CENTER_DETECTOR_RADIUS:
        if detector_disk is None or oracle_disk is None:
            return None
        return {"geometry_type": "disk", "center_xy": list(oracle_disk[:2]), "radius": float(detector_disk[2])}
    if arm is GeometryArm.DETECTOR_BOX:
        return None if detector_box is None else {"geometry_type": "aabb", **dict(detector_box)}
    if arm is GeometryArm.DETECTOR_POLYGON:
        return None if detector_polygon is None else {"geometry_type": "polygon", **dict(detector_polygon)}
    if arm is GeometryArm.DETECTOR_MASK:
        return None if detector_mask is None else {"geometry_type": "mask_reference", **dict(detector_mask)}
    raise AssertionError(f"unhandled geometry arm {arm}")


def _polygon_area_centroid(
    points: Sequence[Sequence[float]],
) -> tuple[float, tuple[float, float]]:
    vertices = [(float(point[0]), float(point[1])) for point in points]
    if len(vertices) < 3:
        raise ValueError("polygon requires at least three points")
    twice_area = 0.0
    cx_numerator = 0.0
    cy_numerator = 0.0
    for left, right in zip(vertices, vertices[1:] + vertices[:1]):
        cross = left[0] * right[1] - right[0] * left[1]
        twice_area += cross
        cx_numerator += (left[0] + right[0]) * cross
        cy_numerator += (left[1] + right[1]) * cross
    if abs(twice_area) <= 1e-12:
        raise ValueError("polygon area must be positive")
    return (
        abs(twice_area) / 2.0,
        (
            cx_numerator / (3.0 * twice_area),
            cy_numerator / (3.0 * twice_area),
        ),
    )


def geometry_area_centroid(
    geometry: Mapping[str, Any] | None,
    *,
    mask_points_xy: Sequence[Sequence[float]] | None = None,
    mask_pixel_area: float | None = None,
) -> tuple[float | None, tuple[float, float] | None]:
    """Return measured area/centroid without silently turning shapes into disks."""
    if geometry is None:
        return None, None
    kind = str(geometry.get("geometry_type"))
    if kind == "disk":
        radius = float(geometry["radius"])
        center = tuple(float(value) for value in geometry["center_xy"])
        return math.pi * radius * radius, (center[0], center[1])
    if kind == "aabb":
        low = tuple(float(value) for value in geometry["min_xy"])
        high = tuple(float(value) for value in geometry["max_xy"])
        if low[0] >= high[0] or low[1] >= high[1]:
            raise ValueError("aabb min_xy must be below max_xy")
        return (
            (high[0] - low[0]) * (high[1] - low[1]),
            ((low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0),
        )
    if kind == "polygon":
        return _polygon_area_centroid(geometry["points"])
    if kind == "mask_reference":
        if not mask_points_xy or mask_pixel_area is None:
            raise ValueError("mask projection requires measured world points and pixel area")
        points = [(float(point[0]), float(point[1])) for point in mask_points_xy]
        return (
            len(points) * float(mask_pixel_area),
            (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            ),
        )
    raise ValueError(f"unsupported geometry type {kind!r}")


def geometry_metrics(
    geometry: Mapping[str, Any] | None,
    oracle_disk: Sequence[float] | None,
    *,
    mask_points_xy: Sequence[Sequence[float]] | None = None,
    mask_pixel_area: float | None = None,
) -> dict[str, float | None]:
    """Representation-independent center and area attribution metrics."""
    if geometry is None or oracle_disk is None:
        return {"center_error": None, "area_error": None}
    area, center = geometry_area_centroid(
        geometry,
        mask_points_xy=mask_points_xy,
        mask_pixel_area=mask_pixel_area,
    )
    ox, oy, radius = (float(value) for value in oracle_disk)
    assert area is not None and center is not None
    return {
        "center_error": math.hypot(center[0] - ox, center[1] - oy),
        "area_error": abs(area - math.pi * radius * radius),
    }


def project_geometry_to_planner_disks(
    geometry: Mapping[str, Any] | None,
    *,
    mask_points_xy: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Conservatively project every registered geometry kind to fixed-MPC disks.

    The fixed planner consumes disk primitives. AABB and polygon arms use their
    enclosing circle. A measured mask uses the enclosing circle of its actual
    foreground pixels; no mask is fabricated when the artifact is absent.
    """
    if geometry is None:
        return ()
    kind = str(geometry.get("geometry_type"))
    if kind == "disk":
        center = geometry["center_xy"]
        return ((float(center[0]), float(center[1]), float(geometry["radius"])),)
    if kind == "aabb":
        low = tuple(float(value) for value in geometry["min_xy"])
        high = tuple(float(value) for value in geometry["max_xy"])
        center = ((low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0)
        radius = math.hypot(high[0] - low[0], high[1] - low[1]) / 2.0
        return ((center[0], center[1], radius),)
    if kind == "polygon":
        _area, center = _polygon_area_centroid(geometry["points"])
        radius = max(
            math.hypot(float(point[0]) - center[0], float(point[1]) - center[1])
            for point in geometry["points"]
        )
        return ((center[0], center[1], radius),)
    if kind == "mask_reference":
        if not mask_points_xy:
            raise ValueError("mask projection requires its measured foreground pixels")
        points = [(float(point[0]), float(point[1])) for point in mask_points_xy]
        center = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        radius = max(
            math.hypot(point[0] - center[0], point[1] - center[1])
            for point in points
        )
        if radius <= 0:
            raise ValueError("mask foreground must span a positive area")
        return ((center[0], center[1], radius),)
    raise ValueError(f"unsupported geometry type {kind!r}")


def disk_metrics(
    detector: Sequence[float] | None,
    oracle: Sequence[float] | None,
) -> dict[str, float | None]:
    """Exact analytic metrics for two disks; missing estimates remain missing."""
    if detector is None or oracle is None:
        return {
            "center_error": None,
            "radius_error": None,
            "area_error": None,
            "iou": None,
            "false_positive_area": None,
            "false_negative_area": None,
            "boundary_distance": None,
        }
    dx, dy, dr = map(float, detector)
    ox, oy, radius = map(float, oracle)
    if dr <= 0 or radius <= 0:
        raise ValueError("disk radii must be positive")
    distance = math.hypot(dx - ox, dy - oy)
    area_d = math.pi * dr * dr
    area_o = math.pi * radius * radius
    if distance >= dr + radius:
        intersection = 0.0
    elif distance <= abs(dr - radius):
        intersection = math.pi * min(dr, radius) ** 2
    else:
        alpha = math.acos((distance * distance + dr * dr - radius * radius) / (2 * distance * dr))
        beta = math.acos((distance * distance + radius * radius - dr * dr) / (2 * distance * radius))
        term = (-distance + dr + radius) * (distance + dr - radius) * (distance - dr + radius) * (distance + dr + radius)
        intersection = dr * dr * alpha + radius * radius * beta - 0.5 * math.sqrt(max(0.0, term))
    union = area_d + area_o - intersection
    return {
        "center_error": distance,
        "radius_error": abs(dr - radius),
        "area_error": abs(area_d - area_o),
        "iou": intersection / union,
        "false_positive_area": area_d - intersection,
        "false_negative_area": area_o - intersection,
        "boundary_distance": distance + abs(dr - radius),
    }


def validate_operating_curve(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require a controlled single-family sweep with complete config hashes."""
    if not rows:
        raise ValueError("operating curve cannot be empty")
    families = {row.get("parameter_family") for row in rows}
    if len(families) != 1 or None in families:
        raise ValueError("operating curve must change exactly one parameter family")
    if any(not row.get("config_hash") for row in rows):
        raise ValueError("every operating-curve row requires a config hash")
    if len({row["config_hash"] for row in rows}) != len(rows):
        raise ValueError("operating-curve config hashes must be unique")


def pareto_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return success-maximizing, violation-minimizing nondominated rows."""
    validate_operating_curve(rows)
    result = []
    for row in rows:
        success = float(row["success_rate"])
        violation = float(row["semantic_violation_rate"])
        dominated = any(
            float(other["success_rate"]) >= success
            and float(other["semantic_violation_rate"]) <= violation
            and (
                float(other["success_rate"]) > success
                or float(other["semantic_violation_rate"]) < violation
            )
            for other in rows
        )
        if not dominated:
            result.append(row)
    return result
