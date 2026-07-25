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
