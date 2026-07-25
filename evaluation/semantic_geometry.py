"""Marker-free semantic geometry contract and coordinate adapters.

This module contains policy-visible estimates only.  Evaluator truth may be
converted by the explicit oracle adapter, but is never embedded as a hidden
field or retained by reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "semantic-geometry-v1"
ADAPTER_VERSION = "semantic-geometry-adapters-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_KEYS = frozenset(
    {
        "evaluator_truth",
        "ground_truth",
        "scene_manifest",
        "semantic_violation",
        "success",
        "reward",
        "raw_env",
        "env",
        "simulator",
    }
)


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class GeometryType(_StringEnum):
    DISK = "disk"
    AABB = "aabb"
    POLYGON = "polygon"
    MASK_REFERENCE = "mask_reference"
    EMPTY = "empty"


class CoordinateFrame(_StringEnum):
    IMAGE_PIXEL = "image_pixel_xy"
    NORMALIZED_IMAGE = "normalized_image_xy"
    PROVIDER_NATIVE = "provider_native_xy"
    WORLD = "world_xy"
    SIMULATOR = "simulator_xy"
    PLANNER_COST_MAP = "planner_cost_map_xy"


class GeometrySource(_StringEnum):
    NONE = "none"
    ORACLE = "oracle"
    FIXTURE = "fixture"
    DETECTOR = "detector"
    VLM_GROUNDING = "vlm_grounding"


class ApplicabilityState(_StringEnum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ProcessingStatus(_StringEnum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    NOT_RUN = "not_run"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be an xy pair")
    return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


def _json_copy(value: Any, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            if key.lower() in _FORBIDDEN_KEYS:
                raise PermissionError(f"forbidden evaluator field in geometry payload: {key}")
            result[key] = _json_copy(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_copy(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value, path)
    raise ValueError(f"{path} contains non-JSON type {type(value).__name__}")


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_copy(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ApplicabilityDecision:
    region_id: str
    state: ApplicabilityState
    capability_basis: tuple[str, ...] = ()
    task_basis: tuple[str, ...] = ()
    confidence: float = 0.0
    reason_code: str = "UNKNOWN"
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        object.__setattr__(self, "state", ApplicabilityState(self.state))
        confidence = _finite(self.confidence, "applicability confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("applicability confidence must be in [0,1]")
        object.__setattr__(self, "confidence", confidence)
        if not self.reason_code:
            raise ValueError("reason_code must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "state": self.state.value,
            "capability_basis": list(self.capability_basis),
            "task_basis": list(self.task_basis),
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "explanation": self.explanation,
        }


def _validate_geometry(kind: GeometryType, geometry: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(geometry, "geometry")
    if kind is GeometryType.EMPTY:
        if value:
            raise ValueError("empty geometry must be {}")
    elif kind is GeometryType.DISK:
        if set(value) != {"center_xy", "radius"}:
            raise ValueError("disk requires center_xy and radius")
        value["center_xy"] = list(_point(value["center_xy"], "disk.center_xy"))
        value["radius"] = _finite(value["radius"], "disk.radius")
        if value["radius"] <= 0:
            raise ValueError("disk radius must be positive")
    elif kind is GeometryType.AABB:
        if set(value) != {"min_xy", "max_xy"}:
            raise ValueError("aabb requires min_xy and max_xy")
        low = _point(value["min_xy"], "aabb.min_xy")
        high = _point(value["max_xy"], "aabb.max_xy")
        if low[0] >= high[0] or low[1] >= high[1]:
            raise ValueError("aabb min_xy must be strictly below max_xy")
        value = {"min_xy": list(low), "max_xy": list(high)}
    elif kind is GeometryType.POLYGON:
        if set(value) != {"points"} or len(value["points"]) < 3:
            raise ValueError("polygon requires at least three points")
        points = [_point(item, "polygon point") for item in value["points"]]
        if len(set(points)) < 3:
            raise ValueError("polygon requires three distinct points")
        value["points"] = [list(item) for item in points]
    elif kind is GeometryType.MASK_REFERENCE:
        required = {"mask_sha256", "width", "height", "artifact_path"}
        if set(value) != required:
            raise ValueError(f"mask_reference requires exactly {sorted(required)}")
        if not _SHA256.fullmatch(str(value["mask_sha256"])):
            raise ValueError("mask_sha256 must be lowercase SHA-256")
        for name in ("width", "height"):
            if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 1:
                raise ValueError(f"mask {name} must be a positive integer")
        if not value["artifact_path"]:
            raise ValueError("mask artifact_path must be non-empty")
    return value


@dataclass(frozen=True)
class SemanticRegion:
    region_id: str
    semantic_class: str
    geometry_type: GeometryType
    geometry: Mapping[str, Any]
    coordinate_frame: CoordinateFrame
    confidence: float
    source: GeometrySource
    source_model: str | None = None
    source_artifact_hash: str | None = None
    applicability: ApplicabilityDecision | None = None
    applicability_confidence: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    adapter_version: str = ADAPTER_VERSION

    def __post_init__(self) -> None:
        if not self.region_id or not self.semantic_class:
            raise ValueError("region_id and semantic_class must be non-empty")
        kind = GeometryType(self.geometry_type)
        frame = CoordinateFrame(self.coordinate_frame)
        source = GeometrySource(self.source)
        object.__setattr__(self, "geometry_type", kind)
        object.__setattr__(self, "coordinate_frame", frame)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "geometry", _validate_geometry(kind, self.geometry))
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        object.__setattr__(self, "confidence", confidence)
        if self.source_artifact_hash is not None and not _SHA256.fullmatch(self.source_artifact_hash):
            raise ValueError("source_artifact_hash must be lowercase SHA-256")
        if self.applicability is not None and self.applicability.region_id != self.region_id:
            raise ValueError("applicability region_id mismatch")
        if self.applicability_confidence is not None:
            app_conf = _finite(self.applicability_confidence, "applicability_confidence")
            if not 0.0 <= app_conf <= 1.0:
                raise ValueError("applicability_confidence must be in [0,1]")
            object.__setattr__(self, "applicability_confidence", app_conf)
        object.__setattr__(self, "provenance", _json_copy(self.provenance, "provenance"))
        if not self.adapter_version:
            raise ValueError("adapter_version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "semantic_class": self.semantic_class,
            "geometry_type": self.geometry_type.value,
            "geometry": dict(self.geometry),
            "coordinate_frame": self.coordinate_frame.value,
            "confidence": self.confidence,
            "source": self.source.value,
            "source_model": self.source_model,
            "source_artifact_hash": self.source_artifact_hash,
            "applicability": None if self.applicability is None else self.applicability.to_dict(),
            "applicability_confidence": self.applicability_confidence,
            "provenance": dict(self.provenance),
            "adapter_version": self.adapter_version,
        }


@dataclass(frozen=True)
class SemanticGeometryPayload:
    request_id: str
    scene_id: str
    image_sha256: str | None
    coordinate_frame_metadata: Mapping[str, Any]
    image_to_world_transform_version: str
    regions: tuple[SemanticRegion, ...]
    parser_status: ProcessingStatus = ProcessingStatus.OK
    adapter_status: ProcessingStatus = ProcessingStatus.OK
    fallback_status: ProcessingStatus = ProcessingStatus.NOT_RUN
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id or not self.scene_id:
            raise ValueError("request_id and scene_id must be non-empty")
        if self.image_sha256 is not None and not _SHA256.fullmatch(self.image_sha256):
            raise ValueError("image_sha256 must be lowercase SHA-256")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported geometry schema {self.schema_version}")
        object.__setattr__(self, "parser_status", ProcessingStatus(self.parser_status))
        object.__setattr__(self, "adapter_status", ProcessingStatus(self.adapter_status))
        object.__setattr__(self, "fallback_status", ProcessingStatus(self.fallback_status))
        ordered = tuple(sorted(self.regions, key=lambda item: item.region_id))
        if len({item.region_id for item in ordered}) != len(ordered):
            raise ValueError("region IDs must be unique")
        object.__setattr__(self, "regions", ordered)
        object.__setattr__(
            self,
            "coordinate_frame_metadata",
            _json_copy(self.coordinate_frame_metadata, "coordinate_frame_metadata"),
        )
        object.__setattr__(self, "provenance", _json_copy(self.provenance, "provenance"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "scene_id": self.scene_id,
            "image_sha256": self.image_sha256,
            "coordinate_frame_metadata": dict(self.coordinate_frame_metadata),
            "image_to_world_transform_version": self.image_to_world_transform_version,
            "regions": [item.to_dict() for item in self.regions],
            "parser_status": self.parser_status.value,
            "adapter_status": self.adapter_status.value,
            "fallback_status": self.fallback_status.value,
            "provenance": dict(self.provenance),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class ImageCoordinateTransform:
    """Versioned pixel/world transform with optional resize letterboxing."""

    image_width: int
    image_height: int
    world_x_min: float
    world_x_max: float
    world_y_min: float
    world_y_max: float
    provider_max: float = 1000.0
    content_left: float = 0.0
    content_top: float = 0.0
    content_width: float | None = None
    content_height: float | None = None
    version: str = "image-world-transform-v1"

    def __post_init__(self) -> None:
        if self.image_width < 2 or self.image_height < 2:
            raise ValueError("image dimensions must be at least 2")
        for name in ("world_x_min", "world_x_max", "world_y_min", "world_y_max", "provider_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.world_x_min >= self.world_x_max or self.world_y_min >= self.world_y_max:
            raise ValueError("world bounds must be increasing")
        if self.provider_max <= 0:
            raise ValueError("provider_max must be positive")
        width = float(self.image_width) if self.content_width is None else _finite(self.content_width, "content_width")
        height = float(self.image_height) if self.content_height is None else _finite(self.content_height, "content_height")
        left = _finite(self.content_left, "content_left")
        top = _finite(self.content_top, "content_top")
        if width <= 0 or height <= 0 or left < 0 or top < 0:
            raise ValueError("invalid letterbox content rectangle")
        if left + width > self.image_width or top + height > self.image_height:
            raise ValueError("letterbox content rectangle exceeds image")
        object.__setattr__(self, "content_width", width)
        object.__setattr__(self, "content_height", height)

    def _validate_pixel(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = _point(point, "pixel")
        if not (0.0 <= x <= self.image_width - 1 and 0.0 <= y <= self.image_height - 1):
            raise ValueError("pixel coordinate is out of image bounds")
        return x, y

    def normalized_to_pixel(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = _point(point, "normalized coordinate")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("normalized coordinate must be in [0,1]")
        return x * (self.image_width - 1), y * (self.image_height - 1)

    def provider_to_pixel(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = _point(point, "provider coordinate")
        if not (0.0 <= x <= self.provider_max and 0.0 <= y <= self.provider_max):
            raise ValueError("provider coordinate is out of bounds")
        return self.normalized_to_pixel((x / self.provider_max, y / self.provider_max))

    def pixel_to_world(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = self._validate_pixel(point)
        if not (
            self.content_left <= x <= self.content_left + self.content_width - 1
            and self.content_top <= y <= self.content_top + self.content_height - 1
        ):
            raise ValueError("pixel lies in letterbox padding")
        nx = (x - self.content_left) / (self.content_width - 1)
        ny = (y - self.content_top) / (self.content_height - 1)
        return (
            self.world_x_min + nx * (self.world_x_max - self.world_x_min),
            self.world_y_max - ny * (self.world_y_max - self.world_y_min),
        )

    def world_to_pixel(self, point: Sequence[float]) -> tuple[float, float]:
        x, y = _point(point, "world coordinate")
        if not (
            self.world_x_min <= x <= self.world_x_max
            and self.world_y_min <= y <= self.world_y_max
        ):
            raise ValueError("world coordinate is out of bounds")
        nx = (x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        ny = (self.world_y_max - y) / (self.world_y_max - self.world_y_min)
        return (
            self.content_left + nx * (self.content_width - 1),
            self.content_top + ny * (self.content_height - 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "image_size": [self.image_width, self.image_height],
            "world_bounds": [
                self.world_x_min,
                self.world_x_max,
                self.world_y_min,
                self.world_y_max,
            ],
            "provider_max": self.provider_max,
            "content_rect": [
                self.content_left,
                self.content_top,
                self.content_width,
                self.content_height,
            ],
        }


def adapt_regions(
    *,
    source: GeometrySource,
    records: Sequence[Mapping[str, Any]],
    request_id: str,
    scene_id: str,
    image_sha256: str | None,
    coordinate_frame: CoordinateFrame,
    transform: ImageCoordinateTransform,
    source_model: str | None = None,
    source_artifact_hash: str | None = None,
    allow_oracle: bool = False,
) -> SemanticGeometryPayload:
    """Adapt detached records to one contract without silently clipping."""
    source = GeometrySource(source)
    if source is GeometrySource.ORACLE and not allow_oracle:
        raise PermissionError("oracle geometry requires an explicit oracle arm")
    if source is GeometrySource.NONE and records:
        raise ValueError("none adapter cannot receive regions")
    regions = []
    for index, record in enumerate(records):
        record = _json_copy(record, f"record[{index}]")
        kind = GeometryType(record.get("geometry_type", "disk"))
        applicability_value = record.get("applicability")
        applicability = (
            None
            if applicability_value is None
            else ApplicabilityDecision(
                region_id=str(record.get("region_id", f"region_{index}")),
                state=applicability_value["state"],
                capability_basis=tuple(applicability_value.get("capability_basis", ())),
                task_basis=tuple(applicability_value.get("task_basis", ())),
                confidence=float(applicability_value.get("confidence", 0.0)),
                reason_code=str(applicability_value.get("reason_code", "UNKNOWN")),
                explanation=applicability_value.get("explanation"),
            )
        )
        regions.append(
            SemanticRegion(
                region_id=str(record.get("region_id", f"region_{index}")),
                semantic_class=str(record.get("semantic_class", record.get("class", "unknown"))),
                geometry_type=kind,
                geometry=record.get("geometry", {}),
                coordinate_frame=coordinate_frame,
                confidence=float(record.get("confidence", 1.0 if source in {GeometrySource.ORACLE, GeometrySource.FIXTURE} else 0.0)),
                source=source,
                source_model=source_model,
                source_artifact_hash=source_artifact_hash,
                applicability=applicability,
                applicability_confidence=(
                    None if applicability is None else applicability.confidence
                ),
                provenance={
                    "information_type": "PRIVILEGED" if source is GeometrySource.ORACLE else "DERIVED_PUBLIC",
                    "adapter_version": ADAPTER_VERSION,
                },
            )
        )
    status = ProcessingStatus.EMPTY if not regions else ProcessingStatus.OK
    return SemanticGeometryPayload(
        request_id=request_id,
        scene_id=scene_id,
        image_sha256=image_sha256,
        coordinate_frame_metadata=transform.to_dict(),
        image_to_world_transform_version=transform.version,
        regions=tuple(regions),
        parser_status=status,
        adapter_status=ProcessingStatus.OK,
        provenance={"source": source.value, "source_model": source_model},
    )


def detector_box_records(
    detections: Sequence[Mapping[str, Any]],
    *,
    transform: ImageCoordinateTransform,
    input_frame: CoordinateFrame,
) -> tuple[dict[str, Any], ...]:
    """Convert provider/normalized/pixel detector boxes to pixel AABBs."""
    result = []
    for index, item in enumerate(detections):
        low = item.get("min_xy")
        high = item.get("max_xy")
        if input_frame is CoordinateFrame.PROVIDER_NATIVE:
            low_px, high_px = transform.provider_to_pixel(low), transform.provider_to_pixel(high)
        elif input_frame is CoordinateFrame.NORMALIZED_IMAGE:
            low_px, high_px = transform.normalized_to_pixel(low), transform.normalized_to_pixel(high)
        elif input_frame is CoordinateFrame.IMAGE_PIXEL:
            low_px, high_px = transform._validate_pixel(low), transform._validate_pixel(high)
        else:
            raise ValueError("detector boxes must originate in an image frame")
        result.append(
            {
                "region_id": str(item.get("region_id", f"detector_{index}")),
                "semantic_class": str(item.get("semantic_class", item.get("class", "unknown"))),
                "geometry_type": "aabb",
                "geometry": {"min_xy": list(low_px), "max_xy": list(high_px)},
                "confidence": float(item.get("confidence", 0.0)),
            }
        )
    return tuple(result)
