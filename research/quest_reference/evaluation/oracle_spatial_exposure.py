"""EXP-01A: provider-free oracle spatial exposure experiment.

Only simulator-oracle binary requirement fields, rasterized trajectory geometry,
and binary capability/rule cards enter this module.  It deliberately has no RGB,
segmentation, VLM, policy, environment, or RL dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "exp01a-oracle-spatial-exposure-v1"
TRAJECTORY_IDS = ("water_cross", "fragile_cross", "safe_bypass", "neutral")
CARD_IDS = ("A", "B", "C", "D")
CARDS = {
    "A": (0, 0),
    "B": (1, 0),
    "C": (0, 1),
    "D": (1, 1),
}
MODEL_ARMS = (
    "global_presence",
    "endpoint",
    "centerline",
    "dense_cost",
    "factorized_no_cf",
    "factorized_full",
    "oracle_formula",
)
LEARNED_ARMS = ("dense_cost", "factorized_no_cf", "factorized_full")
NONSTRUCTURED_ARMS = ("global_presence", "endpoint", "centerline", "dense_cost")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def semantic_cost(
    water_exposure: Any,
    fragile_exposure: Any,
    waterproof: Any,
    protect_fragile: Any,
) -> Any:
    """Frozen EXP-01A semantic-cost equation."""
    ew = np.asarray(water_exposure, dtype=np.float64)
    ef = np.asarray(fragile_exposure, dtype=np.float64)
    kappa = np.asarray(waterproof, dtype=np.float64)
    rule = np.asarray(protect_fragile, dtype=np.float64)
    if np.any((ew < 0) | (ew > 1)) or np.any((ef < 0) | (ef > 1)):
        raise ValueError("exposures must lie in [0,1]")
    if np.any((kappa != 0) & (kappa != 1)) or np.any((rule != 0) & (rule != 1)):
        raise ValueError("cards must be binary")
    water = ew * (1.0 - kappa)
    fragile = ef * rule
    result = 1.0 - (1.0 - water) * (1.0 - fragile)
    return float(result) if result.ndim == 0 else result


def grid_xy(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if height < 8 or width < 8:
        raise ValueError("raster must be at least 8x8")
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    return np.meshgrid(xs, ys)


def ellipse_mask(
    height: int,
    width: int,
    center_xy: Sequence[float],
    radii_xy: Sequence[float],
) -> np.ndarray:
    xx, yy = grid_xy(height, width)
    cx, cy = map(float, center_xy)
    rx, ry = map(float, radii_xy)
    if rx <= 0 or ry <= 0:
        raise ValueError("ellipse radii must be positive")
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.uint8)


def _distance_to_polyline(
    xx: np.ndarray, yy: np.ndarray, points: Sequence[Sequence[float]]
) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("trajectory requires at least two points")
    best = np.full(xx.shape, np.inf, dtype=np.float64)
    for left, right in zip(points, points[1:]):
        x0, y0 = map(float, left)
        x1, y1 = map(float, right)
        dx, dy = x1 - x0, y1 - y0
        denom = dx * dx + dy * dy
        if denom <= 0:
            raise ValueError("trajectory cannot contain zero-length segments")
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / denom, 0.0, 1.0)
        distance = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
        best = np.minimum(best, distance)
    return best


def rasterize_polyline(
    height: int,
    width: int,
    points: Sequence[Sequence[float]],
    radius: float,
) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius cannot be negative")
    xx, yy = grid_xy(height, width)
    pixel_radius = max(1.0 / max(height - 1, width - 1), float(radius))
    return (_distance_to_polyline(xx, yy, points) <= pixel_radius).astype(np.uint8)


def exposure(mask: np.ndarray, requirement: np.ndarray) -> float:
    if mask.shape != requirement.shape or mask.ndim != 2:
        raise ValueError("mask and requirement must be same-sized 2-D arrays")
    denominator = int(np.count_nonzero(mask))
    if denominator == 0:
        raise ValueError("swept footprint cannot be empty")
    return float(np.asarray(requirement, dtype=np.float64)[mask.astype(bool)].sum() / denominator)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    points: tuple[tuple[float, float], ...]
    footprint: np.ndarray
    centerline: np.ndarray
    endpoint: np.ndarray
    exposure: tuple[float, float]
    centerline_exposure: tuple[float, float]
    endpoint_exposure: tuple[float, float]

    def __post_init__(self) -> None:
        if self.trajectory_id not in TRAJECTORY_IDS:
            raise ValueError("unknown trajectory")
        for array in (self.footprint, self.centerline, self.endpoint):
            if array.ndim != 2 or not np.isin(array, (0, 1)).all():
                raise ValueError("trajectory rasters must be binary 2-D arrays")


@dataclass(frozen=True)
class OracleSpatialScene:
    scene_id: str
    split: str
    fields: np.ndarray
    trajectories: tuple[TrajectoryRecord, ...]
    robot_radius: float
    geometry: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "locked_test"}:
            raise ValueError("unknown split")
        if self.fields.ndim != 3 or self.fields.shape[0] != 2 or not np.isin(self.fields, (0, 1)).all():
            raise ValueError("fields must be a 2xHxW binary tensor")
        if tuple(item.trajectory_id for item in self.trajectories) != TRAJECTORY_IDS:
            raise ValueError("scene must contain four trajectories in frozen order")
        patterns = {item.trajectory_id: item.exposure for item in self.trajectories}
        eps = 1e-12
        if not (
            patterns["water_cross"][0] > 0 and patterns["water_cross"][1] <= eps
            and patterns["fragile_cross"][0] <= eps and patterns["fragile_cross"][1] > 0
            and max(patterns["safe_bypass"]) <= eps
            and max(patterns["neutral"]) <= eps
        ):
            raise ValueError(f"invalid trajectory exposure pattern: {patterns}")

    @property
    def geometry_hash(self) -> str:
        return stable_hash(dict(self.geometry))


def _route_masks(
    height: int,
    width: int,
    points: tuple[tuple[float, float], ...],
    radius: float,
    fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float], tuple[float, float]]:
    footprint = rasterize_polyline(height, width, points, radius)
    centerline = rasterize_polyline(height, width, points, 0.0)
    endpoint = np.zeros((height, width), dtype=np.uint8)
    x, y = points[-1]
    col = int(round((x + 1.0) * 0.5 * (width - 1)))
    row = int(round((y + 1.0) * 0.5 * (height - 1)))
    endpoint[np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)] = 1
    return (
        footprint,
        centerline,
        endpoint,
        tuple(exposure(footprint, field) for field in fields),
        tuple(exposure(centerline, field) for field in fields),
        tuple(exposure(endpoint, field) for field in fields),
    )


def generate_scene(seed: int, split: str, scene_index: int, *, size: int = 48) -> OracleSpatialScene:
    """Generate one deterministic scene; locked test changes shape and footprint jointly."""
    split_code = {"train": 11, "validation": 23, "locked_test": 47}
    if split not in split_code:
        raise ValueError("unknown split")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), split_code[split], int(scene_index)]))
    if split == "locked_test":
        robot_radius = float(rng.uniform(0.080, 0.105))
        water_radii = (float(rng.uniform(0.105, 0.135)), float(rng.uniform(0.065, 0.090)))
        fragile_radii = (float(rng.uniform(0.065, 0.090)), float(rng.uniform(0.105, 0.135)))
        grazing_probability = 0.80
    else:
        robot_radius = float(rng.uniform(0.045, 0.065))
        radius = float(rng.uniform(0.105, 0.145))
        water_radii = (radius, radius)
        radius = float(rng.uniform(0.105, 0.145))
        fragile_radii = (radius, radius)
        grazing_probability = 0.35
    # Randomly assign semantic and non-semantic routes to four corridors.  This
    # removes fixed position and path-length shortcuts while preserving common
    # start/goal and the four requested trajectory meanings.
    corridor_values = np.asarray((-0.69, -0.23, 0.23, 0.69), dtype=np.float64)
    permutation = rng.permutation(4)
    assigned_lane = {
        trajectory_id: float(corridor_values[int(permutation[index])] + rng.uniform(-0.012, 0.012))
        for index, trajectory_id in enumerate(TRAJECTORY_IDS)
    }
    water_center = (
        -0.24 + float(rng.uniform(-0.035, 0.035)),
        assigned_lane["water_cross"] + float(rng.uniform(-0.018, 0.018)),
    )
    fragile_center = (
        0.24 + float(rng.uniform(-0.035, 0.035)),
        assigned_lane["fragile_cross"] + float(rng.uniform(-0.018, 0.018)),
    )
    fields = np.stack((
        ellipse_mask(size, size, water_center, water_radii),
        ellipse_mask(size, size, fragile_center, fragile_radii),
    ))

    def crossing_lane(center_y: float, vertical_radius: float, sign: float) -> tuple[float, str]:
        if rng.random() < grazing_probability:
            # Leave the centerline outside the analytic ellipse while giving
            # the discrete swept footprint enough overlap at 48x48 resolution.
            return center_y + sign * (vertical_radius + 0.10 * robot_radius), "footprint_only"
        return center_y + float(rng.uniform(-0.25, 0.25)) * vertical_radius, "interior"

    water_lane, water_mode = crossing_lane(water_center[1], water_radii[1], 1.0)
    fragile_lane, fragile_mode = crossing_lane(fragile_center[1], fragile_radii[1], -1.0)
    start, goal = (-0.92, 0.0), (0.92, 0.0)
    route_points = {
        "water_cross": (start, (-0.84, water_lane), (0.60, water_lane), (0.84, 0.0), goal),
        "fragile_cross": (start, (-0.84, fragile_lane), (0.60, fragile_lane), (0.84, 0.0), goal),
        "safe_bypass": (start, (-0.84, assigned_lane["safe_bypass"]), (0.60, assigned_lane["safe_bypass"]), (0.84, 0.0), goal),
        "neutral": (start, (-0.84, assigned_lane["neutral"]), (0.60, assigned_lane["neutral"]), (0.84, 0.0), goal),
    }
    trajectories = []
    for trajectory_id in TRAJECTORY_IDS:
        points = route_points[trajectory_id]
        masks = _route_masks(size, size, points, robot_radius, fields)
        target_index = 0 if trajectory_id == "water_cross" else (1 if trajectory_id == "fragile_cross" else None)
        if target_index is not None and masks[3][target_index] <= 0:
            # The continuous shapes overlap by construction, but a very small
            # ellipse can vanish at the exact grazing side on a coarse raster.
            # Fall back deterministically to an interior crossing and record it.
            lane = water_center[1] if target_index == 0 else fragile_center[1]
            points = (start, (-0.84, lane), (0.60, lane), (0.84, 0.0), goal)
            route_points[trajectory_id] = points
            masks = _route_masks(size, size, points, robot_radius, fields)
            if target_index == 0:
                water_mode = "raster_fallback_interior"
            else:
                fragile_mode = "raster_fallback_interior"
        trajectories.append(TrajectoryRecord(trajectory_id, points, *masks))
    geometry = {
        "size": size,
        "water_center": water_center,
        "water_radii": water_radii,
        "fragile_center": fragile_center,
        "fragile_radii": fragile_radii,
        "robot_radius": robot_radius,
        "assigned_lanes": assigned_lane,
        "water_cross_mode": water_mode,
        "fragile_cross_mode": fragile_mode,
        "points": {key: value for key, value in route_points.items()},
    }
    return OracleSpatialScene(
        scene_id=f"exp01a-{split}-{scene_index:05d}",
        split=split,
        fields=fields,
        trajectories=tuple(trajectories),
        robot_radius=robot_radius,
        geometry=geometry,
    )


@dataclass(frozen=True)
class SpatialDataset:
    splits: Mapping[str, tuple[OracleSpatialScene, ...]]

    def __post_init__(self) -> None:
        if set(self.splits) != {"train", "validation", "locked_test"}:
            raise ValueError("dataset requires three splits")
        ids = [scene.scene_id for scenes in self.splits.values() for scene in scenes]
        hashes = [scene.geometry_hash for scenes in self.splits.values() for scene in scenes]
        if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
            raise ValueError("scenes and geometries must be disjoint across splits")

    def audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "splits": {
                split: {
                    "scene_count": len(scenes),
                    "row_count": len(scenes) * 16,
                    "scene_ids": [scene.scene_id for scene in scenes],
                    "geometry_hashes": [scene.geometry_hash for scene in scenes],
                    "field_hashes": [hashlib.sha256(scene.fields.tobytes()).hexdigest() for scene in scenes],
                    "footprint_hashes": [
                        hashlib.sha256(b"".join(item.footprint.tobytes() for item in scene.trajectories)).hexdigest()
                        for scene in scenes
                    ],
                }
                for split, scenes in self.splits.items()
            },
        }


def generate_dataset(seed: int, counts: Mapping[str, int], *, size: int = 48) -> SpatialDataset:
    required = {"train", "validation", "locked_test"}
    if set(counts) != required or any(int(value) < 1 for value in counts.values()):
        raise ValueError("counts require positive train, validation, and locked_test sizes")
    return SpatialDataset({
        split: tuple(generate_scene(seed, split, index, size=size) for index in range(int(counts[split])))
        for split in ("train", "validation", "locked_test")
    })


def rows_for_scenes(scenes: Sequence[OracleSpatialScene]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        for trajectory_index, trajectory in enumerate(scene.trajectories):
            for card_id in CARD_IDS:
                waterproof, protect_fragile = CARDS[card_id]
                rows.append({
                    "scene_index": scene_index,
                    "scene_id": scene.scene_id,
                    "trajectory_index": trajectory_index,
                    "trajectory_id": trajectory.trajectory_id,
                    "card_id": card_id,
                    "waterproof": waterproof,
                    "protect_fragile": protect_fragile,
                    "water_exposure": trajectory.exposure[0],
                    "fragile_exposure": trajectory.exposure[1],
                    "semantic_cost": semantic_cost(*trajectory.exposure, waterproof, protect_fragile),
                })
    return rows


class DenseCost(nn.Module):
    """Unstructured field+footprint+cards CNN/MLP baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
        nn.init.constant_(self.fc2.bias, -4.0)

    def forward(self, spatial: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
        value = F.avg_pool2d(spatial, kernel_size=2)
        value = F.silu(self.conv1(value))
        value = F.silu(self.conv2(value)).mean(dim=(-2, -1))
        value = F.silu(self.fc1(torch.cat((value, cards), dim=1)))
        return torch.sigmoid(self.fc2(value)).squeeze(-1)


class _RelationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(value)).squeeze(-1)


class FactorizedCost(nn.Module):
    """Separate water-capability and fragile-rule relation heads."""

    def __init__(self) -> None:
        super().__init__()
        self.water_head = _RelationHead()
        self.fragile_head = _RelationHead()

    def forward_components(self, exposures: torch.Tensor, cards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Exposure is an explicit physical gate.  The learned relation heads
        # estimate incompatibility/rule strength, never manufacture risk where
        # the swept footprint has zero overlap.
        water = exposures[:, 0] * self.water_head(torch.stack((exposures[:, 0], cards[:, 0]), dim=1))
        fragile = exposures[:, 1] * self.fragile_head(torch.stack((exposures[:, 1], cards[:, 1]), dim=1))
        return water, fragile

    def forward(self, exposures: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
        water, fragile = self.forward_components(exposures, cards)
        return 1.0 - (1.0 - water) * (1.0 - fragile)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_parameter_budget(maximum_gap: float = 0.10) -> dict[str, int]:
    counts = {"dense_cost": parameter_count(DenseCost()), "factorized": parameter_count(FactorizedCost())}
    gap = abs(counts["dense_cost"] - counts["factorized"]) / max(counts.values())
    if gap > maximum_gap:
        raise ValueError(f"parameter budget violated: {counts}, gap={gap:.3f}")
    return counts


def build_model(arm: str, seed: int, initial_state: Mapping[str, torch.Tensor] | None = None) -> nn.Module:
    if arm not in LEARNED_ARMS:
        raise ValueError("arm is not learned")
    torch.manual_seed(int(seed))
    model: nn.Module = DenseCost() if arm == "dense_cost" else FactorizedCost()
    if initial_state is not None:
        model.load_state_dict(copy.deepcopy(dict(initial_state)))
    return model.cpu()


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.002
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 12
    invariance_weight: float = 0.2
    ranking_weight: float = 0.2
    invariance_epsilon: float = 0.005
    ranking_margin: float = 0.02


@dataclass(frozen=True)
class _RowCache:
    rows: tuple[Mapping[str, Any], ...]
    spatial_unique: np.ndarray
    spatial_indices: np.ndarray
    exposures: np.ndarray
    cards: np.ndarray
    targets: np.ndarray


def _make_cache(scenes: Sequence[OracleSpatialScene]) -> _RowCache:
    rows = tuple(rows_for_scenes(scenes))
    spatial_unique = np.asarray([
        np.concatenate((scene.fields, trajectory.footprint[None]), axis=0)
        for scene in scenes
        for trajectory in scene.trajectories
    ], dtype=np.uint8)
    return _RowCache(
        rows=rows,
        spatial_unique=spatial_unique,
        spatial_indices=np.asarray([
            int(row["scene_index"]) * len(TRAJECTORY_IDS) + int(row["trajectory_index"])
            for row in rows
        ], dtype=np.int64),
        exposures=np.asarray([
            (row["water_exposure"], row["fragile_exposure"]) for row in rows
        ], dtype=np.float32),
        cards=np.asarray([
            (row["waterproof"], row["protect_fragile"]) for row in rows
        ], dtype=np.float32),
        targets=np.asarray([row["semantic_cost"] for row in rows], dtype=np.float32),
    )


def _batch_tensors(
    cache: _RowCache, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = np.asarray(indices, dtype=np.int64)
    return (
        torch.tensor(cache.spatial_unique[cache.spatial_indices[selected]], dtype=torch.float32),
        torch.from_numpy(cache.exposures[selected]),
        torch.from_numpy(cache.cards[selected]),
        torch.from_numpy(cache.targets[selected]),
    )


def _forward(model: nn.Module, spatial: torch.Tensor, exposures: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
    return model(spatial, cards) if isinstance(model, DenseCost) else model(exposures, cards)


def _cf_loss(
    predictions: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
    config: TrainingConfig,
) -> torch.Tensor:
    selected = set(map(int, indices))
    lookup = {
        (row["scene_id"], row["trajectory_id"], row["card_id"]): position
        for position, row in enumerate(rows)
        if position in selected
    }
    # Mini-batch-local pairs are diagnostic regularizers; base supervision remains complete.
    invariant, ranking = [], []
    local = {int(global_index): local_index for local_index, global_index in enumerate(indices)}
    for global_index in indices:
        row = rows[int(global_index)]
        card = row["card_id"]
        kappa, rule = CARDS[card]
        swaps = []
        if row["trajectory_id"] in {"water_cross", "neutral"}:
            swaps.append(next(key for key, value in CARDS.items() if value == (kappa, 1 - rule)))
        if row["trajectory_id"] in {"fragile_cross", "neutral"}:
            swaps.append(next(key for key, value in CARDS.items() if value == (1 - kappa, rule)))
        for other_card in swaps:
            other = lookup.get((row["scene_id"], row["trajectory_id"], other_card))
            if other is not None and other > int(global_index):
                invariant.append(torch.abs(predictions[local[int(global_index)]] - predictions[local[other]]))
        if row["trajectory_id"] == "water_cross" and kappa == 0:
            other_card = next(key for key, value in CARDS.items() if value == (1, rule))
            compatible = lookup.get((row["scene_id"], "water_cross", other_card))
            if compatible is not None:
                ranking.append(F.relu(config.ranking_margin - predictions[local[int(global_index)]] + predictions[local[compatible]]))
            bypass = lookup.get((row["scene_id"], "safe_bypass", card))
            if bypass is not None:
                ranking.append(F.relu(config.ranking_margin - predictions[local[int(global_index)]] + predictions[local[bypass]]))
        if row["trajectory_id"] == "fragile_cross" and rule == 1:
            other_card = next(key for key, value in CARDS.items() if value == (kappa, 0))
            unprotected = lookup.get((row["scene_id"], "fragile_cross", other_card))
            if unprotected is not None:
                ranking.append(F.relu(config.ranking_margin - predictions[local[int(global_index)]] + predictions[local[unprotected]]))
            bypass = lookup.get((row["scene_id"], "safe_bypass", card))
            if bypass is not None:
                ranking.append(F.relu(config.ranking_margin - predictions[local[int(global_index)]] + predictions[local[bypass]]))
    zero = predictions.sum() * 0.0
    inv = torch.stack([F.relu(value - config.invariance_epsilon) for value in invariant]).mean() if invariant else zero
    rank = torch.stack(ranking).mean() if ranking else zero
    return config.invariance_weight * inv + config.ranking_weight * rank


def predict_learned(
    model: nn.Module,
    scenes: Sequence[OracleSpatialScene],
    rows: Sequence[Mapping[str, Any]],
    batch_size: int = 512,
    cache: _RowCache | None = None,
) -> np.ndarray:
    cache = cache or _make_cache(scenes)
    if len(cache.rows) != len(rows):
        raise ValueError("cache/row mismatch")
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            indices = np.arange(start, min(start + batch_size, len(rows)))
            spatial, exposures, cards, _targets = _batch_tensors(cache, indices)
            result.extend(_forward(model, spatial, exposures, cards).cpu().numpy().tolist())
    return np.asarray(result, dtype=np.float64)


def train_model(
    arm: str,
    train_scenes: Sequence[OracleSpatialScene],
    validation_scenes: Sequence[OracleSpatialScene],
    config: TrainingConfig,
    seed: int,
    initial_state: Mapping[str, torch.Tensor] | None = None,
    train_cache: _RowCache | None = None,
    validation_cache: _RowCache | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if config.batch_size < 16 or config.batch_size % 16 != 0:
        raise ValueError("batch_size must be a positive multiple of 16 to preserve scene factorials")
    model = build_model(arm, seed, initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_cache = train_cache or _make_cache(train_scenes)
    validation_cache = validation_cache or _make_cache(validation_scenes)
    train_rows, validation_rows = train_cache.rows, validation_cache.rows
    rng = np.random.default_rng(seed)
    best_state, best_mae, stale = copy.deepcopy(model.state_dict()), math.inf, 0
    history = []
    for epoch in range(config.max_epochs):
        # Keep each 16-row scene factorial together so full-CF pairs are
        # actually present in a mini-batch.  Scene order is still randomized.
        scene_order = rng.permutation(len(train_scenes))
        order = np.concatenate([
            np.arange(scene_index * 16, (scene_index + 1) * 16) for scene_index in scene_order
        ])
        model.train()
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            spatial, exposures, cards, targets = _batch_tensors(train_cache, indices)
            predictions = _forward(model, spatial, exposures, cards)
            loss = F.mse_loss(predictions, targets)
            if arm == "factorized_full":
                loss = loss + _cf_loss(predictions, train_rows, indices, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_prediction = predict_learned(
            model, validation_scenes, validation_rows, config.batch_size, validation_cache
        )
        validation_mae = float(np.mean(np.abs(validation_prediction - validation_cache.targets)))
        history.append(validation_mae)
        if validation_mae < best_mae - 1e-7:
            best_mae, best_state, stale = validation_mae, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, {"epochs": len(history), "best_validation_mae": best_mae, "validation_mae_history": history}


def deterministic_predictions(arm: str, scenes: Sequence[OracleSpatialScene], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if arm not in {"global_presence", "endpoint", "centerline", "oracle_formula"}:
        raise ValueError("not a deterministic arm")
    result = []
    for row in rows:
        scene = scenes[int(row["scene_index"])]
        trajectory = scene.trajectories[int(row["trajectory_index"])]
        if arm == "global_presence":
            estimate = tuple(float(field.any()) for field in scene.fields)
        elif arm == "endpoint":
            estimate = trajectory.endpoint_exposure
        elif arm == "centerline":
            estimate = trajectory.centerline_exposure
        else:
            estimate = trajectory.exposure
        result.append(semantic_cost(*estimate, row["waterproof"], row["protect_fragile"]))
    return np.asarray(result, dtype=np.float64)


def _tie_score(delta: float, tolerance: float = 1e-9) -> float:
    return 1.0 if delta > tolerance else (0.5 if abs(delta) <= tolerance else 0.0)


def scene_metrics(rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, *, false_safe_threshold: float = 0.02) -> dict[str, np.ndarray]:
    if len(rows) != len(predictions):
        raise ValueError("prediction length mismatch")
    by_scene: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_scene.setdefault(str(row["scene_id"]), []).append(index)
    output = {key: [] for key in ("dangerous_inversion", "cross_bypass_accuracy", "action_regret", "false_safe", "semantic_cost_mae", "irrelevant_swap_delta")}
    for scene_id in sorted(by_scene):
        indices = by_scene[scene_id]
        lookup = {(rows[i]["trajectory_id"], rows[i]["card_id"]): i for i in indices}
        inversion, ranking, regret, false_safe, invariant = [], [], [], [], []
        for card_id in CARD_IDS:
            card_indices = [lookup[(trajectory_id, card_id)] for trajectory_id in TRAJECTORY_IDS]
            true = np.asarray([rows[i]["semantic_cost"] for i in card_indices])
            pred = predictions[card_indices]
            dangerous = np.flatnonzero(true > 1e-12)
            safe = np.flatnonzero(true <= 1e-12)
            for high in dangerous:
                for low in safe:
                    inversion.append(1.0 - _tie_score(float(pred[high] - pred[low])))
            minimum = float(pred.min())
            selected = np.flatnonzero(np.isclose(pred, minimum, atol=1e-9, rtol=0.0))
            regret.append(float(true[selected].mean() - true.min()))
            threshold_dangerous = np.flatnonzero(true > false_safe_threshold)
            false_safe.extend((pred[threshold_dangerous] <= false_safe_threshold).astype(float).tolist())
        for card_id in CARD_IDS:
            kappa, rule = CARDS[card_id]
            if kappa == 0:
                high, low = lookup[("water_cross", card_id)], lookup[("safe_bypass", card_id)]
                ranking.append(_tie_score(float(predictions[high] - predictions[low])))
            if rule == 1:
                high, low = lookup[("fragile_cross", card_id)], lookup[("safe_bypass", card_id)]
                ranking.append(_tie_score(float(predictions[high] - predictions[low])))
        for trajectory_id, axis in (("water_cross", 1), ("fragile_cross", 0), ("neutral", 0), ("neutral", 1)):
            for card_id in CARD_IDS:
                card = CARDS[card_id]
                swapped = list(card)
                swapped[axis] = 1 - swapped[axis]
                other = next(key for key, value in CARDS.items() if value == tuple(swapped))
                if CARD_IDS.index(card_id) < CARD_IDS.index(other):
                    invariant.append(abs(float(predictions[lookup[(trajectory_id, card_id)]] - predictions[lookup[(trajectory_id, other)]])))
        output["dangerous_inversion"].append(float(np.mean(inversion)) if inversion else 0.0)
        output["cross_bypass_accuracy"].append(float(np.mean(ranking)))
        output["action_regret"].append(float(np.mean(regret)))
        output["false_safe"].append(float(np.mean(false_safe)) if false_safe else 0.0)
        output["semantic_cost_mae"].append(float(np.mean(np.abs(predictions[indices] - np.asarray([rows[i]["semantic_cost"] for i in indices])))))
        output["irrelevant_swap_delta"].append(float(np.mean(invariant)))
    return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}


def exposure_mae(arm: str, scenes: Sequence[OracleSpatialScene]) -> float | None:
    if arm in {"dense_cost"}:
        return None
    errors = []
    for scene in scenes:
        for trajectory in scene.trajectories:
            if arm == "global_presence":
                estimate = tuple(float(field.any()) for field in scene.fields)
            elif arm == "endpoint":
                estimate = trajectory.endpoint_exposure
            elif arm == "centerline":
                estimate = trajectory.centerline_exposure
            else:
                estimate = trajectory.exposure
            errors.extend(abs(left - right) for left, right in zip(estimate, trajectory.exposure))
    return float(np.mean(errors))


def percentile_bootstrap(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or resamples < 100:
        raise ValueError("bootstrap requires >=2 values and >=100 resamples")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[samples].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def summarize_metrics(metrics: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(value)) for key, value in metrics.items()}


def evaluate_experiment(
    dataset: SpatialDataset,
    training: TrainingConfig,
    *,
    model_seed: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    assert_parameter_budget(0.10)
    predictions: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in MODEL_ARMS}
    histories: dict[str, Any] = {}
    caches = {split: _make_cache(dataset.splits[split]) for split in dataset.splits}
    factor_base = build_model("factorized_no_cf", model_seed)
    factor_initial = copy.deepcopy(factor_base.state_dict())
    models = {}
    for arm in LEARNED_ARMS:
        initial = factor_initial if arm in {"factorized_no_cf", "factorized_full"} else None
        models[arm], histories[arm] = train_model(
            arm,
            dataset.splits["train"],
            dataset.splits["validation"],
            training,
            model_seed + (0 if arm != "dense_cost" else 1000),
            initial,
            caches["train"],
            caches["validation"],
        )
    split_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split in ("validation", "locked_test"):
        scenes = dataset.splits[split]
        rows = rows_for_scenes(scenes)
        split_metrics[split] = {}
        for arm in MODEL_ARMS:
            if arm in LEARNED_ARMS:
                prediction = predict_learned(models[arm], scenes, rows, training.batch_size, caches[split])
            else:
                prediction = deterministic_predictions(arm, scenes, rows)
            predictions[arm][split] = prediction
            split_metrics[split][arm] = scene_metrics(rows, prediction)
    validation_summary = {arm: summarize_metrics(split_metrics["validation"][arm]) for arm in MODEL_ARMS}
    strongest = min(
        NONSTRUCTURED_ARMS,
        key=lambda arm: (
            validation_summary[arm]["dangerous_inversion"],
            validation_summary[arm]["action_regret"],
            validation_summary[arm]["semantic_cost_mae"],
        ),
    )
    test = split_metrics["locked_test"]
    def compare(reference: str, candidate: str, seed_offset: int) -> dict[str, Any]:
        comparison = {}
        for metric_index, metric in enumerate(("dangerous_inversion", "action_regret")):
            differences = test[reference][metric] - test[candidate][metric]
            comparison[metric] = {
                "mean_reduction": float(differences.mean()),
                "paired_scene_bootstrap_ci95": list(percentile_bootstrap(
                    differences,
                    seed=bootstrap_seed + seed_offset + metric_index,
                    resamples=bootstrap_resamples,
                )),
            }
        return comparison

    comparisons = compare(strongest, "factorized_no_cf", 0)
    dense_comparisons = compare("dense_cost", "factorized_no_cf", 10)
    cf_comparisons = compare("factorized_no_cf", "factorized_full", 20)
    test_summary = {}
    for arm in MODEL_ARMS:
        test_summary[arm] = summarize_metrics(test[arm])
        test_summary[arm]["exposure_mae"] = exposure_mae(arm, dataset.splits["locked_test"])
        test_summary[arm]["parameter_count"] = (
            0 if arm not in LEARNED_ARMS else parameter_count(models[arm])
        )
    oracle_error = test_summary["oracle_formula"]["semantic_cost_mae"]
    valid = oracle_error <= 1e-12 and all(
        len(scenes) * 16 == len(rows_for_scenes(scenes)) for scenes in dataset.splits.values()
    )
    significant = all(item["paired_scene_bootstrap_ci95"][0] > 0 for item in comparisons.values())
    near_ceiling = test_summary["factorized_no_cf"]["cross_bypass_accuracy"] >= 0.98
    hypothesis = "SUPPORTED" if valid and significant and near_ceiling else "NOT_SUPPORTED"
    if hypothesis == "SUPPORTED":
        program = "EXP01B_AUTHORIZED"
    else:
        dense_tie = all(
            item["paired_scene_bootstrap_ci95"][0] <= 0 <= item["paired_scene_bootstrap_ci95"][1]
            for item in dense_comparisons.values()
        )
        program = "REASSESS_FACTORIZATION" if dense_tie else "EXP01A_STOP"
    cf_keep = (
        any(item["paired_scene_bootstrap_ci95"][0] > 0 for item in cf_comparisons.values())
        and not any(item["paired_scene_bootstrap_ci95"][1] < 0 for item in cf_comparisons.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "validity": "VALID" if valid else "INVALID_RUN",
        "hypothesis_outcome": hypothesis if valid else "NOT_INTERPRETABLE",
        "program_decision": program if valid else "REPAIR_AND_RERUN",
        "strongest_nonstructured_baseline_selected_on_validation": strongest,
        "parameter_counts": assert_parameter_budget(0.10),
        "validation": validation_summary,
        "locked_test": test_summary,
        "paired_comparisons": comparisons,
        "dense_vs_factorized_no_cf": dense_comparisons,
        "full_cf_diagnostic": {
            "comparisons": cf_comparisons,
            "recommendation": "KEEP_CF" if cf_keep else "DROP_CF",
            "affects_program_decision": False,
        },
        "conditions": {
            "oracle_formula_exact": oracle_error <= 1e-12,
            "factorized_ranking_near_ceiling": near_ceiling,
            "both_primary_ci_lower_bounds_positive": significant,
            "full_cf_policy": "KEEP_ONLY_IF_SIGNIFICANTLY_HELPFUL_AND_NOT_HARMFUL",
        },
        "training": histories,
    }
