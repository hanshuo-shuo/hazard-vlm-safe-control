"""EXP-01A-D frozen-checkpoint, test-only decision evaluation.

This module contains no optimizer, loss, training loop, RGB, VLM, or RL code.
Candidate sets are generated only from a frozen seed and oracle geometry.  A
learned arm can be evaluated only when its checkpoint hash was already recorded
by the immutable source EXP-01A manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from evaluation.oracle_spatial_exposure import (
    CARDS,
    CARD_IDS,
    DenseCost,
    FactorizedCost,
    ellipse_mask,
    exposure,
    parameter_count,
    rasterize_polyline,
    semantic_cost,
)


SCHEMA_VERSION = "exp01ad-frozen-decision-v1"
CANDIDATE_IDS = ("water_heavy", "fragile_heavy", "balanced", "dual_high")
LEARNED_ARMS = ("dense_cost", "factorized_no_cf", "factorized_full")
MODEL_ARMS = (*LEARNED_ARMS, "oracle_formula")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DecisionCandidate:
    candidate_id: str
    points: tuple[tuple[float, float], ...]
    footprint: np.ndarray
    exposure: tuple[float, float]

    def __post_init__(self) -> None:
        if self.candidate_id not in CANDIDATE_IDS:
            raise ValueError("unknown candidate")
        if self.footprint.ndim != 2 or not np.isin(self.footprint, (0, 1)).all():
            raise ValueError("footprint must be a binary raster")
        if not all(0 <= value <= 1 for value in self.exposure):
            raise ValueError("candidate exposures must lie in [0,1]")


@dataclass(frozen=True)
class DecisionScene:
    scene_id: str
    scene_index: int
    generation_attempt: int
    fields: np.ndarray
    candidates: tuple[DecisionCandidate, ...]
    robot_radius: float
    geometry: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.fields.ndim != 3 or self.fields.shape[0] != 2:
            raise ValueError("fields must be 2xHxW")
        if tuple(item.candidate_id for item in self.candidates) != CANDIDATE_IDS:
            raise ValueError("candidate order changed")
        audit_candidate_set(self)

    def record(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_index": self.scene_index,
            "generation_attempt": self.generation_attempt,
            "geometry": dict(self.geometry),
            "field_sha256": hashlib.sha256(self.fields.tobytes()).hexdigest(),
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "points": item.points,
                    "exposure": item.exposure,
                    "footprint_sha256": hashlib.sha256(item.footprint.tobytes()).hexdigest(),
                }
                for item in self.candidates
            ],
        }


def _unique_argmin(values: Sequence[float], margin: float) -> int | None:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    if float(values[int(order[1])]) - float(values[int(order[0])]) < margin:
        return None
    return int(order[0])


def audit_candidate_set(scene: DecisionScene, *, optimum_margin: float = 0.002) -> None:
    exposures = np.asarray([item.exposure for item in scene.candidates], dtype=np.float64)
    if np.any(exposures <= 0):
        raise ValueError("zero-exposure bypass is forbidden")
    costs = {
        card_id: np.asarray([
            semantic_cost(*item.exposure, *CARDS[card_id]) for item in scene.candidates
        ])
        for card_id in CARD_IDS
    }
    # B is forced to all-zero by the frozen semantic equation.  A, C, and D
    # must contain positive, distinct costs and unique optima.
    if not np.allclose(costs["B"], 0.0, atol=0.0, rtol=0.0):
        raise ValueError("card B truth-table invariant changed")
    for card_id in ("A", "C", "D"):
        if np.any(costs[card_id] <= 0) or len(np.unique(np.round(costs[card_id], 12))) < 2:
            raise ValueError("candidate set lacks positive distinct oracle costs")
    optima = {card_id: _unique_argmin(costs[card_id], optimum_margin) for card_id in ("A", "C", "D")}
    if any(value is None for value in optima.values()) or len(set(optima.values())) != 3:
        raise ValueError("oracle optimum must change across rule and capability interventions")
    # Frozen semantic roles make the intervention direction auditable.
    expected = {
        "A": CANDIDATE_IDS.index("fragile_heavy"),
        "C": CANDIDATE_IDS.index("balanced"),
        "D": CANDIDATE_IDS.index("water_heavy"),
    }
    if optima != expected:
        raise ValueError(f"unexpected oracle-optimum roles: {optima}")


def generate_decision_scene(
    seed: int,
    scene_index: int,
    *,
    size: int = 48,
    max_attempts: int = 512,
    optimum_margin: float = 0.002,
) -> DecisionScene:
    """Generate without accessing a model, checkpoint, or prediction."""
    templates = {
        "water_heavy": (0.05, 1.08),
        "fragile_heavy": (1.08, 0.05),
        "balanced": (0.72, 0.72),
        "dual_high": (0.20, 0.20),
    }
    for attempt in range(max_attempts):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(scene_index), int(attempt), 101]))
        robot_radius = float(rng.uniform(0.065, 0.095))
        water_center = (-0.34 + float(rng.uniform(-0.025, 0.025)), float(rng.uniform(-0.20, 0.20)))
        fragile_center = (0.34 + float(rng.uniform(-0.025, 0.025)), float(rng.uniform(-0.20, 0.20)))
        water_radii = (float(rng.uniform(0.12, 0.17)), float(rng.uniform(0.10, 0.15)))
        fragile_radii = (float(rng.uniform(0.12, 0.17)), float(rng.uniform(0.10, 0.15)))
        fields = np.stack((
            ellipse_mask(size, size, water_center, water_radii),
            ellipse_mask(size, size, fragile_center, fragile_radii),
        ))
        candidates = []
        geometry_candidates: dict[str, Any] = {}
        for candidate_id in CANDIDATE_IDS:
            water_scale, fragile_scale = templates[candidate_id]
            water_sign = -1.0 if rng.random() < 0.5 else 1.0
            fragile_sign = -1.0 if rng.random() < 0.5 else 1.0
            water_y = water_center[1] + water_sign * water_scale * water_radii[1]
            fragile_y = fragile_center[1] + fragile_sign * fragile_scale * fragile_radii[1]
            jitter = float(rng.uniform(-0.008, 0.008))
            points = (
                (-0.94, 0.0),
                (-0.58, water_y + jitter),
                (-0.14, water_y + jitter),
                (0.14, fragile_y - jitter),
                (0.58, fragile_y - jitter),
                (0.94, 0.0),
            )
            footprint = rasterize_polyline(size, size, points, robot_radius)
            value = (exposure(footprint, fields[0]), exposure(footprint, fields[1]))
            candidates.append(DecisionCandidate(candidate_id, points, footprint, value))
            geometry_candidates[candidate_id] = {"points": points, "template": templates[candidate_id]}
        geometry = {
            "size": size,
            "robot_radius": robot_radius,
            "water_center": water_center,
            "water_radii": water_radii,
            "fragile_center": fragile_center,
            "fragile_radii": fragile_radii,
            "candidate_geometry": geometry_candidates,
        }
        try:
            scene = DecisionScene(
                scene_id=f"exp01ad-test-{scene_index:05d}",
                scene_index=int(scene_index),
                generation_attempt=attempt,
                fields=fields,
                candidates=tuple(candidates),
                robot_radius=robot_radius,
                geometry=geometry,
            )
            audit_candidate_set(scene, optimum_margin=optimum_margin)
            return scene
        except ValueError:
            continue
    raise RuntimeError(f"unable to generate valid scene {scene_index} in {max_attempts} attempts")


def generate_candidate_artifact(
    *, seed: int, scene_count: int, size: int, optimum_margin: float
) -> tuple[tuple[DecisionScene, ...], dict[str, Any]]:
    if scene_count < 2:
        raise ValueError("scene_count must be at least two")
    scenes = tuple(
        generate_decision_scene(seed, index, size=size, optimum_margin=optimum_margin)
        for index in range(scene_count)
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "PROSPECTIVELY_FROZEN_CANDIDATE_SET",
        "prediction_accessed": False,
        "checkpoint_accessed": False,
        "seed": int(seed),
        "scene_count": scene_count,
        "rows_per_scene": len(CANDIDATE_IDS) * len(CARD_IDS),
        "row_count": scene_count * len(CANDIDATE_IDS) * len(CARD_IDS),
        "scenes": [scene.record() for scene in scenes],
    }
    return scenes, artifact


def verify_candidate_artifact(
    artifact: Mapping[str, Any], *, seed: int, scene_count: int, size: int, optimum_margin: float
) -> tuple[DecisionScene, ...]:
    scenes, expected = generate_candidate_artifact(
        seed=seed, scene_count=scene_count, size=size, optimum_margin=optimum_margin
    )
    if stable_hash(artifact) != stable_hash(expected):
        raise PermissionError("frozen candidate artifact does not match prospective generator contract")
    return scenes


def audit_source_artifacts(config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = config["source_exp01a"]
    manifest_path = (root / source["manifest_path"]).resolve()
    results_path = (root / source["results_path"]).resolve()
    if file_sha256(manifest_path) != source["manifest_sha256"]:
        raise PermissionError("source EXP-01A manifest changed after EXP-01A-D freeze")
    if file_sha256(results_path) != source["results_sha256"]:
        raise PermissionError("source EXP-01A results changed after EXP-01A-D freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("run_mode") != "FORMAL" or results.get("validity") != "VALID":
        raise PermissionError("source EXP-01A result is not a valid formal run")
    return manifest, results


def load_frozen_checkpoints(
    source_manifest: Mapping[str, Any], *, root: Path
) -> dict[str, nn.Module]:
    """Load only hashes prospectively registered by the source experiment."""
    registered = source_manifest.get("checkpoints")
    if not isinstance(registered, Mapping):
        raise FileNotFoundError(
            "source EXP-01A manifest contains no checkpoint registry; retraining or reconstructed weights are forbidden"
        )
    if set(registered) != set(LEARNED_ARMS):
        raise PermissionError("source checkpoint registry must contain exactly the three learned arms")
    models: dict[str, nn.Module] = {}
    for arm in LEARNED_ARMS:
        record = registered[arm]
        path = (root / str(record["path"])).resolve()
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise PermissionError(f"missing or modified frozen checkpoint: {arm}")
        model: nn.Module = DenseCost() if arm == "dense_cost" else FactorizedCost()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError(f"checkpoint is not a state dict: {arm}")
        model.load_state_dict(state, strict=True)
        if parameter_count(model) != int(record["parameter_count"]):
            raise PermissionError(f"parameter count mismatch: {arm}")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[arm] = model
    return models


def truth_matrix(scene: DecisionScene) -> np.ndarray:
    return np.asarray([
        [semantic_cost(*candidate.exposure, *CARDS[card_id]) for card_id in CARD_IDS]
        for candidate in scene.candidates
    ], dtype=np.float64)


def predict_models(
    models: Mapping[str, nn.Module], scenes: Sequence[DecisionScene]
) -> dict[str, np.ndarray]:
    result = {arm: [] for arm in MODEL_ARMS}
    with torch.inference_mode():
        for scene in scenes:
            spatial, exposures, cards = [], [], []
            for candidate in scene.candidates:
                for card_id in CARD_IDS:
                    spatial.append(np.concatenate((scene.fields, candidate.footprint[None]), axis=0))
                    exposures.append(candidate.exposure)
                    cards.append(CARDS[card_id])
            spatial_tensor = torch.tensor(np.asarray(spatial), dtype=torch.float32)
            exposure_tensor = torch.tensor(exposures, dtype=torch.float32)
            card_tensor = torch.tensor(cards, dtype=torch.float32)
            for arm in LEARNED_ARMS:
                model = models[arm]
                prediction = (
                    model(spatial_tensor, card_tensor)
                    if isinstance(model, DenseCost)
                    else model(exposure_tensor, card_tensor)
                )
                result[arm].append(prediction.numpy().reshape(len(CANDIDATE_IDS), len(CARD_IDS)))
            result["oracle_formula"].append(truth_matrix(scene))
    return {arm: np.asarray(values, dtype=np.float64) for arm, values in result.items()}


def _tie_accuracy(delta: float, tolerance: float = 1e-9) -> float:
    return 1.0 if delta > tolerance else (0.5 if abs(delta) <= tolerance else 0.0)


def decision_metrics(
    scenes: Sequence[DecisionScene],
    predictions: np.ndarray,
    *,
    dangerous_excess_margin: float,
    false_safe_threshold: float,
) -> dict[str, np.ndarray]:
    if predictions.shape != (len(scenes), len(CANDIDATE_IDS), len(CARD_IDS)):
        raise ValueError("prediction shape mismatch")
    output = {key: [] for key in (
        "action_regret",
        "dangerous_selection_rate",
        "false_safe_rate",
        "pairwise_ranking_accuracy",
        "semantic_cost_mae",
    )}
    for scene_index, scene in enumerate(scenes):
        truth = truth_matrix(scene)
        prediction = predictions[scene_index]
        regrets, dangerous, false_safe, ranking = [], [], [], []
        for card_index in range(len(CARD_IDS)):
            true_card = truth[:, card_index]
            pred_card = prediction[:, card_index]
            selected = np.flatnonzero(np.isclose(pred_card, pred_card.min(), atol=1e-9, rtol=0.0))
            regrets.append(float(true_card[selected].mean() - true_card.min()))
            dangerous.append(float(np.mean(
                true_card[selected] > true_card.min() + dangerous_excess_margin
            )))
            for left in range(len(CANDIDATE_IDS)):
                for right in range(left + 1, len(CANDIDATE_IDS)):
                    true_delta = float(true_card[left] - true_card[right])
                    if abs(true_delta) > 1e-12:
                        predicted_delta = float(pred_card[left] - pred_card[right])
                        ranking.append(_tie_accuracy(math.copysign(1.0, true_delta) * predicted_delta))
        eligible = truth > false_safe_threshold
        false_safe.extend((prediction[eligible] <= false_safe_threshold).astype(float).tolist())
        output["action_regret"].append(float(np.mean(regrets)))
        output["dangerous_selection_rate"].append(float(np.mean(dangerous)))
        output["false_safe_rate"].append(float(np.mean(false_safe)) if false_safe else 0.0)
        output["pairwise_ranking_accuracy"].append(float(np.mean(ranking)))
        output["semantic_cost_mae"].append(float(np.mean(np.abs(prediction - truth))))
    return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}


def paired_bootstrap_ci(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or resamples < 100:
        raise ValueError("paired bootstrap requires >=2 scenes and >=100 resamples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


def evaluate_gate(
    scenes: Sequence[DecisionScene],
    predictions: Mapping[str, np.ndarray],
    *,
    dangerous_excess_margin: float,
    false_safe_threshold: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    per_scene = {
        arm: decision_metrics(
            scenes,
            predictions[arm],
            dangerous_excess_margin=dangerous_excess_margin,
            false_safe_threshold=false_safe_threshold,
        )
        for arm in MODEL_ARMS
    }
    summary = {
        arm: {metric: float(values.mean()) for metric, values in metrics.items()}
        for arm, metrics in per_scene.items()
    }
    comparisons = {}
    for metric_index, metric in enumerate(("action_regret", "dangerous_selection_rate")):
        reduction = per_scene["dense_cost"][metric] - per_scene["factorized_no_cf"][metric]
        comparisons[metric] = {
            "mean_reduction": float(reduction.mean()),
            "paired_scene_bootstrap_ci95": list(paired_bootstrap_ci(
                reduction, seed=bootstrap_seed + metric_index, resamples=bootstrap_resamples
            )),
        }
    significant = [
        metric for metric, item in comparisons.items()
        if item["paired_scene_bootstrap_ci95"][0] > 0
    ]
    other_not_worse = all(
        item["mean_reduction"] >= -1e-12 and item["paired_scene_bootstrap_ci95"][1] >= 0
        for item in comparisons.values()
    )
    oracle_exact = summary["oracle_formula"]["semantic_cost_mae"] <= 1e-12
    unlock = bool(significant) and other_not_worse and oracle_exact
    cf_diagnostic = {}
    for metric_index, metric in enumerate(("action_regret", "dangerous_selection_rate")):
        reduction = per_scene["factorized_no_cf"][metric] - per_scene["factorized_full"][metric]
        cf_diagnostic[metric] = {
            "mean_reduction": float(reduction.mean()),
            "paired_scene_bootstrap_ci95": list(paired_bootstrap_ci(
                reduction, seed=bootstrap_seed + 20 + metric_index, resamples=bootstrap_resamples
            )),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "validity": "VALID" if oracle_exact else "INVALID_RUN",
        "program_decision": "EXP01B_AUTHORIZED" if unlock else "EXP01B_NOT_AUTHORIZED",
        "metrics": summary,
        "dense_vs_factorized_no_cf": comparisons,
        "gate": {
            "significantly_improved_primary_metrics": significant,
            "other_primary_not_worse": other_not_worse,
            "oracle_formula_exact": oracle_exact,
            "passed": unlock,
        },
        "factorized_full_diagnostic": {
            "comparisons": cf_diagnostic,
            "affects_program_decision": False,
        },
    }
