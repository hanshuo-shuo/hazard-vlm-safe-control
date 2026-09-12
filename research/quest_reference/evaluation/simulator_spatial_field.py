"""Simulator-supervised RGB requirement-field experiment for EXP-01B."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from evaluation.oracle_spatial_decision import (
    CANDIDATE_IDS,
    DecisionScene,
    generate_decision_scene,
    paired_bootstrap_ci,
)
from evaluation.oracle_spatial_exposure import CARDS, CARD_IDS, DenseCost, FactorizedCost
from evaluation.oracle_spatial_exposure import generate_scene as generate_canonical_scene


SCHEMA_VERSION = "exp01b-simulator-field-v1"
APPEARANCE_SPECS = {
    "aqua_sand": {"floor": (194, 181, 151), "water": (25, 135, 205), "fragile": (190, 105, 48), "wave": 3, "crack": 5, "light": 1.00},
    "blue_slate": {"floor": (156, 165, 174), "water": (35, 92, 185), "fragile": (175, 142, 62), "wave": 4, "crack": 6, "light": 0.92},
    "teal_clay": {"floor": (183, 156, 132), "water": (20, 150, 150), "fragile": (175, 72, 65), "wave": 5, "crack": 4, "light": 1.05},
    "violet_night": {"floor": (75, 73, 91), "water": (105, 62, 190), "fragile": (65, 178, 112), "wave": 7, "crack": 8, "light": 0.72},
    "silver_neon": {"floor": (202, 207, 211), "water": (80, 180, 225), "fragile": (225, 55, 155), "wave": 6, "crack": 9, "light": 1.12},
    "indigo_lime": {"floor": (105, 98, 119), "water": (48, 48, 155), "fragile": (165, 205, 55), "wave": 8, "crack": 7, "light": 0.82},
    "monochrome_wave": {"floor": (175, 175, 175), "water": (95, 125, 145), "fragile": (135, 115, 105), "wave": 9, "crack": 10, "light": 0.98},
}


def stable_hash(value: Any) -> str:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    return (np.asarray(image.resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.float32)


def render_requirement_rgb(
    fields: np.ndarray,
    *,
    family: str,
    image_size: int,
    render_seed: int,
    erase_channel: int | None = None,
) -> np.ndarray:
    """Render terrain before any trajectory overlay; erasure reveals neutral floor."""
    if family not in APPEARANCE_SPECS:
        raise ValueError(f"unknown appearance family: {family}")
    if fields.shape[0] != 2:
        raise ValueError("two requirement channels required")
    spec = APPEARANCE_SPECS[family]
    rng = np.random.default_rng(int(render_seed))
    yy, xx = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    floor = np.asarray(spec["floor"], dtype=np.float32)
    light = float(spec["light"]) * (0.88 + 0.18 * xx / max(1, image_size - 1))
    noise = rng.normal(0.0, 3.0, size=(image_size, image_size, 1))
    image = floor[None, None, :] * light[..., None] + noise
    masks = [_resize_mask(fields[index], image_size) for index in range(2)]
    if erase_channel is not None:
        masks[int(erase_channel)] = np.zeros_like(masks[int(erase_channel)])
    water_color = np.asarray(spec["water"], dtype=np.float32)
    fragile_color = np.asarray(spec["fragile"], dtype=np.float32)
    wave = np.sin((xx + 0.55 * yy) * (2.0 * np.pi * float(spec["wave"]) / image_size))
    water_texture = water_color[None, None, :] + wave[..., None] * np.asarray((18, 24, 28))
    crack_period = int(spec["crack"])
    cracks = ((xx + 2 * yy + render_seed % crack_period) % crack_period == 0) | ((2 * xx - yy) % (crack_period + 3) == 0)
    fragile_texture = np.broadcast_to(fragile_color, image.shape).copy()
    fragile_texture[cracks] *= 0.55
    for mask, texture in ((masks[0], water_texture), (masks[1], fragile_texture)):
        alpha = (0.82 * mask)[..., None]
        image = image * (1.0 - alpha) + texture * alpha
    return np.clip(image, 0, 255).astype(np.uint8)


def lowres_field(fields: np.ndarray, field_size: int) -> np.ndarray:
    tensor = torch.tensor(fields[None], dtype=torch.float32)
    return F.interpolate(tensor, size=(field_size, field_size), mode="area")[0].numpy()


def lowres_footprint(mask: np.ndarray, field_size: int) -> np.ndarray:
    tensor = torch.tensor(mask[None, None], dtype=torch.float32)
    return F.interpolate(tensor, size=(field_size, field_size), mode="area")[0, 0].numpy()


def pooled_exposure(fields: np.ndarray, footprint: np.ndarray) -> tuple[float, float]:
    denominator = float(np.asarray(footprint, dtype=np.float64).sum())
    if denominator <= 0:
        raise ValueError("footprint must be nonempty")
    return tuple(float((fields[index] * footprint).sum() / denominator) for index in range(2))


@dataclass(frozen=True)
class RGBScene:
    split: str
    family: str
    scene: DecisionScene
    rgb: np.ndarray
    field_target: np.ndarray
    footprints: np.ndarray
    exposures: np.ndarray
    render_seed: int


def build_rgb_scenes(
    split: str,
    spec: Mapping[str, Any],
    *,
    image_size: int,
    field_size: int,
) -> tuple[RGBScene, ...]:
    families = tuple(spec["appearance_families"])
    scenes = []
    for index in range(int(spec["scene_count"])):
        scene = generate_decision_scene(int(spec["scene_seed"]), index, size=48)
        family = families[index % len(families)]
        render_seed = int(spec["scene_seed"]) * 100_000 + index
        target = lowres_field(scene.fields, field_size)
        footprints = np.asarray([lowres_footprint(item.footprint, field_size) for item in scene.candidates])
        exposures = np.asarray([pooled_exposure(target, footprint) for footprint in footprints])
        scenes.append(RGBScene(
            split=split,
            family=family,
            scene=scene,
            rgb=render_requirement_rgb(scene.fields, family=family, image_size=image_size, render_seed=render_seed),
            field_target=target,
            footprints=footprints,
            exposures=exposures,
            render_seed=render_seed,
        ))
    return tuple(scenes)


class ResNet18Field(nn.Module):
    """ResNet18 encoder plus a lightweight skip decoder at output stride 4."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lateral4 = nn.Conv2d(512, 64, 1)
        self.lateral3 = nn.Conv2d(256, 64, 1)
        self.lateral2 = nn.Conv2d(128, 64, 1)
        self.lateral1 = nn.Conv2d(64, 64, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1)
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(self.stem(rgb))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        value = self.lateral4(x4)
        value = F.interpolate(value, x3.shape[-2:], mode="bilinear", align_corners=False) + self.lateral3(x3)
        value = F.interpolate(value, x2.shape[-2:], mode="bilinear", align_corners=False) + self.lateral2(x2)
        value = F.interpolate(value, x1.shape[-2:], mode="bilinear", align_corners=False) + self.lateral1(x1)
        return self.refine(value)


def field_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return bce + dice, bce, dice


def _rgb_tensor(scenes: Sequence[RGBScene]) -> torch.Tensor:
    return torch.tensor(np.asarray([item.rgb for item in scenes]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def _target_tensor(scenes: Sequence[RGBScene]) -> torch.Tensor:
    return torch.tensor(np.asarray([item.field_target for item in scenes]), dtype=torch.float32)


@dataclass(frozen=True)
class FieldTrainingConfig:
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    max_epochs: int = 24
    patience: int = 6


def train_field_model(
    train: Sequence[RGBScene], validation: Sequence[RGBScene], *, seed: int, config: FieldTrainingConfig
) -> tuple[ResNet18Field, dict[str, Any]]:
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    model = ResNet18Field().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_rgb, train_target = _rgb_tensor(train), _target_tensor(train)
    validation_rgb, validation_target = _rgb_tensor(validation), _target_tensor(validation)
    rng = np.random.default_rng(seed)
    best_state, best_loss, stale = copy.deepcopy(model.state_dict()), math.inf, 0
    history = []
    for epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), config.batch_size):
            index = torch.tensor(order[start:start + config.batch_size], dtype=torch.long)
            logits = model(train_rgb[index])
            loss, _bce, _dice = field_loss(logits, train_target[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation_loss = float(field_loss(model(validation_rgb), validation_target)[0])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": len(history), "best_validation_loss": best_loss, "history": history}


def predict_fields(model: nn.Module, scenes: Sequence[RGBScene], *, batch_size: int = 64) -> np.ndarray:
    rgb = _rgb_tensor(scenes)
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(scenes), batch_size):
            output.append(torch.sigmoid(model(rgb[start:start + batch_size])).numpy())
    return np.concatenate(output, axis=0)


def predict_rgb_arrays(model: nn.Module, rgb_arrays: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    rgb = torch.tensor(np.asarray(rgb_arrays), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rgb), batch_size):
            output.append(torch.sigmoid(model(rgb[start:start + batch_size])).numpy())
    return np.concatenate(output, axis=0)


def field_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    binary_prediction = prediction >= 0.5
    binary_target = target >= 0.5
    ious = []
    for channel in range(2):
        intersection = np.logical_and(binary_prediction[:, channel], binary_target[:, channel]).sum()
        union = np.logical_or(binary_prediction[:, channel], binary_target[:, channel]).sum()
        ious.append(float(intersection / union) if union else 1.0)
    return {"water_iou": ious[0], "fragile_iou": ious[1], "field_miou": float(np.mean(ious))}


def risk_truth(exposures: np.ndarray) -> np.ndarray:
    return np.asarray([
        [1.0 - (1.0 - value[0] * (1 - CARDS[card][0])) * (1.0 - value[1] * CARDS[card][1]) for card in CARD_IDS]
        for value in exposures
    ], dtype=np.float64)


def relation_predictions(model: FactorizedCost, exposures: np.ndarray) -> np.ndarray:
    values, cards = [], []
    for exposure_value in exposures:
        for card in CARD_IDS:
            values.append(exposure_value)
            cards.append(CARDS[card])
    with torch.inference_mode():
        prediction = model(
            torch.tensor(np.asarray(values), dtype=torch.float32),
            torch.tensor(np.asarray(cards), dtype=torch.float32),
        )
    return prediction.numpy().reshape(len(exposures), len(CARD_IDS))


def dense_predictions(model: DenseCost, fields: np.ndarray, footprints: np.ndarray) -> np.ndarray:
    spatial, cards = [], []
    for footprint in footprints:
        for card in CARD_IDS:
            spatial.append(np.concatenate((fields, footprint[None]), axis=0))
            cards.append(CARDS[card])
    with torch.inference_mode():
        prediction = model(torch.tensor(np.asarray(spatial), dtype=torch.float32), torch.tensor(cards, dtype=torch.float32))
    return prediction.numpy().reshape(len(footprints), len(CARD_IDS))


def fit_cards_only(train: Sequence[RGBScene], seen_cards: Sequence[str]) -> np.ndarray:
    features, targets = [], []
    for scene in train:
        truth = risk_truth(scene.exposures)
        for card in seen_cards:
            card_index = CARD_IDS.index(card)
            for candidate_index in range(len(CANDIDATE_IDS)):
                kappa, rule = CARDS[card]
                features.append((1.0, float(kappa), float(rule)))
                targets.append(truth[candidate_index, card_index])
    return np.linalg.lstsq(np.asarray(features), np.asarray(targets), rcond=None)[0]


def cards_only_predictions(coefficients: np.ndarray) -> np.ndarray:
    per_card = np.asarray([
        float(np.dot(coefficients, (1.0, *CARDS[card]))) for card in CARD_IDS
    ])
    return np.tile(np.clip(per_card, 0.0, 1.0)[None], (len(CANDIDATE_IDS), 1))


def _score_scene(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    card_indices: Sequence[int],
    false_safe_threshold: float,
    dangerous_margin: float,
) -> dict[str, float]:
    regret, dangerous, ranking = [], [], []
    for card_index in card_indices:
        true_card, predicted_card = truth[:, card_index], prediction[:, card_index]
        selected = np.flatnonzero(np.isclose(predicted_card, predicted_card.min(), atol=1e-9, rtol=0.0))
        regret.append(float(true_card[selected].mean() - true_card.min()))
        dangerous.append(float(np.mean(true_card[selected] > true_card.min() + dangerous_margin)))
        for left in range(len(CANDIDATE_IDS)):
            for right in range(left + 1, len(CANDIDATE_IDS)):
                delta = float(true_card[left] - true_card[right])
                if abs(delta) <= 1e-12:
                    continue
                predicted_delta = float(predicted_card[left] - predicted_card[right]) * math.copysign(1.0, delta)
                ranking.append(1.0 if predicted_delta > 1e-9 else (0.5 if abs(predicted_delta) <= 1e-9 else 0.0))
    selected_truth = truth[:, card_indices]
    selected_prediction = prediction[:, card_indices]
    eligible = selected_truth > false_safe_threshold
    false_safe = float(np.mean(selected_prediction[eligible] <= false_safe_threshold)) if eligible.any() else 0.0
    return {
        "regret": float(np.mean(regret)),
        "dangerous_selection": float(np.mean(dangerous)),
        "dangerous_inversion": 1.0 - float(np.mean(ranking)),
        "pair_ranking": float(np.mean(ranking)),
        "false_safe": false_safe,
        "semantic_cost_mae": float(np.mean(np.abs(selected_prediction - selected_truth))),
    }


def evaluate_split(
    scenes: Sequence[RGBScene],
    predicted_fields: np.ndarray,
    *,
    factorized: FactorizedCost,
    factorized_full: FactorizedCost,
    dense: DenseCost,
    cards_coefficients: np.ndarray,
    card_ids: Sequence[str],
    false_safe_threshold: float,
    dangerous_margin: float,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    target_fields = np.asarray([item.field_target for item in scenes])
    field_result = field_metrics(predicted_fields, target_fields)
    per_scene = {arm: {metric: [] for metric in ("regret", "dangerous_selection", "dangerous_inversion", "pair_ranking", "false_safe", "semantic_cost_mae")} for arm in ("cards_only", "global_presence", "dense_cost", "factorized_no_cf", "factorized_full", "oracle_field")}
    exposure_errors = {"global_presence": [], "factorized_no_cf": [], "factorized_full": [], "oracle_field": []}
    risk_predictions = {arm: [] for arm in per_scene}
    card_indices = [CARD_IDS.index(card) for card in card_ids]
    for scene_index, scene in enumerate(scenes):
        predicted = predicted_fields[scene_index]
        predicted_exposure = np.asarray([pooled_exposure(predicted, footprint) for footprint in scene.footprints])
        global_exposure = np.tile(predicted.mean(axis=(-2, -1))[None], (len(CANDIDATE_IDS), 1))
        truth = risk_truth(scene.exposures)
        predictions = {
            "cards_only": cards_only_predictions(cards_coefficients),
            "global_presence": relation_predictions(factorized, global_exposure),
            "dense_cost": dense_predictions(dense, predicted, scene.footprints),
            "factorized_no_cf": relation_predictions(factorized, predicted_exposure),
            "factorized_full": relation_predictions(factorized_full, predicted_exposure),
            "oracle_field": relation_predictions(factorized, scene.exposures),
        }
        for arm, values in predictions.items():
            risk_predictions[arm].append(values)
            metrics = _score_scene(
                truth, values, card_indices=card_indices,
                false_safe_threshold=false_safe_threshold, dangerous_margin=dangerous_margin,
            )
            for metric, value in metrics.items():
                per_scene[arm][metric].append(value)
        exposure_errors["global_presence"].append(float(np.mean(np.abs(global_exposure - scene.exposures))))
        exposure_errors["factorized_no_cf"].append(float(np.mean(np.abs(predicted_exposure - scene.exposures))))
        exposure_errors["factorized_full"].append(float(np.mean(np.abs(predicted_exposure - scene.exposures))))
        exposure_errors["oracle_field"].append(0.0)
    arrays = {arm: {metric: np.asarray(values) for metric, values in metrics.items()} for arm, metrics in per_scene.items()}
    summary = {}
    for arm, metrics in arrays.items():
        summary[arm] = {metric: float(values.mean()) for metric, values in metrics.items()}
        summary[arm]["field_miou"] = None if arm == "cards_only" else (1.0 if arm == "oracle_field" else field_result["field_miou"])
        summary[arm]["exposure_mae"] = None if arm in {"cards_only", "dense_cost"} else float(np.mean(exposure_errors[arm]))
    return summary, arrays, {arm: np.asarray(values) for arm, values in risk_predictions.items()}


def bootstrap_comparisons(
    arrays: Mapping[str, Mapping[str, np.ndarray]], *, seed: int, resamples: int
) -> dict[str, Any]:
    result = {}
    for baseline_index, baseline in enumerate(("cards_only", "global_presence", "dense_cost")):
        result[baseline] = {}
        for metric_index, metric in enumerate(("dangerous_inversion", "false_safe", "regret")):
            reduction = arrays[baseline][metric] - arrays["factorized_no_cf"][metric]
            result[baseline][metric] = {
                "mean_reduction": float(reduction.mean()),
                "paired_scene_bootstrap_ci95": list(paired_bootstrap_ci(
                    reduction, seed=seed + 10 * baseline_index + metric_index, resamples=resamples
                )),
            }
    return result


def same_image_paired_check(
    risk_predictions: Mapping[str, np.ndarray], truths: Sequence[np.ndarray]
) -> dict[str, dict[str, float]]:
    """Water crossing vs relative water avoidance under identical RGB and card A."""
    water = CANDIDATE_IDS.index("water_heavy")
    avoid = CANDIDATE_IDS.index("fragile_heavy")
    card = CARD_IDS.index("A")
    result = {}
    truth_accuracy = float(np.mean([truth[water, card] > truth[avoid, card] for truth in truths]))
    for arm, values in risk_predictions.items():
        deltas = values[:, water, card] - values[:, avoid, card]
        result[arm] = {
            "accuracy": float(np.mean(deltas > 1e-9) + 0.5 * np.mean(np.abs(deltas) <= 1e-9)),
            "mean_predicted_risk_delta": float(np.mean(deltas)),
            "oracle_pair_validity": truth_accuracy,
        }
    return result


def terrain_erasure_check(
    model: nn.Module,
    scenes: Sequence[RGBScene],
    original_prediction: np.ndarray,
    *,
    image_size: int,
) -> dict[str, dict[str, float]]:
    result = {}
    for channel, name in enumerate(("water", "fragile")):
        erased_rgb = np.asarray([
            render_requirement_rgb(
                item.scene.fields,
                family=item.family,
                image_size=image_size,
                render_seed=item.render_seed,
                erase_channel=channel,
            )
            for item in scenes
        ])
        erased_prediction = predict_rgb_arrays(model, erased_rgb)
        deltas = []
        for index, item in enumerate(scenes):
            candidate_index = int(np.argmax(item.exposures[:, channel]))
            footprint = item.footprints[candidate_index]
            original_exposure = pooled_exposure(original_prediction[index], footprint)[channel]
            erased_exposure = pooled_exposure(erased_prediction[index], footprint)[channel]
            deltas.append(original_exposure - erased_exposure)
        values = np.asarray(deltas)
        result[name] = {
            "original_greater_rate": float(np.mean(values > 1e-9)),
            "mean_exposure_drop": float(values.mean()),
        }
    return result


def irrelevant_swap_check(
    field_model: nn.Module,
    factorized: FactorizedCost,
    *,
    family: str,
    seed: int,
    scene_count: int,
    image_size: int,
    field_size: int,
    epsilon: float = 0.005,
) -> dict[str, float]:
    rgbs, records = [], []
    for index in range(scene_count):
        scene = generate_canonical_scene(seed, "locked_test", index, size=48)
        render_seed = seed * 100_000 + index
        rgbs.append(render_requirement_rgb(
            scene.fields, family=family, image_size=image_size, render_seed=render_seed
        ))
        records.append(scene)
    prediction = predict_rgb_arrays(field_model, np.asarray(rgbs))
    deltas = []
    for scene_index, scene in enumerate(records):
        route_predictions = {}
        for route_index, route in enumerate(scene.trajectories):
            footprint = lowres_footprint(route.footprint, field_size)
            value = np.asarray([pooled_exposure(prediction[scene_index], footprint)])
            route_predictions[route.trajectory_id] = relation_predictions(factorized, value)[0]
        # Water-only: protect-fragile is irrelevant.
        for kappa in (0, 1):
            left = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (kappa, 0))
            right = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (kappa, 1))
            deltas.append(abs(route_predictions["water_cross"][left] - route_predictions["water_cross"][right]))
        # Fragile-only: waterproof is irrelevant.
        for rule in (0, 1):
            left = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (0, rule))
            right = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (1, rule))
            deltas.append(abs(route_predictions["fragile_cross"][left] - route_predictions["fragile_cross"][right]))
        # Neutral: both hypercube axes are irrelevant.
        neutral = route_predictions["neutral"]
        for left, right in ((0, 1), (0, 2), (1, 3), (2, 3)):
            deltas.append(abs(neutral[left] - neutral[right]))
    values = np.asarray(deltas)
    return {
        "mean_absolute_swap_delta": float(values.mean()),
        "invariance_accuracy_at_0.005": float(np.mean(values <= epsilon)),
    }
