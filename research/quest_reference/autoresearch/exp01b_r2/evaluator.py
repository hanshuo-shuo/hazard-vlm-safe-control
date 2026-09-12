"""Fixed evaluator for the EXP-01B-R2 development loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from evaluation.oracle_spatial_exposure import CARDS, CARD_IDS, DenseCost, FactorizedCost
from evaluation.oracle_spatial_exposure import generate_scene as generate_canonical_scene
from evaluation.simulator_spatial_field import (
    evaluate_split,
    field_metrics,
    fit_cards_only,
    lowres_footprint,
    pooled_exposure,
    relation_predictions,
    render_requirement_rgb,
    risk_truth,
    same_image_paired_check,
)


ROOT = Path(__file__).resolve().parents[2]


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _load_state(path: Path, model: torch.nn.Module) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_fixed_models(config: Mapping[str, Any]) -> dict[str, torch.nn.Module]:
    paths = config["source_relation_checkpoints"]
    return {
        "dense_cost": _load_state(ROOT / paths["dense_cost"], DenseCost()),
        "factorized_no_cf": _load_state(ROOT / paths["factorized_no_cf"], FactorizedCost()),
        "factorized_full": _load_state(ROOT / paths["factorized_full"], FactorizedCost()),
    }


def _rgb_tensor(rgb_arrays: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(rgb_arrays), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def predict_rgb_arrays(
    model: torch.nn.Module,
    rgb_arrays: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    rgb = _rgb_tensor(rgb_arrays)
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rgb), batch_size):
            logits = model(rgb[start:start + batch_size].to(device, non_blocking=True))
            expected = (len(logits), 2, 16, 16)
            if tuple(logits.shape) != expected:
                raise ValueError(f"candidate output shape {tuple(logits.shape)} != {expected}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("candidate produced non-finite logits")
            values.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def predict_scenes(
    model: torch.nn.Module,
    scenes: Sequence[Any],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    return predict_rgb_arrays(
        model,
        np.asarray([scene.rgb for scene in scenes]),
        device=device,
        batch_size=batch_size,
    )


def _terrain_erasure_check(
    model: torch.nn.Module,
    scenes: Sequence[Any],
    original_prediction: np.ndarray,
    *,
    device: torch.device,
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
        erased_prediction = predict_rgb_arrays(model, erased_rgb, device=device)
        deltas = []
        for index, item in enumerate(scenes):
            candidate_index = int(np.argmax(item.exposures[:, channel]))
            footprint = item.footprints[candidate_index]
            original = pooled_exposure(original_prediction[index], footprint)[channel]
            erased = pooled_exposure(erased_prediction[index], footprint)[channel]
            deltas.append(original - erased)
        values = np.asarray(deltas)
        result[name] = {
            "original_greater_rate": float(np.mean(values > 1e-9)),
            "mean_exposure_drop": float(values.mean()),
        }
    return result


def _irrelevant_swap_check(
    model: torch.nn.Module,
    factorized: FactorizedCost,
    *,
    family: str,
    seed: int,
    scene_count: int,
    image_size: int,
    field_size: int,
    device: torch.device,
    epsilon: float = 0.005,
) -> dict[str, float]:
    rgbs, records = [], []
    for index in range(scene_count):
        scene = generate_canonical_scene(seed, "locked_test", index, size=48)
        render_seed = seed * 100_000 + index
        rgbs.append(render_requirement_rgb(
            scene.fields,
            family=family,
            image_size=image_size,
            render_seed=render_seed,
        ))
        records.append(scene)
    prediction = predict_rgb_arrays(model, np.asarray(rgbs), device=device)
    deltas = []
    for scene_index, scene in enumerate(records):
        route_predictions = {}
        for route in scene.trajectories:
            footprint = lowres_footprint(route.footprint, field_size)
            value = np.asarray([pooled_exposure(prediction[scene_index], footprint)])
            route_predictions[route.trajectory_id] = relation_predictions(factorized, value)[0]
        for kappa in (0, 1):
            left = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (kappa, 0))
            right = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (kappa, 1))
            deltas.append(abs(route_predictions["water_cross"][left] - route_predictions["water_cross"][right]))
        for rule in (0, 1):
            left = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (0, rule))
            right = next(index for index, card in enumerate(CARD_IDS) if CARDS[card] == (1, rule))
            deltas.append(abs(route_predictions["fragile_cross"][left] - route_predictions["fragile_cross"][right]))
        neutral = route_predictions["neutral"]
        for left, right in ((0, 1), (0, 2), (1, 3), (2, 3)):
            deltas.append(abs(neutral[left] - neutral[right]))
    values = np.asarray(deltas)
    return {
        "mean_absolute_swap_delta": float(values.mean()),
        "invariance_accuracy_at_0.005": float(np.mean(values <= epsilon)),
    }


def _truths(scenes: Sequence[Any]) -> list[np.ndarray]:
    return [risk_truth(scene.exposures) for scene in scenes]


def evaluate_candidate(
    model: torch.nn.Module,
    datasets: Mapping[str, Sequence[Any]],
    *,
    fixed_models: Mapping[str, torch.nn.Module],
    source_config: Mapping[str, Any],
    tier: Mapping[str, Any],
    model_seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cards_coefficients = fit_cards_only(
        datasets["train"],
        source_config["splits"]["train"]["risk_cards_seen"],
    )
    result: dict[str, Any] = {}
    saved_arrays: dict[str, np.ndarray] = {}
    shortcut_count = int(tier["shortcut_scenes"])
    for split in ("iid", "appearance_ood", "joint_ood"):
        scenes = datasets[split]
        predicted_fields = predict_scenes(model, scenes, device=device)
        card_ids = (
            (source_config["splits"][split]["primary_held_out_card"],)
            if split == "joint_ood"
            else tuple(source_config["splits"][split]["evaluation_cards"])
        )
        summary, arrays, risk_predictions = evaluate_split(
            scenes,
            predicted_fields,
            factorized=fixed_models["factorized_no_cf"],
            factorized_full=fixed_models["factorized_full"],
            dense=fixed_models["dense_cost"],
            cards_coefficients=cards_coefficients,
            card_ids=card_ids,
            false_safe_threshold=float(source_config["evaluation"]["false_safe_threshold"]),
            dangerous_margin=float(source_config["evaluation"]["dangerous_excess_margin"]),
        )
        for arm, metrics in arrays.items():
            for metric, values in metrics.items():
                saved_arrays[f"seed_{model_seed}__{split}__{arm}__{metric}"] = values

        permutation = np.roll(np.arange(len(scenes)), 1)
        shuffled_fields = predict_rgb_arrays(
            model,
            np.asarray([scenes[index].rgb for index in permutation]),
            device=device,
        )
        shuffled_summary, _, _ = evaluate_split(
            scenes,
            shuffled_fields,
            factorized=fixed_models["factorized_no_cf"],
            factorized_full=fixed_models["factorized_full"],
            dense=fixed_models["dense_cost"],
            cards_coefficients=cards_coefficients,
            card_ids=card_ids,
            false_safe_threshold=float(source_config["evaluation"]["false_safe_threshold"]),
            dangerous_margin=float(source_config["evaluation"]["dangerous_excess_margin"]),
        )
        target = np.asarray([item.field_target for item in scenes])
        normal_field = field_metrics(predicted_fields, target)
        shuffled_field = field_metrics(shuffled_fields, target)
        limited_scenes = scenes[:shortcut_count]
        checks = {
            "same_image_cross_vs_avoid": same_image_paired_check(risk_predictions, _truths(scenes)),
            "image_shuffle": {
                "normal_field_miou": normal_field["field_miou"],
                "shuffled_field_miou": shuffled_field["field_miou"],
                "delta_field_miou": normal_field["field_miou"] - shuffled_field["field_miou"],
                "normal_factorized_pair_ranking": summary["factorized_no_cf"]["pair_ranking"],
                "shuffled_factorized_pair_ranking": shuffled_summary["factorized_no_cf"]["pair_ranking"],
                "delta_pair_ranking": (
                    summary["factorized_no_cf"]["pair_ranking"]
                    - shuffled_summary["factorized_no_cf"]["pair_ranking"]
                ),
            },
            "terrain_erasure": _terrain_erasure_check(
                model,
                limited_scenes,
                predicted_fields[:shortcut_count],
                device=device,
                image_size=int(source_config["image_size"]),
            ),
            "irrelevant_swap": _irrelevant_swap_check(
                model,
                fixed_models["factorized_no_cf"],
                family=scenes[0].family,
                seed=int(source_config["splits"][split]["scene_seed"]) + 7000,
                scene_count=shortcut_count,
                image_size=int(source_config["image_size"]),
                field_size=int(source_config["field_size"]),
                device=device,
            ),
        }
        result[split] = {"summary": summary, "checks": checks}
    return result, saved_arrays


def aggregate_seed_results(seed_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    split_names = ("iid", "appearance_ood", "joint_ood")
    metrics = ("regret", "false_safe", "pair_ranking", "field_miou", "exposure_mae")
    aggregate: dict[str, Any] = {"splits": {}}
    for split in split_names:
        aggregate["splits"][split] = {}
        arms = seed_results[next(iter(seed_results))][split]["summary"].keys()
        for arm in arms:
            aggregate["splits"][split][arm] = {}
            for metric in metrics:
                values = [item[split]["summary"][arm].get(metric) for item in seed_results.values()]
                numeric = [float(value) for value in values if value is not None]
                aggregate["splits"][split][arm][metric] = None if not numeric else float(np.mean(numeric))

    shortcut_values = {
        "min_image_shuffle_pair_delta": [],
        "min_terrain_erasure_positive_rate": [],
        "min_irrelevant_swap_accuracy": [],
        "min_same_image_accuracy": [],
    }
    for seed_result in seed_results.values():
        for split in ("appearance_ood", "joint_ood"):
            checks = seed_result[split]["checks"]
            shortcut_values["min_image_shuffle_pair_delta"].append(checks["image_shuffle"]["delta_pair_ranking"])
            shortcut_values["min_terrain_erasure_positive_rate"].extend([
                checks["terrain_erasure"]["water"]["original_greater_rate"],
                checks["terrain_erasure"]["fragile"]["original_greater_rate"],
            ])
            shortcut_values["min_irrelevant_swap_accuracy"].append(
                checks["irrelevant_swap"]["invariance_accuracy_at_0.005"]
            )
            shortcut_values["min_same_image_accuracy"].append(
                checks["same_image_cross_vs_avoid"]["factorized_no_cf"]["accuracy"]
            )
    aggregate["shortcuts"] = {key: float(min(values)) for key, values in shortcut_values.items()}
    appearance = aggregate["splits"]["appearance_ood"]["factorized_no_cf"]
    joint = aggregate["splits"]["joint_ood"]["factorized_no_cf"]
    aggregate["objective"] = {
        "appearance_ood_regret": appearance["regret"],
        "worst_dev_regret": max(float(appearance["regret"]), float(joint["regret"])),
        "worst_dev_false_safe": max(float(appearance["false_safe"]), float(joint["false_safe"])),
        "worst_dev_pair_ranking": min(float(appearance["pair_ranking"]), float(joint["pair_ranking"])),
        "worst_dev_field_miou": min(float(appearance["field_miou"]), float(joint["field_miou"])),
    }
    return aggregate
