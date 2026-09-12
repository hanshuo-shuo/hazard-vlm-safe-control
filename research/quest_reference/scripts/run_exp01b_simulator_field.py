#!/usr/bin/env python3
"""Train and evaluate EXP-01B simulator-supervised spatial requirement fields."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.oracle_spatial_exposure import DenseCost, FactorizedCost  # noqa: E402
from evaluation.simulator_spatial_field import (  # noqa: E402
    APPEARANCE_SPECS,
    SCHEMA_VERSION,
    FieldTrainingConfig,
    bootstrap_comparisons,
    build_rgb_scenes,
    evaluate_split,
    field_metrics,
    fit_cards_only,
    irrelevant_swap_check,
    predict_fields,
    predict_rgb_arrays,
    render_requirement_rgb,
    same_image_paired_check,
    stable_hash,
    terrain_erasure_check,
    train_field_model,
)


SPLIT_FILE = ROOT / "configs" / "exp01b_splits.json"
RELATION_CHECKPOINT = ROOT / "results" / "exp01ad_retrained_replication" / "checkpoints" / "factorized_no_cf.pt"
FULL_CHECKPOINT = ROOT / "results" / "exp01ad_retrained_replication" / "checkpoints" / "factorized_full.pt"
DENSE_CHECKPOINT = ROOT / "results" / "exp01ad_retrained_replication" / "checkpoints" / "dense_cost.pt"
EXP01AD_RESULTS = ROOT / "results" / "exp01ad_retrained_replication" / "RESULTS.json"
DEFAULT_OUTPUT = ROOT / "results" / "exp01b_simulator_field"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_splits(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported EXP-01B split schema")
    required = {"train", "validation", "iid", "appearance_ood", "joint_ood"}
    if set(config.get("splits", {})) != required:
        raise ValueError("five frozen splits required")
    train = set(config["splits"]["train"]["appearance_families"])
    appearance = set(config["splits"]["appearance_ood"]["appearance_families"])
    joint = set(config["splits"]["joint_ood"]["appearance_families"])
    if train & appearance or train & joint or appearance & joint:
        raise ValueError("OOD appearance families must be strictly disjoint")
    if config["splits"]["joint_ood"].get("primary_held_out_card") != "D":
        raise ValueError("joint-OOD held-out card changed")
    if "D" in config["splits"]["train"]["risk_cards_seen"]:
        raise ValueError("joint-OOD card cannot be seen in EXP-01B train")


def _load_model(path: Path, model: torch.nn.Module) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite EXP-01B artifact: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _field_config(config: Mapping[str, Any]) -> FieldTrainingConfig:
    value = config["training"]
    return FieldTrainingConfig(
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        batch_size=int(value["batch_size"]),
        max_epochs=int(value["max_epochs"]),
        patience=int(value["patience"]),
    )


def _truths(scenes):
    from evaluation.simulator_spatial_field import risk_truth
    return [risk_truth(item.exposures) for item in scenes]


def _combined_report(exp01ad: Mapping[str, Any], exp01b: Mapping[str, Any]) -> str:
    joint = exp01b["splits"]["joint_ood"]["summary"]
    lines = [
        "# EXP-01A-D-R + EXP-01B Combined Result",
        "",
        "## Bottom line",
        "",
        exp01b["conclusion"],
        "",
        "## EXP-01A-D-R decision result",
        "",
        "| Model | Regret ↓ | Dangerous selection ↓ | Pair ranking ↑ | Cost MAE ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("dense_cost", "factorized_no_cf", "factorized_full", "oracle_formula"):
        item = exp01ad["metrics"][arm]
        lines.append(
            f"| {arm} | {item['action_regret']:.6f} | {item['dangerous_selection_rate']:.6f} | "
            f"{item['pairwise_ranking_accuracy']:.6f} | {item['semantic_cost_mae']:.6f} |"
        )
    lines.extend([
        "",
        "## EXP-01B joint-OOD",
        "",
        "| Model | Field mIoU ↑ | Exposure MAE ↓ | Pair ranking ↑ | False-safe ↓ | Regret ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ("cards_only", "global_presence", "dense_cost", "factorized_no_cf", "factorized_full", "oracle_field"):
        item = joint[arm]
        field = "—" if item["field_miou"] is None else f"{item['field_miou']:.4f}"
        exposure = "—" if item["exposure_mae"] is None else f"{item['exposure_mae']:.6f}"
        lines.append(
            f"| {arm} | {field} | {exposure} | {item['pair_ranking']:.4f} | "
            f"{item['false_safe']:.4f} | {item['regret']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _refuse_nonempty(output)
    config = read_json(SPLIT_FILE)
    audit_splits(config)
    image_size, field_size = int(config["image_size"]), int(config["field_size"])
    datasets = {
        split: build_rgb_scenes(
            split, spec, image_size=image_size, field_size=field_size
        )
        for split, spec in config["splits"].items()
    }
    split_audit = {
        split: {
            "scene_count": len(scenes),
            "scene_ids": [item.scene.scene_id for item in scenes],
            "families": sorted({item.family for item in scenes}),
            "rgb_sha256": hashlib.sha256(b"".join(item.rgb.tobytes() for item in scenes)).hexdigest(),
            "field_sha256": hashlib.sha256(b"".join(item.field_target.tobytes() for item in scenes)).hexdigest(),
        }
        for split, scenes in datasets.items()
    }
    write_json(output / "SPLITS.json", {
        "schema_version": SCHEMA_VERSION,
        "source_split_sha256": file_sha256(SPLIT_FILE),
        "splits": split_audit,
    })

    field_model, history = train_field_model(
        datasets["train"], datasets["validation"],
        seed=int(config["seed"]), config=_field_config(config),
    )
    field_checkpoint = output / "field_resnet18.pt"
    torch.save(field_model.state_dict(), field_checkpoint)
    factorized = _load_model(RELATION_CHECKPOINT, FactorizedCost())
    factorized_full = _load_model(FULL_CHECKPOINT, FactorizedCost())
    dense = _load_model(DENSE_CHECKPOINT, DenseCost())
    cards_coefficients = fit_cards_only(
        datasets["train"], config["splits"]["train"]["risk_cards_seen"]
    )

    result_splits = {}
    saved_predictions = {}
    evaluation = config["evaluation"]
    for split in ("iid", "appearance_ood", "joint_ood"):
        scenes = datasets[split]
        predicted_fields = predict_fields(field_model, scenes)
        card_ids = (
            (config["splits"][split]["primary_held_out_card"],)
            if split == "joint_ood"
            else tuple(config["splits"][split]["evaluation_cards"])
        )
        summary, arrays, risk_predictions = evaluate_split(
            scenes,
            predicted_fields,
            factorized=factorized,
            factorized_full=factorized_full,
            dense=dense,
            cards_coefficients=cards_coefficients,
            card_ids=card_ids,
            false_safe_threshold=float(evaluation["false_safe_threshold"]),
            dangerous_margin=float(evaluation["dangerous_excess_margin"]),
        )
        permutation = np.roll(np.arange(len(scenes)), 1)
        shuffled_fields = predict_rgb_arrays(field_model, np.asarray([scenes[index].rgb for index in permutation]))
        shuffled_summary, _shuffled_arrays, _shuffled_risk = evaluate_split(
            scenes,
            shuffled_fields,
            factorized=factorized,
            factorized_full=factorized_full,
            dense=dense,
            cards_coefficients=cards_coefficients,
            card_ids=card_ids,
            false_safe_threshold=float(evaluation["false_safe_threshold"]),
            dangerous_margin=float(evaluation["dangerous_excess_margin"]),
        )
        target = np.asarray([item.field_target for item in scenes])
        normal_field = field_metrics(predicted_fields, target)
        shuffled_field = field_metrics(shuffled_fields, target)
        checks = {
            "same_image_cross_vs_avoid": same_image_paired_check(risk_predictions, _truths(scenes)),
            "image_shuffle": {
                "normal_field_miou": normal_field["field_miou"],
                "shuffled_field_miou": shuffled_field["field_miou"],
                "delta_field_miou": normal_field["field_miou"] - shuffled_field["field_miou"],
                "normal_factorized_pair_ranking": summary["factorized_no_cf"]["pair_ranking"],
                "shuffled_factorized_pair_ranking": shuffled_summary["factorized_no_cf"]["pair_ranking"],
                "delta_pair_ranking": summary["factorized_no_cf"]["pair_ranking"] - shuffled_summary["factorized_no_cf"]["pair_ranking"],
            },
            "terrain_erasure": terrain_erasure_check(
                field_model,
                scenes[:int(evaluation["erasure_scene_count_per_split"])],
                predicted_fields[:int(evaluation["erasure_scene_count_per_split"])],
                image_size=image_size,
            ),
            "irrelevant_swap": irrelevant_swap_check(
                field_model,
                factorized,
                family=scenes[0].family,
                seed=int(config["splits"][split]["scene_seed"]) + 7000,
                scene_count=int(evaluation["irrelevant_swap_scene_count_per_split"]),
                image_size=image_size,
                field_size=field_size,
            ),
        }
        result_splits[split] = {"summary": summary, "checks": checks}
        if split == "joint_ood":
            result_splits[split]["paired_bootstrap"] = bootstrap_comparisons(
                arrays,
                seed=int(evaluation["bootstrap_seed"]),
                resamples=int(evaluation["bootstrap_resamples"]),
            )
        saved_predictions[f"{split}_field"] = predicted_fields
        for arm, values in risk_predictions.items():
            saved_predictions[f"{split}_{arm}_risk"] = values

    comparisons = result_splits["joint_ood"]["paired_bootstrap"]
    gate_pass = all(
        comparisons[baseline][metric]["paired_scene_bootstrap_ci95"][0] > 0
        for baseline in ("cards_only", "global_presence", "dense_cost")
        for metric in ("dangerous_inversion", "false_safe", "regret")
    )
    factor = result_splits["joint_ood"]["summary"]["factorized_no_cf"]
    conclusion = (
        "EXP-01B supports RGB-grounded spatial requirements: the simulator-supervised field drives "
        "Factorized/no-CF to the best non-oracle joint-OOD decision performance."
        if gate_pass else
        "EXP-01B does not satisfy the strict all-baseline/all-primary continuation condition; inspect "
        "the paired decision metrics and shortcut checks rather than field mIoU alone."
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "EXP-01B",
        "training": history,
        "splits": result_splits,
        "joint_ood_gate_passed": gate_pass,
        "program_decision": "CLOSED_LOOP_AUTHORIZED" if gate_pass else "CLOSED_LOOP_NOT_AUTHORIZED",
        "conclusion": conclusion,
        "joint_factorized_snapshot": factor,
    }
    prediction_path = output / "PREDICTIONS.npz"
    np.savez_compressed(prediction_path, **saved_predictions)
    write_json(output / "RESULTS.json", result)
    write_json(output / "MANIFEST.json", {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "EXP-01B",
        "split_file_sha256": file_sha256(SPLIT_FILE),
        "field_checkpoint_sha256": file_sha256(field_checkpoint),
        "relation_checkpoint_sha256": file_sha256(RELATION_CHECKPOINT),
        "full_checkpoint_sha256": file_sha256(FULL_CHECKPOINT),
        "dense_checkpoint_sha256": file_sha256(DENSE_CHECKPOINT),
        "predictions_sha256": file_sha256(prediction_path),
        "rgb_supervision": "simulator_oracle_fields_only",
        "vlm_calls": 0,
        "rl_steps": 0,
    })
    exp01ad = read_json(EXP01AD_RESULTS)
    (output / "COMBINED_REPORT.md").write_text(_combined_report(exp01ad, result), encoding="utf-8")
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "decision": outcome["program_decision"],
        "joint_ood_gate_passed": outcome["joint_ood_gate_passed"],
        "conclusion": outcome["conclusion"],
    }, indent=2))
