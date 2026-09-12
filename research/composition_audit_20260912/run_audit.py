#!/usr/bin/env python3
"""New diagnostic around byte-preserved Quest reference code, not a legacy edit."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import types

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "quest_reference"


def install_reference_namespace():
    # Standalone process only: bypass the reference package's broad __init__,
    # whose unrelated controller dependencies are deliberately not imported.
    for name, path in (("evaluation", REFERENCE / "evaluation"),
                       ("autoresearch", REFERENCE / "autoresearch"),
                       ("scripts", REFERENCE / "scripts")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


install_reference_namespace()
from autoresearch.exp01b_r2.train import train_candidate
from autoresearch.exp01b_r2.evaluator import load_fixed_models, predict_rgb_arrays
from evaluation.simulator_spatial_field import (
    build_rgb_scenes, lowres_field, pooled_exposure, render_requirement_rgb,
    risk_truth, relation_predictions, dense_predictions, field_metrics,
    fit_cards_only, cards_only_predictions, _score_scene,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_reference(protocol):
    manifest = json.loads((REFERENCE / "autoresearch/exp01b_r2/protected_hashes.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(REFERENCE / name) != expected:
            raise ValueError(f"Protected reference mismatch: {name}")
    if sha(REFERENCE / "autoresearch/exp01b_r2/train.py") != protocol["reference_train_sha256"]:
        raise ValueError("Captured training recipe changed")


@dataclass(frozen=True)
class Probe:
    pair_id: str
    family: str
    rgb: np.ndarray
    field_target: np.ndarray
    footprints: np.ndarray
    exposures: np.ndarray
    permutation: np.ndarray


def make_pairs(base, seed):
    originals, swapped = [], []
    rng = np.random.default_rng(seed)
    for index, item in enumerate(base):
        permutation = rng.permutation(4)
        footprints = item.footprints[permutation].copy()
        for is_swapped, destination in ((False, originals), (True, swapped)):
            fields = item.scene.fields[::-1].copy() if is_swapped else item.scene.fields
            target = lowres_field(fields, 16)
            rgb = render_requirement_rgb(fields, family=item.family, image_size=64,
                                         render_seed=item.render_seed)
            exposures = np.asarray([pooled_exposure(target, f) for f in footprints])
            destination.append(Probe(f"{seed}:{index}", item.family, rgb, target,
                                     footprints, exposures, permutation))
        assert np.array_equal(originals[-1].footprints, swapped[-1].footprints)
        assert np.array_equal(originals[-1].field_target[::-1], swapped[-1].field_target)
        assert np.allclose(originals[-1].exposures[:, ::-1], swapped[-1].exposures,
                           atol=1e-12, rtol=0)
        assert not np.array_equal(originals[-1].rgb, swapped[-1].rgb)
    return originals, swapped


def score_arrays(truth, predictions, cards, protocol):
    result = []
    for t, p in zip(truth, predictions):
        row = _score_scene(t, p, card_indices=cards,
                           false_safe_threshold=protocol["false_safe_threshold"],
                           dangerous_margin=protocol["dangerous_excess_margin"])
        eligible = t[:, cards] > protocol["false_safe_threshold"]
        row["false_safe_eligible"] = int(eligible.sum())
        row["false_safe_count"] = int(((p[:, cards] <= protocol["false_safe_threshold"]) & eligible).sum())
        result.append(row)
    arrays = {key: np.asarray([row[key] for row in result]) for key in result[0]}
    summary = {key: float(value.mean()) for key, value in arrays.items()
               if key not in ("false_safe_count", "false_safe_eligible")}
    eligible = int(arrays["false_safe_eligible"].sum())
    count = int(arrays["false_safe_count"].sum())
    summary.update(false_safe_eligible=eligible, false_safe_count=count,
                   false_safe_micro=(count / eligible if eligible else None))
    return summary, arrays


def paired_correct(truth_original, truth_swapped, pred_original, pred_swapped):
    # Probability that two independently uniform tie-resolving decisions both
    # choose their own unique optimum; do not reward arbitrary argmin ordering.
    valid, scores = [], []
    for t0, t1, p0, p1 in zip(truth_original, truth_swapped, pred_original, pred_swapped):
        row_valid, row_score = [], []
        for card in range(4):
            b0 = np.flatnonzero(np.isclose(t0[:, card], t0[:, card].min(), atol=1e-12, rtol=0))
            b1 = np.flatnonzero(np.isclose(t1[:, card], t1[:, card].min(), atol=1e-12, rtol=0))
            v = len(b0) == len(b1) == 1 and b0[0] != b1[0]
            row_valid.append(v)
            if not v:
                row_score.append(0.0)
                continue
            a0 = np.flatnonzero(np.isclose(p0[:, card], p0[:, card].min(), atol=1e-9, rtol=0))
            a1 = np.flatnonzero(np.isclose(p1[:, card], p1[:, card].min(), atol=1e-9, rtol=0))
            row_score.append(float(b0[0] in a0) / len(a0) * float(b1[0] in a1) / len(a1))
        valid.append(row_valid)
        scores.append(row_score)
    return np.asarray(valid), np.asarray(scores)


def predict_arms(model, probes, fixed, mean_field, role_risk, cards_risk, device):
    predicted_fields = predict_rgb_arrays(model, np.asarray([p.rgb for p in probes]), device=device)
    truth, predictions, exposures = [], {}, []
    for probe, field in zip(probes, predicted_fields):
        e = np.asarray([pooled_exposure(field, f) for f in probe.footprints])
        prior_e = np.asarray([pooled_exposure(mean_field, f) for f in probe.footprints])
        t = risk_truth(probe.exposures)
        row = {
            "cards_only": cards_risk,
            "role_metadata_diagnostic": role_risk[probe.permutation],
            "mean_field_no_rgb": risk_truth(prior_e),
            "dense_cost": dense_predictions(fixed["dense_cost"], field, probe.footprints),
            "factorized_no_cf": relation_predictions(fixed["factorized_no_cf"], e),
            "factorized_full": relation_predictions(fixed["factorized_full"], e),
            "predicted_formula": risk_truth(e),
            "oracle_formula": t.copy(),
            "oracle_learned": relation_predictions(fixed["factorized_no_cf"], probe.exposures),
        }
        truth.append(t)
        exposures.append(e)
        for arm, prediction in row.items():
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite predictions for {arm}")
            predictions.setdefault(arm, []).append(prediction)
    predictions = {k: np.asarray(v) for k, v in predictions.items()}
    return np.asarray(truth), predictions, predicted_fields, np.asarray(exposures)


def run(seed_index, out, protocol):
    verify_reference(protocol)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training audit requires allocated GPU; use --self-test on CPU")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = protocol["model_seeds"][seed_index]
    started = time.monotonic()
    write_json(out / "START.json", {"seed": seed, "protocol_sha256": sha(HERE / "protocol.json"),
                                   "runner_sha256": sha(__file__), "python": platform.python_version(),
                                   "torch": torch.__version__, "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(), "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID")})
    train = build_rgb_scenes("train", protocol["train"], image_size=64, field_size=16)
    validation = build_rgb_scenes("validation", protocol["validation"], image_size=64, field_size=16)
    print(f"seed={seed} data ready; training captured recipe", flush=True)
    model, history = train_candidate(train, validation, seed=seed, device=device, budget=protocol["budget"])
    checkpoint = out / "visual_field.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    write_json(out / "TRAINING.json", history)
    print(f"seed={seed} trained {history['epochs']} epochs in {history['elapsed_seconds']:.1f}s", flush=True)
    config = json.loads((REFERENCE / "autoresearch/exp01b_r2/config.json").read_text())
    fixed = load_fixed_models(config)
    mean_field = np.asarray([x.field_target for x in train]).mean(axis=0)
    role_risk = np.asarray([risk_truth(x.exposures) for x in train]).mean(axis=0)
    cards_risk = cards_only_predictions(fit_cards_only(train, ["A", "B", "C"]))
    results, metrics_arrays, saved = {}, {}, {}
    fingerprints = {}
    for split, spec in protocol["fresh_splits"].items():
        base = build_rgb_scenes(split, spec, image_size=64, field_size=16)
        original, swapped = make_pairs(base, spec["scene_seed"])
        split_results, paired_predictions, paired_truth = {}, {}, {}
        for variant, probes in (("original", original), ("swapped", swapped)):
            key = f"{split}__{variant}"
            truth, predictions, fields, exposure = predict_arms(
                model, probes, fixed, mean_field, role_risk, cards_risk, device)
            target = np.asarray([p.field_target for p in probes])
            footprints = np.asarray([p.footprints for p in probes])
            fingerprints[key] = {
                "rgb_sha256": hashlib.sha256(np.asarray([p.rgb for p in probes]).tobytes()).hexdigest(),
                "field_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                "footprint_sha256": hashlib.sha256(footprints.tobytes()).hexdigest(),
                "pair_ids": [p.pair_id for p in probes],
                "families": [p.family for p in probes],
            }
            split_results[variant] = {"field": field_metrics(fields, target), "arms": {}}
            saved[f"{key}__truth"] = truth
            saved[f"{key}__fields"] = fields
            saved[f"{key}__target"] = target
            saved[f"{key}__footprints"] = footprints
            saved[f"{key}__exposure"] = exposure
            saved[f"{key}__permutation"] = np.asarray([p.permutation for p in probes])
            for arm, prediction in predictions.items():
                saved[f"{key}__{arm}__prediction"] = prediction
                split_results[variant]["arms"][arm] = {}
                for card_name, cards in (("all", [0, 1, 2, 3]), ("D", [3])):
                    summary, arrays = score_arrays(truth, prediction, cards, protocol)
                    split_results[variant]["arms"][arm][card_name] = summary
                    for metric, values in arrays.items():
                        metrics_arrays[f"{key}__{arm}__{card_name}__{metric}"] = values
            paired_predictions[variant], paired_truth[variant] = predictions, truth
            # Save an identical representative pair per split, only once across seeds.
            if seed_index == 0:
                from PIL import Image
                for i in range(4):
                    Image.fromarray(probes[i].rgb).save(out / f"{key}__example_{i}.png")
        split_results["paired"] = {}
        for arm in protocol["arms"]:
            valid, correct = paired_correct(paired_truth["original"], paired_truth["swapped"],
                                            paired_predictions["original"][arm], paired_predictions["swapped"][arm])
            split_results["paired"][arm] = {
                "eligible_changed_card_pairs": int(valid.sum()),
                "both_correct_probability": float(correct.sum() / valid.sum()) if valid.any() else None,
            }
            metrics_arrays[f"{split}__paired__{arm}__eligible"] = valid
            metrics_arrays[f"{split}__paired__{arm}__both_correct"] = correct
        results[split] = split_results
        print(f"seed={seed} evaluated {split}: {split_results['paired']['predicted_formula']}", flush=True)
    np.savez_compressed(out / "PER_SCENE_METRICS.npz", **metrics_arrays)
    np.savez_compressed(out / "PREDICTIONS.npz", **saved)
    write_json(out / "DATASET_AUDIT.json", fingerprints)
    write_json(out / "RESULTS.json", {"status": "COMPLETE", "experiment_id": protocol["experiment_id"],
                                    "seed": seed, "checkpoint_sha256": sha(checkpoint),
                                    "protocol_sha256": sha(HERE / "protocol.json"),
                                    "elapsed_seconds": time.monotonic() - started, "splits": results})
    write_json(out / "MANIFEST.json", {f.name: sha(f) for f in out.iterdir() if f.is_file()})
    print(f"seed={seed} COMPLETE", flush=True)


def self_test(protocol):
    verify_reference(protocol)
    torch.set_num_threads(2)
    base = build_rgb_scenes("self_test", {"scene_seed": 92817, "scene_count": 12,
                            "appearance_families": ["aqua_sand", "violet_night"]}, image_size=64, field_size=16)
    original, swapped = make_pairs(base, 7182)
    t0 = np.asarray([risk_truth(x.exposures) for x in original])
    t1 = np.asarray([risk_truth(x.exposures) for x in swapped])
    summary, _ = score_arrays(t0, t0, [0, 1, 2, 3], protocol)
    assert summary["regret"] == 0 and summary["false_safe_count"] == 0
    valid, correct = paired_correct(t0, t1, t0, t1)
    assert valid.any() and np.array_equal(correct[valid], np.ones(valid.sum()))
    _, blind = paired_correct(t0, t1, t0, t0)
    assert blind.sum() == 0  # unchanged unique choice cannot solve changed optimum
    _, ties = paired_correct(t0, t1, np.zeros_like(t0), np.zeros_like(t1))
    assert np.allclose(ties[valid], 1 / 16)
    from evaluation.oracle_spatial_exposure import CARDS, CARD_IDS, semantic_cost
    for exposure in [np.zeros((4, 2)), np.ones((4, 2)), np.random.default_rng(82).uniform(size=(4, 2))]:
        expected = np.asarray([[semantic_cost(*e, *CARDS[c]) for c in CARD_IDS] for e in exposure])
        np.testing.assert_allclose(risk_truth(exposure), expected, atol=1e-7)
    # Candidate reordering must not change any metric.
    p = np.random.default_rng(44).uniform(size=t0.shape)
    left, _ = score_arrays(t0, p, [0, 1, 2, 3], protocol)
    right, _ = score_arrays(t0[:, ::-1], p[:, ::-1], [0, 1, 2, 3], protocol)
    for key in left:
        if left[key] is not None:
            assert np.isclose(left[key], right[key])
    print(json.dumps({"status": "PASS", "protected_hashes": 15, "paired_scenes": 12,
                      "checks": ["field/exposure swap", "fixed footprints", "changed RGB", "oracle zero error",
                                 "pair eligibility", "blind failure", "uniform ties", "analytic formula", "candidate permutation"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-index", type=int, choices=range(3))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if args.self_test:
        self_test(protocol)
    elif args.seed_index is not None and args.output:
        run(args.seed_index, args.output.resolve(), protocol)
    else:
        parser.error("Provide --self-test or --seed-index and --output")
