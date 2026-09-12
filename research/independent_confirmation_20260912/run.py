"""One recipe/seed; checkpoint and calibration are persisted before test access."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import traceback
import numpy as np
import torch
from common import HERE, reference, protocol, verify_freeze, write_json, manifest, sha
from generator import exposures, formula
from metrics import per_scene, summarize, field_counts, calibrate, upper_cost, coverage_counts


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def relation(model, values):
    flat = values.reshape(-1, 2)
    cards = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    with torch.inference_mode():
        pred = model(torch.tensor(np.repeat(flat, 4, axis=0), dtype=torch.float32),
                     torch.tensor(np.tile(cards, (len(flat), 1)), dtype=torch.float32)).numpy()
    return pred.reshape(*values.shape[:-1], 4)


def main(root, task_index):
    p = protocol()
    verify_freeze(root)
    recipe = p["recipes"][task_index // len(p["model_seeds"])]
    index = task_index % len(p["model_seeds"])
    seed = p["model_seeds"][index]
    out = root / "runs" / recipe / f"seed_{index}"
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated Quest GPU is required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.monotonic()
    events = []
    data_manifest = json.loads((root / "datasets/MANIFEST.json").read_text())

    def load(name):
        path = root / "datasets" / f"{name}.npz"
        if sha(path) != data_manifest[path.name]:
            raise ValueError(f"Dataset hash mismatch: {name}")
        events.append({"event": f"load:{name}", "at_utc": timestamp()})
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    def predict(model, data):
        shape = data["rgb"].shape[:2]
        fields = reference.predict_rgb_arrays(model, data["rgb"].reshape(-1, 64, 64, 3), device=device)
        fields = fields.reshape(*shape, 2, 16, 16)
        return fields, exposures(fields, data["footprints"])

    try:
        write_json(out / "START.json", {"at_utc": timestamp(), "recipe": recipe, "model_seed": seed,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": platform.python_version(),
                   "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(), "protocol_sha256": sha(HERE / "protocol.json"),
                   "freeze_sha256": sha(root / "FREEZE.json"), "dataset_manifest_sha256": sha(root / "datasets/MANIFEST.json")})
        train, validation = load(f"{recipe}_train"), load(f"{recipe}_validation")
        model, history = reference.train_candidate(
            [SimpleNamespace(rgb=x[0], field_target=y[0]) for x, y in zip(train["rgb"], train["target"])],
            [SimpleNamespace(rgb=x[0], field_target=y[0]) for x, y in zip(validation["rgb"], validation["target"])],
            seed=seed, device=device, budget=p["budget"])
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "visual_field.pt")
        write_json(out / "TRAINING.json", history)
        events.append({"event": "checkpoint_exported", "at_utc": timestamp()})
        print(recipe, seed, "trained", history["epochs"], history["elapsed_seconds"], flush=True)
        calibration = load("calibration")
        cfields, cexposure = predict(model, calibration)
        cpred = formula(cexposure)
        offset, scores, rank = calibrate(calibration["truth"], cpred, p["calibration_rule"]["alpha"])
        np.savez_compressed(out / "CALIBRATION.npz", fields=cfields, prediction=cpred, scores=scores)
        write_json(out / "CALIBRATION.json", {"at_utc": timestamp(), "offset": offset, "rank": rank,
                                             "n_scene_pairs": len(scores), "alpha": .05,
                                             "test_accessed": False, "checkpoint_sha256": sha(out / "visual_field.pt")})
        events.append({"event": "calibration_persisted", "at_utc": timestamp()})
        config = json.loads((reference.REFERENCE / "autoresearch/exp01b_r2/config.json").read_text())
        fixed = reference.load_fixed_models(config)
        mean_field = train["target"].mean((0, 1))
        constant = formula(train["exposure"].mean((0, 1, 2)))
        results, metrics = {}, {}
        for split in p["test_splits"]:
            data = load(split)
            fields, exposure = predict(model, data)
            prior_fields = np.broadcast_to(mean_field, fields.shape)
            # Within-variant cyclic image shuffle, retaining the queried footprints.
            shuffled_exposure = exposures(np.roll(fields, 1, axis=0), data["footprints"])
            predictions = {"formula": formula(exposure), "learned_no_cf": relation(fixed["factorized_no_cf"], exposure),
                           "learned_cf": relation(fixed["factorized_full"], exposure),
                           "mean_field_no_rgb": formula(exposures(prior_fields, data["footprints"])),
                           "cards_only": np.broadcast_to(constant, data["truth"].shape).copy(),
                           "shuffled_rgb": formula(shuffled_exposure), "oracle_formula": formula(data["exposure"])}
            saved = {"fields": fields, "exposure": exposure, **predictions}
            counts = field_counts(fields, data["target"])
            for k, v in counts.items():
                metrics[f"{split}__field__{k}"] = v
            results[split] = {"arms": {}, "field_miou": float((counts["intersection"].sum((0, 1)) /
                                          np.maximum(counts["union"].sum((0, 1)), 1)).mean()),
                              "original_field_miou": float((counts["intersection"][:, 0].sum(0) /
                                          np.maximum(counts["union"][:, 0].sum(0), 1)).mean())}
            for arm, prediction in predictions.items():
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(arm)
                rows = per_scene(data["truth"], prediction, p)
                results[split]["arms"][arm] = summarize(rows)
                for k, v in rows.items():
                    metrics[f"{split}__{arm}__{k}"] = v
            upper = upper_cost(predictions["formula"], offset)
            saved["calibrated_formula"] = upper
            for name, pred in [("raw", predictions["formula"]), ("calibrated", upper)]:
                metrics[f"{split}__{name}__coverage"] = coverage_counts(data["truth"], pred, p)
                metrics[f"{split}__{name}__pair_bound_failure"] = (data["truth"][..., [0, 2, 3]] >
                     pred[..., [0, 2, 3]] + 1e-12).any((1, 2, 3)).astype(int)
            np.savez_compressed(out / f"{split}_PREDICTIONS.npz", **saved)
            print(recipe, seed, split, results[split]["arms"]["formula"], flush=True)
        np.savez_compressed(out / "METRICS.npz", **metrics)
        write_json(out / "EVENTS.json", events)
        write_json(out / "RESULTS.json", {"status": "COMPLETE", "recipe": recipe, "model_seed": seed,
                  "at_utc": timestamp(), "elapsed_seconds": time.monotonic() - started,
                  "offset": offset, "checkpoint_sha256": sha(out / "visual_field.pt"), "splits": results})
        write_json(out / "MANIFEST.json", manifest(out))
    except Exception:
        write_json(out / "FAILURE.json", {"at_utc": timestamp(), "traceback": traceback.format_exc(), "events": events})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True, choices=range(15))
    args = parser.parse_args()
    main(args.root, args.task_index)
