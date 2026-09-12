"""Independent slow reconstruction plus Quest CPU checkpoint replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import protocol, reference, verify_freeze, verify_manifest, write_json, sha
from autoresearch.exp01b_r2.train import build_candidate


def recompute(truth, pred):
    regrets, original, false, eligible, correct, valid = [], [], [], [], [], []
    for t, p in zip(truth, pred):
        r, o, successes, keep = [], [], 0., 0
        for c in [0, 2, 3]:
            bests, choices, qualifying = [], [], True
            for v in range(2):
                ids = np.flatnonzero(np.abs(p[v, :, c] - p[v, :, c].min()) <= 1e-9)
                delta = t[v, ids, c].mean() - t[v, :, c].min()
                r.append(delta)
                if v == 0:
                    o.append(delta)
                order = np.argsort(t[v, :, c])
                qualifying &= t[v, order[1], c] - t[v, order[0], c] >= .002
                bests.append(order[0])
                choices.append(ids)
            qualifying &= bests[0] != bests[1]
            if qualifying:
                keep += 1
                successes += float(bests[0] in choices[0]) / len(choices[0]) * float(bests[1] in choices[1]) / len(choices[1])
        e = t[..., [0, 2, 3]] > .02
        false.append(int((e & (p[..., [0, 2, 3]] <= .02)).sum()))
        eligible.append(int(e.sum()))
        regrets.append(np.mean(r)); original.append(np.mean(o))
        correct.append(successes); valid.append(keep)
    return {"regret": regrets, "original_regret": original, "false_safe_count": false,
            "false_safe_eligible": eligible, "paired_correct_sum": correct, "paired_eligible": valid}


def main(root):
    p = protocol()
    verify_freeze(root)
    torch.set_num_threads(4)
    data = {s: np.load(root / "datasets" / f"{s}.npz", allow_pickle=False) for s in p["test_splits"]}
    calibration = np.load(root / "datasets/calibration.npz", allow_pickle=False)
    for split, d in data.items():
        target_exposure = (d["target"][:, :, None].astype(float) * d["footprints"][:, None, :, None]).sum((-2, -1))
        target_exposure /= d["footprints"].astype(float).sum((-2, -1))[:, None, :, None]
        target_truth = np.asarray([[reference.risk_truth(e) for e in scene] for scene in target_exposure])
        np.testing.assert_allclose(target_truth, d["truth"], atol=1e-12, rtol=0)
    checked, replays = 0, []
    for recipe in p["recipes"]:
        for i, seed in enumerate(p["model_seeds"]):
            run = root / "runs" / recipe / f"seed_{i}"
            checked += verify_manifest(run)
            metrics = np.load(run / "METRICS.npz", allow_pickle=False)
            info = json.loads((run / "CALIBRATION.json").read_text())
            cal = np.load(run / "CALIBRATION.npz", allow_pickle=False)
            errors = np.maximum(calibration["truth"][..., [0, 2, 3]] - cal["prediction"][..., [0, 2, 3]], 0).reshape(300, -1).max(1)
            np.testing.assert_allclose(info["offset"], np.sort(errors)[285], atol=1e-15)
            for split, d in data.items():
                saved = np.load(run / f"{split}_PREDICTIONS.npz", allow_pickle=False)
                # Independent multiplication/summation rather than the runner's einsum.
                exposure = (saved["fields"][:, :, None].astype(float) * d["footprints"][:, None, :, None]).sum((-2, -1))
                exposure /= d["footprints"].astype(float).sum((-2, -1))[:, None, :, None]
                np.testing.assert_allclose(exposure, saved["exposure"], atol=1e-12, rtol=0)
                reconstructed = np.asarray([[reference.risk_truth(e) for e in scene] for scene in exposure])
                np.testing.assert_allclose(reconstructed, saved["formula"], atol=1e-12, rtol=0)
                for arm in p["arms"]:
                    rows = recompute(d["truth"], saved[arm])
                    for k, value in rows.items():
                        np.testing.assert_allclose(value, metrics[f"{split}__{arm}__{k}"], atol=1e-12, rtol=0)
                upper = np.minimum(saved["formula"] + info["offset"], 1)
                upper[..., 1] = 0
                np.testing.assert_array_equal(upper, saved["calibrated_formula"])
                # Verify coverage/conditional-unsafe counts from saved cost arrays.
                for name, pred in [("raw", saved["formula"]), ("calibrated", upper)]:
                    rows = np.zeros((len(pred), len(p["coverage_thresholds"]), 3))
                    for n in range(len(pred)):
                        for v in range(2):
                            for c in [0, 2, 3]:
                                best = pred[n, v, :, c].min()
                                ids = np.flatnonzero(abs(pred[n, v, :, c] - best) <= 1e-9)
                                for j, threshold in enumerate(p["coverage_thresholds"]):
                                    if best <= threshold:
                                        rows[n, j] += [1, (d["truth"][n, v, ids, c] > .02).mean(), d["truth"][n, v, ids, c].mean()]
                    np.testing.assert_allclose(rows, metrics[f"{split}__{name}__coverage"], atol=1e-12, rtol=0)
                saved.close()
            model = build_candidate(seed).cpu()
            model.load_state_dict(torch.load(run / "visual_field.pt", weights_only=True, map_location="cpu"))
            d = data["random_appearance"]
            replay = reference.predict_rgb_arrays(model, d["rgb"][:2].reshape(-1, 64, 64, 3), device=torch.device("cpu"))
            gpu = np.load(run / "random_appearance_PREDICTIONS.npz", allow_pickle=False)["fields"][:2].reshape(replay.shape)
            error = float(np.max(abs(replay - gpu)))
            binary = float(np.mean((replay >= .5) == (gpu >= .5)))
            # Compatibility criteria fixed in this source before training; GPU arrays remain authoritative.
            if error > .01 or binary < .995:
                raise ValueError(f"Checkpoint numerical compatibility failed: {recipe}/{i}: {error}, {binary}")
            replays.append({"recipe": recipe, "seed": seed, "max_pixel_difference": error,
                            "binary_agreement": binary, "checkpoint_sha256": sha(run / "visual_field.pt")})
            print("verified", recipe, seed, error, binary, flush=True)
    write_json(root / "reports/VERIFICATION.json", {"status": "PASS", "verified_run_files": checked,
               "models": len(replays), "replayed_images": len(replays) * 4,
               "replay_device": "Quest allocated CPU", "gpu_predictions_replaced": False,
               "metrics_independently_recomputed": ["regret", "original_regret", "false_safe counts", "paired correctness", "coverage/unsafe/cost counts", "formula", "calibration order statistic"],
               "replays": replays})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
