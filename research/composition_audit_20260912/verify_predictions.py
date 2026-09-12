#!/usr/bin/env python3
"""Independent array checks plus CPU replay of all six exported visual models."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from PIL import Image
import run_audit as audit
from autoresearch.exp01b_r2.train import CandidateFieldModel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/c3_composition_audit_20260912"


def formula(e):
    w, f = e[..., 0], e[..., 1]
    return np.stack((w, np.zeros_like(w), w + f - w*f, f), axis=-1)


def main():
    torch.set_num_threads(2)
    records = []
    for group in ("runs", "repair_runs"):
        for index in range(3):
            folder = DATA / group / f"seed_{index}"
            p = dict(np.load(folder / "PREDICTIONS.npz", allow_pickle=False))
            m = dict(np.load(folder / "PER_SCENE_METRICS.npz", allow_pickle=False))
            largest = 0.0
            selection_differences = {}
            for split in ("fresh_iid", "fresh_appearance", "fresh_joint"):
                for variant in ("original", "swapped"):
                    key = f"{split}__{variant}"
                    target = p[f"{key}__target"]
                    foot = p[f"{key}__footprints"]
                    e = np.einsum('nchw,nkhw->nkc', target, foot) / foot.sum(axis=(-2,-1))[..., None]
                    # Independently reconstruct truth from stored target fields
                    # and footprints, rather than from the generator's roles.
                    error = float(np.max(np.abs(formula(e) - p[f"{key}__truth"])))
                    largest = max(largest, error)
                    assert error < 2e-7
                    pred_e = np.einsum('nchw,nkhw->nkc', p[f"{key}__fields"], foot) / foot.sum(axis=(-2,-1))[..., None]
                    np.testing.assert_allclose(formula(pred_e), p[f"{key}__predicted_formula__prediction"], atol=2e-7, rtol=0)
                    truth = p[f"{key}__truth"]
                    for arm in audit.json.loads((audit.HERE / "protocol.json").read_text())["arms"]:
                        pred = p[f"{key}__{arm}__prediction"]
                        minimum = pred.min(axis=1, keepdims=True)
                        chosen = np.isclose(pred, minimum, atol=1e-9, rtol=0)
                        regret = (((truth * chosen).sum(axis=1)/chosen.sum(axis=1)) - truth.min(axis=1)).mean(axis=1)
                        np.testing.assert_allclose(regret, m[f"{key}__{arm}__all__regret"], atol=2e-12, rtol=0)
                        false_safe = ((truth > .02) & (pred <= .02)).sum(axis=(1,2))
                        np.testing.assert_array_equal(false_safe, m[f"{key}__{arm}__all__false_safe_count"])
                    learned = p[f"{key}__factorized_no_cf__prediction"]
                    direct = p[f"{key}__predicted_formula__prediction"]
                    a = np.isclose(learned, learned.min(axis=1,keepdims=True), atol=1e-9, rtol=0)
                    b = np.isclose(direct, direct.min(axis=1,keepdims=True), atol=1e-9, rtol=0)
                    different = np.any(a != b, axis=1)
                    selection_differences[key] = {
                        "all_cards_of_800": int(different.sum()),
                        "nontrivial_A_C_D_of_600": int(different[:, [0, 2, 3]].sum()),
                        "zero_cost_B_of_200": int(different[:, 1].sum()),
                    }
            model = CandidateFieldModel().eval()
            model.load_state_dict(torch.load(folder / "visual_field.pt", map_location="cpu", weights_only=True), strict=True)
            images = []
            expected = []
            for variant in ("original", "swapped"):
                key = f"fresh_appearance__{variant}"
                images.append(np.asarray(Image.open(DATA / "runs/seed_0" / f"{key}__example_0.png").convert("RGB")))
                expected.append(p[f"{key}__fields"][0])
            with torch.inference_mode():
                x = torch.tensor(np.asarray(images), dtype=torch.float32).permute(0,3,1,2)/255
                replay = torch.sigmoid(model(x)).numpy()
            replay_error = float(np.max(np.abs(replay - np.asarray(expected))))
            # Cross-device convolution implementations need not be bit identical.
            # Report the discrepancy; archived GPU arrays remain authoritative.
            agreement = float(np.mean((replay > .5) == (np.asarray(expected) > .5)))
            assert np.isfinite(replay).all() and agreement >= .995
            records.append({"group": group, "seed_index": index,
                "target_reconstruction_max_error": largest, "cpu_replay_max_error": replay_error,
                "cpu_replay_binary_field_agreement": agreement,
                "learned_vs_formula_different_choice_sets": selection_differences})
    value = {"status": "PASS", "six_exported_checkpoints_cpu_replayed": True,
             "image_replays": 12, "scored_image_observations": 7200,
             "checks": ["target cost reconstruction", "analytic predicted cost reconstruction", "all arm regrets", "false-safe counts", "checkpoint inference"],
             "records": records}
    (Path(__file__).parent / "PREDICTION_VERIFICATION.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
