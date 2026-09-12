"""Five frozen checkpoints, paired logit intervention and nested residual audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import *

def main(root, index):
    p = protocol()
    verify_freeze(root, "E1")
    verify_manifest(root / "e1_data")
    prior = Path(p["prior_root"]) / "runs/independent" / f"seed_{index}"
    verify_manifest(prior)
    out = root / "e1_runs" / f"seed_{index}"
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Quest GPU allocation required")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_candidate(index).cuda()
    model.load_state_dict(torch.load(prior / "visual_field.pt", weights_only=True, map_location="cuda"))
    state_hash = sha(prior / "visual_field.pt")
    results, scores_saved = {}, {}
    for split in p["e1"]["splits"]:
        d = load_npz(root / "e1_data" / f"{split}.npz")
        logits = infer_logits(model, d["rgb"], torch.device("cuda"))
        saved, result = {}, {}
        for name, shift in [("inherited_minus2", 0), ("restored_zero", 2)]:
            fields = probabilities(logits, shift)
            pred = formula(exposure(fields, d["footprints"]))
            saved[f"{name}__fields"], saved[f"{name}__prediction"] = fields, pred
            result[name] = {}
            for truth_name, truth in [("low16", d["truth"]), ("reference", d["truth_ref512"])]:
                key = f"{name}__{truth_name}"
                scores = nested_scores(truth, pred, d)
                if split == "calibration":
                    for unit, value in scores.items():
                        scores_saved[f"{key}__{unit}"] = value
                summary = {"raw": summarize(rows(truth, pred, d)), "calibrated": {}}
                for unit in p["e1"]["units"]:
                    source = scores_saved[f"{key}__{unit}"]
                    q = quantile(source)
                    summary["calibrated"][unit] = {"q": q, **summarize(rows(truth, pred, d, offset=q))}
                result[name][truth_name] = summary
            # Boundary-only replacement is explicitly privileged and not an acceptance method.
            replaced = np.where(d["boundary"], d["target"], fields)
            inside = (d["target"] > 0) & ~d["boundary"]
            deficit = np.maximum(d["target"] - fields, 0)
            result[name]["boundary_diagnostic"] = {
                "boundary_positive_deficit_sum": float(deficit[d["boundary"]].sum()),
                "interior_positive_deficit_sum": float(deficit[inside].sum()),
                "raw_oracle_boundary_replacement": summarize(rows(d["truth_ref512"], formula(exposure(replaced, d["footprints"])), d))}
        ids_a = choose(saved["inherited_minus2__prediction"], d["tie_uniform"])
        ids_b = choose(saved["restored_zero__prediction"], d["tie_uniform"])
        result["selector_changed_fraction"] = float((deployed(ids_a, d) != deployed(ids_b, d)).mean())
        old_path = prior / ("CALIBRATION.npz" if split == "calibration" else f"{split}_PREDICTIONS.npz")
        old_fields = load_npz(old_path)["fields"]
        error = float(abs(saved["inherited_minus2__fields"] - old_fields).max())
        if error > .01:
            raise ValueError("Frozen-checkpoint replay exceeded fixed numerical tolerance")
        result["prior_replay_max_difference"] = error
        truth = d["truth_ref512"]
        minimum = truth.min(2)
        result["oracle"] = {"deployed": summarize(rows(truth, truth, d)),
            "all_variant_card_coverage": float((minimum[..., CARDS] <= .02).mean()),
            "safe_candidate_count_histogram": np.bincount((truth <= .02).sum(2)[..., CARDS].ravel(), minlength=5).tolist(),
            "oracle_margin_quantiles": np.quantile(.02 - minimum[..., CARDS], [0, .05, .25, .5, .75, .95, 1]).tolist()}
        saved["oracle_minimum"] = minimum
        np.savez_compressed(out / f"{split}.npz", **saved)
        results[split] = result
        print(index, split, result["restored_zero"]["reference"]["raw"], flush=True)
    np.savez_compressed(out / "CALIBRATION_SCORES.npz", **scores_saved)
    write_json(out / "RESULTS.json", {"status": "COMPLETE", "at_utc": now(), "prior_checkpoint_sha256": state_hash,
                                     "training_performed": False, "splits": results})
    write_json(out / "MANIFEST.json", manifest(out))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True, choices=range(5))
    args = parser.parse_args()
    main(args.root, args.index)
