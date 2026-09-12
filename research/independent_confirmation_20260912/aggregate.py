"""Crossed seed/scene resampling, maintaining matched recipes and pair blocks."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import protocol, verify_freeze, verify_manifest, write_json, sha


def interval(num, den, seed_weights, scene_weights):
    num, den = np.asarray(num, float), np.asarray(den, float)
    point = float(num.sum() / den.sum())
    seed_values = num.sum(1) / den.sum(1)
    crossed = ((seed_weights @ num) * scene_weights).sum(1) / ((seed_weights @ den) * scene_weights).sum(1)
    conditional = (scene_weights @ num.mean(0)) / (scene_weights @ den.mean(0))
    return {"mean": point, "per_seed": seed_values.tolist(),
            "crossed_ci95": np.quantile(crossed, [.025, .975]).tolist(),
            "conditional_scene_ci95": np.quantile(conditional, [.025, .975]).tolist()}, crossed, conditional


def main(root):
    p = protocol()
    verify_freeze(root)
    checked = verify_manifest(root / "datasets")
    out = root / "reports"
    out.mkdir(exist_ok=False)
    rows, results, training = {}, {}, {}
    for recipe in p["recipes"]:
        rows[recipe], results[recipe], training[recipe] = [], [], []
        for index, seed in enumerate(p["model_seeds"]):
            run = root / "runs" / recipe / f"seed_{index}"
            checked += verify_manifest(run)
            result = json.loads((run / "RESULTS.json").read_text())
            if result["status"] != "COMPLETE" or result["model_seed"] != seed:
                raise ValueError("Incomplete or wrong seed")
            events = json.loads((run / "EVENTS.json").read_text())
            names = [x["event"] for x in events]
            assert names.index("checkpoint_exported") < names.index("load:calibration") < names.index("calibration_persisted")
            assert all(names.index("calibration_persisted") < names.index(f"load:{s}") for s in p["test_splits"])
            with np.load(run / "METRICS.npz", allow_pickle=False) as f:
                rows[recipe].append({k: f[k] for k in f.files})
            results[recipe].append(result)
            h = json.loads((run / "TRAINING.json").read_text())
            training[recipe].append({"seed": seed, "epochs": h["epochs"], "optimizer_steps": h["epochs"] * 40,
                    "seconds": h["elapsed_seconds"], "offset": result["offset"], "checkpoint_sha256": result["checkpoint_sha256"]})
    summary = {"experiment_id": p["experiment_id"], "evidence_class": p["evidence_class"],
               "freeze_sha256": sha(root / "FREEZE.json"), "checked_manifest_files": checked,
               "model_seeds": p["model_seeds"], "training": training, "splits": {}, "primary": {}}
    rng = np.random.default_rng(p["uncertainty"]["seed"])
    count, seeds = p["uncertainty"]["resamples"], len(p["model_seeds"])
    seed_weights = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=count) / seeds
    for split, spec in p["test_splits"].items():
        n = spec["scene_count"]
        scene_weights = rng.multinomial(n, np.full(n, 1 / n), size=count)
        estimates = {}
        s = {"scene_pairs": n, "recipes": {}, "contrasts": {}}
        for recipe in p["recipes"]:
            def values(arm, name):
                return np.asarray([r[f"{split}__{arm}__{name}"] for r in rows[recipe]])
            value = {"arms": {}, "uncertainty": {}, "coverage": {}}
            for arm in p["arms"]:
                per_seed = [r["splits"][split]["arms"][arm] for r in results[recipe]]
                value["arms"][arm] = {k: float(np.mean([x[k] for x in per_seed]))
                                      if per_seed[0][k] is not None else None for k in per_seed[0]}
                value["arms"][arm]["per_seed"] = per_seed
            for metric, numerator, denominator in [
                ("regret", values("formula", "regret"), np.ones((seeds, n))),
                ("false_safe_micro", values("formula", "false_safe_count"), values("formula", "false_safe_eligible")),
                ("paired_correct", values("formula", "paired_correct_sum"), values("formula", "paired_eligible")),
                ("original_regret", values("formula", "original_regret"), np.ones((seeds, n)))]:
                stat, crossed, conditional = interval(numerator, denominator, seed_weights, scene_weights)
                value["uncertainty"][metric] = stat
                estimates[recipe, metric] = stat, crossed, conditional
            value["field_miou_per_seed"] = [r["splits"][split]["field_miou"] for r in results[recipe]]
            value["field_miou"] = float(np.mean(value["field_miou_per_seed"]))
            value["original_field_miou"] = float(np.mean([r["splits"][split]["original_field_miou"] for r in results[recipe]]))
            for name in ["raw", "calibrated"]:
                data = values(name, "coverage")
                totals = data.sum((0, 1))
                curve = []
                for threshold, (accepted, unsafe, cost) in zip(p["coverage_thresholds"], totals):
                    curve.append({"threshold": threshold, "coverage": float(accepted / (seeds * n * 6)),
                                  "accepted_seed_decisions": float(accepted), "total_seed_decisions": seeds * n * 6,
                                  "unsafe_selected_probability_sum": float(unsafe),
                                  "conditional_unsafe_rate": float(unsafe / accepted) if accepted else None,
                                  "conditional_mean_cost": float(cost / accepted) if accepted else None})
                value["coverage"][name] = {"curve": curve,
                    "pair_bound_failure_rate": float(values(name, "pair_bound_failure").mean()),
                    "pair_bound_failure_per_seed": values(name, "pair_bound_failure").mean(1).tolist()}
            s["recipes"][recipe] = value
        for left, right in [("legacy", "balanced"), ("balanced", "independent")]:
            contrast = {}
            for metric in ["regret", "false_safe_micro", "paired_correct", "original_regret"]:
                a, ax, ac = estimates[left, metric]
                b, bx, bc = estimates[right, metric]
                contrast[metric] = {"mean_left_minus_right": a["mean"] - b["mean"],
                    "per_seed": (np.array(a["per_seed"]) - b["per_seed"]).tolist(),
                    "crossed_ci95": np.quantile(ax - bx, [.025, .975]).tolist(),
                    "conditional_scene_ci95": np.quantile(ac - bc, [.025, .975]).tolist()}
            s["contrasts"][f"{left}_minus_{right}"] = contrast
        for recipe in p["recipes"]:
            for arm in ["learned_no_cf", "learned_cf"]:
                diff = np.asarray([r[f"{split}__{arm}__regret"] - r[f"{split}__formula__regret"] for r in rows[recipe]])
                stat, _, _ = interval(diff, np.ones_like(diff), seed_weights, scene_weights)
                s["recipes"][recipe].setdefault("learned_minus_formula", {})[arm] = stat
        summary["splits"][split] = s
    main = summary["splits"][p["primary"]["split"]]["contrasts"]["legacy_minus_balanced"]
    regret_pass = main["regret"]["crossed_ci95"][0] > 0
    false_safe_increase_upper = -main["false_safe_micro"]["crossed_ci95"][0]
    safety_pass = false_safe_increase_upper <= .01
    summary["primary"] = {"definition": p["primary"], "regret_reduction": main["regret"],
                           "false_safe_increase_upper_ci95": false_safe_increase_upper,
                           "regret_pass": regret_pass, "false_safe_guard_pass": safety_pass,
                           "joint_claim_pass": regret_pass and safety_pass}
    write_json(out / "SUMMARY.json", summary)
    print(json.dumps(summary["primary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
