#!/usr/bin/env python3
"""Paired analysis with fixed training seeds and intact scene intervention blocks."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci(values, seed=2026091299):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def load_runs(root, protocol):
    results, arrays, audits = [], [], []
    for index, seed in enumerate(protocol["model_seeds"]):
        path = root / f"seed_{index}"
        for name, expected in read(path / "MANIFEST.json").items():
            if digest(path / name) != expected:
                raise ValueError(f"Artifact mismatch: {path / name}")
        r = read(path / "RESULTS.json")
        assert r["status"] == "COMPLETE" and r["seed"] == seed
        assert r["protocol_sha256"] == digest(HERE / ("balanced_repair/protocol.json" if "REPAIR" in protocol["experiment_id"] else "protocol.json"))
        assert digest(path / "visual_field.pt") == r["checkpoint_sha256"]
        results.append(r)
        arrays.append(dict(np.load(path / "PER_SCENE_METRICS.npz", allow_pickle=False)))
        audits.append(read(path / "DATASET_AUDIT.json"))
    assert audits[0] == audits[1] == audits[2]
    return results, arrays, audits[0]


def summarize(results, arrays, protocol):
    splits = {}
    for split in protocol["fresh_splits"]:
        row = {}
        for variant in ("original", "swapped"):
            row[variant] = {"field": {}, "arms": {}}
            for metric in results[0]["splits"][split][variant]["field"]:
                row[variant]["field"][metric] = float(np.mean([r["splits"][split][variant]["field"][metric] for r in results]))
            for arm in protocol["arms"]:
                row[variant]["arms"][arm] = {}
                for card in ("all", "D"):
                    summaries = [r["splits"][split][variant]["arms"][arm][card] for r in results]
                    row[variant]["arms"][arm][card] = {
                        metric: float(np.mean([s[metric] for s in summaries]))
                        for metric in summaries[0] if summaries[0][metric] is not None}
        row["paired"] = {}
        row["balanced"] = {}
        for arm in protocol["arms"]:
            per_seed = [r["splits"][split]["paired"][arm]["both_correct_probability"] for r in results]
            eligible = arrays[0][f"{split}__paired__{arm}__eligible"]
            assert all(np.array_equal(a[f"{split}__paired__{arm}__eligible"], eligible) for a in arrays)
            score = np.mean([a[f"{split}__paired__{arm}__both_correct"] for a in arrays], axis=0)
            # Exactly 2 eligible card changes per generated pair here. Retain
            # numerator/denominator resampling if a future generator differs.
            rng = np.random.default_rng(protocol["bootstrap_seed"])
            draws = rng.integers(len(score), size=(10000, len(score)))
            boot = score.sum(axis=1)[draws].sum(axis=1) / eligible.sum(axis=1)[draws].sum(axis=1)
            row["paired"][arm] = {"mean": float(np.mean(per_seed)), "per_seed": per_seed,
                "ci95": [float(v) for v in np.quantile(boot, [.025, .975])],
                "eligible_card_pairs_per_seed": int(eligible.sum()), "scene_pairs": len(score)}
            row["balanced"][arm] = {}
            for card in ("all", "D"):
                row["balanced"][arm][card] = {}
                for metric in ("regret", "false_safe", "pair_ranking", "semantic_cost_mae", "dangerous_selection"):
                    per_seed_scene = np.asarray([
                        .5 * (a[f"{split}__original__{arm}__{card}__{metric}"] + a[f"{split}__swapped__{arm}__{card}__{metric}"])
                        for a in arrays])
                    row["balanced"][arm][card][metric] = {
                        "mean": float(per_seed_scene.mean()), "ci95": ci(per_seed_scene.mean(axis=0)),
                        "per_seed": per_seed_scene.mean(axis=1).tolist()}
        row["comparisons"] = {}
        for left, right in (("factorized_no_cf", "predicted_formula"), ("factorized_no_cf", "factorized_full")):
            differences = np.asarray([
                .5 * sum(a[f"{split}__{v}__{left}__all__regret"] - a[f"{split}__{v}__{right}__all__regret"]
                         for v in ("original", "swapped")) for a in arrays])
            row["comparisons"][f"{left}_minus_{right}"] = {
                "mean": float(differences.mean()), "ci95": ci(differences.mean(axis=0)),
                "per_seed": differences.mean(axis=1).tolist(), "positive_favors": right}
        splits[split] = row
    return {"experiment_id": protocol["experiment_id"], "seeds": protocol["model_seeds"],
            "unique_scene_pairs": 600, "unique_rgb_images": 1200,
            "uncertainty_scope": "Paired scene bootstrap conditional on the three fixed training seeds; no multiplicity adjustment for secondary diagnostics",
            "checkpoint_sha256": [r["checkpoint_sha256"] for r in results], "splits": splits}


def main(root, output):
    protocol = read(HERE / "protocol.json")
    repair_protocol = read(HERE / "balanced_repair/protocol.json")
    base_r, base_a, audit = load_runs(root / "runs", protocol)
    repair_r, repair_a, repair_audit = load_runs(root / "repair_runs", repair_protocol)
    assert audit == repair_audit
    result = {"baseline": summarize(base_r, base_a, protocol),
              "balanced_repair": summarize(repair_r, repair_a, repair_protocol),
              "repair_comparisons": {}, "dataset_hashes_match_all_six_models": True}
    for split in protocol["fresh_splits"]:
        differences = np.asarray([
            .5 * sum(b[f"{split}__{v}__predicted_formula__all__regret"] - r[f"{split}__{v}__predicted_formula__all__regret"]
                     for v in ("original", "swapped")) for b, r in zip(base_a, repair_a)])
        result["repair_comparisons"][split] = {
            "baseline_minus_repair_regret": float(differences.mean()),
            "ci95": ci(differences.mean(axis=0)), "per_seed": differences.mean(axis=1).tolist(),
            "scope": "Adaptive descriptive comparison on consumed probes"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "primary": result["baseline"]["splits"]["fresh_appearance"]["comparisons"],
                      "repair": result["repair_comparisons"], "baseline_pair": result["baseline"]["splits"]["fresh_appearance"]["paired"]["predicted_formula"],
                      "repair_pair": result["balanced_repair"]["splits"]["fresh_appearance"]["paired"]["predicted_formula"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
