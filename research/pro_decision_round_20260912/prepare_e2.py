"""Fixed continuous geometry; 48/64 supervision shares exactly one RGB array."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from common import *
from geometry import continuous_scene, pool_task, reference_audit

def build_bank(seed, count, geometry, families, p, references, exact_balance=False):
    records = [continuous_scene(seed, i, geometry, families, p) for i in range(count)]
    if exact_balance:
        swapped = set(np.random.default_rng(seed + 71).permutation(count)[:count//2].tolist())
        for i, r in enumerate(records):
            r["assignment"] = [1, 0] if i in swapped else [0, 1]
            r.pop("continuous_sha256")
            r["continuous_sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    with ProcessPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(pool_task, [(r, p, references) for r in records], chunksize=8))
    arrays = {k: np.asarray([v[k] for v in values]) for k in values[0]}
    arrays.update(deployment(seed, count))
    return arrays, records

def main(root):
    p = protocol()
    verify_freeze(root, "E2")
    previous = json.loads((root / "e1_report/SUMMARY.json").read_text())
    if not previous["proceed_to_e2"]:
        raise RuntimeError("E1 stop rule: measurement precision or feasibility failed")
    out = root / "e2_data"
    out.mkdir(exist_ok=False)
    audit, seen_rgb, seen_geometry = {}, {}, {}
    cases = [(f"{g}_{split}", spec, g, p["appearance"]["train"], False, True)
             for g in p["e2"]["geometry"] for split, spec in [("train", p["e2"]["train"]), ("validation", p["e2"]["validation"])]]
    cases += [(name, spec, "independent", p["appearance"]["train" if name == "iid" else "target"], True, False)
              for name, spec in p["e2"]["tests"].items()]
    for name, spec, geometry, families, refs, balance in cases:
        arrays, records = build_bank(spec["seed"], spec["count"], geometry, families, p, refs, balance)
        image_hashes = set(array_sha(x) for pair in arrays["rgb"] for x in pair)
        geometry_hashes = set(r["continuous_sha256"] for r in records)
        for earlier in seen_rgb:
            if image_hashes & seen_rgb[earlier] or geometry_hashes & seen_geometry[earlier]:
                raise ValueError(f"Dataset identity overlap {earlier}/{name}")
        seen_rgb[name], seen_geometry[name] = image_hashes, geometry_hashes
        np.testing.assert_array_equal(arrays["target48"][:, 0, ::-1], arrays["target48"][:, 1])
        np.testing.assert_array_equal(arrays["target64"][:, 0, ::-1], arrays["target64"][:, 1])
        np.savez_compressed(out / f"{name}.npz", **arrays)
        write_json(out / f"{name}.json", records)
        audit[name] = {"scenes": len(records), "rgb_sha256": array_sha(arrays["rgb"]),
            "same_rgb_for_both_source_rasters": True, "training_uses_only_variant_zero": not refs,
            "source_target_difference_mean": float(abs(arrays["target48"] - arrays["target64"]).mean())}
        if refs:
            audit[name]["reference"] = reference_audit(arrays, p)
            if not audit[name]["reference"]["pass"]:
                raise RuntimeError(f"Reference precision gate failed for {name}")
            audit[name]["native_truth_label_flip_vs_reference"] = {str(r): float(((arrays[f"truth_native{r}"][..., CARDS] > .02) !=
                                      (arrays["truth_ref512"][..., CARDS] > .02)).mean()) for r in [48,64]}
        print(name, audit[name], flush=True)
    write_json(out / "AUDIT.json", {"cross_split_overlap": 0, "datasets": audit})
    write_json(out / "MANIFEST.json", manifest(out))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
