"""Reconstruct continuous-reference truth for consumed diagnostic datasets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from common import protocol, verify_freeze, verify_manifest, load_npz, write_json, manifest, deployment, sha
from geometry import raster, low, boundary_band, reference_audit

def reference_task(args):
    record, p = args
    torch.set_num_threads(1)
    result = {}
    for size in [64, p["reference_check_size"], p["reference_size"]]:
        f, w, e = raster(record, size)
        if size == 64:
            result["target_reconstructed"] = np.stack([low(f), low(f)[::-1]])
        else:
            from common import formula
            result[f"truth_ref{size}"] = formula(e)
            if size == p["reference_size"]:
                result["footprints_common"] = low(w)
    return result

def main(root):
    p = protocol()
    verify_freeze(root, "E1")
    prior = Path(p["prior_root"])
    verify_manifest(prior / "datasets")
    out = root / "e1_data"
    out.mkdir(exist_ok=False)
    audit = {}
    for split in p["e1"]["splits"]:
        d = load_npz(prior / "datasets" / f"{split}.npz")
        records = json.loads((prior / "datasets" / f"{split}.json").read_text())
        with ProcessPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(reference_task, [(r, p) for r in records], chunksize=8))
        more = {k: np.asarray([x[k] for x in values]) for k in values[0]}
        np.testing.assert_array_equal(more.pop("target_reconstructed"), d["target"])
        d.update(more)
        d.update(deployment(int(records[0]["id"].split(":")[0]), len(records)))
        d["boundary"] = boundary_band(d["target"])
        np.savez_compressed(out / f"{split}.npz", **d)
        audit[split] = {"source_sha256": sha(prior / "datasets" / f"{split}.npz"),
                        "reference": reference_audit(d, p),
                        "low16_vs_reference_flip": float(((d["truth"][..., [0, 2, 3]] > .02) !=
                                (d["truth_ref512"][..., [0, 2, 3]] > .02)).mean())}
        print(split, audit[split], flush=True)
    write_json(out / "AUDIT.json", audit)
    write_json(out / "MANIFEST.json", manifest(out))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
