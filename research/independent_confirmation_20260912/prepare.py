"""One CPU job materializes immutable datasets, geometric QA and hash registry."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from common import protocol, verify_freeze, write_json, manifest
from generator import make_legacy, make_random
from metrics import pair_validity


def main(root):
    verify_freeze(root)
    torch.set_num_threads(4)
    p = protocol()
    out = root / "datasets"
    out.mkdir(exist_ok=False)
    audit, identities = {}, {}

    def save(name, arrays, records, group):
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise ValueError("Nonfinite dataset")
        if arrays["rgb"].shape[1] == 2:
            np.testing.assert_array_equal(arrays["target"][:, 0, ::-1], arrays["target"][:, 1])
            valid, optimum = pair_validity(arrays["truth"], p["changed_optimum_minimum_gap"])
        else:
            valid, optimum = None, None
        rgb_ids = set(h for r in records for h in r["rgb_sha256"])
        geometry_ids = set(r["unordered_geometry_sha256"] for r in records)
        for other, (other_group, other_rgb, other_geometry) in identities.items():
            # Matched recipes intentionally share train/validation geometry.
            if other_group != group and (rgb_ids & other_rgb or geometry_ids & other_geometry):
                raise ValueError(f"Cross-split overlap {name} / {other}")
        identities[name] = group, rgb_ids, geometry_ids
        np.savez_compressed(out / f"{name}.npz", **arrays)
        write_json(out / f"{name}.json", records)
        audit[name] = {"scene_count": len(records), "images": int(np.prod(arrays["rgb"].shape[:2])),
                       "eligible_changed_card_pairs": int(valid.sum()) if valid is not None else None,
                       "zero_exposure_entries": int((arrays["exposure"] == 0).sum()),
                       "dangerous_entries_acd": int((arrays["truth"][..., [0, 2, 3]] > .02).sum()),
                       "family_counts": {f: sum(r["family"] == f for r in records)
                                         for f in sorted(set(r["family"] for r in records))}}
        if optimum is not None:
            audit[name]["optimum_candidate_counts_by_card_original"] = {
                str(c): np.bincount(optimum[:, 0, c], minlength=4).tolist() for c in [0, 2, 3]}
        if name.startswith("random_") or name == "shape_appearance":
            if valid.sum() < 60:
                raise ValueError(f"Insufficient eligible interventions: {name}")
        print(name, audit[name], flush=True)
        if arrays["rgb"].shape[1] == 2:
            # Deterministic first eight pairs, with legitimate paths shown separately.
            panel = Image.new("RGB", (8 * 256, 3 * 256), "white")
            for i in range(8):
                for v in range(2):
                    panel.paste(Image.fromarray(arrays["rgb"][i, v]).resize((256, 256)), (i * 256, v * 256))
                footprint = arrays["footprints"][i]
                color = np.zeros((16, 16, 3), dtype=float)
                for f, c in zip(footprint, [[220, 40, 60], [30, 160, 40], [30, 80, 240], [220, 160, 20]]):
                    color += f[..., None] * np.asarray(c)
                panel.paste(Image.fromarray(np.clip(color, 0, 255).astype("uint8")).resize((256, 256)), (i * 256, 512))
            panel.save(out / f"{name}_first8.png")

    for split in ["train", "validation"]:
        for recipe in p["recipes"]:
            if recipe == "independent":
                data, records = make_random(p[f"independent_{split}"], p, paired=False)
            else:
                data, records = make_legacy(p[split], p, split=split, balanced=recipe == "balanced", paired=False)
            save(f"{recipe}_{split}", data, records, split)
    data, records = make_random(p["calibration"], p)
    save("calibration", data, records, "calibration")
    for split, spec in p["test_splits"].items():
        if spec.get("generator") == "legacy":
            data, records = make_legacy(spec, p, split=split)
        else:
            data, records = make_random(spec, p)
        save(split, data, records, split)
    write_json(out / "DATASET_AUDIT.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                           "cross_split_overlap": 0, "datasets": audit})
    write_json(out / "MANIFEST.json", manifest(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
