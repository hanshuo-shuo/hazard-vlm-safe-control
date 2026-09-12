import argparse
import json
from pathlib import Path
import numpy as np
from common import *

def main(root):
    p = protocol()
    verify_freeze(root, "E2")
    m, runs = {}, {}
    for geom in p["e2"]["geometry"]:
        for raster_size in [48,64]:
            key = f"{geom}_{raster_size}"
            m[key], runs[key] = [], []
            for i in range(5):
                run = root / "e2_runs" / key / f"seed_{i}"
                verify_manifest(run)
                m[key].append(load_npz(run / "METRICS.npz"))
                runs[key].append(json.loads((run / "RESULTS.json").read_text()))
                h = json.loads((run / "TRAINING.json").read_text())
                assert h["steps"] == 960 and h["checkpoint"] == "final"
        assert all(a["rgb_training_sha256"] == b["rgb_training_sha256"] for a,b in zip(runs[f"{geom}_48"], runs[f"{geom}_64"]))
    summary = {"status":"COMPLETE", "at_utc":now(), "per_seed":runs, "contrasts":{}, "geometry_primary_pass":True}
    for split in p["e2"]["tests"]:
        values = {key:np.asarray([r[f"{split}__restored_zero__common__regret"] for r in group]) for key,group in m.items()}
        contrasts = {}
        for raster_size in [48,64]:
            diff = values[f"balanced_anchor_{raster_size}"] - values[f"independent_{raster_size}"]
            stat = crossed_ratio(diff, np.ones_like(diff), seed=p["bootstrap"]["seed"]+raster_size, resamples=p["bootstrap"]["resamples"])
            stat["meaningful_geometry_advantage"] = stat["ci95"][0] > .001
            contrasts[f"geometry_at_{raster_size}"] = stat
            if split == "appearance":
                summary["geometry_primary_pass"] &= stat["meaningful_geometry_advantage"]
        for geom in p["e2"]["geometry"]:
            diff = values[f"{geom}_48"] - values[f"{geom}_64"]
            stat = crossed_ratio(diff, np.ones_like(diff), seed=p["bootstrap"]["seed"]+123, resamples=p["bootstrap"]["resamples"])
            stat["within_equivalence_band"] = stat["ci95"][0] >= -.0005 and stat["ci95"][1] <= .0005
            contrasts[f"raster_48_minus_64_{geom}"] = stat
        interaction = (values["balanced_anchor_48"]-values["independent_48"]) - (values["balanced_anchor_64"]-values["independent_64"])
        contrasts["interaction"] = crossed_ratio(interaction, np.ones_like(interaction), seed=p["bootstrap"]["seed"]+333)
        summary["contrasts"][split] = contrasts
    out = root / "e2_report"
    out.mkdir(exist_ok=False)
    write_json(out / "SUMMARY.json", summary)
    write_json(out / "MANIFEST.json", manifest(out))
    print(json.dumps({"geometry_primary_pass":summary["geometry_primary_pass"], "contrasts":summary["contrasts"]}, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
