"""Separate development, certification and final test scene banks."""
import argparse
from pathlib import Path
import numpy as np
from common import *
from prepare_e2 import build_bank
from geometry import reference_audit

def main(root):
    p=protocol()
    verify_freeze(root,"E3")
    out=root/"e3_data"
    out.mkdir(exist_ok=False)
    seen_rgb, seen_geometry={},{}
    # Check against every E2 train/validation/test image and continuous identity.
    for path in (root/"e2_data").glob("*.npz"):
        d=load_npz(path)
        seen_rgb[f"e2:{path.stem}"]=set(array_sha(x) for pair in d["rgb"] for x in pair)
        records=json.loads(path.with_suffix(".json").read_text())
        seen_geometry[f"e2:{path.stem}"]=set(r["continuous_sha256"] for r in records)
    audit={}
    for name in ["dev","iid_calibration","iid_test","target_calibration","target_test"]:
        spec=p["e3"][name]
        families=p["appearance"]["target" if name.startswith("target") else "train"]
        d, records=build_bank(spec["seed"],spec["count"],"independent",families,p,True)
        ids=set(array_sha(x) for pair in d["rgb"] for x in pair)
        geometry=set(r["continuous_sha256"] for r in records)
        for previous in seen_rgb:
            if ids & seen_rgb[previous] or geometry & seen_geometry[previous]:
                raise ValueError(f"Cross-split overlap: {name}/{previous}")
        seen_rgb[name],seen_geometry[name]=ids,geometry
        audit[name]={"scenes":len(records),"reference":reference_audit(d,p),
                     "one_certification_decision_per_scene":True,"unique_rgb":len(ids)}
        if not audit[name]["reference"]["pass"]:
            raise RuntimeError(f"Reference stability failed: {name}")
        np.savez_compressed(out/f"{name}.npz",**d)
        write_json(out/f"{name}.json",records)
        print(name,audit[name],flush=True)
    write_json(out/"AUDIT.json",{"cross_split_overlap":0,"datasets":audit})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
