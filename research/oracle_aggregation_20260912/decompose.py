"""Same-scene oracle discovery. All calls run on Quest; prior assets read-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pro_decision_round_20260912"))
from common import (sha, now, write_json, load_npz, formula, exposure, choose,
                    selected, deployed, rows, summarize, manifest, verify_manifest)
from geometry import raster, field_mask

def block_mean(a, size):
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape[-2:]
    if h % size or w % size:
        raise ValueError("Only exact nonoverlapping pooling is allowed")
    return a.reshape(*a.shape[:-2], size, h//size, size, w//size).mean((-3, -1))

def oracle_one(record):
    torch.set_num_threads(1)
    p, f, eref = raster(record, 512)
    pl, fl = block_mean(p, 16), block_mean(f, 16)
    den = fl.sum((1, 2))
    epool = np.einsum("chw,khw->kc", pl, fl) / den[:, None]
    # Independently form within-block E[PF] and covariance, not epool-eref.
    cov = np.stack([block_mean(p * w, 16) - pl * block_mean(w, 16) for w in f])
    delta = -cov.sum((-2, -1)) / den[:, None]
    envelope = np.sqrt(pl[None]*(1-pl[None])*fl[:, None]*(1-fl[:, None])).sum((-2,-1))/den[:,None]
    joint_boundary = (((pl[None] > 0)&(pl[None] < 1)) & ((fl[:,None] > 0)&(fl[:,None] < 1))).any((-2,-1))
    p1024 = np.asarray([field_mask(k,c,r,a,1024) for k,c,r,a in
                       zip(record["shapes"],record["centers"],record["radii"],record["angles"])])[record["assignment"]]
    pcheck = block_mean(p1024, 512)
    echeck = np.einsum("chw,khw->kc", pcheck, f.astype(float))/f.sum((1,2))[:,None]
    pair = lambda x: np.stack([x, x[:, ::-1]])
    return dict(ref_e=eref, pool_e=pair(epool), covariance_delta=pair(delta),
                envelope=pair(envelope), boundary=pair(joint_boundary), check_e=pair(echeck),
                common_footprints=fl, pooled_property=pl)

def main(root):
    p = json.loads((HERE/"protocol.json").read_text())
    prior = Path(p["prior_root"])
    root.mkdir(parents=True, exist_ok=True)
    out = root/"discovery"
    out.mkdir(exist_ok=False)
    source = {str(f): sha(f) for f in sorted(HERE.glob("*.py"))}
    source[str(HERE/"protocol.json")] = sha(HERE/"protocol.json")
    write_json(root/"FREEZE_DISCOVERY.json", {"at_utc":now(),"sources":source,"already_consumed":True})
    verify_manifest(prior/"e3_data")
    input_hashes = {}
    for i in range(5):
        verify_manifest(prior/"e3_runs"/f"seed_{i}")
    for name in p["splits"]:
        data_path = prior/"e3_data"/f"{name}.npz"
        record_path = data_path.with_suffix(".json")
        d = load_npz(data_path)
        records = json.loads(record_path.read_text())
        input_hashes[str(data_path)] = sha(data_path)
        input_hashes[str(record_path)] = sha(record_path)
        if len(records) != p["count_per_split"]:
            raise ValueError("Frozen count changed")
        values=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for j,v in enumerate(pool.map(oracle_one,records,chunksize=8)):
                values.append(v)
                if (j+1)%250 == 0:
                    print(name,j+1,now(),flush=True)
        a = {k:np.asarray([v[k] for v in values]) for k in values[0]}
        np.testing.assert_array_equal(a["common_footprints"],d["footprints_common"])
        np.testing.assert_allclose(formula(a["ref_e"]),d["truth_ref512"],atol=1e-14,rtol=0)
        np.testing.assert_allclose(a["pool_e"]-a["ref_e"],a["covariance_delta"],atol=1e-14,rtol=0)
        a["target_e"] = exposure(d["target64"],d["footprints_common"])
        a["ref_cost"],a["pool_cost"],a["target_cost"] = [formula(a[k]) for k in ["ref_e","pool_e","target_e"]]
        a["check_cost"] = formula(a["check_e"])
        a["old_ref256_cost"] = d["truth_ref256"]
        for k in ["deploy_variant","deploy_card","tie_uniform"]:
            a[k]=d[k]
        for post in ["restored","minus2"]:
            costs=[]
            for i in range(5):
                path=prior/"e3_runs"/f"seed_{i}"/f"{name.split('_')[0]}_TEST_PREDICTIONS.npz"
                input_hashes[str(path)]=sha(path)
                costs.append(load_npz(path)[post])
            a[f"{post}_cost"]=np.asarray(costs)
            a[f"{post}_e"]=a[f"{post}_cost"][...,[0,3]]
            np.testing.assert_allclose(formula(a[f"{post}_e"]),a[f"{post}_cost"],atol=1e-14,rtol=0)
        np.savez_compressed(out/f"{name}.npz",**a)
        print("saved",name,flush=True)
    write_json(out/"INPUT_HASHES.json",input_hashes)
    write_json(out/"MANIFEST.json",manifest(out))

if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
