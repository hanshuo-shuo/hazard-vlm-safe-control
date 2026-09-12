"""One prospective IID bank; frozen inference, no optimization or sample search."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from decompose import (HERE, oracle_one, block_mean, field_mask, raster, load_npz,
                       write_json, now, sha, formula, exposure, rows, choose,
                       selected, deployed, manifest)
from common import protocol, deployment, reference, build_candidate, infer_logits, probabilities, cp_upper
from geometry import continuous_scene
from report import report

def canonical(record):
    patches=sorted([dict(center=c,radii=r,angle=a,shape=s) for c,r,a,s in
        zip(record["centers"],record["radii"],record["angles"],record["shapes"])],key=lambda x:json.dumps(x,sort_keys=True))
    paths=sorted(record["points"],key=lambda x:json.dumps(x))
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return digest(patches),digest(dict(patches=patches,paths=paths,radius=record["radius"]))

def materialize(record):
    a=oracle_one(record)
    f=np.asarray([field_mask(k,c,r,t,64) for k,c,r,t in
        zip(record["shapes"],record["centers"],record["radii"],record["angles"])])[record["assignment"]]
    targets=block_mean(f,16)
    a["target64"]=np.stack([targets,targets[::-1]])
    a["rgb"]=np.asarray([reference.render_requirement_rgb(v,family=record["family"],image_size=64,
                       render_seed=record["render_seed"]) for v in [f,f[::-1].copy()]])
    a["old_ref256_cost"]=formula(raster(record,256)[2])
    return a

def hypotheses(a, cp):
    truth=a["ref_cost"]
    r=rows(truth,a["pool_cost"],a)
    stable_oracle=(deployed(truth.min(2),a)<=.02)&(deployed(a["check_cost"].min(2),a)<=.02)
    lost=stable_oracle&~r["safe_accepted"]
    n=len(lost);k=int(lost.sum());alpha=cp["uncertainty"]["one_sided_alpha_each"]
    lower=1-cp_upper(n-k,n,alpha)
    flip=[];conservative=[];small=[];large=[]
    bc=a["boundary"]
    bcost=np.stack([bc[...,0],np.zeros_like(bc[...,0]),bc.any(-1),bc[...,1]],axis=-1)
    for pred in a["restored_cost"]:
        ids=choose(pred,a["tie_uniform"])
        fixed=lambda c:deployed(selected(c,ids),a)
        ref,pool=fixed(truth),fixed(a["pool_cost"])
        f=(ref>.02)!=(pool>.02)
        b=fixed(bcost).astype(bool);s=abs(pool-.02)<=.005
        flip.append(f);conservative.append((ref<=.02)&(pool>.02));small.append(b&s);large.append(b&~s)
    flip,conservative,small,large=map(np.asarray,[flip,conservative,small,large])
    rng=np.random.default_rng(cp["uncertainty"]["bootstrap_seed"])
    count=cp["uncertainty"]["bootstrap_resamples"]
    sw=rng.multinomial(5,np.ones(5)/5,size=count)/5
    nw=rng.multinomial(n,np.ones(n)/n,size=count)
    def resample(x):return ((sw@x.astype(float))*nw).sum(1)
    den=resample(flip);ds=resample(small);dl=resample(large)
    valid=(den>0)&(ds>0)&(dl>0)
    ratio=resample(conservative)[valid]/den[valid]
    diff=resample(flip&small)[valid]/ds[valid]-resample(flip&large)[valid]/dl[valid]
    h2=float(np.quantile(ratio,alpha)) if valid.all() else None
    h3=float(np.quantile(diff,alpha)) if valid.all() else None
    checkdiff=abs(a["check_cost"]-truth)[...,[0,2,3]]
    checkflip=((a["check_cost"]>.02)!=(truth>.02))[...,[0,2,3]]
    guard={"label_difference":float(checkflip.mean()),"cost_difference_p95":float(np.quantile(checkdiff,.95)),
           "covariance_identity_max_abs":float(abs(a["pool_e"]-a["ref_e"]-a["covariance_delta"]).max())}
    guard["pass"]=guard["label_difference"]<=.01 and guard["cost_difference_p95"]<=.002 and guard["covariance_identity_max_abs"]<=1e-12
    result={"H1":{"events":k,"scenes":n,"estimate":k/n,"lower":lower,"pass":lower>.01},
            "H2":{"conservative_changes":int(conservative.sum()),"all_changes":int(flip.sum()),
                  "estimate":float(conservative.sum()/flip.sum()) if flip.sum() else None,"lower":h2,"pass":h2 is not None and h2>.90},
            "H3":{"small_events":int((flip&small).sum()),"small_observations":int(small.sum()),
                  "large_events":int((flip&large).sum()),"large_observations":int(large.sum()),
                  "estimate":float((flip&small).sum()/small.sum()-(flip&large).sum()/large.sum()) if small.sum() and large.sum() else None,
                  "lower":h3,"pass":h3 is not None and h3>.10},
            "undefined_bootstraps":int((~valid).sum()),"numerical_guard":guard,
            "one_sided_alpha_each":alpha,"at_utc":now(),"scope":"Prospective mechanism confirmation on normal synthetic IID; not cross-task transfer or a new safety certificate"}
    result["decision"]="SUPPORTED_FIXED_PREDICTION" if guard["pass"] and all(result[k]["pass"] for k in ["H1","H2","H3"]) else "NOT_FULLY_SUPPORTED_STOP_EXTENSION"
    return result

def main(root):
    cp=json.loads((HERE/"confirmation_protocol.json").read_text());p=protocol()
    prior=Path(json.loads((HERE/"protocol.json").read_text())["prior_root"])
    out=root/"confirmation";out.mkdir(exist_ok=False)
    checkpoints=[prior/"e2_runs/independent_64"/f"seed_{i}/model.pt" for i in range(5)]
    freeze={"at_utc":now(),"protocol_sha256":sha(HERE/"confirmation_protocol.json"),
            "sources":{str(f):sha(f) for f in sorted(HERE.glob("*.py"))},
            "checkpoints":{str(f):sha(f) for f in checkpoints},"new_records_generated":False}
    write_json(out/"FREEZE.json",freeze)
    spec=cp["dataset"];n=spec["count"]
    records=[continuous_scene(spec["seed"],i,"independent",p["appearance"]["train"],p) for i in range(n)]
    old=[set(),set()];prior_records={}
    for stage in ["e2_data","e3_data"]:
        for path in sorted((prior/stage).glob("*.json")):
            if path.name in ["AUDIT.json","MANIFEST.json"]:continue
            prior_records[str(path)]=sha(path)
            for record in json.loads(path.read_text()):
                for j,h in enumerate(canonical(record)):old[j].add(h)
    new=[set(),set()]
    for record in records:
        for j,h in enumerate(canonical(record)):
            if h in old[j] or h in new[j]:raise ValueError("Geometry overlap: stop without scene replacement")
            new[j].add(h)
    write_json(out/"IDENTITY.json",{"prior_unique":[len(s) for s in old],"new_unique":[len(s) for s in new],
               "exact_overlap":0,"scope":"Anonymous patches/paths; not arbitrary geometric symmetry","prior_record_hashes":prior_records})
    write_json(out/"records.json",records)
    values=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,v in enumerate(pool.map(materialize,records,chunksize=8)):
            values.append(v)
            if (j+1)%250==0:print("confirmation geometry",j+1,now(),flush=True)
    a={k:np.asarray([v[k] for v in values]) for k in values[0]}
    a.update(deployment(spec["seed"],n))
    a["target_e"]=exposure(a["target64"],a["common_footprints"])
    for prefix in ["ref","pool","target","check"]:a[f"{prefix}_cost"]=formula(a[f"{prefix}_e"])
    predictions={k:[] for k in ["restored","minus2"]}
    torch.set_num_threads(4)
    for i,path in enumerate(checkpoints):
        assert sha(path)==freeze["checkpoints"][str(path)]
        model=build_candidate(p["model_seeds"][i]);model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        logits=infer_logits(model,a["rgb"],torch.device("cpu"))
        np.savez_compressed(out/f"logits_{i}.npz",logits=logits)
        for post,shift in [("restored",0),("minus2",-2)]:
            predictions[post].append(formula(exposure(probabilities(logits,shift),a["common_footprints"])))
        print("frozen model inference",i,now(),flush=True)
    for post,values in predictions.items():
        a[f"{post}_cost"]=np.asarray(values);a[f"{post}_e"]=a[f"{post}_cost"][...,[0,3]]
    np.savez_compressed(out/"data.npz",**a)
    write_json(out/"HYPOTHESES.json",hypotheses(a,cp))
    write_json(out/"SUMMARY.json",{"at_utc":now(),"stage":"ONE_PROSPECTIVE_CONFIRMATION_NO_TRAINING","splits":{"iid_confirmation":report(a)}})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
