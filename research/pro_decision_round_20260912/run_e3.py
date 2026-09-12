"""Finite-bank selective-risk certification before a single final test evaluation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import *

def certify(truth,pred,d,p):
    delta=p["delta_family"]/(len(p["model_seeds"])*2*len(p["e3"]["acceptance_thresholds"]))
    candidates=[]
    for t in p["e3"]["acceptance_thresholds"]:
        r=rows(truth,pred,d,threshold=t)
        k,n=int(r["unsafe_accepted"].sum()),int(r["accepted"].sum())
        upper=cp_upper(k,n,delta)
        candidates.append({"threshold":t,"accepted":n,"unsafe_accepted":k,"upper":upper,
                           "certified":n>0 and upper<=p["epsilon"]})
    valid=[r for r in candidates if r["certified"]]
    chosen=sorted(valid,key=lambda r:(-r["accepted"],r["threshold"]))[0] if valid else None
    return {"delta_per_rule":delta,"epsilon":p["epsilon"],"family_size":120,
            "chosen":chosen,"candidate_rules":candidates}

def main(root,index):
    p=protocol()
    verify_freeze(root,"E3")
    prior=root/"e2_runs/independent_64"/f"seed_{index}"
    verify_manifest(prior)
    out=root/"e3_runs"/f"seed_{index}"
    out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Quest GPU allocation required")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    model=build_candidate(p["model_seeds"][index]).cuda()
    model.load_state_dict(torch.load(prior/"model.pt",weights_only=True,map_location="cuda"))
    dataset_manifest=json.loads((root/"e3_data/MANIFEST.json").read_text())
    events=[]
    def load(name):
        path=root/"e3_data"/f"{name}.npz"
        if sha(path)!=dataset_manifest[path.name]:
            raise ValueError("Dataset hash changed")
        events.append({"event":f"load:{name}","at_utc":now()})
        return load_npz(path)
    def predict(d):
        logits=infer_logits(model,d["rgb"],torch.device("cuda"))
        return {"restored":formula(exposure(probabilities(logits),d["footprints_common"])),
                "minus2":formula(exposure(probabilities(logits,-2),d["footprints_common"]))}
    dev=load("dev")
    dev_prediction=predict(dev)
    write_json(out/"DEVELOPMENT.json",{post:summarize(rows(dev["truth_ref512"],pred,dev)) for post,pred in dev_prediction.items()})
    write_json(out/"SELECTOR_FREEZE.json",{"at_utc":now(),"model_sha256":sha(prior/"model.pt"),
        "model_index":index,"seed":p["model_seeds"][index],"selector":p["e3"]["main_selector"],
        "thresholds":p["e3"]["acceptance_thresholds"],"rule_bank_predeclared_in_protocol":True,
        "development_used_to_tune":False,"calibration_accessed":False,"test_accessed":False,
        "deployment":p["e3"]["deployment"],"truth_reference":512,"tau":.02})
    events.append({"event":"selector_frozen","at_utc":now()})
    certificates={}
    for domain in ["iid","target"]:
        d=load(f"{domain}_calibration")
        prediction=predict(d)
        certificate=certify(d["truth_ref512"],prediction["restored"],d,p)
        for post in ["restored","minus2"]:
            scores=nested_scores(d["truth_ref512"],prediction[post],d)["pair_24_entries"]
            certificate[f"pair_q_{post}"]=quantile(scores)
        certificate.update(at_utc=now(),calibration_domain=domain,label_permission="source" if domain=="iid" else "labeled_target_adaptation")
        certificates[domain]=certificate
        np.savez_compressed(out/f"{domain}_CALIBRATION.npz",**prediction)
        write_json(out/f"{domain}_CERTIFICATE.json",certificate)
        events.append({"event":f"certificate:{domain}","at_utc":now()})
    results,metrics={},{}
    for domain in ["iid","target"]:
        d=load(f"{domain}_test")
        pred=predict(d)
        truth=d["truth_ref512"]
        scenarios=[("source",certificates["iid"])]
        if domain=="target":
            scenarios.append(("target_adapted",certificates["target"]))
        saved={**pred}
        for scenario,cert in scenarios:
            threshold=cert["chosen"]["threshold"] if cert["chosen"] is not None else -1.
            arm_rows={"raw_minus2":rows(truth,pred["minus2"],d),"raw_restored":rows(truth,pred["restored"],d),
                      "pair_conformal_restored":rows(truth,pred["restored"],d,offset=cert["pair_q_restored"]),
                      "selective_certified":rows(truth,pred["restored"],d,threshold=threshold)}
            oracle=rows(truth,truth,d)
            minimum=deployed(truth.min(2),d)
            oracle.update(accepted=minimum<=.02,unsafe_accepted=np.zeros(len(minimum),bool),safe_accepted=minimum<=.02,
                          actual_cost=minimum,estimated_cost=minimum,regret=np.zeros(len(minimum)))
            arm_rows["oracle"]=oracle
            key=f"{domain}__{scenario}"
            results[key]={"certificate_domain":cert["calibration_domain"],
                          "claim_scope":"matched-distribution certification" if (domain=="iid" or scenario=="target_adapted") else "distribution-shift stress test; no risk guarantee",
                          "arms":{arm:summarize(r) for arm,r in arm_rows.items()},"selected_rule":cert["chosen"]}
            for arm,r in arm_rows.items():
                for k,v in r.items():
                    metrics[f"{key}__{arm}__{k}"]=v
            print(index,key,results[key]["arms"]["selective_certified"],flush=True)
        np.savez_compressed(out/f"{domain}_TEST_PREDICTIONS.npz",**saved)
    np.savez_compressed(out/"METRICS.npz",**metrics)
    write_json(out/"EVENTS.json",events)
    write_json(out/"RESULTS.json",{"status":"COMPLETE","at_utc":now(),"seed":p["model_seeds"][index],"scenarios":results})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--index",type=int,choices=range(5),required=True)
    args=parser.parse_args()
    main(args.root,args.index)
