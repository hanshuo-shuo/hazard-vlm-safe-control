"""Independent scalar choices/counts, binomial p-values and CPU checkpoint replay."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from common import *

def scalar_choices(pred, uniforms):
    ids=np.empty((len(pred),2,4),int)
    for n in range(len(pred)):
        for v in range(2):
            for c in range(4):
                a=pred[n,v,:,c]
                tied=np.flatnonzero(abs(a-a.min())<=1e-9)
                ids[n,v,c]=tied[min(int(uniforms[n,v,c]*len(tied)),len(tied)-1)]
    return ids

def direct_exposure(fields, footprints):
    numerator=(fields[:,:,None].astype(float)*footprints[:,None,:,None]).sum((-2,-1))
    return numerator/footprints.astype(float).sum((-2,-1))[:,None,:,None]

def direct_formula(e):
    output=np.empty((*e.shape[:-1],4),float)
    output[...,0]=e[...,0]
    output[...,1]=0
    output[...,2]=1-(1-e[...,0])*(1-e[...,1])
    output[...,3]=e[...,1]
    return output

def scalar_rows(truth,pred,d,threshold=.02,offset=0):
    ids=scalar_choices(pred,d["tie_uniform"])
    costs,preds,oracle,chosen=[],[],[],[]
    for n,(v,c) in enumerate(zip(d["deploy_variant"],d["deploy_card"])):
        k=ids[n,v,c]
        costs.append(truth[n,v,k,c]);preds.append(pred[n,v,k,c]);oracle.append(truth[n,v,:,c].min());chosen.append(k)
    costs,preds,oracle=np.array(costs),np.array(preds),np.array(oracle)
    accept=np.minimum(preds+offset,1)<=threshold
    return {"accepted":accept,"unsafe_accepted":accept&(costs>.02),"safe_accepted":accept&(costs<=.02),
            "oracle":oracle<=.02,"actual_cost":costs,"estimated_cost":preds,"oracle_cost":oracle,
            "regret":costs-oracle,"chosen":np.asarray(chosen)}

def main(root):
    p=protocol()
    for stage in ["E1","E2","E3","REPORT"]:
        verify_freeze(root,stage)
    checked=sum(verify_manifest(root/name) for name in ["e1_data","e2_data","e3_data","e1_report","e2_report","e3_report"])
    torch.set_num_threads(4)
    e1={s:load_npz(root/"e1_data"/f"{s}.npz") for s in p["e1"]["splits"]}
    for i in range(5):
        run=root/"e1_runs"/f"seed_{i}"
        checked+=verify_manifest(run)
        result=json.loads((run/"RESULTS.json").read_text())
        for split,d in e1.items():
            saved=load_npz(run/f"{split}.npz")
            for post in p["e1"]["postprocessing"]:
                pred=direct_formula(direct_exposure(saved[f"{post}__fields"],d["footprints"]))
                np.testing.assert_allclose(pred,saved[f"{post}__prediction"],atol=1e-12,rtol=0)
                for name,truth in [("low16",d["truth"]),("reference",d["truth_ref512"])]:
                    actual=scalar_rows(truth,saved[f"{post}__prediction"],d)
                    expected=result["splits"][split][post][name]["raw"]
                    for k in ["accepted","unsafe_accepted","safe_accepted","oracle_safe"]:
                        source="oracle" if k=="oracle_safe" else k
                        assert int(actual[source].sum())==expected[k]
            a=saved["inherited_minus2__fields"].astype(float)
            transformed=(math.exp(2)*a)/(1-a+math.exp(2)*a)
            np.testing.assert_allclose(transformed,saved["restored_zero__fields"],atol=2e-6,rtol=0)
        print("E1 verified",i,flush=True)
    e2={s:load_npz(root/"e2_data"/f"{s}.npz") for s in p["e2"]["tests"]}
    replay=[]
    for geom in p["e2"]["geometry"]:
        for size in [48,64]:
            for i,seed in enumerate(p["model_seeds"]):
                run=root/"e2_runs"/f"{geom}_{size}"/f"seed_{i}"
                checked+=verify_manifest(run)
                metrics=load_npz(run/"METRICS.npz")
                for split,d in e2.items():
                    saved=load_npz(run/f"{split}_PREDICTIONS.npz")
                    for post in ["restored_zero","inherited_minus2"]:
                        for mode,fp,truth in [("common",d["footprints_common"],d["truth_ref512"]),
                                              ("native",d[f"footprints{size}"],d[f"truth_native{size}"])]:
                            pred=saved[f"{post}__{mode}__prediction"]
                            np.testing.assert_allclose(direct_formula(direct_exposure(saved[f"{post}__fields"],fp)),pred,atol=1e-12,rtol=0)
                            ids=scalar_choices(pred,d["tie_uniform"])
                            picked=np.take_along_axis(truth,ids[:,:,None,:],axis=2).squeeze(2)
                            regret=(picked-truth.min(2))[...,[0,2,3]].mean((1,2))
                            prefix=f"{split}__{post}__{mode}"
                            np.testing.assert_allclose(regret,metrics[f"{prefix}__regret"],atol=1e-12,rtol=0)
                            false=((truth[...,CARDS]>.02)&(pred[...,CARDS]<=.02)).sum((1,2,3))
                            np.testing.assert_array_equal(false,metrics[f"{prefix}__false_count"])
                model=build_candidate(seed).cpu()
                model.load_state_dict(torch.load(run/"model.pt",map_location="cpu",weights_only=True))
                d=e2["iid"]
                cpu=probabilities(infer_logits(model,d["rgb"][:2],torch.device("cpu")))
                gpu=load_npz(run/"iid_PREDICTIONS.npz")["restored_zero__fields"][:2]
                error=float(abs(cpu-gpu).max()); agreement=float(((cpu>=.5)==(gpu>=.5)).mean())
                if error>.01 or agreement<.995:
                    raise ValueError("CPU compatibility check failed")
                replay.append({"geometry":geom,"source":size,"seed":seed,"max_probability_difference":error,"binary_agreement":agreement})
                print("E2 verified",geom,size,i,error,flush=True)
    del e1,e2
    e3={s:load_npz(root/"e3_data"/f"{s}.npz") for s in ["iid_calibration","target_calibration","iid_test","target_test"]}
    certified_pvalues=[]
    for i in range(5):
        run=root/"e3_runs"/f"seed_{i}"
        checked+=verify_manifest(run)
        metrics=load_npz(run/"METRICS.npz")
        certs={s:json.loads((run/f"{s}_CERTIFICATE.json").read_text()) for s in ["iid","target"]}
        for domain,cert in certs.items():
            d=e3[f"{domain}_calibration"]
            pred=load_npz(run/f"{domain}_CALIBRATION.npz")["restored"]
            base=scalar_rows(d["truth_ref512"],pred,d,threshold=1.)
            for rule in cert["candidate_rules"]:
                accept=base["estimated_cost"]<=rule["threshold"]
                m=int(accept.sum());k=int((accept&(base["actual_cost"]>.02)).sum())
                assert m==rule["accepted"] and k==rule["unsafe_accepted"]
                if m and k/m<=.05:
                    tail=sum(math.comb(m,j)*(.05**j)*(.95**(m-j)) for j in range(k+1))
                    independently_valid=tail<=cert["delta_per_rule"]
                else:
                    tail=1.;independently_valid=False
                assert rule["certified"]==independently_valid
                if independently_valid:
                    certified_pvalues.append({"seed_index":i,"domain":domain,"threshold":rule["threshold"],"binomial_pvalue":tail})
            valid=[r for r in cert["candidate_rules"] if r["certified"]]
            expected=sorted(valid,key=lambda r:(-r["accepted"],r["threshold"]))[0] if valid else None
            assert expected==cert["chosen"]
        for domain in ["iid","target"]:
            d=e3[f"{domain}_test"];truth=d["truth_ref512"]
            pred=load_npz(run/f"{domain}_TEST_PREDICTIONS.npz")
            for name,cert in [("source",certs["iid"])]+([("target_adapted",certs["target"])] if domain=="target" else []):
                settings=[("raw_minus2",pred["minus2"],.02,0), ("raw_restored",pred["restored"],.02,0),
                          ("pair_conformal_restored",pred["restored"],.02,cert["pair_q_restored"]),
                          ("selective_certified",pred["restored"],cert["chosen"]["threshold"] if cert["chosen"] else -1.,0)]
                for arm,values,t,q in settings:
                    r=scalar_rows(truth,values,d,t,q)
                    for k,v in r.items():
                        np.testing.assert_allclose(v,metrics[f"{domain}__{name}__{arm}__{k}"],atol=1e-12,rtol=0)
        print("E3 verified",i,flush=True)
    out=root/"final_report"
    out.mkdir(exist_ok=False)
    write_json(out/"VERIFICATION.json",{"status":"PASS","at_utc":now(),"verified_manifest_files":checked,
        "verifier_sha256":sha(__file__),"e1_models":5,"e2_checkpoints":20,"cpu_replayed_images":80,
        "e3_certification_models":5,"certified_rule_pvalues":certified_pvalues,"checkpoint_replays":replay,
        "independent_checks":["scalar tie choices","mean exposures and public formula","regret and false-safe counts",
                              "calibration n/k and exact binomial p-values","selected rule consistency","accepted safety/eta numerators",
                              "checkpoint replay","source and artifact hashes"]})

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
