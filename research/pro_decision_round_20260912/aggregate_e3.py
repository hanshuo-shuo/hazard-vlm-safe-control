import argparse
from pathlib import Path
import numpy as np
from common import *

def ratio(num,den,seed):
    value=crossed_ratio(num,den,seed=seed)
    if value["mean"] is None and np.asarray(den).sum()>0:
        value["mean"]=float(np.asarray(num).sum()/np.asarray(den).sum())
    return value

def main(root):
    p=protocol()
    verify_freeze(root,"E3")
    models,metrics,certificates=[],[],[]
    for i in range(5):
        run=root/"e3_runs"/f"seed_{i}"
        verify_manifest(run)
        models.append(json.loads((run/"RESULTS.json").read_text()))
        metrics.append(load_npz(run/"METRICS.npz"))
        certificates.append({domain:json.loads((run/f"{domain}_CERTIFICATE.json").read_text()) for domain in ["iid","target"]})
        events=[v["event"] for v in json.loads((run/"EVENTS.json").read_text())]
        assert events.index("selector_frozen")<events.index("load:iid_calibration")
        assert events.index("selector_frozen")<events.index("load:target_calibration")
        assert max(events.index("certificate:iid"),events.index("certificate:target"))<events.index("load:iid_test")
    summary={"status":"COMPLETE","at_utc":now(),"per_seed":models,"certificates":certificates,"scenarios":{}}
    for scenario in ["iid__source","target__source","target__target_adapted"]:
        group={}
        for arm in p["e3"]["arms"]:
            def v(name):
                return np.asarray([m[f"{scenario}__{arm}__{name}"] for m in metrics])
            accepted,unsafe,safe,oracle=v("accepted"),v("unsafe_accepted"),v("safe_accepted"),v("oracle")
            ones=np.ones_like(accepted)
            group[arm]={"coverage":ratio(accepted,ones,seed=p["bootstrap"]["seed"]),
                        "joint_unsafe":ratio(unsafe,ones,seed=p["bootstrap"]["seed"]),
                        "conditional_unsafe":ratio(unsafe,accepted,seed=p["bootstrap"]["seed"]),
                        "eta":ratio(safe,oracle,seed=p["bootstrap"]["seed"]),
                        "oracle_coverage":float(oracle.mean()),
                        "per_seed_counts":[{"n":int(len(a)),"accepted":int(a.sum()),"unsafe":int(u.sum()),"safe":int(s.sum()),"oracle":int(o.sum())}
                                           for a,u,s,o in zip(accepted,unsafe,safe,oracle)]}
        domain="target" if scenario=="target__target_adapted" else "iid"
        certified=all(c[domain]["chosen"] is not None for c in certificates)
        eta=group["selective_certified"]["eta"]["ci95"]
        scope_matched=scenario!="target__source"
        summary["scenarios"][scenario]={"arms":group,"all_models_have_certificate":certified,
                                       "matched_distribution":scope_matched,
                                       "gate_pass":bool(scope_matched and certified and eta is not None and eta[0]>=.5)}
    summary["primary_gate_pass"]=summary["scenarios"]["iid__source"]["gate_pass"]
    out=root/"e3_report"
    out.mkdir(exist_ok=False)
    write_json(out/"SUMMARY.json",summary)
    write_json(out/"MANIFEST.json",manifest(out))
    print(json.dumps({"primary_gate_pass":summary["primary_gate_pass"],"scenarios":{k:{"gate_pass":v["gate_pass"],"selective":v["arms"]["selective_certified"]} for k,v in summary["scenarios"].items()}},indent=2),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
