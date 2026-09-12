"""Signed, paired discovery summaries; no certification and no tuned strata."""
import argparse
from pathlib import Path
import numpy as np
from decompose import (HERE, sha, load_npz, write_json, now, formula, choose, selected,
                       deployed, rows, summarize, manifest, verify_manifest)

def interval(a, b=None):
    a=np.atleast_2d(a).astype(float)
    if b is None: b=np.ones_like(a)
    b=np.broadcast_to(b,a.shape).astype(float)
    m,n=a.shape
    rng=np.random.default_rng(26120912)
    sw=rng.multinomial(m,np.ones(m)/m,size=2000)/m
    nw=rng.multinomial(n,np.ones(n)/n,size=2000)
    den=((sw@b)*nw).sum(1)
    good=den>0
    samples=(((sw@a)*nw).sum(1))[good]/den[good]
    return {"mean":float(a.sum()/b.sum()) if b.sum() else None,
            "ci95":np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
            "per_model":(a.sum(1)/np.maximum(b.sum(1),1)).tolist(),
            "numerator":float(a.sum()),"denominator":float(b.sum()),
            "undefined_bootstraps":int((~good).sum())}

def transitions(before,after):
    return {"different":interval((before>.02)!=(after>.02)),
            "false_safe":interval((before>.02)&(after<=.02)),
            "false_danger":interval((before<=.02)&(after>.02))}

def stat_error(x):
    return {"signed":interval(x),"absolute":interval(abs(x)),
            "p05_p50_p95":np.quantile(x,[.05,.5,.95]).tolist()}

def report(a):
    n=len(a["ref_cost"])
    stages={"reference512":a["ref_cost"],"pool512":a["pool_cost"],"target64":a["target_cost"],
            "restored":a["restored_cost"],"minus2":a["minus2_cost"]}
    ids=[choose(c,a["tie_uniform"]) for c in a["restored_cost"]]
    def fixed(c):
        bank=c if c.ndim==5 else np.broadcast_to(c,(5,*c.shape))
        return np.asarray([deployed(selected(v,j),a) for v,j in zip(bank,ids)])
    fc={k:fixed(v) for k,v in stages.items()}
    check=fixed(a["check_cost"])
    oldcheck=fixed(a["old_ref256_cost"])
    emodel=fc["restored"]-fc["target64"]
    esource=fc["target64"]-fc["pool512"]
    epool=fc["pool512"]-fc["reference512"]
    etotal=fc["restored"]-fc["reference512"]
    np.testing.assert_allclose(emodel+esource+epool,etotal,rtol=0,atol=1e-14)
    measurement=esource+epool
    # Selected action's two channels, independent of the deployed card formula.
    def fixed_e(e):
        bank=e if e.ndim==5 else np.broadcast_to(e,(5,*e.shape))
        return np.asarray([v[np.arange(n),a["deploy_variant"],deployed(j,a)] for v,j in zip(bank,ids)])
    ex={k:fixed_e(a[k]) for k in ["ref_e","pool_e","target_e","restored_e"]}
    eparts={"model":ex["restored_e"]-ex["target_e"],"source":ex["target_e"]-ex["pool_e"],
            "pooling":ex["pool_e"]-ex["ref_e"]}
    env=a["envelope"]
    envcost=np.stack([env[...,0],np.zeros_like(env[...,0]),env.sum(-1),env[...,1]],axis=-1)
    bc=a["boundary"]
    bcost=np.stack([bc[...,0],np.zeros_like(bc[...,0]),bc.any(-1),bc[...,1]],axis=-1)
    envelope=fixed(envcost); boundary=fixed(bcost).astype(bool)
    margin=abs(fc["pool512"]-.02)
    flip=(fc["reference512"]>.02)!=(fc["pool512"]>.02)
    robust=flip&((fc["reference512"]>.02)==(check>.02))
    strata={"boundary_small_margin":boundary&(margin<=.005),
            "boundary_larger_margin":boundary&(margin>.005),
            "no_joint_boundary":~boundary,
            "envelope_below_margin":envelope<margin-1e-12,
            "envelope_reaches_margin":envelope>=margin-1e-12}
    strat={k:{"observations":int(v.sum()),"flip_rate":interval(flip&v,v),
              "robust_flip_rate":interval(robust&v,v),"signed_error":interval(epool*v,v)} for k,v in strata.items()}
    reselect={}
    refrows=rows(a["ref_cost"],a["ref_cost"],a)
    for k,c in stages.items():
        bank=c if c.ndim==5 else c[None]
        rr=[rows(a["ref_cost"],v,a) for v in bank]
        stack=lambda key:np.asarray([v[key] for v in rr])
        accepted,unsafe,safe,oracle=[stack(key) for key in ["accepted","unsafe_accepted","safe_accepted","oracle"]]
        reselect[k]={"per_model":[summarize(v) for v in rr],"acceptance":interval(accepted),
                     "conditional_unsafe":interval(unsafe,accepted),"eta":interval(safe,oracle),
                     "regret":interval(stack("regret")),
                     "acceptance_diff_vs_ref":interval(accepted!=refrows["accepted"]),
                     "lost_safe_opportunity":interval(oracle&~safe),
                     "candidate_diff_vs_ref":interval(stack("chosen")!=refrows["chosen"])}
    trans={k:transitions(fc["reference512"],v) for k,v in fc.items() if k!="reference512"}
    adjacent={"pool_to_target":transitions(fc["pool512"],fc["target64"]),
              "target_to_model":transitions(fc["target64"],fc["restored"])}
    code=sum((fc[k]>.02).astype(int)*(2**i) for i,k in enumerate(["reference512","pool512","target64","restored"]))
    return {"n_scenes":n,"n_models":5,"unit":"one stored deployed variant/card per scene; model rows share scenes",
            "fixed_restored_selection":{"transitions_vs_reference":trans,"adjacent_transitions":adjacent,
                "signed_cost_terms":{k:stat_error(v) for k,v in {"model":emodel,"source":esource,"pooling":epool,"total":etotal}.items()},
                "signed_exposure_terms":{k:{c:stat_error(v[...,i]) for i,c in enumerate(["water","fragile"])} for k,v in eparts.items()},
                "opposing_model_and_measurement":interval((emodel*measurement)<-1e-18),
                "sum_component_absolute_minus_absolute_total":interval(abs(emodel)+abs(esource)+abs(epool)-abs(etotal)),
                "danger_pattern_counts_bit_order_ref_pool_target_model":{str(i):int((code==i).sum()) for i in range(16)},
                "strata":strat,"pooling_flips_stable_to_fixed_footprint_check":interval(robust),
                "fixed_footprint_numeric_label_change":interval((fc["reference512"]>.02)!=(check>.02)),
                "fixed_footprint_numeric_abs_error":stat_error(check-fc["reference512"]),
                "old_full_raster256_label_change":interval((fc["reference512"]>.02)!=(oldcheck>.02)),
                "pooling_flips_with_no_joint_boundary":int((flip&~boundary).sum()),
                "pooling_flips_outside_covariance_envelope":int((flip&(envelope<margin-1e-12)).sum())},
            "reselection":reselect,
            "all_candidate_diagnostic":{"unit":"all 2 variants x 4 candidates x 3 nontrivial cards; descriptive only",
                "pooling_label_change":float(np.mean(((a["pool_cost"]>.02)!=(a["ref_cost"]>.02))[...,[0,2,3]])),
                "covariance_identity_max_abs":float(abs((a["pool_e"]-a["ref_e"])-a["covariance_delta"]).max())}}

def main(root):
    out=root/"report";out.mkdir(exist_ok=False)
    write_json(out/"SOURCE_FREEZE.json",{"at_utc":now(),"sha256":sha(__file__)})
    verify_manifest(root/"discovery")
    result={"at_utc":now(),"stage":"DISCOVERY_ONLY_NO_NEW_CERTIFICATE","splits":{}}
    for name in ["iid_test","target_test"]:
        result["splits"][name]=report(load_npz(root/"discovery"/f"{name}.npz"))
        print(name, "summary complete",flush=True)
    write_json(out/"SUMMARY.json",result)
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
