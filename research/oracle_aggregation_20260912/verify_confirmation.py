"""Independent scene counts and exact direction check, preserving original output."""
import argparse
import json
from pathlib import Path
import math
import numpy as np
from scipy.stats import beta
from decompose import HERE, load_npz, write_json, now, sha, verify_manifest, manifest, raster
from verify import scalar_cost

def select(costs,u):
    values=[float(v) for v in costs]
    ties=[i for i,v in enumerate(values) if abs(v-min(values))<=1e-9]
    return ties[int(u*len(ties))]

def main(root):
    src=root/"confirmation"
    hashes=verify_manifest(src)
    freeze=json.loads((src/"FREEZE.json").read_text())
    for category in ["sources","checkpoints"]:
        for name,digest in freeze[category].items(): assert sha(name)==digest
    assert sha(HERE/"confirmation_protocol.json")==freeze["protocol_sha256"]
    direction_freeze=json.loads((root/"DIRECTION_ADDENDUM_FREEZE.json").read_text())
    assert not direction_freeze["confirmation_hypotheses_exist"]
    assert sha(HERE/"confirmation_direction_addendum.json")==direction_freeze["source_sha256"]
    a=load_npz(src/"data.npz");n=len(a["ref_cost"])
    h=json.loads((src/"HYPOTHESES.json").read_text())
    s=json.loads((src/"SUMMARY.json").read_text())["splits"]["iid_confirmation"]
    alpha=h["one_sided_alpha_each"]
    scalar_decisions=0;h1=0
    for level,key in [("reference512","ref_cost"),("pool512","pool_cost"),("target64","target_cost"),
                      ("restored","restored_cost"),("minus2","minus2_cost")]:
        bank=a[key] if a[key].ndim==5 else a[key][None]
        for mi,costs in enumerate(bank):
            counts=dict(accepted=0,unsafe_accepted=0,safe_accepted=0,oracle_safe=0)
            for i in range(n):
                v,k=int(a["deploy_variant"][i]),int(a["deploy_card"][i]);u=float(a["tie_uniform"][i,v,k])
                q=select(costs[i,v,:,k],u)
                accept=costs[i,v,q,k]<=.02;danger=a["ref_cost"][i,v,q,k]>.02
                oracle=min(a["ref_cost"][i,v,:,k])<=.02
                for name,event in [("accepted",accept),("unsafe_accepted",accept and danger),
                                   ("safe_accepted",accept and not danger),("oracle_safe",oracle)]:counts[name]+=int(event)
                scalar_decisions+=1
                if level=="pool512":
                    stable=oracle and min(a["check_cost"][i,v,:,k])<=.02
                    h1+=int(stable and not (accept and not danger))
            for key,val in counts.items():assert val==s["reselection"][level]["per_model"][mi][key]
    assert h1==h["H1"]["events"]
    lower_h1=float(beta.ppf(alpha,h1,n-h1+1)) if h1 else 0.
    assert abs(lower_h1-h["H1"]["lower"])<1e-12
    change_scene=bad_direction_scene=flips=conservative=smalln=smallf=largen=largef=0
    for i in range(n):
        v,k=int(a["deploy_variant"][i]),int(a["deploy_card"][i]);u=float(a["tie_uniform"][i,v,k])
        changed=bad=False
        for mi in range(5):
            q=select(a["restored_cost"][mi,i,v,:,k],u)
            ref=float(a["ref_cost"][i,v,q,k]);pool=float(a["pool_cost"][i,v,q,k])
            flip=(ref>.02)!=(pool>.02);direction=ref<=.02 and pool>.02
            flips+=int(flip);conservative+=int(direction);changed|=flip;bad|=flip and not direction
            channels={0:[0],2:[0,1],3:[1]}[k]
            boundary=any(bool(a["boundary"][i,v,q,c]) for c in channels)
            if boundary and abs(pool-.02)<=.005:smalln+=1;smallf+=int(flip)
            if boundary and abs(pool-.02)>.005:largen+=1;largef+=int(flip)
        change_scene+=int(changed);bad_direction_scene+=int(bad)
    assert flips==h["H2"]["all_changes"] and conservative==h["H2"]["conservative_changes"]
    for key,val in [("small_events",smallf),("small_observations",smalln),("large_events",largef),("large_observations",largen)]:assert h["H3"][key]==val
    good=change_scene-bad_direction_scene
    lower_direction=float(beta.ppf(alpha,good,bad_direction_scene+1)) if good else 0.
    direct_checks=0;maxdiff=0.
    records=json.loads((src/"records.json").read_text())
    for i in np.linspace(0,n-1,16,dtype=int):
        p,f,_=raster(records[i],512)
        for q in range(4):
            ep=[]
            for c in range(2):
                num=0.
                for y in range(16):
                    for x in range(16):
                        num+=float(p[c,y*32:(y+1)*32,x*32:(x+1)*32].sum())*float(f[q,y*32:(y+1)*32,x*32:(x+1)*32].sum())/1024
                ep.append(num/float(f[q].sum()))
            for v in range(2):
                for k in [0,2,3]:
                    delta=abs(scalar_cost(ep if v==0 else ep[::-1],k)-a["pool_cost"][i,v,q,k])
                    assert delta<1e-13;maxdiff=max(maxdiff,delta);direct_checks+=1
    out=root/"confirmation_verification";out.mkdir(exist_ok=False)
    direction={"changed_scenes":change_scene,"all_changes_conservative_scenes":good,
               "any_anti_conservative_change_scenes":bad_direction_scene,"estimate":good/change_scene if change_scene else None,
               "exact_lower":lower_direction,"pass":lower_direction>.90,
               "scope":"Fixed five-model bank; conditional on at least one selected-action label change in a scene",
               "original_bootstrap_lower":h["H2"]["lower"],"bootstrap_boundary_warning":h["H2"]["lower"]==1.0}
    write_json(out/"VERIFICATION.json",{"status":"PASS","at_utc":now(),"artifact_hashes":hashes,
        "frozen_sources_unchanged":len(freeze["sources"]),"checkpoint_hashes_unchanged":len(freeze["checkpoints"]),
        "scalar_decisions":scalar_decisions,"blockwise_candidate_cost_checks":direct_checks,"block_cost_max_abs":maxdiff,
        "independent_beta_h1_lower":lower_h1,"exact_scene_direction_check":direction,
        "final_decision":"SUPPORTED_WITH_EXACT_SCENE_DIRECTION_CHECK" if h["decision"]=="SUPPORTED_FIXED_PREDICTION" and direction["pass"] else "NOT_FULLY_SUPPORTED_STOP_EXTENSION"})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
