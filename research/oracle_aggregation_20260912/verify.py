"""Independent scalar decision replay and selected blockwise oracle checks."""
import argparse
import json
from pathlib import Path
import unittest
import numpy as np
from decompose import (HERE, load_npz, write_json, now, sha, verify_manifest,
                       manifest, raster)
import test_decomposition

def scalar_cost(e,card):
    w,f=map(float,e)
    return {0:w,1:0.,2:w+f-w*f,3:f}[card]

def main(root):
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_decomposition)
    run=unittest.TextTestRunner(verbosity=2).run(suite)
    if not run.wasSuccessful(): raise AssertionError("Contract test failure")
    summary=json.loads((root/"report/SUMMARY.json").read_text())
    hashes=verify_manifest(root/"discovery")+verify_manifest(root/"report")
    inputs=json.loads((root/"discovery/INPUT_HASHES.json").read_text())
    for name,expected in inputs.items():
        assert sha(name)==expected
    prior=Path(json.loads((HERE/"protocol.json").read_text())["prior_root"])
    scalar_decisions=0; block_checks=0; max_diff=0.
    for name in ["iid_test","target_test"]:
        a=load_npz(root/"discovery"/f"{name}.npz")
        n=len(a["ref_cost"])
        for level,key in [("reference512","ref_cost"),("pool512","pool_cost"),("target64","target_cost"),
                          ("restored","restored_cost"),("minus2","minus2_cost")]:
            bank=a[key] if a[key].ndim==5 else a[key][None]
            for mi,costs in enumerate(bank):
                accepted=bad=safe=oracle=0; regret=0.
                for j in range(n):
                    v,k=int(a["deploy_variant"][j]),int(a["deploy_card"][j])
                    cs=[float(costs[j,v,q,k]) for q in range(4)]
                    ties=[q for q in range(4) if abs(cs[q]-min(cs))<=1e-9]
                    action=ties[int(float(a["tie_uniform"][j,v,k])*len(ties))]
                    actual=float(a["ref_cost"][j,v,action,k]); minimum=min(a["ref_cost"][j,v,:,k])
                    accept=cs[action]<=.02
                    accepted+=accept;bad+=accept and actual>.02;safe+=accept and actual<=.02
                    oracle+=minimum<=.02;regret+=actual-minimum;scalar_decisions+=1
                expected=summary["splits"][name]["reselection"][level]["per_model"][mi]
                for k,val in [("accepted",accepted),("unsafe_accepted",bad),("safe_accepted",safe),("oracle_safe",oracle)]:
                    assert val==expected[k],(name,level,mi,k,val,expected[k])
                assert abs(regret/n-expected["regret"])<1e-12
        records=json.loads((prior/"e3_data"/f"{name}.json").read_text())
        for j in np.linspace(0,n-1,16,dtype=int):
            p,f,_=raster(records[j],512)
            for q in range(4):
                den=float(f[q].sum()); ee=[]
                for channel in range(2):
                    num=0.
                    for y in range(16):
                        for x in range(16):
                            ps=p[channel,y*32:(y+1)*32,x*32:(x+1)*32]
                            fs=f[q,y*32:(y+1)*32,x*32:(x+1)*32]
                            num+=float(ps.sum())*float(fs.sum())/1024
                    ee.append(num/den)
                for v in range(2):
                    for k in [0,2,3]:
                        diff=abs(scalar_cost(ee if v==0 else ee[::-1],k)-a["pool_cost"][j,v,q,k])
                        max_diff=max(max_diff,diff);block_checks+=1
                        assert diff<1e-13
    out=root/"verification";out.mkdir(exist_ok=False)
    write_json(out/"VERIFICATION.json",{"status":"PASS","at_utc":now(),"distinct_tests":run.testsRun,
               "scalar_decisions":scalar_decisions,"blockwise_candidate_cost_checks":block_checks,
               "block_cost_max_abs":max_diff,"artifact_hashes":hashes,"prior_inputs_unchanged":len(inputs),
               "scope":"Computational consistency; does not independently confirm discovery or certify a new rule"})
    write_json(out/"MANIFEST.json",manifest(out))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    main(ap.parse_args().root)
