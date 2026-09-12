"""Independent scalar decision and polygon checks; no repeated fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from run import HERE,load_npz,write_json,now,sha,verify_manifest,manifest,rasterize_polyline
from plic import integrate,normals,intercept,fractions
from test_plic import polygon_area

def argmin(values,u):
    v=[float(x) for x in values];m=min(v)
    tied=[i for i,x in enumerate(v) if abs(x-m)<=1e-9]
    return tied[int(float(u)*len(tied))]

def main(root):
    freeze=json.loads((root/'OPERATOR_FREEZE.json').read_text())
    for name,h in freeze['implementation_hashes'].items():assert sha(HERE/name)==h
    assert sha(HERE/'protocol.json')==freeze['protocol_sha256']
    s=json.loads((root/'report/SUMMARY.json').read_text())
    p=json.loads((HERE/'protocol.json').read_text());events=[json.loads(line) for line in (root/'report/EVENTS.jsonl').read_text().splitlines()]
    scalar=0;polygon_checks=0;max_polygon=0.;lift_max=0.;hashes={}
    for phase in ['development','appended']:
        hashes[phase]=verify_manifest(root/phase)
        a=load_npz(root/phase/'data.npz');n=len(a['truth'])
        for path,h in json.loads((root/phase/'AUDIT.json').read_text())['source_hashes'].items():assert sha(path)==h
        for track,indices in [('target',[0]),('learned',list(range(1,6)))]:
            for arm in ['uniform','shift','plic']:
                count=dict(recovered_safe=0,newly_lost_safe=0,new_unsafe_accepts=0,resolved_unsafe_accepts=0)
                for j,mi in enumerate(indices):
                    accepted=safe=bad=oracle=0
                    for i in range(n):
                        v,k=int(a['deploy_variant'][i]),int(a['deploy_card'][i]);u=a['tie_uniform'][i,v,k]
                        basecost=a['uniform_cost'][mi,i,v,:,k];bq=argmin(basecost,u)
                        bc=float(a['truth'][i,v,bq,k]);ba=float(basecost[bq])<=.02
                        bsa=ba and bc<=.02;bua=ba and bc>.02
                        cc=a['plic_cost'][mi,i,v,:,k] if arm=='plic' else basecost
                        q=argmin(cc,u) if arm=='plic' else bq
                        estimated=float(cc[q])-(float(a['offsets'][mi]) if arm=='shift' else 0.)
                        ac=estimated<=.02;actual=float(a['truth'][i,v,q,k]);sa=ac and actual<=.02;ua=ac and actual>.02
                        accepted+=ac;safe+=sa;bad+=ua;oracle+=min(a['truth'][i,v,:,k])<=.02
                        count['recovered_safe']+=sa and not bsa;count['newly_lost_safe']+=bsa and not sa
                        count['new_unsafe_accepts']+=ua and not bua;count['resolved_unsafe_accepts']+=bua and not ua
                        scalar+=1
                    row=s['phases'][phase][track]['arms'][arm]['per_model'][j]
                    assert (accepted,safe,bad,oracle)==(row['accepted'],row['safe_accepted'],row['unsafe_accepted'],row['oracle_safe'])
                for key,total in count.items():assert total==s['phases'][phase][track]['arms'][arm][key]['numerator']
        rp=Path(p['prior_models_root'])/'e3_data/iid_test.json' if phase=='development' else Path(p['prior_oracle_root'])/'confirmation/records.json'
        records=json.loads(rp.read_text())
        for i in np.linspace(0,n-1,8,dtype=int):
            legal=records[i];f=np.asarray([rasterize_polyline(512,512,pts,legal['radius']) for pts in legal['points']],float)
            for mi in [0,1]:
                fields=a['coarse'][mi,i,0]
                u,r,audit=integrate(fields,f)
                np.testing.assert_allclose(r.T,a['plic_e'][mi,i,0],atol=1e-12,rtol=0)
                lifted=np.repeat(np.repeat(fields,32,-2),32,-1)
                direct=np.einsum('cij,kij->ck',lifted,f)/f.sum((1,2))
                lift_max=max(lift_max,float(abs(direct-u).max()))
                nx,ny,valid=normals(fields)
                for c in range(2):
                    mixed=np.argwhere((fields[c]>0)&(fields[c]<1)&valid[c])
                    if not len(mixed):continue
                    y,x=map(int,mixed[len(mixed)//2]);value=fields[c,y,x];nxx,nyy=nx[c,y,x],ny[c,y,x]
                    alpha=float(intercept(value,nxx,nyy));fine=fractions([value],[nxx],[nyy])[0]
                    for fy in range(32):
                        for fx in range(32):
                            local=(alpha-nxx*fx/32-nyy*fy/32)*32
                            expected=polygon_area(nxx,nyy,local)
                            error=abs(expected-fine[fy,fx]);max_polygon=max(max_polygon,error)
                            assert error<1e-10;polygon_checks+=1
    # Independently count the stricter event pattern from raw scalar fields.
    group=[e for e in events if e['phase']=='appended' and e['track']=='learned' and e['arm']=='plic' and e['new_unsafe_accept']]
    matched=[]
    for e in group:
        mb=e['fixed_target_before']-e['actual_before'];ma=e['fixed_target_after']-e['actual_before']
        lb=e['fixed_uniform_estimate']-e['fixed_target_before']
        eb=e['fixed_uniform_estimate']-e['actual_before'];ea=e['fixed_arm_estimate']-e['actual_before']
        if e['uniform_action']==e['arm_action'] and mb*lb<-1e-18 and abs(ma)<abs(mb)-1e-12 and abs(ea)>abs(eb)+1e-12:matched.append(e)
    expected=json.loads((root/'report/EVENT_COHORTS.json').read_text())['learned/plic/new_unsafe_accept']
    assert len(matched)==expected['same_action_opposed_terms_improved_measurement_worse_total']
    out=root/'verification';out.mkdir(exist_ok=False)
    write_json(out/'VERIFICATION.json',dict(status='PASS',at_utc=now(),scalar_decisions=scalar,
        independent_fine_polygon_areas=polygon_checks,max_polygon_difference=max_polygon,uniform_lift_max_difference=lift_max,
        stage_hashes=hashes,new_unsafe_learned_events=len(group),specific_cancellation_pattern_events=len(matched),
        source_freeze_unchanged=True,scope='Appended empirical comparison; not a fresh confirmation or risk certificate'))
    write_json(out/'MANIFEST.json',manifest(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
