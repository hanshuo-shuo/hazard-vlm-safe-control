"""One same-information comparison; consumed development and appended bank."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'pro_decision_round_20260912'))
from common import (load_npz,sha,write_json,now,manifest,verify_manifest,formula,
                    rows,summarize,choose,selected,deployed,build_candidate,infer_logits,probabilities)
from evaluation.oracle_spatial_exposure import rasterize_polyline
from plic import integrate
import test_plic

def worker(task):
    coarse,legal=task
    # Only action information crosses this boundary, no property geometry.
    f=np.asarray([rasterize_polyline(512,512,p,legal['radius']) for p in legal['points']],float)
    u,r,a=integrate(coarse,f)
    return u.swapaxes(-2,-1),r.swapaxes(-2,-1),a

def load_inputs(root,phase,p):
    modelroot=Path(p['prior_models_root']);old=Path(p['prior_oracle_root'])
    input_hashes={}
    if phase=='development':
        path=modelroot/'e3_data/iid_test.npz';recordpath=path.with_suffix('.json')
        d=load_npz(path);oldcost=load_npz(old/'discovery/iid_test.npz')
        records=json.loads(recordpath.read_text());fields=[d['target64']]
        torch.set_num_threads(4)
        for i,seed in enumerate([20261117,20261129,20261143,20261157,20261171]):
            ckpt=modelroot/'e2_runs/independent_64'/f'seed_{i}/model.pt'
            input_hashes[str(ckpt)]=sha(ckpt)
            model=build_candidate(seed);model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True))
            logits=infer_logits(model,d['rgb'],torch.device('cpu'))
            fields.append(probabilities(logits));print('CPU replay',i,now(),flush=True)
        truth=d['truth_ref512'];baseline=oldcost['restored_cost']
    else:
        path=old/'confirmation/data.npz';recordpath=old/'confirmation/records.json'
        d=load_npz(path);records=json.loads(recordpath.read_text());fields=[d['target64']]
        for i in range(5):
            f=old/'confirmation'/f'logits_{i}.npz';input_hashes[str(f)]=sha(f)
            fields.append(probabilities(load_npz(f)['logits']))
        truth=d['ref_cost'];baseline=d['restored_cost']
    for f in [path,recordpath]:input_hashes[str(f)]=sha(f)
    coarse=np.asarray(fields)
    assert coarse.shape==(6,2000,2,2,16,16)
    metadata={k:d[k] for k in ['deploy_variant','deploy_card','tie_uniform']}
    return coarse,records,truth,baseline,metadata,input_hashes

def fit_shift(costs,truth,d,p):
    result={}
    for track,bank in [('target',costs[:1]),('learned',costs[1:])]:
        base=[rows(truth,c,d) for c in bank]
        maxbad=sum(int(r['unsafe_accepted'].sum()) for r in base)
        candidates=[]
        for offset in p['global_shift']['offset_bank']:
            rr=[rows(truth,c,d,offset=-offset) for c in bank]
            good=sum(int(r['safe_accepted'].sum()) for r in rr)
            bad=sum(int(r['unsafe_accepted'].sum()) for r in rr)
            candidates.append(dict(offset=offset,safe_accepted=good,unsafe_accepted=bad,eligible=bad<=maxbad))
        eligible=[r for r in candidates if r['eligible']]
        winner=sorted(eligible,key=lambda r:(-r['safe_accepted'],abs(r['offset']),r['offset']))[0]
        result[track]=dict(selected=winner,candidates=candidates,baseline_unsafe_accepted=maxbad)
    return result

def main(root,phase):
    p=json.loads((HERE/'protocol.json').read_text());root.mkdir(parents=True,exist_ok=True)
    out=root/phase;out.mkdir(exist_ok=False)
    if phase=='development':
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_plic)
        test=unittest.TextTestRunner(verbosity=2).run(suite)
        if not test.wasSuccessful():raise AssertionError('PLIC contract failure')
        write_json(root/'SOURCE_FREEZE.json',dict(at_utc=now(),parent=p['parent_commit'],
            files={f.name:sha(f) for f in sorted(HERE.iterdir()) if f.is_file()},distinct_tests=test.testsRun))
    else:
        freeze=json.loads((root/'OPERATOR_FREEZE.json').read_text())
        for name,h in freeze['implementation_hashes'].items():assert sha(HERE/name)==h
        plicoffset=freeze['offsets']
    coarse,records,truth,oldbaseline,d,inputs=load_inputs(root,phase,p)
    values=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks=((coarse[:,i],dict(points=r['points'],radius=r['radius'])) for i,r in enumerate(records))
        for i,v in enumerate(pool.map(worker,tasks,chunksize=4)):
            values.append(v)
            if (i+1)%100==0:print(phase,'PLIC',i+1,now(),flush=True)
    u=np.asarray([v[0] for v in values]).swapaxes(0,1)
    r=np.asarray([v[1] for v in values]).swapaxes(0,1)
    uc,rc=formula(u),formula(r)
    if phase=='development':
        fitted=fit_shift(uc,truth,d,p)
        offsets=[fitted['target']['selected']['offset']]+[fitted['learned']['selected']['offset']]*5
        write_json(root/'OFFSET_DEVELOPMENT.json',fitted)
        write_json(root/'OPERATOR_FREEZE.json',dict(at_utc=now(),phase='before appended bank loaded',offsets=offsets,
            implementation_hashes={f.name:sha(f) for f in sorted(HERE.glob('*.py'))},protocol_sha256=sha(HERE/'protocol.json'),
            note='Appended bank already consumed by earlier study; no new untouched confirmation claimed'))
    else:offsets=plicoffset
    a={**d,'truth':truth,'uniform_cost':uc,'plic_cost':rc,'uniform_e':u,'plic_e':r,'coarse':coarse,
       'offsets':np.asarray(offsets),'scene_ids':np.asarray([r['id'] for r in records])}
    np.savez_compressed(out/'data.npz',**a)
    audit=dict(at_utc=now(),n=2000,tracks=6,source_hashes=inputs,
        max_inverse_mass_error=max(v[2]['max_inverse_mass_error'] for v in values),
        max_fine_mass_error=max(v[2]['max_fine_mass_error'] for v in values),
        gradient_fallback_cells=sum(v[2]['gradient_fallback_cells'] for v in values),
        reconstructed_query_cells=sum(v[2]['reconstructed_query_cells'] for v in values),
        learned_uniform_vs_prior_max_abs=float(abs(uc[1:]-oldbaseline).max()),
        learned_uniform_vs_prior_label_difference=int(((uc[1:]>.02)!=(oldbaseline>.02))[...,[0,2,3]].sum()),
        new_scenes=0,new_training=0,new_certificates=0)
    if phase=='appended':np.testing.assert_allclose(uc[1:],oldbaseline,rtol=0,atol=1e-12)
    write_json(out/'AUDIT.json',audit);write_json(out/'MANIFEST.json',manifest(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=['development','appended'],required=True)
    args=ap.parse_args();main(args.root,args.phase)
