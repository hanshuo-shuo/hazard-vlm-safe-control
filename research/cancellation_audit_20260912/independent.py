"""One frozen independent input-by-operator contrast; no new optimization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OLD=Path('/projects/p33100/siosio/hazard_oracle_aggregation_20260912')
MODELS=Path('/projects/p33100/siosio/hazard_pro_decision_round_20260912')
PLIC=Path('/projects/p33100/siosio/hazard_same_information_reconstruction_20260912')
sys.path.insert(0,str(OLD/'research/oracle_aggregation_20260912'))
from confirm import materialize,canonical
from common import protocol,deployment,formula,exposure,build_candidate,infer_logits,probabilities
from geometry import continuous_scene
sys.path.insert(0,str(PLIC/'research/same_information_reconstruction_20260912'))
from plic import integrate
from evaluation.oracle_spatial_exposure import rasterize_polyline
sys.path.insert(0,str(HERE))
from audit import sha,write,extract,analyze


def now():return datetime.now(timezone.utc).isoformat()


def worker(task):
    coarse,points,radius=task
    footprints=np.asarray([rasterize_polyline(512,512,path,radius) for path in points],float)
    u,r,a=integrate(coarse,footprints)
    return u.swapaxes(-2,-1),r.swapaxes(-2,-1),a


def effect(d,weights):
    r=d['r'];sa=lambda score:((score<=.02)&(r<=.02)).astype(int)
    v=sa(d['q1'])-sa(d['q0'])-sa(d['t1'])+sa(d['t0'])
    per_scene=v.mean(0);draws=(weights@per_scene)/len(per_scene)
    return dict(mean=float(per_scene.mean()),one_sided_upper95=float(np.quantile(draws,.95)),
        ci95=np.quantile(draws,[.025,.975]).tolist(),pass_criterion=bool(np.quantile(draws,.95)<-.005),
        safe_gain_learned=float((sa(d['q1'])-sa(d['q0'])).mean()),
        safe_gain_target=float((sa(d['t1'])-sa(d['t0'])).mean()))


def main(root):
    spec=json.loads((HERE/'independent_protocol.json').read_text());p=protocol()
    out=root/'independent';out.mkdir(exist_ok=False)
    ckpts=[MODELS/'e2_runs/independent_64'/f'seed_{i}/model.pt' for i in range(5)]
    imported={str(Path(mod.__file__).resolve()):sha(Path(mod.__file__).resolve()) for mod in list(sys.modules.values())
        if getattr(mod,'__file__',None) and str(Path(mod.__file__).resolve()).startswith('/projects/p33100/siosio/')
        and Path(mod.__file__).is_file()}
    write(out/'FREEZE.json',dict(at_utc=now(),new_records_generated=False,
        protocol_sha256=sha(HERE/'independent_protocol.json'),script_sha256=sha(HERE/'independent.py'),
        imported_project_files=imported,checkpoint_hashes={str(f):sha(f) for f in ckpts},
        old_generator_protocol_sha256=sha(MODELS/'research/pro_decision_round_20260912/protocol.json'),
        numpy=np.__version__,torch=torch.__version__))
    n=spec['dataset']['count'];seed=spec['dataset']['seed']
    records=[continuous_scene(seed,i,'independent',p['appearance']['train'],p) for i in range(n)]
    old=[set(),set()];files={}
    paths=[]
    for stage in ('e2_data','e3_data'):
        paths += [f for f in sorted((MODELS/stage).glob('*.json')) if f.name not in ('AUDIT.json','MANIFEST.json')]
    paths.append(OLD/'confirmation/records.json')
    for path in paths:
        files[str(path)]=sha(path)
        for record in json.loads(path.read_text()):
            for j,h in enumerate(canonical(record)):old[j].add(h)
    fresh=[set(),set()]
    for record in records:
        for j,h in enumerate(canonical(record)):
            assert h not in old[j] and h not in fresh[j], 'Identity overlap; abort, do not replace scene'
            fresh[j].add(h)
    write(out/'IDENTITY.json',dict(at_utc=now(),prior_unique=[len(x) for x in old],new_unique=[len(x) for x in fresh],
        exact_overlap=0,record_sources=files,scope='Anonymous exact patch and complete-scene identity; not arbitrary symmetry equivalence'))
    write(out/'records.json',records)
    values=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,v in enumerate(pool.map(materialize,records,chunksize=8)):
            values.append(v)
            if (i+1)%500==0:print('geometry',i+1,now(),flush=True)
    data={k:np.asarray([v[k] for v in values]) for k in values[0]};del values
    data.update(deployment(seed,n));coarse=[data['target64']]
    torch.set_num_threads(4)
    for i,path in enumerate(ckpts):
        freeze=json.loads((out/'FREEZE.json').read_text())
        assert sha(path)==freeze['checkpoint_hashes'][str(path)]
        model=build_candidate(p['model_seeds'][i]);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        logits=infer_logits(model,data['rgb'],torch.device('cpu'))
        np.savez_compressed(out/f'logits_{i}.npz',logits=logits)
        coarse.append(probabilities(logits));print('frozen inference',i,now(),flush=True)
    coarse=np.asarray(coarse)
    uniform=np.stack([formula(exposure(field,data['common_footprints'])) for field in coarse])
    truth=formula(data['ref_e']);check=formula(data['check_e'])
    a={k:data[k] for k in ('deploy_variant','deploy_card','tie_uniform')}
    a.update(truth=truth,uniform_cost=uniform,plic_cost=uniform)
    before=extract(a);flag=(before['t0']-before['r'])*(before['q0']-before['t0']) < -1e-18
    np.savez_compressed(out/'BASELINE_FLAGS.npz',q0=before['q0'],t0=before['t0'],r=before['r'],action=before['action'],flag=flag)
    write(out/'BASELINE_FREEZE.json',dict(at_utc=now(),sha256=sha(out/'BASELINE_FLAGS.npz'),
        plic_computed=False,note='Oracle flag frozen before PLIC outputs; still requires reference labels'))
    values=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks=((coarse[:,i],r['points'],r['radius']) for i,r in enumerate(records))
        for i,v in enumerate(pool.map(worker,tasks,chunksize=4)):
            values.append(v)
            if (i+1)%500==0:print('fixed PLIC',i+1,now(),flush=True)
    measured_u=np.asarray([v[0] for v in values]).swapaxes(0,1)
    plic_e=np.asarray([v[1] for v in values]).swapaxes(0,1)
    mass=max(max(v[2]['max_inverse_mass_error'],v[2]['max_fine_mass_error']) for v in values)
    np.testing.assert_allclose(formula(measured_u),uniform,atol=1e-12,rtol=0)
    a['plic_cost']=formula(plic_e);d=extract(a)
    for key in ('q0','t0','r','action'):np.testing.assert_array_equal(d[key],before[key])
    np.testing.assert_array_equal((d['t0']-d['r'])*(d['q0']-d['t0']) < -1e-18,flag)
    weights=np.random.default_rng(2612091702).multinomial(n,np.ones(n)/n,size=5000)
    primary=effect(d,weights)
    check_data={**a,'truth':check};sensitivity=effect(extract(check_data),weights)
    discrepancy=np.abs(truth-check)[...,[0,2,3]]
    label_difference=float(((truth>.02)!=(check>.02))[...,[0,2,3]].mean())
    numeric=dict(label_difference=label_difference,p95=float(np.quantile(discrepancy,.95)),mass_max=mass,
        uniform_repeat_max=float(np.abs(formula(measured_u)-uniform).max()))
    numeric['pass']=label_difference<=.01 and numeric['p95']<=.002 and mass<=1e-10 and sensitivity['pass_criterion']
    result=dict(at_utc=now(),primary=primary,finer_property_sensitivity=sensitivity,numeric_guard=numeric,
        independent_scenes=n,fixed_models=5,model_scene_entries=5*n,
        decision='SUPPORTED_FIXED_INPUT_OPERATOR_INTERACTION' if primary['pass_criterion'] and numeric['pass'] else 'NOT_SUPPORTED_OR_UNCERTAIN',
        scope='Single independent computational interaction prediction; fixed model bank and generator. Not decoder superiority, deployable cancellation prediction, or a safety certificate.')
    write(out/'PRIMARY.json',result)
    secondary,_,_=analyze(d,json.loads((HERE/'protocol.json').read_text()))
    write(out/'SECONDARY_COHORTS.json',secondary)
    a.update(truth_check=check,coarse=coarse,uniform_e=measured_u,plic_e=plic_e,rgb=data['rgb'],scene_ids=np.asarray([r['id'] for r in records]))
    np.savez_compressed(out/'data.npz',**a)
    np.savez_compressed(out/'DECISIONS.npz',**d)
    freeze=json.loads((out/'FREEZE.json').read_text())
    for path,h in freeze['imported_project_files'].items():assert sha(path)==h
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
