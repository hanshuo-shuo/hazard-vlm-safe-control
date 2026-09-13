"""Independent replay and publication of the one newly frozen interaction test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from independent import (sha,write,MODELS,protocol,build_candidate,infer_logits,probabilities,
                         rasterize_polyline,integrate)
from verify_publish import argmin
from decompose import raster,formula


def main(root):
    stage=root/'independent';out=root/'independent_verification';out.mkdir(exist_ok=False)
    for name,h in json.loads((stage/'MANIFEST.json').read_text()).items():assert sha(stage/name)==h
    freeze=json.loads((stage/'FREEZE.json').read_text())
    for name,h in freeze['imported_project_files'].items():assert sha(name)==h
    for name,h in freeze['checkpoint_hashes'].items():assert sha(name)==h
    with np.load(stage/'data.npz') as z:a={k:z[k] for k in z.files}
    with np.load(stage/'DECISIONS.npz') as z:d={k:z[k] for k in z.files}
    with np.load(stage/'BASELINE_FLAGS.npz') as z:before={k:z[k] for k in z.files}
    baseline_freeze=json.loads((stage/'BASELINE_FREEZE.json').read_text())
    assert sha(stage/'BASELINE_FLAGS.npz')==baseline_freeze['sha256']
    result=json.loads((stage/'PRIMARY.json').read_text());records=json.loads((stage/'records.json').read_text())
    n=len(records);sums=np.zeros(n);check_sums=np.zeros(n)
    cells={name:dict(safe=0,unsafe=0,accepted=0) for name in ('target_uniform','target_plic','learned_uniform','learned_plic')}
    value_rows={name:[] for name in cells};scalars=0;flags=0
    for m in range(5):
        for i in range(n):
            v,k=int(a['deploy_variant'][i]),int(a['deploy_card'][i]);u=a['tie_uniform'][i,v,k]
            action=argmin(a['uniform_cost'][m+1,i,v,:,k],u)
            r=float(a['truth'][i,v,action,k]);rc=float(a['truth_check'][i,v,action,k])
            q0=float(a['uniform_cost'][m+1,i,v,action,k]);q1=float(a['plic_cost'][m+1,i,v,action,k])
            t0=float(a['uniform_cost'][0,i,v,action,k]);t1=float(a['plic_cost'][0,i,v,action,k])
            assert int(d['action'][m,i])==int(before['action'][m,i])==action
            for key,x in dict(q0=q0,q1=q1,t0=t0,t1=t1,r=r).items():assert d[key][m,i]==x;scalars+=1
            for key,x in dict(q0=q0,t0=t0,r=r).items():assert before[key][m,i]==x
            flag=(t0-r)*(q0-t0)<-1e-18;assert flag==before['flag'][m,i];flags+=flag
            safe=lambda x,truth:int(x<=.02 and truth<=.02)
            sums[i]+=safe(q1,r)-safe(q0,r)-safe(t1,r)+safe(t0,r)
            check_sums[i]+=safe(q1,rc)-safe(q0,rc)-safe(t1,rc)+safe(t0,rc)
            for name,x in dict(target_uniform=t0,target_plic=t1,learned_uniform=q0,learned_plic=q1).items():
                cells[name]['accepted']+=x<=.02;cells[name]['safe']+=safe(x,r);cells[name]['unsafe']+=x<=.02 and r>.02
                value_rows[name].append(x)
    weights=np.random.default_rng(2612091702).multinomial(n,np.ones(n)/n,size=5000)
    for vector,key in [(sums,'primary'),(check_sums,'finer_property_sensitivity')]:
        mean=float(vector.sum()/(5*n));upper=float(np.quantile((weights@vector)/(5*n),.95))
        assert abs(mean-result[key]['mean'])<1e-14 and abs(upper-result[key]['one_sided_upper95'])<1e-14
        assert bool(upper<-.005)==result[key]['pass_criterion']
    small=np.linspace(0,n-1,8,dtype=int);ref_error=0.;plic_error=0.
    for i in small:
        ref=formula(raster(records[i],512)[2]);ref_error=max(ref_error,float(np.abs(ref-a['truth'][i]).max()))
        legal=records[i];f=np.asarray([rasterize_polyline(512,512,pts,legal['radius']) for pts in legal['points']],float)
        uu,rr,aa=integrate(a['coarse'][:2,i],f)
        expected=formula(rr.swapaxes(-2,-1));plic_error=max(plic_error,float(np.abs(expected-a['plic_cost'][:2,i]).max()))
    assert ref_error<1e-12 and plic_error<1e-12
    torch.set_num_threads(4);p=protocol();model_error=0.
    for i in range(5):
        model=build_candidate(p['model_seeds'][i]);model.load_state_dict(torch.load(MODELS/'e2_runs/independent_64'/f'seed_{i}/model.pt',map_location='cpu',weights_only=True))
        pred=probabilities(infer_logits(model,a['rgb'][small],torch.device('cpu')))
        model_error=max(model_error,float(np.abs(pred-a['coarse'][i+1,small]).max()))
    assert model_error<5e-6
    write(out/'VERIFICATION.json',dict(status='PASS',scalar_records=5*n,original_score_comparisons=scalars,
        frozen_baseline_flags=int(flags),bootstrap_primary_and_sensitivity_reproduced=True,
        reference_recomputed_scenes=len(small),reference_max_abs=ref_error,PLIC_replay_max_abs=plic_error,
        CPU_model_replay_images=5*len(small)*2,CPU_field_max_abs=model_error,source_and_checkpoint_hashes_unchanged=True))
    write(out/'CELLS.json',cells)
    primary=result['primary'];passed=result['decision']=='SUPPORTED_FIXED_INPUT_OPERATOR_INTERACTION'
    verb='supports' if passed else 'does not establish'
    paragraph=(f"On this independent bank, reconstruction increases safe acceptance by {primary['safe_gain_target']*100:.2f} pp with target inputs and {primary['safe_gain_learned']*100:.2f} pp with learned inputs. "
        f"Their difference is {primary['mean']*100:.2f} pp, with one-sided 95\\% bootstrap upper bound {primary['one_sided_upper95']*100:.2f} pp. "
        f"This {verb} the predeclared attenuation prediction; the same decision holds under the finer-property check. "
        "These are percentage points of all scenes, averaged over the fixed model bank, not differences at matched risk. "
        "The result tests a computational input--operator interaction and does not validate the opposition flag as a deployment predictor.\n")
    (out/'intervention_results.tex').write_text('% Generated from the frozen independent primary result.\n'+paragraph)
    abstract=(f"A separate frozen input-by-operator intervention test on 2,000 independent scenes {verb} attenuation of the reconstruction's safe-acceptance gain with learned input "
        f"(interaction {primary['mean']*100:.2f} pp; one-sided upper bound {primary['one_sided_upper95']*100:.2f} pp).\n")
    (out/'intervention_abstract.tex').write_text('% Independent interaction test, not score superiority.\n'+abstract)
    tex=[r'\begin{table}[h]',r'\centering\small',r'\caption{Actual four-cell intervention on the independent bank, at the same original learned-uniform-selected actions. Counts sum five fixed models on 2,000 shared scenes. These are actual operator outputs, not the algebraic residual hybrids.}',r'\label{tab:independentcells}',r'\begin{tabular}{lrrr}',r'\toprule',r'Input and aggregator & Accepted & Safe & Unsafe\\',r'\midrule']
    for name,r in cells.items():tex.append(f"{name.replace('_',' ')} & {r['accepted']} & {r['safe']} & {r['unsafe']}\\\\")
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'independent_cells.tex').write_text('\n'.join(tex)+'\n')
    secondary=json.loads((stage/'SECONDARY_COHORTS.json').read_text());st=secondary['overlap_standardization']
    full=secondary['full_pattern_denominators'];ra=secondary['cohorts']['original_rejected']
    lines=['# Independent input-by-operator result','',f"Decision: **{result['decision']}**.",'',
        f"Safe-accept gain, target: {primary['safe_gain_target']*100:.3f} pp; learned: {primary['safe_gain_learned']*100:.3f} pp.",
        f"Interaction: {primary['mean']*100:.3f} pp; one-sided 95% upper: {primary['one_sided_upper95']*100:.3f} pp.",
        f"Finer-property sensitivity interaction: {result['finer_property_sensitivity']['mean']*100:.3f} pp; upper: {result['finer_property_sensitivity']['one_sided_upper95']*100:.3f} pp.",'',
        '## Secondary frozen oracle-cohort analysis','',
        f"Originally rejected flagged: {ra['flagged']['new_harms']}/{ra['flagged']['n']}; unflagged: {ra['unflagged']['new_harms']}/{ra['unflagged']['n']} newly unsafe.",
        f"Overlap-standardized difference: {st['difference']*100:.3f} pp; pointwise descriptive interval: {[v*100 for v in st['pointwise_scene_ci95']]}.",
        f"Both components improve in {full['both_improve_all']} entries; total absolute error worsens in {full['both_improve_worse_all']}; newly unsafe: {full['both_improve_new_harm']}.",'',
        'These flags were saved before computing PLIC but require reference labels. They are oracle-conditioned associations, not label-free risk prediction or causal shares. The single primary test is the computational safe-accept interaction; secondary outcomes do not create additional confirmed claims.','']
    (out/'TABLES.md').write_text('\n'.join(lines))
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print((out/'VERIFICATION.json').read_text());print((out/'TABLES.md').read_text())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
