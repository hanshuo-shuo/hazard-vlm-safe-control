"""Full-scene benefits and harms, plus fixed-action signed event evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from run import load_npz,write_json,now,manifest,verify_manifest,rows,summarize,choose,selected,deployed,HERE

class Bootstrap:
    def __init__(self,m,n):
        rng=np.random.default_rng(2612091401)
        self.sw=rng.multinomial(m,np.ones(m)/m,size=2000)/m
        self.nw=rng.multinomial(n,np.ones(n)/n,size=2000)
    def ratio(self,a,b=None):
        a=np.atleast_2d(a).astype(float)
        b=np.ones_like(a) if b is None else np.broadcast_to(b,a.shape).astype(float)
        den=((self.sw@b)*self.nw).sum(1);valid=den>0
        val=(((self.sw@a)*self.nw).sum(1))[valid]/den[valid]
        return dict(mean=float(a.sum()/b.sum()) if b.sum() else None,
            ci95=np.quantile(val,[.025,.975]).tolist() if len(val) else None,
            numerator=float(a.sum()),denominator=float(b.sum()),undefined=int((~valid).sum()))

def analyze(a,phase):
    truth=a['truth'];d=a;n=len(truth)
    event_rows=[];result={}
    for track,indices in [('target',[0]),('learned',list(range(1,6)))]:
        boot=Bootstrap(len(indices),n)
        all_rows={};fixed={};rowmetrics={}
        for arm in ['uniform','shift','plic']:
            rr=[];fc=[]
            for i in indices:
                c=a['plic_cost'][i] if arm=='plic' else a['uniform_cost'][i]
                offset=-a['offsets'][i] if arm=='shift' else 0.
                rr.append(rows(truth,c,d,offset=offset))
                ids=choose(a['uniform_cost'][i],a['tie_uniform'])
                fc.append(deployed(selected(c,ids),d)+offset)
            all_rows[arm]=rr;fixed[arm]=np.asarray(fc)
        base=all_rows['uniform']
        stack=lambda rr,key:np.asarray([r[key] for r in rr])
        bs=stack(base,'safe_accepted');bu=stack(base,'unsafe_accepted');oracle=stack(base,'oracle')
        actual=stack(base,'actual_cost')
        for arm,rr in all_rows.items():
            safe=stack(rr,'safe_accepted');bad=stack(rr,'unsafe_accepted');accepted=stack(rr,'accepted')
            recovery=safe&~bs;loss=bs&~safe;newbad=bad&~bu;resolvedbad=bu&~bad
            fa=fixed[arm]<=.02;fbad=fa&(actual>.02);fsafe=fa&(actual<=.02)
            fba=fixed['uniform']<=.02
            rowmetrics[arm]=dict(per_model=[summarize(r) for r in rr],
                acceptance=boot.ratio(accepted),conditional_unsafe=boot.ratio(bad,accepted),eta=boot.ratio(safe,oracle),
                recovered_safe=boot.ratio(recovery),newly_lost_safe=boot.ratio(loss),net_safe_gain=boot.ratio(safe.astype(int)-bs),
                new_unsafe_accepts=boot.ratio(newbad),resolved_unsafe_accepts=boot.ratio(resolvedbad),
                safe_opportunities_lost=boot.ratio(oracle&~safe),
                recovered_fraction_of_baseline_losses=boot.ratio(recovery,oracle&~bs),
                new_unsafe_same_action=boot.ratio(newbad&(stack(rr,'chosen')==stack(base,'chosen'))),
                fixed_selection=dict(mean_cost_change=boot.ratio(fixed[arm]-fixed['uniform']),
                    safe_to_danger_label=boot.ratio((fixed['uniform']<=.02)&(fixed[arm]>.02)),
                    danger_to_safe_label=boot.ratio((fixed['uniform']>.02)&(fixed[arm]<=.02)),
                    recovered_safe=boot.ratio(fsafe&~(fba&(actual<=.02))),
                    newly_lost_safe=boot.ratio((fba&(actual<=.02))&~fsafe),
                    new_unsafe_accepts=boot.ratio(fbad&~(fba&(actual>.02)))))
            if arm=='uniform':continue
            for j,i in enumerate(indices):
                ids=choose(a['uniform_cost'][i],a['tie_uniform'])
                at=lambda c:deployed(selected(c,ids),a)
                target_uniform=at(a['uniform_cost'][0]);target_plic=at(a['plic_cost'][0])
                target_after=target_plic if arm=='plic' else target_uniform-a['offsets'][0]
                before_error=fixed['uniform'][j]-actual[j];after_error=fixed[arm][j]-actual[j]
                meas_before=target_uniform-actual[j];meas_after=target_after-actual[j]
                model_before=fixed['uniform'][j]-target_uniform;model_after=fixed[arm][j]-target_after
                # This is an exact fixed-action decomposition, not a causal label.
                np.testing.assert_allclose((meas_after-meas_before)+(model_after-model_before),after_error-before_error,atol=1e-12,rtol=0)
                for q in np.where(recovery[j]|loss[j]|newbad[j]|resolvedbad[j])[0]:
                    event_rows.append(dict(phase=phase,track=track,model_index=i-1 if i else None,arm=arm,
                        scene_index=int(q),scene_id=str(a['scene_ids'][q]),variant=int(a['deploy_variant'][q]),card=int(a['deploy_card'][q]),
                        recovered_safe=bool(recovery[j,q]),newly_lost_safe=bool(loss[j,q]),new_unsafe_accept=bool(newbad[j,q]),resolved_unsafe=bool(resolvedbad[j,q]),
                        uniform_action=int(base[j]['chosen'][q]),arm_action=int(rr[j]['chosen'][q]),
                        actual_before=float(base[j]['actual_cost'][q]),actual_after=float(rr[j]['actual_cost'][q]),
                        fixed_uniform_estimate=float(fixed['uniform'][j,q]),fixed_arm_estimate=float(fixed[arm][j,q]),
                        fixed_target_before=float(target_uniform[q]),fixed_target_after=float(target_after[q]),
                        fixed_measurement_error_before=float(meas_before[q]),fixed_measurement_error_after=float(meas_after[q]),
                        fixed_model_error_before=float(model_before[q]),fixed_model_error_after=float(model_after[q]),
                        baseline_terms_opposed=bool(meas_before[q]*model_before[q]<-1e-18),
                        fixed_measurement_absolute_error_improves=bool(abs(meas_after[q])<abs(meas_before[q])-1e-12),
                        fixed_total_absolute_error_worsens=bool(abs(after_error[q])>abs(before_error[q])+1e-12)))
        result[track]=dict(n_scenes=n,n_models=len(indices),arms=rowmetrics)
    return result,event_rows

def main(root):
    out=root/'report';out.mkdir(exist_ok=False)
    summary=dict(at_utc=now(),status='RETROSPECTIVE_APPENDED_BASELINES_NO_NEW_CONFIRMATION',phases={})
    events=[]
    for phase in ['development','appended']:
        verify_manifest(root/phase)
        r,e=analyze(load_npz(root/phase/'data.npz'),phase)
        summary['phases'][phase]=r;events.extend(e)
        print('summarized',phase,now(),flush=True)
    write_json(out/'SUMMARY.json',summary)
    with (out/'EVENTS.jsonl').open('w') as f:
        for e in events:f.write(json.dumps(e,allow_nan=False)+'\n')
    cohorts={}
    for track in ['target','learned']:
        for arm in ['shift','plic']:
            for event in ['recovered_safe','newly_lost_safe','new_unsafe_accept']:
                group=[e for e in events if e['phase']=='appended' and e['track']==track and e['arm']==arm and e[event]]
                joint=[e for e in group if e['uniform_action']==e['arm_action'] and e['baseline_terms_opposed'] and e['fixed_measurement_absolute_error_improves'] and e['fixed_total_absolute_error_worsens']]
                cohorts[f'{track}/{arm}/{event}']=dict(events=len(group),same_action_opposed_terms_improved_measurement_worse_total=len(joint),
                    scope='Descriptive fixed-action pattern with explicit event identities; not proof that cancellation is the only cause')
    write_json(out/'EVENT_COHORTS.json',cohorts)
    write_json(out/'MANIFEST.json',manifest(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
