"""Independent scalar audit plus the four actually evaluated module settings."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import sha,write,HERE


def argmin(c,u):
    minimum=min(float(v) for v in c)
    tied=[i for i,v in enumerate(c) if abs(float(v)-minimum)<=1e-9]
    return tied[int(float(u)*len(tied))]


def main(root):
    out=root/'verification_publication';out.mkdir(exist_ok=False)
    write(root/'FACTORIAL_FREEZE.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(HERE/'verify_publish.py'),addendum_sha256=sha(HERE/'factorial_addendum.json'),
        timing='After first audit, before computing four-cell factorial summaries; consumed data'))
    p=json.loads((HERE/'protocol.json').read_text())
    s=json.loads((root/'audit/SUMMARY.json').read_text())
    for name,h in json.loads((root/'AUDIT_FREEZE.json').read_text())['sources'].items():assert sha(HERE/name)==h
    for name,h in json.loads((root/'audit/MANIFEST.json').read_text()).items():assert sha(root/'audit'/name)==h
    with np.load(root/'audit/DECISIONS.npz') as z:d={k:z[k] for k in z.files}
    checked=0;score_checks=0;scalar_cells={};max_error=0.
    for phase in p['banks']:
        path=Path(p['input_root'])/phase/'data.npz';assert sha(path)==s['input_hashes'][str(path)]
        with np.load(path) as z:a={k:z[k] for k in ('uniform_cost','plic_cost','truth','tie_uniform','deploy_card','deploy_variant')}
        totals={arm:dict(accepted=0,safe=0,unsafe=0,new_harms=0,outside_unit_interval=0) for arm in ('q00','q10','q01','q11')}
        st=defaultdict(lambda:[0,0,0,0]);coh=defaultdict(lambda:[0,0]);kmax=0.
        for mi in range(5):
            for i in range(2000):
                v,k=int(a['deploy_variant'][i]),int(a['deploy_card'][i])
                action=argmin(a['uniform_cost'][mi+1,i,v,:,k],a['tie_uniform'][i,v,k])
                values=dict(q0=float(a['uniform_cost'][mi+1,i,v,action,k]),q1=float(a['plic_cost'][mi+1,i,v,action,k]),
                    t0=float(a['uniform_cost'][0,i,v,action,k]),t1=float(a['plic_cost'][0,i,v,action,k]),r=float(a['truth'][i,v,action,k]))
                assert action==d[phase+'__action'][mi,i]
                for name,value in values.items():assert value==d[phase+'__'+name][mi,i];score_checks+=1
                q0,q1,t0,t1,r=[values[name] for name in ('q0','q1','t0','t1','r')]
                m0,l0,m1,l1=t0-r,q0-t0,t1-r,q1-t1
                flag=m0*l0 < -1e-18;eligible=q0>.02;harm=eligible and r>.02 and q1<=.02
                c0=abs(m0)+abs(l0)-abs(m0+l0);c1=abs(m1)+abs(l1)-abs(m1+l1)
                residual=(abs(q1-r)-abs(q0-r))-((abs(m1)+abs(l1)-abs(m0)-abs(l0))-(c1-c0))
                max_error=max(max_error,abs(residual))
                for cohort,condition in [('all',True),('original_rejected',eligible),('newly_accepted',eligible and q1<=.02)]:
                    for group,condition2 in [('flagged',flag),('unflagged',not flag),('positive_m_negative_l',flag and m0>0),('negative_m_positive_l',flag and m0<0)]:
                        if condition and condition2:coh[cohort,group][0]+=1;coh[cohort,group][1]+=harm
                if eligible:
                    ma=bisect_right(p['standardization']['bin_inner_edges'],q0-.02)
                    change=bisect_right(p['standardization']['bin_inner_edges'],abs(q1-q0))
                    code=((((mi*4+k)*2+int(q1-q0<0))*7+ma)*7+change)
                    st[code][int(flag)]+=1;st[code][2+int(flag)]+=harm
                hybrid=dict(q00=r+m0+l0,q10=r+m1+l0,q01=r+m0+l1,q11=r+m1+l1)
                for name,score in hybrid.items():
                    aa=score<=.02;bb=aa and r>.02
                    totals[name]['accepted']+=aa;totals[name]['unsafe']+=bb;totals[name]['safe']+=aa and r<=.02
                    totals[name]['new_harms']+=bb and eligible
                    totals[name]['outside_unit_interval']+=score<0 or score>1
                checked+=1
        for (cohort,group),(n,h) in coh.items():
            expected=s['phases'][phase]['cohorts'][cohort][group]
            assert (n,h)==(expected['n'],expected['new_harms'])
        for arm,metrics in totals.items():
            for key,value in metrics.items():assert value==s['phases'][phase]['hybrids'][arm][key],(phase,arm,key,value)
        expected=s['phases'][phase]['overlap_standardization'];num=[0.,0.];den=0.
        for cell in expected['cells']:
            n0,n1,y0,y1=st[cell['stratum']]
            assert (n0,n1,y0,y1)==(cell['unflagged'],cell['flagged'],cell['unflagged_harm'],cell['flagged_harm'])
            w=min(n0,n1)
            if w:den+=w;num[0]+=w*y0/n0;num[1]+=w*y1/n1
        assert abs((num[1]-num[0])/den-expected['difference'])<1e-12
        scalar_cells[phase]=len(st)
    assert max_error<1e-12
    with gzip.open(root/'audit/ALL_EVENTS.csv.gz','rt') as f:
        rows=list(csv.DictReader(f))
        assert len(rows)==checked
        for row in rows:
            phase=row['phase'];mi,i=int(row['model']),int(row['scene'])
            for col,key in [('reference','r'),('q0','q0'),('q1','q1'),('target0','t0'),('target1','t1')]:
                assert float(row[col])==d[phase+'__'+key][mi,i]
    write(out/'VERIFICATION.json',dict(status='PASS',scalar_records=checked,original_score_comparisons=score_checks,
        full_csv_rows=len(rows),independent_standardization_strata=scalar_cells,max_identity_error=max_error,
        source_and_inputs_unchanged=True,scope='Independent numerical check, not human review or causal identification'))
    factorial={}
    for phase in p['banks']:
        at=lambda k:d[phase+'__'+k];truth=at('r');oracle=at('oracle');cards=at('card')
        values={'target_uniform':at('t0'),'target_plic':at('t1'),'learned_uniform':at('q0'),'learned_plic':at('q1')}
        metrics={name:dict(cost=x,absolute_error=np.abs(x-truth),safe=(x<=.02)&(truth<=.02),unsafe=(x<=.02)&(truth>.02)) for name,x in values.items()}
        cells={name:dict(accepted=int((x<=.02).sum()),safe=int(metrics[name]['safe'].sum()),unsafe=int(metrics[name]['unsafe'].sum()),
            eta=float(metrics[name]['safe'].sum()/oracle.sum()),mae=float(metrics[name]['absolute_error'].mean())) for name,x in values.items()}
        interactions={}
        rng=np.random.default_rng(2612091602);sw=rng.multinomial(2000,np.ones(2000)/2000,size=1000)
        for metric in ('cost','absolute_error','safe','unsafe'):
            v={k:x[metric].astype(float) for k,x in metrics.items()}
            interaction=(v['learned_plic']-v['learned_uniform'])-(v['target_plic']-v['target_uniform'])
            draws=(sw@interaction.mean(0))/2000
            interactions[metric]=dict(mean=float(interaction.mean()),pointwise_scene_ci95=np.quantile(draws,[.025,.975]).tolist(),
                per_model=interaction.mean(1).tolist(),per_card={str(k):float(interaction[cards==k].mean()) for k in (0,2,3)})
        factorial[phase]=dict(cells=cells,interactions=interactions,
            scope='Actually executed input-field by aggregator intervention, same learned-selected action; oracle input, not independent residual-module causation or deployed prediction')
    write(out/'FACTORIAL.json',factorial)
    lines=['# Full-cohort cancellation audit','',
        'Both banks are consumed. Five fixed models share 2,000 scenes per bank. The baseline opposition flag uses reference labels; all associations are oracle diagnostics.','',
        '| Bank | Originally rejected flagged: harm/n | Unflagged: harm/n | Raw rates | Overlap-standardized difference (pp) | Pointwise 95% interval (pp) |',
        '|---|---:|---:|---|---:|---|']
    for phase,r in s['phases'].items():
        a=r['cohorts']['original_rejected']['flagged'];b=r['cohorts']['original_rejected']['unflagged'];st=r['overlap_standardization'];lo,hi=st['pointwise_scene_ci95']
        lines.append(f"| {phase} | {a['new_harms']}/{a['n']} | {b['new_harms']}/{b['n']} | {a['harm_rate']*100:.3f}% / {b['harm_rate']*100:.3f}% | {st['difference']*100:+.3f} | [{lo*100:+.3f},{hi*100:+.3f}] |")
    lines += ['','## Full-population improvement denominators','',
        '| Bank | All records | Originally opposed | Both components improve | Both improve but total error worsens | Among these, newly unsafe |',
        '|---|---:|---:|---:|---:|---:|']
    for phase,r in s['phases'].items():
        a=r['full_pattern_denominators'];lines.append(f"| {phase} | {a['all_entries']} | {a['flagged']} | {a['both_improve_all']} | {a['both_improve_worse_all']} ({a['both_improve_worse_all']/a['both_improve_all']*100:.2f}%) | {a['both_improve_new_harm']} |")
    lines += ['','## Algebraic hybrids (not physical module interventions)','',
        '| Bank | Score | Safe accepted | Unsafe accepted | New harms | Outside [0,1] |','|---|---|---:|---:|---:|---:|']
    for phase,r in s['phases'].items():
        for arm,a in r['hybrids'].items():lines.append(f"| {phase} | {arm} | {a['safe']} | {a['unsafe']} | {a['new_harms']} | {a['outside_unit_interval']} |")
    lines += ['','## Actual input-field × aggregator intervention','',
        '| Bank | Input and aggregator | Safe accepted | Unsafe accepted | Eta | MAE |','|---|---|---:|---:|---:|---:|']
    for phase,r in factorial.items():
        for cell,a in r['cells'].items():lines.append(f"| {phase} | {cell} | {a['safe']} | {a['unsafe']} | {a['eta']*100:.3f}% | {a['mae']:.6f} |")
    lines += ['', '| Bank | Interaction: learned operator effect minus target operator effect | Mean | Pointwise 95% interval |','|---|---|---:|---|']
    for phase,r in factorial.items():
        for metric,a in r['interactions'].items():lines.append(f"| {phase} | {metric} | {a['mean']:+.6f} | {a['pointwise_scene_ci95']} |")
    lines += ['','The actual four-cell intervention does not make the residual terms independently manipulable modules. Hybrids are un-clipped scalar diagnostics and may be outside [0,1]. No outcome-defined subgroup is promoted to a deployment predictor.','']
    (out/'TABLES.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,2,figsize=(6.5,2.8),layout='constrained')
    for j,(phase,r) in enumerate(s['phases'].items()):
        a=r['cohorts']['original_rejected']['flagged'];b=r['cohorts']['original_rejected']['unflagged'];st=r['overlap_standardization'];lo,hi=st['pointwise_scene_ci95']
        ax[0].plot(j-.13,(a['harm_rate']-b['harm_rate'])*100,'o',color='#8b8b8b',label='Raw' if j==0 else None)
        ax[0].errorbar(j+.13,st['difference']*100,yerr=np.array([[st['difference']-lo],[hi-st['difference']]])*100,fmt='o',color='#2878a8',capsize=3,label='Overlap standardized' if j==0 else None)
    ax[0].axhline(0,color='black',lw=.6);ax[0].set(xticks=[0,1],xticklabels=['Development','Appended'],ylabel='Flagged - unflagged harm rate (pp)',title='A. Originally rejected actions');ax[0].legend(frameon=False,fontsize=6.8)
    labels=['Uniform','Measurement\nterm only','Model-relative\nterm only','Full PLIC']
    xx=np.arange(4)
    for phase,offset,color in [('development',-.18,'#86a9c1'),('appended',.18,'#c4764a')]:
        yy=[s['phases'][phase]['hybrids'][key]['new_harms'] for key in ('q00','q10','q01','q11')]
        ax[1].bar(xx+offset,yy,width=.36,color=color,label=phase.capitalize())
    ax[1].set(xticks=xx,xticklabels=labels,ylabel='Newly unsafe model-scene decisions',title='B. Algebraic score substitutions');ax[1].tick_params(axis='x',labelsize=6.5);ax[1].legend(frameon=False,fontsize=6.8)
    fig.savefig(out/'cohort_and_substitutions.png',dpi=220);fig.savefig(out/'cohort_and_substitutions.svg');plt.close(fig)
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print((out/'TABLES.md').read_text(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
