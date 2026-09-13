"""Independent original-array replay, direct sorted-label counts, and figures."""
import argparse
from bisect import bisect_right
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import sha, write, HERE


def scalar_choice(cost, u):
    minimum = min(float(x) for x in cost)
    tied = [i for i,x in enumerate(cost) if abs(float(x)-minimum)<=1e-9]
    return tied[int(float(u)*len(tied))]


def main(root):
    out=root/'verification_publication';out.mkdir(exist_ok=False)
    for name,h in json.loads((root/'ANALYSIS_FREEZE.json').read_text())['files'].items():
        assert sha(HERE/name)==h
    for name,h in json.loads((root/'analysis/MANIFEST.json').read_text()).items():
        assert sha(root/'analysis'/name)==h
    p=json.loads((HERE/'protocol.json').read_text());s=json.loads((root/'analysis/SUMMARY.json').read_text())
    with np.load(root/'analysis/DECISIONS.npz') as z:d={k:z[k] for k in z.files}
    replay=0
    for phase in p['banks']:
        path=Path(p['input_root'])/phase/'data.npz'
        assert sha(path)==s['inputs'][str(path)]
        with np.load(path) as archive:
            a={name:archive[name] for name in ('truth','deploy_variant','deploy_card','tie_uniform','uniform_cost','plic_cost')}
            for mi in range(6):
                for n in range(2000):
                    v,k=int(a['deploy_variant'][n]),int(a['deploy_card'][n])
                    u=float(a['tie_uniform'][n,v,k])
                    uniform=a['uniform_cost'][mi,n,v,:,k]
                    base=scalar_choice(uniform,u)
                    for mode in ('fixed','reselected'):
                        for arm in ('uniform','plic'):
                            c=a[arm+'_cost'][mi,n,v,:,k]
                            action=base if mode=='fixed' else scalar_choice(c,u)
                            key=f'{phase}__{mode}__{arm}__'
                            assert d[key+'action'][mi,n]==action
                            assert d[key+'score'][mi,n]==float(c[action])
                            assert d[key+'actual'][mi,n]==float(a['truth'][n,v,action,k])
                            replay+=1
    cache={};raw={};checked=0
    with gzip.open(root/'analysis/ALL_CURVES.csv.gz','rt') as f:
        for row in csv.DictReader(f):
            phase,mode,track,arm=[row[k] for k in ('phase','selector','track','arm')]
            key=(phase,mode,track,arm)
            if key not in cache:
                ids=[0] if track=='target' else list(range(1,6)) if track=='learned' else [int(track.split('_')[1])+1]
                prefix=f'{phase}__{mode}__{arm}__'
                scores=d[prefix+'score'][ids].ravel();bad=d[prefix+'bad'][ids].ravel()
                cache[key]=(sorted(scores[~bad].tolist()),sorted(scores[bad].tolist()))
                raw[key]=[]
            safe_scores,bad_scores=cache[key]
            if row['threshold']=='reject-all':
                assert row['conditional_risk']=='undefined' and int(row['accepted'])==0
                continue
            t=float(row['threshold']);safe=bisect_right(safe_scores,t);bad=bisect_right(bad_scores,t)
            assert (safe,bad,safe+bad)==tuple(int(row[k]) for k in ('safe','unsafe','accepted'))
            assert abs(bad/(safe+bad)-float(row['conditional_risk']))<1e-14
            raw[key].append((t,safe,bad,float(row['conditional_risk']),float(row['eta'])))
            checked+=1
    for phase in p['banks']:
        for mode in ('fixed','reselected'):
            for track,rec in s['phases'][phase][mode].items():
                for arm in ('uniform','plic'):
                    family=raw[phase,mode,track,arm]
                    for beta,row in zip(p['risk_budgets'],rec['arms'][arm]):
                        eligible=[r for r in family if r[3]<=beta]
                        if eligible:
                            winner=min(eligible,key=lambda r:(-r[1],r[2],r[0]))
                            assert (row['threshold'],row['safe'],row['unsafe'])==winner[:3]
                        else:assert row['accepted']==0 and row['risk'] is None
    write(out/'VERIFICATION.json',dict(status='PASS',scalar_score_replays=replay,
        exact_threshold_rows_verified=checked,source_and_input_hashes_unchanged=True,
        method='Scalar action choices from original arrays; every threshold count independently checked by binary search in separately sorted safe and unsafe score lists; all envelope optima checked by full family scan',
        scope='No independent data, causal validation, or risk certificate'))
    plt.rcParams.update({'font.size':7.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'uniform':'#2878a8','plic':'#d87324'}
    fig,axes=plt.subplots(2,2,figsize=(5.5,4.05),layout='constrained')
    for col,track in enumerate(('target','learned')):
        ax=axes[0,col]
        for phase,style in [('development','--'),('appended','-')]:
            for arm in ('uniform','plic'):
                a=np.asarray(raw[phase,'fixed',track,arm]);order=np.argsort(a[:,3],kind='stable')
                risk=a[order,3];eta=np.maximum.accumulate(a[order,4]);keep=risk<=.05
                risk=np.r_[0,risk[keep],.05];eta=np.r_[0,eta[keep],eta[keep][-1]]
                ax.step(risk*100,eta*100,where='post',color=colors[arm],ls=style,lw=1.15)
        ax.set(xlim=(.5,5),ylim=(94,100.2),title=['A. Perfect coarse target','B. Five learned fields'][col],ylabel='Safe opportunity use (%)')
        ax.set_xticks([.5,1,2,3,4,5]);ax.grid(alpha=.15)
        ax=axes[1,col]
        for phase,color,style in [('development','#6c6c6c','--'),('appended','#7b3f95','-')]:
            r=s['phases'][phase]['fixed'][track];delta=np.asarray(r['delta_eta'])*100;ci=np.asarray(r['pointwise_descriptive_ci95'])*100
            ax.plot(np.asarray(p['risk_budgets'])*100,delta,color=color,ls=style,lw=1,label=phase.capitalize())
            ax.fill_between(np.asarray(p['risk_budgets'])*100,ci[:,0],ci[:,1],color=color,alpha=.12)
        ax.axhline(0,color='black',lw=.65);ax.set(xlim=(.5,5),xlabel='Empirical risk ceiling (%)',ylabel='PLIC - uniform use (pp)',title=['C. Target difference','D. Learned difference'][col]);ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=colors[a],label=a.title()) for a in ('uniform','plic')]
    handles += [Line2D([0],[0],color='gray',ls=st,label=ph) for ph,st in [('Development','--'),('Appended','-')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    for ext in ('png','svg'):fig.savefig(out/f'matched_risk.{ext}',dpi=230)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5),layout='constrained')
    for ri,phase in enumerate(p['banks']):
        for ci,track in enumerate(('target','learned')):
            ax=axes[ri,ci]
            for arm in ('uniform','plic'):
                a=np.asarray(raw[phase,'fixed',track,arm])
                ax.plot(a[:,3]*100,a[:,4]*100,color=colors[arm],lw=.8,label=arm.title())
            ax.set(title=f'{phase.capitalize()}: {track}',xlabel='Actual empirical unsafe / accepted (%)',ylabel='Safe opportunity use (%)');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.savefig(out/'raw_full_families.png',dpi=180);plt.close(fig)
    lines=['# Full-family empirical risk comparison','',
        'Both banks were already consumed. Thresholds are outcome-selected within each bank; these are descriptive empirical envelopes, not certificates or out-of-sample rules. Zero acceptance has undefined risk. All equal scores enter together.','']
    for mode in ('fixed','reselected'):
        for phase in p['banks']:
            lines += [f'## {mode}, {phase}','', '| Track | Risk ceiling | Uniform eta | PLIC eta | Delta (pp) | Pointwise descriptive 95% interval (pp) |','|---|---:|---:|---:|---:|---|']
            for track in ('target','learned'):
                r=s['phases'][phase][mode][track]
                for j,beta in enumerate(p['risk_budgets']):
                    a,b=r['arms']['uniform'][j],r['arms']['plic'][j];lo,hi=r['pointwise_descriptive_ci95'][j]
                    lines.append(f"| {track} | {beta*100:.1f}% | {a['eta']*100:.3f}% | {b['eta']*100:.3f}% | {r['delta_eta'][j]*100:+.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] |")
            lines.append('')
    lines += ['## Development-selected thresholds on consumed appended bank','',
              'These thresholds are selected only on development and then applied unchanged; actual appended risk is reported and need not meet the development ceiling.','',
              '| Selector | Track | Development ceiling | Arm | Threshold | Actual appended risk | Appended eta |',
              '|---|---|---:|---|---:|---:|---:|']
    for mode,groups in s['development_selected_on_consumed_appended'].items():
        for track,arms in groups.items():
            for arm,rr in arms.items():
                for beta,r in zip(p['risk_budgets'],rr):
                    lines.append(f"| {mode} | {track} | {beta*100:.1f}% | {arm} | {r['threshold']} | {r['risk']*100:.3f}% | {r['eta']*100:.3f}% |")
    lines += ['',f"Continuation decision: **{s['continuation_gate']['decision']}**.",
        'Positive point estimates do not establish equivalence, inferiority, or a stable advantage. No new scene is generated when the fixed screen fails.','']
    (out/'TABLES.md').write_text('\n'.join(lines))
    tex=['% Generated on Quest from frozen retrospective curve summary.',r'\begin{table}[h]',r'\centering\small',
         r'\caption{Fixed-action empirical envelopes on the consumed appended bank. Differences and pointwise bootstrap intervals are in percentage points of safe opportunity use. Thresholds are selected with these same outcome labels.}',
         r'\label{tab:matched}',r'\begin{tabular}{lrrrrl}',r'\toprule',
         r'Input & Risk ceiling & Uniform $\eta$ & PLIC $\eta$ & Difference & 95\% interval\\',r'\midrule']
    for track in ('target','learned'):
        r=s['phases']['appended']['fixed'][track]
        for j in [1,3,5,7]:
            a,b=r['arms']['uniform'][j],r['arms']['plic'][j];lo,hi=r['pointwise_descriptive_ci95'][j]
            tex.append(f"{track.title()} & {p['risk_budgets'][j]*100:.0f}\\% & {a['eta']*100:.2f}\\% & {b['eta']*100:.2f}\\% & {r['delta_eta'][j]*100:+.2f} & [{lo*100:+.2f},{hi*100:+.2f}]\\\\")
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'matched_table.tex').write_text('\n'.join(tex)+'\n')
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps(json.loads((out/'VERIFICATION.json').read_text()),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
