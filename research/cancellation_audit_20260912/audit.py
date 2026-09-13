"""Full-population denominators and oracle residual substitution accounting."""
import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def choose(c,u):
    tied=np.isclose(c,c.min(-1,keepdims=True),atol=1e-9,rtol=0)
    rank=np.floor(u*tied.sum(-1)).astype(int)
    return np.argmax(tied&(tied.cumsum(-1)==rank[...,None]+1),axis=-1)


def extract(a):
    n=len(a['truth']);ix=np.arange(n);v=a['deploy_variant'];k=a['deploy_card']
    costs={arm:np.stack([field[ix,v,:,k] for field in a[arm+'_cost']]) for arm in ('uniform','plic')}
    assert costs['uniform'].shape==(6,n,4)
    actions=choose(costs['uniform'][1:],a['tie_uniform'][ix,v,k])
    at=lambda c:np.take_along_axis(np.broadcast_to(c,(5,n,4)),actions[...,None],axis=-1)[...,0]
    return dict(q0=at(costs['uniform'][1:]),q1=at(costs['plic'][1:]),t0=at(costs['uniform'][0]),
                t1=at(costs['plic'][0]),r=at(a['truth'][ix,v,:,k]),action=actions,
                oracle=np.broadcast_to(a['truth'][ix,v,:,k].min(-1)<=.02,(5,n)),
                card=np.broadcast_to(k,(5,n)),variant=np.broadcast_to(v,(5,n)))


def derive(d):
    q0,q1,t0,t1,r=[d[k] for k in ('q0','q1','t0','t1','r')]
    m0,l0,m1,l1=t0-r,q0-t0,t1-r,q1-t1
    flag=m0*l0 < -1e-18
    risk=q0>.02;harm=(r>.02)&risk&(q1<=.02)
    improve_m=np.abs(m1)<np.abs(m0)-1e-12
    improve_l=np.abs(l1)<np.abs(l0)-1e-12
    worse=np.abs(q1-r)>np.abs(q0-r)+1e-12
    k0=np.abs(m0)+np.abs(l0)-np.abs(m0+l0)
    k1=np.abs(m1)+np.abs(l1)-np.abs(m1+l1)
    identity=(np.abs(q1-r)-np.abs(q0-r))-((np.abs(m1)+np.abs(l1)-np.abs(m0)-np.abs(l0))-(k1-k0))
    return dict(m0=m0,l0=l0,m1=m1,l1=l1,flag=flag,at_risk=risk,harm=harm,
                new_accept=risk&(q1<=.02),improve_m=improve_m,improve_l=improve_l,worse=worse,
                pattern=flag&improve_m&worse,both_improve=improve_m&improve_l,
                q00=q0,q10=q0+(t1-t0),q01=q1-(t1-t0),q11=q1,k0=k0,k1=k1,
                identity_max=float(np.abs(identity).max()))


def strata(d,edges):
    m,n=d['q0'].shape
    margin=np.maximum(d['q0']-.02,0);correction=d['q1']-d['q0']
    a=np.digitize(margin,edges);b=np.digitize(np.abs(correction),edges)
    code=((((np.arange(m)[:,None]*4+d['card'])*2+(correction<0))*7+a)*7+b)
    return code.astype(int),margin,np.abs(correction)


def standardized(flag,harm,eligible,code,weights=None,return_cells=False):
    f,y,e,c=[np.asarray(x).ravel() for x in (flag,harm,eligible,code)]
    w=np.ones(len(f)) if weights is None else np.asarray(weights).ravel()
    size=int(c.max())+1;ns=[];ys=[]
    for group in (False,True):
        select=e&(f==group)
        ns.append(np.bincount(c,weights=w*select,minlength=size))
        ys.append(np.bincount(c,weights=w*select*y,minlength=size))
    common=(ns[0]>0)&(ns[1]>0);overlap=np.minimum(ns[0],ns[1]);total=float(overlap.sum())
    rates=[np.divide(y,n,out=np.zeros_like(y),where=n>0) for y,n in zip(ys,ns)]
    mean=[float((overlap*r).sum()/total) if total else None for r in rates]
    result=dict(risk_unflagged=mean[0],risk_flagged=mean[1],difference=mean[1]-mean[0] if total else None,
                overlapping_strata=int(common.sum()),overlap_weight=total,
                support_unflagged=float(ns[0][common].sum()/ns[0].sum()) if ns[0].sum() else None,
                support_flagged=float(ns[1][common].sum()/ns[1].sum()) if ns[1].sum() else None)
    if return_cells:
        result['cells']=[dict(stratum=int(i),unflagged=int(ns[0][i]),flagged=int(ns[1][i]),
            unflagged_harm=int(ys[0][i]),flagged_harm=int(ys[1][i]),weight=float(overlap[i])) for i in np.flatnonzero(ns[0]+ns[1])]
    return result


def counts(mask,x):
    n=int(mask.sum());h=int((mask&x['harm']).sum())
    return dict(n=n,new_harms=h,harm_rate=h/n if n else None,
        new_accepts=int((mask&x['new_accept']).sum()),old_pattern=int((mask&x['pattern']).sum()),
        both_components_improve=int((mask&x['both_improve']).sum()),
        both_improve_total_worsens=int((mask&x['both_improve']&x['worse']).sum()))


def analyze(d,p):
    x=derive(d);shape=d['q0'].shape
    code,margin,correction=strata(d,p['standardization']['bin_inner_edges'])
    cohort={}
    for name,base in [('all',np.ones(shape,bool)),('original_rejected',x['at_risk']),('newly_accepted',x['new_accept'])]:
        cohort[name]={}
        for group,flag in [('flagged',x['flag']),('positive_m_negative_l',x['flag']&(x['m0']>0)),
                           ('negative_m_positive_l',x['flag']&(x['m0']<0)),('unflagged',~x['flag'])]:
            cohort[name][group]=counts(base&flag,x)
    st=standardized(x['flag'],x['harm'],x['at_risk'],code,return_cells=True)
    # Covariate means use the identical overlap weights, exposing residual
    # imbalance inside these deliberately coarsened bins.
    for name,value in [('predicted_margin',margin),('absolute_correction',correction)]:
        raw_means=[];weighted=[]
        for flag in (False,True):
            mask=x['at_risk']&(x['flag']==flag)
            raw_means.append(float(value[mask].mean()) if mask.any() else None)
            total=0.;weight=0.
            for cell in st['cells']:
                if not cell['weight']:continue
                select=mask&(code==cell['stratum'])
                total+=cell['weight']*value[select].mean();weight+=cell['weight']
            weighted.append(float(total/weight) if weight else None)
        st[name]=dict(raw_unflagged_flagged=raw_means,overlap_unflagged_flagged=weighted)
    rng=np.random.default_rng(2612091601);differences=[];raw=[];n=shape[1]
    for _ in range(1000):
        sw=rng.multinomial(n,np.ones(n)/n);w=np.broadcast_to(sw,shape)
        rr=standardized(x['flag'],x['harm'],x['at_risk'],code,w)
        if rr['difference'] is not None:differences.append(rr['difference'])
        rates=[]
        for flag in (False,True):
            mask=x['at_risk']&(x['flag']==flag);den=(w*mask).sum()
            rates.append((w*mask*x['harm']).sum()/den if den else np.nan)
        if np.isfinite(rates).all():raw.append(rates[1]-rates[0])
    st['pointwise_scene_ci95']=np.quantile(differences,[.025,.975]).tolist() if differences else None
    st['undefined_draws']=1000-len(differences)
    hybrids={};binary={}
    for arm in ('q00','q10','q01','q11'):
        score=x[arm];accept=score<=.02;unsafe=accept&(d['r']>.02);safe=accept&(d['r']<=.02)
        binary[arm]=unsafe.astype(int)
        hybrids[arm]=dict(accepted=int(accept.sum()),safe=int(safe.sum()),unsafe=int(unsafe.sum()),
            new_harms=int((unsafe&x['at_risk']).sum()),risk=float(unsafe.sum()/accept.sum()),
            eta=float(safe.sum()/d['oracle'].sum()),mae=float(np.abs(score-d['r']).mean()),
            outside_unit_interval=int(((score<0)|(score>1)).sum()))
    interaction=binary['q11']-binary['q10']-binary['q01']+binary['q00']
    hybrid_event_account=dict(
        full_new_harms=int(x['harm'].sum()),
        also_harm_under_measurement_only=int((x['harm']&(x['q10']<=.02)).sum()),
        also_harm_under_model_residual_only=int((x['harm']&(x['q01']<=.02)).sum()),
        requires_both_scalar_changes=int((x['harm']&(x['q10']>.02)&(x['q01']>.02)).sum()),
        measurement_only_new_harms_removed_by_full=int((x['at_risk']&(d['r']>.02)&(x['q10']<=.02)&(x['q11']>.02)).sum()),
        signed_threshold_interaction_sum=int(interaction.sum()),
        positive_interaction_entries=int((interaction>0).sum()),negative_interaction_entries=int((interaction<0).sum()))
    for arm in ('q00','q10','q01','q11'):
        assert np.isfinite(x[arm]).all()
    pattern=dict(all_entries=int(np.prod(shape)),flagged=int(x['flag'].sum()),
        old_pattern_all=int(x['pattern'].sum()),old_pattern_new_harm=int((x['pattern']&x['harm']).sum()),
        both_improve_all=int(x['both_improve'].sum()),both_improve_worse_all=int((x['both_improve']&x['worse']).sum()),
        both_improve_new_harm=int((x['both_improve']&x['harm']).sum()),identity_max=x['identity_max'])
    assert x['identity_max']<1e-12
    return dict(cohorts=cohort,overlap_standardization=st,
        unadjusted_rejected_risk_difference_ci95=np.quantile(raw,[.025,.975]).tolist(),
        hybrids=hybrids,hybrid_event_account=hybrid_event_account,full_pattern_denominators=pattern),x,code


def main(root):
    p=json.loads((HERE/'protocol.json').read_text());root.mkdir(parents=True,exist_ok=True)
    out=root/'audit';out.mkdir(exist_ok=False)
    write(root/'AUDIT_FREEZE.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
        sources={f.name:sha(f) for f in HERE.iterdir() if f.is_file()},
        scope='Newly requested full-cohort audit on already consumed data; prior stop remains unchanged'))
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),input_hashes={},phases={})
    arrays={}
    with gzip.open(out/'ALL_EVENTS.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['phase','model','scene','card','action','reference','q0','q1','target0','target1',
            'm0','l0','m1','l1','flag','original_rejected','new_harm','old_pattern','both_improve','total_worse',
            'measurement_only','model_residual_only','stratum'])
        for phase in p['banks']:
            path=Path(p['input_root'])/phase/'data.npz';h=sha(path)
            assert h==json.loads((path.parent/'MANIFEST.json').read_text())['data.npz']
            report['input_hashes'][str(path)]=h
            with np.load(path) as z:a={k:z[k] for k in ('truth','deploy_variant','deploy_card','tie_uniform','uniform_cost','plic_cost')}
            d=extract(a);summary,x,code=analyze(d,p);report['phases'][phase]=summary
            arrays.update({phase+'__'+k:v for k,v in d.items()})
            for m in range(5):
                for i in range(2000):
                    at=lambda key:d[key][m,i];xx=lambda key:x[key][m,i]
                    writer.writerow([phase,m,i,int(at('card')),int(at('action')),at('r'),at('q0'),at('q1'),at('t0'),at('t1'),
                        xx('m0'),xx('l0'),xx('m1'),xx('l1'),int(xx('flag')),int(xx('at_risk')),int(xx('harm')),
                        int(xx('pattern')),int(xx('both_improve')),int(xx('worse')),xx('q10'),xx('q01'),int(code[m,i])])
            print(phase,json.dumps({k:summary[k] for k in ('full_pattern_denominators','hybrid_event_account')}),flush=True)
    report['decision']='ORACLE_DIAGNOSTIC_ONLY_NO_AUTOMATIC_NEW_SCENES'
    report['identification']=dict(flag_requires_reference=True,correction_is_operator_dependent=True,
        hybrids_are_physical_module_interventions=False,predictor_validated=False,causal_fraction_identified=False,
        note='Full denominators and standardized associations do not by themselves solve observability or causal identification')
    write(out/'SUMMARY.json',report);np.savez_compressed(out/'DECISIONS.npz',**arrays)
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
