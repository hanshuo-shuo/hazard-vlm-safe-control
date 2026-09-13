"""Complete the pre-specified independent secondary factorial reporting."""
from pathlib import Path
import json
import numpy as np
from audit import sha,write

ROOT=Path('/projects/p33100/siosio/hazard_cancellation_audit_20260912')


def main():
    source=ROOT/'independent/DECISIONS.npz';out=ROOT/'independent_factorial';out.mkdir(exist_ok=False)
    expected=json.loads((ROOT/'independent/MANIFEST.json').read_text())['DECISIONS.npz'];assert sha(source)==expected
    with np.load(source) as z:d={k:z[k] for k in z.files}
    scores=dict(target_uniform=d['t0'],target_plic=d['t1'],learned_uniform=d['q0'],learned_plic=d['q1'])
    masks={'all':np.ones_like(d['r'],bool)}
    masks.update({f'model_{i}':np.broadcast_to(np.arange(5)[:,None]==i,d['r'].shape) for i in range(5)})
    masks.update({f'card_{name}':d['card']==k for k,name in [(0,'A'),(2,'C'),(3,'D')]})
    metrics={name:dict(cost=x,absolute_error=abs(x-d['r']),safe=((x<=.02)&(d['r']<=.02)).astype(float),
                      unsafe=((x<=.02)&(d['r']>.02)).astype(float)) for name,x in scores.items()}
    result=dict(input_sha256=expected,source_sha256=sha(Path(__file__)),
        scope='Pre-specified secondary factorial results; descriptive, not additional confirmed hypotheses. Aggregate intervals reuse the primary paired-scene draws, no model-population claim.',groups={})
    weights=np.random.default_rng(2612091702).multinomial(2000,np.ones(2000)/2000,size=5000)
    for group,mask in masks.items():
        cells={};interactions={};count=int(mask.sum())
        for name,x in scores.items():
            accepted=int(((x<=.02)&mask).sum());bad=int((metrics[name]['unsafe']*mask).sum());good=int((metrics[name]['safe']*mask).sum())
            cells[name]=dict(n=count,accepted=accepted,safe=good,unsafe=bad,conditional_unsafe=bad/accepted if accepted else None,
                eta=good/int((d['oracle']&mask).sum()),mean_cost=float(x[mask].mean()),mae=float(metrics[name]['absolute_error'][mask].mean()))
        for metric in ('cost','absolute_error','safe','unsafe'):
            v={name:m[metric] for name,m in metrics.items()}
            interaction=v['learned_plic']-v['learned_uniform']-v['target_plic']+v['target_uniform']
            row=dict(mean=float(interaction[mask].mean()))
            if group=='all':row['pointwise_scene_ci95']=np.quantile((weights@interaction.mean(0))/2000,[.025,.975]).tolist()
            interactions[metric]=row
        result['groups'][group]=dict(cells=cells,interactions=interactions)
    for name,cell in json.loads((ROOT/'independent_verification/CELLS.json').read_text()).items():
        for key,value in cell.items():assert result['groups']['all']['cells'][name][key]==value
    primary=json.loads((ROOT/'independent/PRIMARY.json').read_text())
    assert abs(result['groups']['all']['interactions']['safe']['mean']-primary['primary']['mean'])<1e-14
    for metric in ('cost','absolute_error','safe','unsafe'):
        for groups in [[f'model_{i}' for i in range(5)],['card_A','card_C','card_D']]:
            weighted=sum(result['groups'][g]['interactions'][metric]['mean']*int(masks[g].sum()) for g in groups)/10000
            assert abs(weighted-result['groups']['all']['interactions'][metric]['mean'])<1e-12
    result['partition_counts_and_primary_verified']=True
    write(out/'SECONDARY_FACTORIAL.json',result)
    lines=['# Independent secondary input-by-operator factorial','',result['scope'],'',
        '| Group | Safe-accept interaction (pp) | Unsafe-accept interaction (pp) | Signed-cost interaction | Absolute-error interaction |',
        '|---|---:|---:|---:|---:|']
    for group,r in result['groups'].items():
        v=r['interactions'];lines.append(f"| {group} | {v['safe']['mean']*100:+.3f} | {v['unsafe']['mean']*100:+.3f} | {v['cost']['mean']:+.6f} | {v['absolute_error']['mean']:+.6f} |")
    lines += ['','All four cell counts, accepted risks, eta and errors for every model and card are in the JSON. Per-model/card values are descriptive; only the frozen aggregate safe-accept primary has confirmatory status.','']
    (out/'TABLES.md').write_text('\n'.join(lines))
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print((out/'TABLES.md').read_text())


if __name__=='__main__':main()
