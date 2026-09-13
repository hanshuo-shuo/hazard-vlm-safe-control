"""Complete empirical threshold families, with identical selection on each arm."""
import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def choose(values, uniform):
    tied = np.isclose(values, values.min(-1, keepdims=True), atol=1e-9, rtol=0)
    rank = np.floor(uniform * tied.sum(-1)).astype(int)
    return np.argmax(tied & (tied.cumsum(-1) == rank[..., None] + 1), axis=-1)


def extract(a, mode):
    n = len(a['truth']); ix = np.arange(n)
    v, k = a['deploy_variant'], a['deploy_card']
    truth = a['truth'][ix, v, :, k]
    u = a['tie_uniform'][ix, v, k]
    costs = {name: np.stack([field[ix, v, :, k] for field in a[name + '_cost']])
             for name in ('uniform', 'plic')}
    assert truth.shape == (n,4) and all(c.shape == (6,n,4) for c in costs.values())
    base = choose(costs['uniform'], u)
    out = {}
    for name, c in costs.items():
        action = base if mode == 'fixed' else choose(c, u)
        scores = np.take_along_axis(c, action[..., None], -1)[..., 0]
        actual = truth[ix[None, :], action]
        out[name] = dict(score=scores, bad=actual > .02, actual=actual, action=action)
    return out, truth.min(-1) <= .02


class Curve:
    def __init__(self, score, bad):
        self.score = np.asarray(score).ravel()
        self.bad = np.asarray(bad).ravel()
        assert np.isfinite(self.score).all()
        self.order = np.argsort(self.score, kind='stable')
        sorted_score = self.score[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(sorted_score) - 1]
        self.threshold = sorted_score[self.ends]

    def counts(self, weights=None):
        w = np.ones(len(self.score), dtype=int) if weights is None else np.asarray(weights).ravel()
        accepted = np.cumsum(w[self.order])[self.ends]
        unsafe = np.cumsum((w * self.bad)[self.order])[self.ends]
        return accepted, accepted - unsafe, unsafe

    def envelope(self, budgets, oracle_count, weights=None):
        accepted, safe, bad = self.counts(weights)
        risk = np.divide(bad, accepted, out=np.full(len(bad), np.inf), where=accepted > 0)
        result = []
        for beta in budgets:
            eligible = np.flatnonzero(risk <= beta)
            if not len(eligible):
                result.append(dict(threshold=None, accepted=0, safe=0, unsafe=0, risk=None, eta=0.))
                continue
            eligible = eligible[safe[eligible] == safe[eligible].max()]
            eligible = eligible[bad[eligible] == bad[eligible].min()]
            q = int(eligible[0])
            result.append(dict(threshold=float(self.threshold[q]), accepted=int(accepted[q]),
                               safe=int(safe[q]), unsafe=int(bad[q]), risk=float(risk[q]),
                               eta=float(safe[q] / oracle_count)))
        return result


def apply(score, bad, threshold, oracle_count):
    accepted = np.zeros_like(bad) if threshold is None else score <= threshold
    total = int(accepted.sum()); unsafe = int((accepted & bad).sum())
    return dict(threshold=threshold, accepted=total, safe=total-unsafe, unsafe=unsafe,
                risk=unsafe/total if total else None, eta=(total-unsafe)/oracle_count)


def boot_difference(curves, oracle, budgets, m, seed, draws):
    rng = np.random.default_rng(seed); n = len(oracle); values = []
    for _ in range(draws):
        scene = rng.multinomial(n, np.ones(n)/n)
        models = rng.multinomial(m, np.ones(m)/m)
        weights = models[:, None] * scene[None, :]
        denominator = int(models.sum() * (scene * oracle).sum())
        rows = [curves[name].envelope(budgets, denominator, weights) for name in ('uniform', 'plic')]
        values.append([b['eta']-a['eta'] for a, b in zip(*rows)])
    return np.quantile(values, [.025, .975], axis=0).T.tolist()


def main(root):
    p = json.loads((HERE/'protocol.json').read_text())
    root.mkdir(parents=True, exist_ok=True)
    out = root/'analysis'; out.mkdir(exist_ok=False)
    write(root/'ANALYSIS_FREEZE.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
          scope='Before reading full threshold curves; consumed data remain retrospective',
          files={f.name:sha(f) for f in HERE.iterdir() if f.is_file()}))
    src = Path(p['input_root']); budgets = p['risk_budgets']
    summary = dict(at_utc=datetime.now(timezone.utc).isoformat(), scope=p['scope'],
                   risk_budgets=budgets, inputs={}, phases={})
    decisions = {}; phase_data = {}
    with gzip.open(out/'ALL_CURVES.csv.gz', 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['phase','selector','track','arm','threshold','accepted','safe','unsafe','conditional_risk','eta'])
        for phase in p['banks']:
            path = src/phase/'data.npz'; expected = json.loads((src/phase/'MANIFEST.json').read_text())['data.npz']
            assert sha(path) == expected
            summary['inputs'][str(path)] = expected
            with np.load(path, allow_pickle=False) as data:
                a = {name:data[name] for name in ('truth','deploy_variant','deploy_card','tie_uniform','uniform_cost','plic_cost')}
            summary['phases'][phase] = {}
            for mode in ('fixed','reselected'):
                rows, oracle = extract(a, mode)
                decisions[f'{phase}__{mode}__oracle'] = oracle
                for arm, values in rows.items():
                    for key, val in values.items():
                        decisions[f'{phase}__{mode}__{arm}__{key}'] = val
                phase_data[phase, mode] = (rows, oracle)
                summary['phases'][phase][mode] = {}
                groups = [('target',[0]),('learned',list(range(1,6)))] + [(f'model_{i}',[i+1]) for i in range(5)]
                for group, ids in groups:
                    m = len(ids); denominator = int(m*oracle.sum()); record = dict(oracle=denominator, arms={})
                    curves = {}
                    for arm in ('uniform','plic'):
                        curves[arm] = c = Curve(rows[arm]['score'][ids], rows[arm]['bad'][ids])
                        accepted,safe,bad = c.counts()
                        writer.writerow([phase,mode,group,arm,'reject-all',0,0,0,'undefined',0])
                        for t,aa,ss,bb in zip(c.threshold,accepted,safe,bad):
                            writer.writerow([phase,mode,group,arm,t,aa,ss,bb,bb/aa,ss/denominator])
                        record['arms'][arm] = c.envelope(budgets, denominator)
                        record['arms'][arm+'_at_02'] = apply(rows[arm]['score'][ids],rows[arm]['bad'][ids],.02,denominator)
                    record['delta_eta'] = [b['eta']-a['eta'] for a,b in zip(record['arms']['uniform'],record['arms']['plic'])]
                    if group in ('target','learned'):
                        record['pointwise_descriptive_ci95'] = boot_difference(curves,oracle,budgets,m,2612091501,1000)
                    summary['phases'][phase][mode][group] = record
            print('completed curves',phase,datetime.now(timezone.utc).isoformat(),flush=True)
    transfer = {}
    for mode in ('fixed','reselected'):
        rows,oracle = phase_data['appended',mode]; transfer[mode] = {}
        for group,ids in [('target',[0]),('learned',list(range(1,6)))]:
            transfer[mode][group] = {}
            for arm in ('uniform','plic'):
                thresholds = summary['phases']['development'][mode][group]['arms'][arm]
                transfer[mode][group][arm] = [apply(rows[arm]['score'][ids],rows[arm]['bad'][ids],r['threshold'],int(len(ids)*oracle.sum())) for r in thresholds]
    summary['development_selected_on_consumed_appended'] = transfer
    gate_points = []
    for q,beta in enumerate(budgets):
        details = {}
        for phase in p['banks']:
            data = summary['phases'][phase]['fixed']; pool = data['learned']
            positive = sum(data[f'model_{i}']['delta_eta'][q]>0 for i in range(5))
            details[phase] = dict(delta_eta=pool['delta_eta'][q],ci95=pool['pointwise_descriptive_ci95'][q],positive_models=positive)
        passed = beta<=.03 and all(r['delta_eta']>=.005 and r['ci95'][0]>0 and r['positive_models']>=4 for r in details.values())
        gate_points.append(dict(beta=beta,passed=passed,banks=details))
    runs=[]; current=[]
    for q,point in enumerate(gate_points):
        if point['passed']: current.append(q)
        else:
            if current:runs.append(current)
            current=[]
    if current:runs.append(current)
    qualified=[r for r in runs if len(r)>=3 and budgets[r[-1]]-budgets[r[0]]>=.01-1e-12]
    summary['continuation_gate']=dict(points=gate_points,qualified_runs=qualified,
        decision='ELIGIBLE_FOR_ONE_FROZEN_INDEPENDENT_POINT' if qualified else 'STOP_NO_NEW_SCENES',
        note='Operational stopping screen on consumed data, not a confirmatory test or a venue recommendation')
    if qualified:
        winner=sorted(qualified,key=lambda r:(-len(r),r[0]))[0]
        summary['continuation_gate']['eligible_beta']=budgets[winner[(len(winner)-1)//2]]
    old=json.loads((src/'report/SUMMARY.json').read_text())
    summary['marginal_new_accepts']={}
    for track in ('target','learned'):
        arm=old['phases']['appended'][track]['arms']['plic']
        safe=arm['recovered_safe']['numerator'];bad=arm['new_unsafe_accepts']['numerator']
        assert arm['newly_lost_safe']['numerator']==arm['resolved_unsafe_accepts']['numerator']==0
        summary['marginal_new_accepts'][track]=dict(new_safe=safe,new_unsafe=bad,danger_fraction=bad/(safe+bad),
            scope='Outcome-conditioned retrospective subset; not the all-accepted risk or a new risk constraint')
    np.savez_compressed(out/'DECISIONS.npz',**decisions)
    write(out/'SUMMARY.json',summary)
    write(out/'MANIFEST.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps(summary['continuation_gate'],indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
