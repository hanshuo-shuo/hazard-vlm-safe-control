"""Shared immutable numerical definitions for the Pro-directed round."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "composition_audit_20260912"))
import run_audit as reference
from autoresearch.exp01b_r2.train import build_candidate, field_loss, _augment_palette
CARDS = np.array([0, 2, 3])

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def protocol():
    return json.loads((HERE / "protocol.json").read_text())

def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def manifest(path):
    return {str(f.relative_to(path)): sha(f) for f in sorted(Path(path).rglob("*"))
            if f.is_file() and f.name != "MANIFEST.json"}

def verify_manifest(path):
    entries = json.loads((Path(path) / "MANIFEST.json").read_text())
    for name, expected in entries.items():
        if sha(Path(path) / name) != expected:
            raise ValueError(f"Artifact identity mismatch: {path}/{name}")
    return len(entries)

def freeze(root, stage, filenames):
    out = Path(root) / f"FREEZE_{stage}.json"
    if out.exists():
        raise FileExistsError(out)
    sources = [HERE / f for f in filenames] + [HERE / "protocol.json", HERE / "PRO_REVIEW.md"]
    sources += [HERE.parent / "composition_audit_20260912/run_audit.py"]
    sources += [f for f in (HERE.parent / "quest_reference").rglob("*")
                if f.is_file() and f.suffix in {".py", ".json", ".pt"} and not f.name.startswith("._")]
    write_json(out, {"stage": stage, "at_utc": now(), "source_sha256": {
        str(f.relative_to(ROOT)): sha(f) for f in sorted(set(sources))}})

def verify_freeze(root, stage):
    entries = json.loads((Path(root) / f"FREEZE_{stage}.json").read_text())
    for name, expected in entries["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return entries

def exposure(fields, footprints):
    # N,V,C,H,W by N,K,H,W, using a spatial-area mean.
    f, w = np.asarray(fields, float), np.asarray(footprints, float)
    den = w.sum((-2, -1))
    if np.any(den <= 0):
        raise ValueError("Empty footprint")
    return np.einsum("nvchw,nkhw->nvkc", f, w) / den[:, None, :, None]

def formula(e):
    ew, ef = np.moveaxis(np.asarray(e, float), -1, 0)
    return np.stack([ew, np.zeros_like(ew), ew + ef - ew * ef, ef], axis=-1)

def choose(cost, uniforms):
    tied = np.isclose(cost, cost.min(2, keepdims=True), atol=1e-9, rtol=0)
    rank = np.floor(uniforms * tied.sum(2)).astype(int)
    return ((np.cumsum(tied, axis=2) > rank[:, :, None]) & tied).argmax(2)

def selected(cost, ids):
    return np.take_along_axis(cost, ids[:, :, None, :], axis=2).squeeze(2)

def deployed(value, d):
    return value[np.arange(len(value)), d["deploy_variant"], d["deploy_card"]]

def deployment(seed, n):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    return {"deploy_variant": rng.integers(2, size=n), "deploy_card": rng.choice(CARDS, size=n),
            "tie_uniform": rng.random((n, 2, 4))}

def rows(truth, pred, d, threshold=.02, offset=0.):
    ids = choose(pred, d["tie_uniform"])
    actual, estimate = selected(truth, ids), selected(pred, ids)
    actual = deployed(actual, d)
    estimate = deployed(estimate, d)
    oracle_cost = deployed(truth.min(2), d)
    accepted = np.minimum(estimate + offset, 1) <= threshold
    unsafe = actual > .02
    return {"accepted": accepted, "unsafe_accepted": accepted & unsafe,
            "safe_accepted": accepted & ~unsafe, "oracle": oracle_cost <= .02,
            "actual_cost": actual, "estimated_cost": estimate, "oracle_cost": oracle_cost,
            "regret": actual - oracle_cost, "chosen": deployed(ids, d)}

def summarize(r):
    accepted, oracle = int(r["accepted"].sum()), int(r["oracle"].sum())
    unsafe, safe = int(r["unsafe_accepted"].sum()), int(r["safe_accepted"].sum())
    return {"n": len(r["accepted"]), "accepted": accepted, "unsafe_accepted": unsafe,
            "safe_accepted": safe, "oracle_safe": oracle, "coverage": accepted / len(r["accepted"]),
            "joint_unsafe": unsafe / len(r["accepted"]), "conditional_unsafe": unsafe / accepted if accepted else None,
            "eta": safe / oracle if oracle else None, "oracle_coverage": oracle / len(r["accepted"]),
            "regret": float(r["regret"].mean())}

def quantile(scores, alpha=.05):
    scores = np.asarray(scores)
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= len(scores) else 1.

def nested_scores(truth, pred, d):
    deficit = np.maximum(truth - pred, 0)
    ids = choose(pred, d["tie_uniform"])
    return {"pair_24_entries": deficit[..., CARDS].max((1, 2, 3)),
            "current_four_candidates": deployed(deficit.max(2), d),
            "fixed_selected_action": deployed(selected(deficit, ids), d)}

def cp_upper(k, n, delta=.05):
    """Exact one-sided binomial upper bound via CDF inversion, no normal approximation."""
    if n == 0 or k == n:
        return 1.
    if not 0 <= k < n or not 0 < delta < 1:
        raise ValueError("Invalid binomial counts/confidence")
    if k == 0:
        return float(-math.expm1(math.log(delta) / n))
    j = np.arange(k + 1, dtype=float)
    coeff = np.asarray([math.lgamma(n + 1) - math.lgamma(int(i) + 1) - math.lgamma(n - int(i) + 1) for i in j])
    lo, hi = k / n, 1.
    for _ in range(70):
        p = (lo + hi) / 2
        terms = coeff + j * math.log(p) + (n - j) * math.log1p(-p)
        m = terms.max()
        log_cdf = m + math.log(np.exp(terms - m).sum())
        if log_cdf > math.log(delta):
            lo = p
        else:
            hi = p
    return float(hi)

def crossed_ratio(num, den, *, seed, resamples=5000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    s, n = num.shape
    rng = np.random.default_rng(seed)
    sw = rng.multinomial(s, np.ones(s) / s, size=resamples) / s
    nw = rng.multinomial(n, np.ones(n) / n, size=resamples)
    bottom = ((sw @ den) * nw).sum(1)
    if (bottom <= 0).any() or den.sum() == 0:
        return {"mean": None, "ci95": None, "undefined_resamples": int((bottom <= 0).sum())}
    estimates = ((sw @ num) * nw).sum(1) / bottom
    return {"mean": float(num.sum() / den.sum()), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "per_seed": (num.sum(1) / den.sum(1)).tolist()}

def infer_logits(model, rgb, device):
    shape = rgb.shape[:2]
    rgb = rgb.reshape(-1, 64, 64, 3)
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rgb), 64):
            x = torch.tensor(rgb[start:start + 64], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
            y = model(x)
            if not torch.isfinite(y).all():
                raise ValueError("Nonfinite logits")
            result.append(y.cpu().numpy())
    return np.concatenate(result).reshape(*shape, 2, 16, 16)

def probabilities(logits, shift=0):
    return torch.sigmoid(torch.from_numpy(logits + shift)).numpy()
