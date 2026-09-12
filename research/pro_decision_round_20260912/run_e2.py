"""The matched 2x2x5 experiment; final checkpoint after exactly 960 updates."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import *
from common import _augment_palette

def train_fixed(train, val, source, seed, p):
    cfg = p["e2"]["training"]
    device = torch.device("cuda")
    model = build_candidate(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rgb = torch.tensor(train["rgb"][:, 0].copy(), dtype=torch.float32).permute(0,3,1,2) / 255
    target = torch.tensor(train[f"target{source}"][:, 0].copy(), dtype=torch.float32)
    vrgb = torch.tensor(val["rgb"][:, 0].copy(), dtype=torch.float32).permute(0,3,1,2) / 255
    vtarget = torch.tensor(val[f"target{source}"][:, 0].copy(), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    history, steps = [], 0
    started = time.monotonic()
    for epoch in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(rgb))
        losses = []
        for start in range(0, len(order), cfg["batch_size"]):
            if time.monotonic() - started > cfg["maximum_seconds"]:
                raise TimeoutError("Fixed training time ceiling exceeded")
            idx = torch.tensor(order[start:start + cfg["batch_size"]])
            x, y = rgb[idx].to(device), target[idx].to(device)
            x = _augment_palette(x, rng=rng)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = field_loss(model(x), y)
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            vloss = float(field_loss(model(vrgb.to(device)), vtarget.to(device))[0])
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation_loss_diagnostic_only": vloss})
    if steps != cfg["optimizer_steps"]:
        raise RuntimeError(f"Expected exactly 960 optimizer steps, got {steps}")
    return model.eval(), {"epochs": cfg["epochs"], "steps": steps, "checkpoint": "final", "post_training_shift": 0,
                         "elapsed_seconds": time.monotonic()-started, "history": history}

def all_metrics(truth, pred, d):
    ids = choose(pred, d["tie_uniform"])
    cost = selected(truth, ids)
    regret = (cost - truth.min(2))[..., CARDS].mean((1,2))
    false = (truth[..., CARDS] > .02) & (pred[..., CARDS] <= .02)
    return {"regret": regret, "false_count": false.sum((1,2,3)), "danger_count": (truth[..., CARDS] > .02).sum((1,2,3)),
            **{f"decision_{k}":v for k,v in rows(truth, pred, d).items()}}

def main(root, task):
    p = protocol()
    verify_freeze(root, "E2")
    verify_manifest(root / "e2_data")
    geometry = p["e2"]["geometry"][task // 10]
    source = p["e2"]["source_rasters"][(task // 5) % 2]
    index = task % 5
    seed = p["model_seeds"][index]
    out = root / "e2_runs" / f"{geometry}_{source}" / f"seed_{index}"
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Quest GPU allocation required")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    write_json(out / "START.json", {"at_utc":now(), "task":task, "seed":seed, "geometry":geometry, "source":source,
                                    "slurm_job_id":os.environ.get("SLURM_JOB_ID"), "gpu":torch.cuda.get_device_name(),
                                    "torch":torch.__version__, "numpy":np.__version__, "freeze_sha256":sha(root / "FREEZE_E2.json")})
    train = load_npz(root / "e2_data" / f"{geometry}_train.npz")
    val = load_npz(root / "e2_data" / f"{geometry}_validation.npz")
    model, history = train_fixed(train, val, source, seed, p)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, out / "model.pt")
    write_json(out / "TRAINING.json", history)
    results, metrics = {}, {}
    for split in p["e2"]["tests"]:
        d = load_npz(root / "e2_data" / f"{split}.npz")
        logits = infer_logits(model, d["rgb"], torch.device("cuda"))
        saved, results[split] = {}, {}
        for post, shift in [("restored_zero",0), ("inherited_minus2",-2)]:
            fields = probabilities(logits, shift)
            saved[f"{post}__fields"] = fields
            results[split][post] = {}
            for mode, fp, truth in [("common", d["footprints_common"], d["truth_ref512"]),
                                    ("native", d[f"footprints{source}"], d[f"truth_native{source}"])]:
                pred = formula(exposure(fields, fp))
                saved[f"{post}__{mode}__prediction"] = pred
                m = all_metrics(truth, pred, d)
                for k,v in m.items():
                    metrics[f"{split}__{post}__{mode}__{k}"] = v
                results[split][post][mode] = {"mean_acd_regret":float(m["regret"].mean()),
                    "false_safe":float(m["false_count"].sum()/m["danger_count"].sum()),
                    "deployment":summarize(rows(truth, pred, d))}
        # Fixed first pair used for compatibility replay; never select examples by success.
        np.savez_compressed(out / f"{split}_PREDICTIONS.npz", **saved)
        print(geometry, source, index, split, results[split]["restored_zero"]["common"], flush=True)
    np.savez_compressed(out / "METRICS.npz", **metrics)
    write_json(out / "RESULTS.json", {"status":"COMPLETE", "at_utc":now(), "geometry":geometry, "source":source,
               "seed":seed, "checkpoint_sha256":sha(out / "model.pt"), "rgb_training_sha256":array_sha(train["rgb"][:,0]),
               "target_training_sha256":array_sha(train[f"target{source}"][:,0]), "splits":results})
    write_json(out / "MANIFEST.json", manifest(out))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=int, choices=range(20), required=True)
    args = parser.parse_args()
    main(args.root, args.task)
