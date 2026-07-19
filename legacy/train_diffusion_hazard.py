"""
Train Conditional DDPM on PointHazardEnv safe demonstrations.

Core design: diffusion knows WHERE IT IS and WHERE HAZARDS ARE,
but NEVER sees the goal.  This turns diffusion into a *safety prior*
rather than a goal-reaching prior.

  cond = [x, y, vx, vy, h1x, h1y, h1r, ..., hKx, hKy, hKr]   (4 + 3K dims)
  action = [fx, fy]                                             (2 dims)
  x0 = [cond | action]                                          (4+3K+2 dims)

The obs vector from the env has layout:
  obs[0:2]          → (x, y)           agent pos
  obs[2:4]          → (vx, vy)         agent vel
  obs[4:6]          → (gx, gy)         GOAL  ← EXCLUDED from cond
  obs[6:6+3K]       → hazard params    ← included in cond

Usage:
    # Train on expert demos (recommended)
    python pointmaze_copilot/train_diffusion_hazard.py \\
        --dataset pointmaze_copilot/expert_hazard_demos.npz \\
        --out_dir pointmaze_copilot/diffusion_hazard \\
        --steps 100000

    # Quick smoke-test (1k steps, cpu)
    python pointmaze_copilot/train_diffusion_hazard.py \\
        --dataset /tmp/expert_hazard_test_demos.npz \\
        --out_dir /tmp/diffusion_hazard_test \\
        --steps 1000 --batch_size 64 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ddpm import ConditionalDDPM, DiffusionConfig, RunningNorm
from hazard.env_pointhazard import PointHazardConfig


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_cond_from_obs(obs: np.ndarray, n_hazards: int) -> np.ndarray:
    """
    Drop goal (obs[:, 4:6]) and return (N, 4 + 3*n_hazards) cond array.

    This assertion documents the architectural separation: diffusion never sees the goal.
    Any future code that accidentally passes goal info here must break this
    assertion:
        assert cond.shape[1] == 4 + 3 * n_hazards  # no goal allowed
    """
    agent_state = obs[:, 0:4]                        # (N, 4)  x, y, vx, vy
    # obs[:, 4:6] is goal — intentionally skipped
    hazard_params = obs[:, 6 : 6 + 3 * n_hazards]   # (N, 3K)
    cond = np.concatenate([agent_state, hazard_params], axis=1)
    assert cond.shape[1] == 4 + 3 * n_hazards, \
        f"cond_dim mismatch: {cond.shape[1]} != {4 + 3 * n_hazards}"
    return cond.astype(np.float32)


def load_hazard_dataset(
    npz_path: str,
    n_hazards: int,
    *,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = np.load(npz_path)
    obs = data["obs"].astype(np.float32)       # (N, obs_dim)
    actions = data["actions"].astype(np.float32)  # (N, 2)

    assert obs.shape[1] == 4 + 2 + 3 * n_hazards, \
        f"Expected obs_dim={4+2+3*n_hazards}, got {obs.shape[1]}"
    assert actions.shape[1] == 2, f"Expected act_dim=2, got {actions.shape[1]}"

    cond = build_cond_from_obs(obs, n_hazards)  # (N, 4+3K)  — no goal

    meta = {}
    if "metadata" in data:
        m = data["metadata"]
        meta = {
            "n_episodes": int(m[0]),
            "n_success": int(m[1]),
            "n_hazard_hit": int(m[2]),
            "n_timeout": int(m[3]),
            "n_transitions": int(m[4]),
        }

    if verbose:
        print(f"Dataset: {npz_path}")
        print(f"  N={obs.shape[0]}  obs_dim={obs.shape[1]}  act_dim={actions.shape[1]}")
        print(f"  cond_dim={cond.shape[1]}  (goal EXCLUDED — by design)")
        if meta:
            print(f"  {meta['n_episodes']} episodes, {meta['n_hazard_hit']} hazard-hit filtered")

    cond_t = torch.from_numpy(cond)
    act_t = torch.from_numpy(actions)
    return cond_t, act_t, meta


# ---------------------------------------------------------------------------
# Fan-out sanity check
# ---------------------------------------------------------------------------

@torch.no_grad()
def fan_out_check(
    ddpm: ConditionalDDPM,
    cond_sample: torch.Tensor,
    n_samples: int = 64,
    *,
    verbose: bool = True,
) -> dict:
    """
    At a random conditioning state, sample n_samples actions unconditionally.
    They should spread across multiple directions (not collapse to one mode).

    Pass criterion: std of sampled actions in both x and y > 0.05.
    """
    cond = cond_sample[:1].expand(n_samples, -1)  # (n, cond_dim)
    samples = ddpm.sample(batch_size=n_samples, cond=cond)
    act_dim = ddpm.cfg.x_dim - ddpm.cfg.cond_dim
    acts = samples[:, ddpm.cfg.cond_dim: ddpm.cfg.cond_dim + act_dim]
    acts_np = acts.cpu().numpy()

    std_x = float(np.std(acts_np[:, 0]))
    std_y = float(np.std(acts_np[:, 1]))
    mean_x = float(np.mean(acts_np[:, 0]))
    mean_y = float(np.mean(acts_np[:, 1]))

    passed = std_x > 0.05 and std_y > 0.05
    result = {"std_x": std_x, "std_y": std_y,
              "mean_x": mean_x, "mean_y": mean_y,
              "passed": passed}
    if verbose:
        status = "PASS" if passed else "WARN"
        print(f"  [fan-out check] std_x={std_x:.3f} std_y={std_y:.3f} "
              f"mean=({mean_x:+.3f},{mean_y:+.3f}) → {status}")
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # --- Load dataset ---
    env_cfg = PointHazardConfig()
    cond_t, act_t, meta = load_hazard_dataset(
        args.dataset, n_hazards=env_cfg.n_hazards)

    cond_dim = cond_t.shape[1]   # 4 + 3*K = 28
    act_dim = act_t.shape[1]     # 2
    x_dim = cond_dim + act_dim   # 30

    # x0: concatenate [cond | action]
    x0_t = torch.cat([cond_t, act_t], dim=1)   # (N, x_dim)

    # --- Running normalisation (whole x0, then condition dims are frozen to
    #     their natural scale; diffusion only noises the action dims anyway) ---
    if args.no_norm:
        norm = None
        print("  Normalisation: DISABLED")
    else:
        norm = RunningNorm.from_data(x0_t)
        print(f"  Normalisation: mean in [{norm.mean.min():.2f}, {norm.mean.max():.2f}]  "
              f"std in [{norm.std.min():.3f}, {norm.std.max():.3f}]")

    # --- Diffusion config ---
    cfg = DiffusionConfig(
        x_dim=x_dim,
        cond_dim=cond_dim,
        num_steps=args.num_diffusion_steps,
        beta_schedule=args.beta_schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        ema_decay=args.ema_decay,
    )
    ddpm = ConditionalDDPM(cfg, device=device, norm=norm)

    # Safety assert: goal is NOT in cond
    assert cfg.cond_dim == 4 + 3 * env_cfg.n_hazards, \
        f"cond_dim={cfg.cond_dim} != 4+3*{env_cfg.n_hazards}. " \
        f"Goal accidentally included?"
    print(f"\nModel: x_dim={x_dim}  cond_dim={cond_dim}  act_dim={act_dim}")
    print(f"       goal EXCLUDED from cond_dim — safety prior design confirmed")
    print(f"       num_steps={cfg.num_steps}  hidden_dim={cfg.hidden_dim}")
    total_params = sum(p.numel() for p in ddpm.model.parameters())
    print(f"       params={total_params:,}")

    # --- DataLoader ---
    dataset = TensorDataset(x0_t)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- Output dir ---
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    best_path = os.path.join(args.out_dir, "riskdiffusion_hazard_best03.pt")
    final_path = os.path.join(args.out_dir, "riskdiffusion_hazard_final03.pt")

    # --- Training loop ---
    loader_iter = iter(loader)
    best_loss = float("inf")
    loss_window: list[float] = []
    t0 = time.time()

    print(f"\nTraining for {args.steps} steps ...")

    for step in range(1, args.steps + 1):
        # Refill iterator
        try:
            (batch,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            (batch,) = next(loader_iter)

        batch = batch.to(device)
        loss = ddpm.train_step(batch)
        loss_window.append(loss)

        # --- Logging ---
        if step % args.log_every == 0:
            avg_loss = float(np.mean(loss_window[-args.log_every:]))
            elapsed = time.time() - t0
            it_s = step / elapsed
            print(f"  step {step:6d}/{args.steps}  loss={avg_loss:.5f}  "
                  f"({it_s:.0f} it/s  {elapsed:.0f}s)")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": avg_loss,
                                    "elapsed": elapsed}) + "\n")

            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
                _save_ckpt(ddpm, cfg, norm, best_path,
                           meta={"step": step, "loss": avg_loss, "best": True})

        # --- Checkpoint every N steps ---
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir,
                                     f"diffusion_hazard_step{step}.pt")
            _save_ckpt(ddpm, cfg, norm, ckpt_path,
                       meta={"step": step,
                             "loss": float(np.mean(loss_window[-args.log_every:]))})

        # --- Fan-out check at end ---
        if step == args.steps:
            print("\nFan-out check (unconditional diversity test):")
            cond_sample = cond_t[:1].to(device)
            fan_out_check(ddpm, cond_sample, n_samples=64)

    # --- Save final ---
    _save_ckpt(ddpm, cfg, norm, final_path,
               meta={"step": args.steps,
                     "loss": float(np.mean(loss_window[-args.log_every:])),
                     "best_loss": best_loss})

    summary = {
        "dataset": args.dataset,
        "n_transitions": int(x0_t.shape[0]),
        "cond_dim": int(cond_dim),
        "act_dim": int(act_dim),
        "x_dim": int(x_dim),
        "goal_in_cond": False,       # by design
        "n_hazards": int(env_cfg.n_hazards),
        "steps": args.steps,
        "best_loss": best_loss,
        "final_loss": float(np.mean(loss_window[-args.log_every:])),
        "device": str(device),
        "diffusion_cfg": {
            "num_steps": cfg.num_steps,
            "beta_schedule": cfg.beta_schedule,
            "beta_min": cfg.beta_min,
            "beta_max": cfg.beta_max,
            "hidden_dim": cfg.hidden_dim,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.  best_loss={best_loss:.5f}")
    print(f"  best ckpt:  {best_path}")
    print(f"  final ckpt: {final_path}")
    print(f"  summary:    {os.path.join(args.out_dir, 'summary.json')}")


# ---------------------------------------------------------------------------
# Checkpoint save / load helpers
# ---------------------------------------------------------------------------

def _save_ckpt(
    ddpm: ConditionalDDPM,
    cfg: DiffusionConfig,
    norm: RunningNorm | None,
    path: str,
    meta: dict | None = None,
):
    payload = {
        "cfg": cfg.__dict__,
        "norm": {"mean": norm.mean.cpu(), "std": norm.std.cpu()} if norm else None,
        "meta": meta or {},
    }
    payload.update(ddpm.state_dict())   # model weights + ema weights
    torch.save({"ddpm": payload}, path)


def load_hazard_ddpm(path: str, *, device: torch.device) -> ConditionalDDPM:
    """Load a hazard diffusion checkpoint into ConditionalDDPM."""
    ckpt = torch.load(path, map_location="cpu")
    dd = ckpt["ddpm"]
    cfg = DiffusionConfig(**dd["cfg"])
    norm = None
    if dd.get("norm") is not None:
        norm = RunningNorm(
            mean=dd["norm"]["mean"].float(),
            std=dd["norm"]["std"].float(),
        )
    ddpm = ConditionalDDPM(cfg, device=device, norm=norm)
    # Load weights (exclude cfg/norm/meta keys)
    state = {k: v for k, v in dd.items()
             if k not in ("cfg", "norm", "meta")}
    ddpm.load_state_dict(state)
    return ddpm


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    print("Smoke test: train 500 steps on /tmp expert demos ...")
    sys.argv = [sys.argv[0],
                "--dataset", "/tmp/expert_hazard_test_demos.npz",
                "--out_dir", "/tmp/diffusion_hazard_smoke",
                "--steps", "500",
                "--batch_size", "32",
                "--log_every", "100",
                "--save_every", "500",
                "--device", "cpu"]
    args = _parse_args()
    train(args)

    # Reload and verify
    final = os.path.join("/tmp/diffusion_hazard_smoke", "diffusion_hazard_final.pt")
    ddpm = load_hazard_ddpm(final, device=torch.device("cpu"))
    print(f"\nReloaded: cond_dim={ddpm.cfg.cond_dim}  x_dim={ddpm.cfg.x_dim}")
    assert ddpm.cfg.cond_dim == 28, f"Expected 28, got {ddpm.cfg.cond_dim}"
    assert ddpm.cfg.x_dim == 30, f"Expected 30, got {ddpm.cfg.x_dim}"

    # Check refine_action works
    fake_cond = torch.zeros(1, 28)
    fake_act = torch.zeros(1, 2)
    refined = ddpm.refine_action(copilot_obs=fake_cond, pilot_action=fake_act, k=10)
    assert refined.shape == (1, 2), f"Bad shape: {refined.shape}"
    print(f"refine_action output: {refined.cpu().numpy()}")
    print("\nSMOKE TEST PASSED")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="hazard/hazard_demos.npz")
    parser.add_argument("--out_dir", type=str,
                        default="hazard/new_diffusion_hazard")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="sigmoid",
                        choices=["sigmoid", "linear"])
    parser.add_argument("--beta_min", type=float, default=1e-4)
    parser.add_argument("--beta_max", type=float, default=0.02)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=200_000)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
