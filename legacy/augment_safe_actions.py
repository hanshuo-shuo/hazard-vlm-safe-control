"""
Augment expert demo dataset with ALL safe actions at each state.

For each (obs, action_expert) in the original dataset, we:
  1. Extract state (pos, vel) and hazard layout from obs.
  2. Generate candidate actions on a grid over [-1,1]^2.
  3. Simulate multi-step lookahead for each candidate.
  4. Keep candidates where the agent stays clear of all hazards.
  5. Output augmented (obs, action_safe) pairs.

This transforms the dataset from "one expert action per state" to
"all safe actions per state", enabling the diffusion to learn the
safe action MANIFOLD rather than a single goal-directed trajectory.

Usage:
    python hazard/augment_safe_actions.py \\
        --input hazard/expert_hazard_demos.npz \\
        --output hazard/augmented_hazard_demos.npz \\
        --grid_res 15 --lookahead 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env_pointhazard import PointHazardConfig


# ---------------------------------------------------------------------------
# Physics simulation (matches env_pointhazard.py exactly)
# ---------------------------------------------------------------------------

def simulate_step(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    cfg: PointHazardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step semi-implicit Euler, matching PointHazardEnv.step()."""
    force = cfg.force_scale * np.clip(action, -1.0, 1.0)
    new_vel = vel + (force - cfg.drag * vel) * cfg.dt
    speed = float(np.linalg.norm(new_vel))
    if speed > cfg.max_speed:
        new_vel = new_vel * (cfg.max_speed / speed)
    new_pos = pos + new_vel * cfg.dt

    # Arena bounce
    a = cfg.arena_half - cfg.agent_radius
    for i in range(2):
        if new_pos[i] < -a:
            new_pos[i] = -a
            new_vel[i] = -new_vel[i] * 0.5
        elif new_pos[i] > a:
            new_pos[i] = a
            new_vel[i] = -new_vel[i] * 0.5

    return new_pos.astype(np.float32), new_vel.astype(np.float32)


def check_hazard_clearance(
    pos: np.ndarray,
    hazards_abs: np.ndarray,
    agent_radius: float,
    margin: float,
) -> bool:
    """True if agent at `pos` is clear of all hazards by `margin`."""
    for hx, hy, hr in hazards_abs:
        d = float(np.linalg.norm(pos - np.array([hx, hy], dtype=np.float32)))
        if d < (agent_radius + float(hr) + margin):
            return False
    return True


def is_action_safe(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    hazards_abs: np.ndarray,
    cfg: PointHazardConfig,
    margin: float = 0.15,
    lookahead: int = 3,
) -> bool:
    """
    Simulate `lookahead` steps: apply `action` on step 0, then zero action
    for remaining steps (conservative: can the agent stop safely?).
    """
    p, v = pos.copy(), vel.copy()
    for step in range(lookahead):
        a = action if step == 0 else np.zeros(2, dtype=np.float32)
        p, v = simulate_step(p, v, a, cfg)
        if not check_hazard_clearance(p, hazards_abs, cfg.agent_radius, margin):
            return False
    return True


# ---------------------------------------------------------------------------
# Vectorized safety check for a batch of candidate actions
# ---------------------------------------------------------------------------

def check_actions_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    candidates: np.ndarray,
    hazards_abs: np.ndarray,
    cfg: PointHazardConfig,
    margin: float = 0.15,
    lookahead: int = 3,
) -> np.ndarray:
    """
    Check safety for N candidate actions at one state.
    Returns boolean mask of shape (N,).
    """
    N = candidates.shape[0]
    safe = np.ones(N, dtype=bool)

    for i in range(N):
        if not safe[i]:
            continue
        safe[i] = is_action_safe(
            pos, vel, candidates[i], hazards_abs, cfg, margin, lookahead
        )
    return safe


# ---------------------------------------------------------------------------
# Main augmentation
# ---------------------------------------------------------------------------

def generate_candidate_grid(grid_res: int) -> np.ndarray:
    """Uniform grid over [-1,1]^2."""
    vals = np.linspace(-1.0, 1.0, grid_res).astype(np.float32)
    gx, gy = np.meshgrid(vals, vals)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def augment_dataset(
    input_path: str,
    output_path: str,
    grid_res: int = 15,
    margin: float = 0.15,
    lookahead: int = 3,
    subsample: int = 0,
    max_per_state: int = 0,
):
    cfg = PointHazardConfig()
    n_haz = cfg.n_hazards

    print(f"Loading {input_path} ...")
    data = np.load(input_path)
    obs_all = data["obs"].astype(np.float32)
    act_all = data["actions"].astype(np.float32)
    N = obs_all.shape[0]
    print(f"  Original: {N} transitions, obs_dim={obs_all.shape[1]}")

    # Candidate grid
    candidates = generate_candidate_grid(grid_res)
    n_candidates = candidates.shape[0]
    print(f"  Candidate grid: {grid_res}x{grid_res} = {n_candidates} actions")
    print(f"  Lookahead: {lookahead} steps, margin: {margin}")

    # Optionally subsample original states for speed
    if subsample > 0 and subsample < N:
        indices = np.random.choice(N, subsample, replace=False)
        indices.sort()
        print(f"  Subsampling {subsample}/{N} states")
    else:
        indices = np.arange(N)

    aug_obs = []
    aug_act = []
    n_safe_total = 0
    n_states = len(indices)

    t0 = time.time()

    for count, idx in enumerate(indices):
        obs = obs_all[idx]
        pos = obs[0:2].copy()
        vel = obs[2:4].copy()
        # obs[4:6] is relative goal — not used for safety
        haz_rel = obs[6:6 + 3 * n_haz].reshape(n_haz, 3).copy()
        # Convert relative hazard (dx, dy, r) to absolute (x, y, r)
        hazards_abs = haz_rel.copy()
        hazards_abs[:, 0] += pos[0]
        hazards_abs[:, 1] += pos[1]

        # Check each candidate
        safe_mask = check_actions_batch(
            pos, vel, candidates, hazards_abs, cfg, margin, lookahead
        )
        safe_actions = candidates[safe_mask]

        if max_per_state > 0 and len(safe_actions) > max_per_state:
            chosen = np.random.choice(len(safe_actions), max_per_state, replace=False)
            safe_actions = safe_actions[chosen]

        if len(safe_actions) > 0:
            # Replicate the obs for each safe action
            obs_repeated = np.tile(obs, (len(safe_actions), 1))
            aug_obs.append(obs_repeated)
            aug_act.append(safe_actions)
            n_safe_total += len(safe_actions)

        if (count + 1) % 5000 == 0 or count == n_states - 1:
            elapsed = time.time() - t0
            avg_safe = n_safe_total / (count + 1)
            print(f"  [{count+1}/{n_states}] "
                  f"avg_safe_per_state={avg_safe:.1f}  "
                  f"total_aug={n_safe_total}  "
                  f"({elapsed:.0f}s)")

    # Concatenate
    aug_obs_arr = np.concatenate(aug_obs, axis=0).astype(np.float32)
    aug_act_arr = np.concatenate(aug_act, axis=0).astype(np.float32)

    print(f"\nAugmented dataset:")
    print(f"  States processed: {n_states}")
    print(f"  Total transitions: {len(aug_obs_arr)}")
    print(f"  Avg safe actions per state: {len(aug_obs_arr) / n_states:.1f}")
    print(f"  obs shape: {aug_obs_arr.shape}")
    print(f"  action shape: {aug_act_arr.shape}")

    # Stats
    print(f"\n  Action stats:")
    for i in range(aug_act_arr.shape[1]):
        print(f"    dim {i}: mean={aug_act_arr[:,i].mean():+.3f}  "
              f"std={aug_act_arr[:,i].std():.3f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        obs=aug_obs_arr,
        actions=aug_act_arr,
        metadata=np.array([n_states, len(aug_obs_arr), grid_res, lookahead, margin]),
    )
    print(f"\nSaved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment expert demos with all safe actions per state")
    parser.add_argument("--input", type=str,
                        default="hazard_demos.npz",
                        help="Input expert demo .npz")
    parser.add_argument("--output", type=str,
                        default="augmented_hazard_demos.npz",
                        help="Output augmented .npz")
    parser.add_argument("--grid_res", type=int, default=15,
                        help="Grid resolution per axis (15 -> 225 candidates)")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="Safety margin on top of agent_radius")
    parser.add_argument("--lookahead", type=int, default=3,
                        help="Number of lookahead steps for safety check")
    parser.add_argument("--subsample", type=int, default=50000,
                        help="Subsample N states from original dataset (0=all)")
    parser.add_argument("--max_per_state", type=int, default=0,
                        help="Cap safe actions per state (0=unlimited)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    augment_dataset(
        input_path=args.input,
        output_path=args.output,
        grid_res=args.grid_res,
        margin=args.margin,
        lookahead=args.lookahead,
        subsample=args.subsample,
        max_per_state=args.max_per_state,
    )


if __name__ == "__main__":
    main()
