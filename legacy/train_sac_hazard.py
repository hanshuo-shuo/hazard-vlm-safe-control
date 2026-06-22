"""
Train SAC on PointHazardEnv via Stable-Baselines3, then collect demo data.

Two modes:
  1. --mode train   : Train SAC from scratch, save model checkpoint.
  2. --mode collect : Load trained SAC, roll out episodes, save (obs, action)
                      dataset as .npz for diffusion training.

The collected dataset intentionally FILTERS OUT episodes where the agent
hits a hazard, so the diffusion only trains on safe trajectories.

Usage:
    # Train SAC (default 200k steps)
    python pointmaze_copilot/train_sac_hazard.py --mode train \\
        --total_steps 200000 --save_path pointmaze_copilot/sac_hazard.zip

    # Collect safe demos from trained SAC
    python pointmaze_copilot/train_sac_hazard.py --mode collect \\
        --load_path pointmaze_copilot/sac_hazard.zip \\
        --n_episodes 3000 --out_path pointmaze_copilot/sac_hazard_demos.npz

    # Collect safe demos from A* expert (no SAC needed)
    python pointmaze_copilot/train_sac_hazard.py --mode collect_expert \\
        --n_episodes 3000 --out_path hazard/expert_hazard_demos.npz

    # Quick end-to-end test
    python pointmaze_copilot/train_sac_hazard.py --mode test
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ---------------------------------------------------------------------------
# Environment factory for SB3
# ---------------------------------------------------------------------------

def _make_env(seed: int = 0):
    from hazard.env_hazard_gym import make_gym_env
    from hazard.env_pointhazard import PointHazardConfig
    cfg = PointHazardConfig()
    env = make_gym_env(cfg=cfg, seed=seed)
    return env


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.noise import NormalActionNoise

    print(f"Training SAC for {args.total_steps} steps ...")
    env = _make_env(seed=args.seed)

    # Extra Gaussian exploration noise on top of SAC's entropy-based exploration
    if args.train_noise > 0:
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.train_noise * np.ones(n_actions),
        )
        print(f"  Using additional Gaussian action noise σ={args.train_noise}")
    else:
        action_noise = None

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        action_noise=action_noise,
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    # Checkpoint every 50k steps
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix="sac_hazard_ckpt",
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=ckpt_cb,
        progress_bar=True,
    )

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")
    env.close()


# ---------------------------------------------------------------------------
# Collect demos from trained SAC
# ---------------------------------------------------------------------------

def collect_sac(args):
    from stable_baselines3 import SAC

    print(f"Loading SAC from {args.load_path} ...")
    env = _make_env(seed=args.seed)
    model = SAC.load(args.load_path, env=env, device="auto")

    _collect_rollouts(
        env=env,
        policy_fn=lambda obs: model.predict(obs, deterministic=args.deterministic)[0],
        n_episodes=args.n_episodes,
        out_path=args.out_path,
        action_noise_std=args.action_noise,
        seed=args.seed,
        source_name="SAC",
    )
    env.close()


# ---------------------------------------------------------------------------
# Collect demos from A* safe expert
# ---------------------------------------------------------------------------

def collect_expert(args):
    from hazard.env_pointhazard import PointHazardConfig, make_env
    from hazard.safe_expert import SafeExpert, SafeExpertConfig

    cfg = PointHazardConfig()
    env = make_env(cfg=cfg, seed=args.seed)
    expert = SafeExpert(env, cfg=SafeExpertConfig())

    def expert_policy_fn(obs):
        return expert.act(obs)

    def reset_hook(obs):
        expert.reset_from_obs(obs)

    _collect_rollouts(
        env=env,
        policy_fn=expert_policy_fn,
        n_episodes=args.n_episodes,
        out_path=args.out_path,
        action_noise_std=args.action_noise,
        seed=args.seed,
        source_name="A*-expert",
        reset_hook=reset_hook,
    )
    env.close()


# ---------------------------------------------------------------------------
# Shared rollout collection
# ---------------------------------------------------------------------------

def _collect_rollouts(
    env,
    policy_fn,
    n_episodes: int,
    out_path: str,
    action_noise_std: float,
    seed: int,
    source_name: str,
    reset_hook=None,
    max_steps: int = 300,
):
    all_obs = []
    all_actions = []

    n_success = 0
    n_hazard = 0
    n_timeout = 0
    n_transitions_total = 0

    t0 = time.time()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        obs = np.asarray(obs, dtype=np.float32)

        if reset_hook is not None:
            reset_hook(obs)

        ep_obs = []
        ep_act = []
        hit_hazard = False
        success = False

        for t in range(max_steps):
            action = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(-1)

            # Optional noise injection
            if action_noise_std > 0:
                noise = np.random.normal(0, action_noise_std, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0).astype(np.float32)

            ep_obs.append(obs.copy())
            ep_act.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)

            if terminated:
                if info.get("termination_reason") == "hazard":
                    hit_hazard = True
                elif info.get("termination_reason") == "goal":
                    success = True
                break
            if truncated:
                break

        # Only keep SAFE episodes (no hazard hit)
        if not hit_hazard and len(ep_obs) > 0:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_act)

        if hit_hazard:
            n_hazard += 1
        elif success:
            n_success += 1
        else:
            n_timeout += 1

        n_transitions_total += len(ep_obs)

        if (ep + 1) % 200 == 0 or ep == n_episodes - 1:
            elapsed = time.time() - t0
            print(f"  [{source_name}] ep {ep+1}/{n_episodes}  "
                  f"success={n_success}  hazard={n_hazard}  timeout={n_timeout}  "
                  f"safe_transitions={len(all_obs)}  "
                  f"({elapsed:.0f}s)")

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.float32)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=obs_arr,
        actions=act_arr,
        metadata=np.array([n_episodes, n_success, n_hazard, n_timeout, len(all_obs)]),
    )

    kept_eps = n_success + n_timeout
    print(f"\nDataset saved to {out_path}")
    print(f"  Source: {source_name}")
    print(f"  Episodes: {n_episodes} total, {kept_eps} safe (filtered {n_hazard} hazard-hit)")
    print(f"  Transitions: {len(all_obs)} safe  (discarded {n_transitions_total - len(all_obs)} from unsafe eps)")
    print(f"  obs shape: {obs_arr.shape}  action shape: {act_arr.shape}")


# ---------------------------------------------------------------------------
# Quick end-to-end test
# ---------------------------------------------------------------------------

def test_e2e(args):
    """Train 5k steps, collect 20 episodes, verify shapes."""
    from stable_baselines3 import SAC

    print("=" * 50)
    print("End-to-end test: train 5k steps + collect 20 eps")
    print("=" * 50)

    env = _make_env(seed=0)
    model = SAC("MlpPolicy", env, verbose=0, learning_starts=500, seed=0, device="cpu")
    model.learn(total_timesteps=5000, progress_bar=True)

    tmp_path = "/tmp/sac_hazard_test.zip"
    model.save(tmp_path)
    print(f"SAC saved to {tmp_path}")

    # Reload and collect
    model2 = SAC.load(tmp_path, env=env, device="cpu")

    out_npz = "/tmp/sac_hazard_test_demos.npz"
    _collect_rollouts(
        env=env,
        policy_fn=lambda obs: model2.predict(obs, deterministic=True)[0],
        n_episodes=20,
        out_path=out_npz,
        action_noise_std=0.0,
        seed=100,
        source_name="SAC-test",
    )

    # Verify
    data = np.load(out_npz)
    print(f"\nVerification:")
    print(f"  obs shape: {data['obs'].shape}")
    print(f"  actions shape: {data['actions'].shape}")
    assert data["obs"].shape[1] == 30, f"Expected obs_dim=30, got {data['obs'].shape[1]}"
    assert data["actions"].shape[1] == 2, f"Expected act_dim=2, got {data['actions'].shape[1]}"
    assert data["obs"].shape[0] == data["actions"].shape[0]
    print("PASS: shapes correct")

    # Also test expert collection
    print("\n--- Expert collection test ---")
    from pointmaze_copilot.env_pointhazard import make_env as make_raw_env
    from pointmaze_copilot.safe_expert import SafeExpert, SafeExpertConfig

    raw_env = make_raw_env(seed=0)
    expert = SafeExpert(raw_env, cfg=SafeExpertConfig())

    out_expert = "/tmp/expert_hazard_test_demos.npz"
    _collect_rollouts(
        env=raw_env,
        policy_fn=lambda obs: expert.act(obs),
        n_episodes=20,
        out_path=out_expert,
        action_noise_std=0.05,
        seed=200,
        source_name="A*-expert",
        reset_hook=lambda obs: expert.reset_from_obs(obs),
    )

    data_e = np.load(out_expert)
    print(f"  Expert obs shape: {data_e['obs'].shape}")
    print(f"  Expert actions shape: {data_e['actions'].shape}")
    assert data_e["obs"].shape[1] == 30
    assert data_e["actions"].shape[1] == 2

    # Expert should have 0 hazard hits
    meta = data_e["metadata"]
    print(f"  Expert metadata: eps={meta[0]} success={meta[1]} hazard={meta[2]} timeout={meta[3]} transitions={meta[4]}")
    assert meta[2] == 0, f"Expert had {meta[2]} hazard hits — should be 0!"
    print("PASS: expert has 0 hazard hits")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)

    env.close()
    raw_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAC training + demo collection for PointHazardEnv")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["train", "collect", "collect_expert", "test"])
    parser.add_argument("--seed", type=int, default=42)

    # Train args
    parser.add_argument("--total_steps", type=int, default=2000_000)
    parser.add_argument("--save_path", type=str, default="hazard/sac_hazard.zip")
    parser.add_argument("--train_noise", type=float, default=0.1,
                        help="Gaussian noise std for exploration during training (0 to disable)")

    # Collect args
    parser.add_argument("--load_path", type=str, default="hazard/sac_hazard.zip")
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--out_path", type=str, default="hazard/hazard_demos.npz")
    parser.add_argument("--action_noise", type=float, default=0.05,
                        help="Gaussian noise std added to actions during collection")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic SAC policy for collection")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "collect":
        collect_sac(args)
    elif args.mode == "collect_expert":
        collect_expert(args)
    elif args.mode == "test":
        test_e2e(args)


if __name__ == "__main__":
    main()
