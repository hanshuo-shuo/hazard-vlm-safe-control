"""
Shared autonomy for PointHazardEnv: PIVOT VLM pilot + diffusion safety copilot.

This is the PIVOT variant of shared_autonomy_hazard.py. Instead of asking the
VLM to output continuous [fx, fy] values (regression), we render numbered
candidate-direction arrows on the image and let the VLM *choose* the best one
(classification). Inspired by:
    PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs

Architecture:
    1. Render environment frame
    2. Generate candidate force directions (evenly spaced)
    3. Annotate frame with numbered orange arrows
    4. VLM selects the best arrow number
    5. Selected action feeds into diffusion copilot (unchanged)

Pilot modes:
    vlm           : PIVOT visual prompting (select from arrows)
    goal_error    : oracle  dx = normalize(goal - pos)
    noisy_expert  : safe A*-expert + Gaussian noise
    random        : uniform random actions

Usage:
    # PIVOT VLM only (no diffusion)
    python hazard/shared_autonomy_hazard_pivot.py \\
        --pilot_mode vlm --disable_diffusion --episodes 20

    # PIVOT VLM + diffusion
    python hazard/shared_autonomy_hazard_pivot.py \\
        --pilot_mode vlm --episodes 50

    # PIVOT VLM + adaptive-k diffusion
    python hazard/shared_autonomy_hazard_pivot.py \\
        --pilot_mode vlm --adaptive_k --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from hazard.env_pointhazard import PointHazardConfig, make_env
from hazard.hazard_renderer import HazardRenderer
from hazard.pivot_vlm import pivot_choose_action
from hazard.safe_expert import SafeExpert, SafeExpertConfig
from ddpm import ConditionalDDPM, DiffusionConfig, RunningNorm


# ---------------------------------------------------------------------------
# Diffusion checkpoint loading (same as original)
# ---------------------------------------------------------------------------

def _load_ddpm_ckpt(path: str, *, device: torch.device) -> ConditionalDDPM:
    ckpt = torch.load(path, map_location="cpu")
    dd = ckpt["ddpm"]
    cfg = DiffusionConfig(**dd["cfg"])
    norm = None
    if dd.get("norm") is not None:
        norm = RunningNorm(mean=dd["norm"]["mean"].float(),
                           std=dd["norm"]["std"].float())
    model = ConditionalDDPM(cfg, device=device, norm=norm)
    model.load_state_dict(dd)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PIVOT shared autonomy on PointHazardEnv: VLM selects from visual candidates + diffusion copilot")

    # Episodes
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    # Pilot mode
    parser.add_argument("--pilot_mode", type=str, default="vlm",
                        choices=["vlm", "goal_error", "noisy_expert", "random"])
    parser.add_argument("--pilot_noise", type=float, default=0.3,
                        help="Noise std for noisy_expert mode")

    # VLM config
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vlm_every", type=int, default=1,
                        help="Call VLM every N steps (hold intent between calls)")
    parser.add_argument("--vlm_retries", type=int, default=1)

    # PIVOT config
    parser.add_argument("--pivot_n_dirs", type=int, default=8,
                        help="Number of direction bins (evenly spaced)")
    parser.add_argument("--pivot_n_mags", type=int, default=2,
                        help="Number of magnitude levels per direction")
    parser.add_argument("--pivot_arrow_len", type=float, default=0.6,
                        help="Arrow length in world units for magnitude=1")

    # Diffusion config
    parser.add_argument("--diff_ckpt", type=str, default="hazard/diffusion_hazard/diffusion_hazard_best.pt")
    parser.add_argument("--fwd_ratio", type=float, default=0.0,
                        help="Forward diffusion ratio in [0,1]: 0=no diffusion, 1=full refine")
    parser.add_argument("--disable_diffusion", action="store_true")
    parser.add_argument("--assist_strength", type=float, default=1.0)

    # Adaptive k (uncertainty-gated guidance scale)
    parser.add_argument("--adaptive_k", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--random_k", action="store_true",
                        help="Random k baseline")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=25)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.0)
    parser.add_argument("--uncertainty_scale", type=float, default=20.0)

    # Adaptive fwd_ratio based on hazard proximity
    parser.add_argument("--adaptive_fwd_ratio", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Increase fwd_ratio when close to hazard (< threshold)")
    parser.add_argument("--fwd_ratio_near", type=float, default=0.8,
                        help="fwd_ratio when within hazard_dist_threshold (more correction)")
    parser.add_argument("--hazard_dist_threshold", type=float, default=0.45,
                        help="Distance threshold: below this use fwd_ratio_near, above use --fwd_ratio")

    # Logging
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--log_dir", type=str, default="hazard/logs_hazard_pivot")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_hazard_pivot")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Environment ---
    env_cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=env_cfg, seed=0)

    # --- Renderer (for PIVOT annotation) ---
    renderer = HazardRenderer.from_env(env)

    # --- VLM ---
    vlm_model = None
    vlm_processor = None
    if args.pilot_mode == "vlm":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"Loading VLM: {args.model_path} ...")
        vlm_processor = AutoProcessor.from_pretrained(args.model_path)
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        vlm_model.eval()
        n_cand = args.pivot_n_dirs * args.pivot_n_mags
        print(f"VLM loaded. PIVOT mode: {args.pivot_n_dirs} dirs x "
              f"{args.pivot_n_mags} mags = {n_cand} candidates")

    # --- Safe expert (for noisy_expert baseline) ---
    expert = None
    if args.pilot_mode == "noisy_expert":
        expert = SafeExpert(env, cfg=SafeExpertConfig())

    # --- Diffusion copilot ---
    use_diffusion = (not args.disable_diffusion) and bool(args.diff_ckpt) and args.fwd_ratio > 0
    ddpm = None
    max_k = 0
    k = 0
    if use_diffusion:
        ddpm = _load_ddpm_ckpt(args.diff_ckpt, device=device)
        cond_dim = ddpm.cfg.cond_dim
        act_dim = ddpm.cfg.x_dim - cond_dim
        max_k = ddpm.cfg.num_steps - 1
        k = int(round(args.fwd_ratio * max_k))
        print(f"Diffusion loaded: cond_dim={cond_dim}, act_dim={act_dim}, "
              f"k={k}/{max_k}")

    # --- Output dirs ---
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.save_log:
        run_dir = os.path.join(args.log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
    if args.save_gif:
        os.makedirs(args.gif_dir, exist_ok=True)

    # --- Evaluation loop ---
    ep_returns: list[float] = []
    ep_successes: list[int] = []
    ep_hazard_hits: list[int] = []
    ep_final_dist: list[float] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        goal = obs[4:6].copy()

        if expert is not None:
            expert.reset_from_obs(obs)

        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []

        # VLM / PIVOT state
        held_intent = np.zeros(2, dtype=np.float32)
        last_vlm_uncertainty = 0.5
        last_vlm_confidence = 0.5
        last_vlm_logprob = float("nan")
        last_vlm_raw = ""
        last_vlm_parsed_ok = False
        last_choice_idx = -1

        log_rows: list[dict] = []

        for t in range(args.max_steps):
            frame = env.render()
            if frame is None:
                frame = np.zeros((480, 480, 3), dtype=np.uint8)
            image = Image.fromarray(frame)

            if args.save_gif and t % args.gif_every == 0:
                frames.append(image.copy())

            vlm_called = False
            result = None

            # --- Compute pilot action ---
            if args.pilot_mode == "vlm":
                if t % max(1, args.vlm_every) == 0:
                    vlm_called = True
                    result = pivot_choose_action(
                        vlm_model, vlm_processor, image, renderer,
                        agent_world_xy=obs[:2],
                        n_directions=args.pivot_n_dirs,
                        n_magnitudes=args.pivot_n_mags,
                        arrow_length_world=args.pivot_arrow_len,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        retries=args.vlm_retries,
                    )

                    if result.parsed_ok:
                        held_intent = result.action.copy()
                    last_vlm_uncertainty = result.uncertainty
                    last_vlm_confidence = result.confidence
                    last_vlm_logprob = result.mean_logprob
                    last_vlm_raw = result.raw_text
                    last_vlm_parsed_ok = result.parsed_ok
                    last_choice_idx = result.choice_idx

                    if args.debug_print:
                        reason_str = (result.reason or "")[:60]
                        print(f"  [pivot t={t}] choice={result.choice_idx+1}/{result.n_candidates} "
                              f"action=({held_intent[0]:+.2f},{held_intent[1]:+.2f}) "
                              f"conf={result.confidence:.3f} ok={result.parsed_ok} "
                              f"reason={reason_str}")

                pilot = np.clip(held_intent, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "goal_error":
                diff = obs[4:6] - obs[:2]
                norm = float(np.linalg.norm(diff))
                if norm > 1e-6:
                    pilot = np.clip(diff / norm, -1.0, 1.0).astype(np.float32)
                else:
                    pilot = np.zeros(2, dtype=np.float32)

            elif args.pilot_mode == "noisy_expert":
                assert expert is not None
                clean = expert.act(obs)
                noise = np.random.normal(0, args.pilot_noise, size=2).astype(np.float32)
                pilot = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "random":
                pilot = np.random.uniform(-1, 1, size=2).astype(np.float32)

            else:
                raise ValueError(f"Unknown pilot_mode: {args.pilot_mode}")

            # --- Diffusion copilot refine ---
            n_haz = env_cfg.n_hazards
            copilot_obs = np.concatenate([obs[:4], obs[6:6 + 3 * n_haz]]).astype(np.float32)
            uncertainty = last_vlm_uncertainty if args.pilot_mode == "vlm" else 0.0

            # Compute distance to nearest hazard (edge distance = center dist - radius)
            agent_pos = obs[:2]
            hazard_data = obs[6:6 + 3 * n_haz].reshape(n_haz, 3)  # (x, y, r)
            haz_centers = hazard_data[:, :2]
            haz_radii = hazard_data[:, 2]
            dists_to_haz = np.linalg.norm(haz_centers - agent_pos, axis=1) - haz_radii
            min_haz_dist = float(np.min(dists_to_haz))

            if not use_diffusion or ddpm is None:
                assist = pilot.copy()
                effective_k = 0
            else:
                # Determine fwd_ratio: adaptive by hazard proximity or fixed
                if args.adaptive_fwd_ratio:
                    if min_haz_dist < args.hazard_dist_threshold:
                        base_k = int(round(args.fwd_ratio_near * max_k))
                    else:
                        base_k = k  # use default --fwd_ratio
                else:
                    base_k = k

                if args.random_k:
                    effective_k = int(np.random.randint(args.k_min, args.k_max + 1))
                elif args.adaptive_k and args.pilot_mode == "vlm":
                    score = float(np.clip(
                        (uncertainty - args.uncertainty_threshold) * args.uncertainty_scale,
                        0.0, 1.0))
                    effective_k = int(args.k_min + (args.k_max - args.k_min) * score)
                else:
                    effective_k = base_k
                effective_k = int(np.clip(effective_k, 0, max_k))

                if effective_k <= 0:
                    assist = pilot.copy()
                else:
                    cop = torch.from_numpy(copilot_obs).to(device).reshape(1, -1)
                    pa = torch.from_numpy(pilot).to(device).reshape(1, -1)
                    a_hat = ddpm.refine_action(
                        copilot_obs=cop, pilot_action=pa, k=effective_k)
                    assist = a_hat.detach().cpu().numpy().reshape(-1).astype(np.float32)

            # Blend
            assist = np.clip(assist, -1.0, 1.0)
            alpha = args.assist_strength
            exec_action = pilot + alpha * (assist - pilot)
            exec_action = np.clip(exec_action, -1.0, 1.0).astype(np.float32)

            # --- Step env ---
            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            ep_return += float(reward)
            last_info = dict(info or {})

            # Log
            if args.save_log:
                row = {
                    "t": t, "pilot_mode": args.pilot_mode,
                    # Agent state
                    "agent_pos": agent_pos.tolist(),
                    "goal": goal.tolist(),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    # Hazard proximity
                    "min_haz_dist": float(min_haz_dist),
                    "adaptive_fwd_ratio_active": bool(args.adaptive_fwd_ratio and min_haz_dist < args.hazard_dist_threshold),
                    # Actions
                    "pilot": pilot.tolist(),
                    "assist": assist.tolist(),
                    "exec": exec_action.tolist(),
                    "pilot_assist_diff": float(np.linalg.norm(assist - pilot)),
                    # Diffusion
                    "k": effective_k if use_diffusion else 0,
                    "fwd_ratio_effective": (effective_k / max_k) if (use_diffusion and max_k > 0) else 0.0,
                    # PIVOT / VLM
                    "vlm_called": vlm_called,
                    "vlm_uncertainty": float(last_vlm_uncertainty),
                    "vlm_confidence": float(last_vlm_confidence),
                    "vlm_logprob": float(last_vlm_logprob),
                    "vlm_parsed_ok": last_vlm_parsed_ok,
                    "pivot_choice": last_choice_idx + 1,  # 1-indexed, -1+1=0 means no call yet
                    "pivot_held_intent": held_intent.tolist(),
                    # Outcome
                    "reward": float(reward),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                }
                if vlm_called and result is not None:
                    row["vlm_raw"] = last_vlm_raw
                    row["vlm_reason"] = result.reason
                    row["pivot_n_candidates"] = result.n_candidates
                    row["pivot_action"] = result.action.tolist()
                    row["pivot_all_scores"] = result.scores.tolist() if hasattr(result, "scores") and result.scores is not None else None
                log_rows.append(row)

            if terminated or truncated:
                break

        # Episode summary
        success = bool(last_info.get("goal_success", False))
        hazard_hit = bool(last_info.get("hazard_hit", False))
        dist = float(last_info.get("dist_to_goal", float("nan")))
        term_reason = last_info.get("termination_reason", "timeout")
        ep_returns.append(ep_return)
        ep_successes.append(int(success))
        ep_hazard_hits.append(int(hazard_hit))
        ep_final_dist.append(dist)

        flag = "OK " if success else ("HIT" if hazard_hit else "---")
        print(f"[ep {ep:03d}] [{flag}] return={ep_return:7.1f} steps={t+1:3d} "
              f"dist={dist:.2f} reason={term_reason} "
              f"goal=({goal[0]:+.1f},{goal[1]:+.1f})")

        # Save log
        if args.save_log:
            log_path = os.path.join(run_dir, f"ep{ep:03d}.jsonl")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "type": "episode_meta", "episode": ep,
                    "pilot_mode": args.pilot_mode,
                    "pivot": True,
                    "pivot_n_dirs": args.pivot_n_dirs,
                    "pivot_n_mags": args.pivot_n_mags,
                    "use_diffusion": use_diffusion, "seed": args.seed + ep,
                    "goal": goal.tolist(),
                }) + "\n")
                for row in log_rows:
                    f.write(json.dumps(row) + "\n")
                f.write(json.dumps({
                    "type": "episode_end", "return": ep_return,
                    "success": success, "hazard_hit": hazard_hit,
                    "dist": dist, "steps": t + 1,
                }) + "\n")

        # Save GIF
        if args.save_gif and frames:
            dur = int(1000 / max(1e-6, args.gif_fps))
            tag = "ok" if success else ("hit" if hazard_hit else "timeout")
            gif_path = os.path.join(args.gif_dir,
                                    f"ep{ep:03d}_{tag}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=dur, loop=0)

    env.close()

    # Final summary
    n = args.episodes
    mean_ret = float(np.mean(ep_returns))
    succ_rate = float(np.mean(ep_successes))
    hit_rate = float(np.mean(ep_hazard_hits))
    timeout_rate = 1.0 - succ_rate - hit_rate
    mean_dist = float(np.nanmean(ep_final_dist))

    print(f"\n{'='*60}")
    print(f"Pilot: {args.pilot_mode} (PIVOT)  Diffusion: {'ON' if use_diffusion else 'OFF'}  "
          f"Episodes: {n}")
    print(f"  PIVOT: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags = "
          f"{args.pivot_n_dirs * args.pivot_n_mags} candidates")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n}  "
          f"({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    if args.pilot_mode == "vlm":
        print(f"  VLM: {args.model_path}  every={args.vlm_every} temp={args.temperature}")
    print(f"{'='*60}")

    # Save overall summary
    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "pivot": True,
            "pivot_n_dirs": args.pivot_n_dirs,
            "pivot_n_mags": args.pivot_n_mags,
            "use_diffusion": use_diffusion,
            "episodes": n,
            "success_rate": succ_rate,
            "hazard_hit_rate": hit_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_ret,
            "mean_final_dist": mean_dist,
            "args": vars(args),
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
