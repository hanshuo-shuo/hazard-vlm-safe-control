"""
VLM-guided control for PointHazardEnv: PIVOT VLM pilot via OpenRouter API.

Same logic as shared_autonomy_hazard_pivot.py but the VLM call is made
through the OpenRouter API (Gemini Flash) instead of a local model.

Usage:
    conda activate llmimproverl
    python hazard/shared_autonomy_hazard_pivot_openrouter.py \\
        --api_key YOUR_OPENROUTER_KEY --episodes 10 --disable_diffusion

    # With diffusion
    python hazard/shared_autonomy_hazard_pivot_openrouter.py \\
        --api_key YOUR_OPENROUTER_KEY --episodes 10
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from env_pointhazard import PointHazardConfig, make_env
from hazard_renderer import HazardRenderer
from pivot_vlm import (
    PivotResult,
    _parse_pivot_selection,
    annotate_candidates,
    build_pivot_prompt,
    generate_candidates,
)
from safe_expert import SafeExpert, SafeExpertConfig
from ddpm import ConditionalDDPM, DiffusionConfig, RunningNorm


# ---------------------------------------------------------------------------
# OpenRouter VLM call
# ---------------------------------------------------------------------------

def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_openrouter(
    api_key: str,
    model: str,
    image: Image.Image,
    prompt_text: str,
    *,
    n_candidates: int,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
) -> tuple[int | None, str | None, str, bool]:
    """Call OpenRouter with a vision message. Returns (choice_0idx, reason, raw_text, parsed_ok)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    b64 = _image_to_base64(image)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    raw_text = response.choices[0].message.content or ""
    choice_idx, reason, parsed_ok = _parse_pivot_selection(raw_text, n_candidates)
    return choice_idx, reason, raw_text, parsed_ok


def pivot_choose_action_api(
    api_key: str,
    model: str,
    base_image: Image.Image,
    renderer,
    agent_world_xy: np.ndarray,
    *,
    n_directions: int = 8,
    n_magnitudes: int = 2,
    arrow_length_world: float = 0.8,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    retries: int = 1,
) -> PivotResult:
    """PIVOT pipeline via OpenRouter API: generate candidates -> annotate -> API call -> action."""
    candidates = generate_candidates(n_directions, n_magnitudes)
    n_cand = len(candidates)

    annotated = annotate_candidates(
        base_image, candidates, renderer, agent_world_xy, arrow_length_world)

    prompt_text = build_pivot_prompt(n_cand)

    choice_idx, reason, raw_text, parsed_ok = None, None, "", False
    for attempt in range(retries + 1):
        p_text = prompt_text
        if attempt > 0:
            p_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"
        choice_idx, reason, raw_text, parsed_ok = _call_openrouter(
            api_key, model, annotated, p_text,
            n_candidates=n_cand,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if parsed_ok:
            break

    if parsed_ok and choice_idx is not None:
        action = candidates[choice_idx].copy()
        confidence = 0.9
    else:
        action = np.zeros(2, dtype=np.float32)
        confidence = 0.3
    uncertainty = 1.0 - confidence

    return PivotResult(
        action=np.clip(action, -1.0, 1.0),
        confidence=confidence,
        uncertainty=uncertainty,
        choice_idx=choice_idx if choice_idx is not None else -1,
        reason=reason,
        raw_text=raw_text,
        parsed_ok=parsed_ok,
        mean_logprob=float("nan"),
        n_candidates=n_cand,
    )


# ---------------------------------------------------------------------------
# Diffusion checkpoint loading
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
        description="PIVOT VLM-guided control via OpenRouter Gemini Flash API")

    # Episodes
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    # Pilot mode
    parser.add_argument("--pilot_mode", type=str, default="vlm",
                        choices=["vlm", "goal_error", "noisy_expert", "random"])
    parser.add_argument("--pilot_noise", type=float, default=0.3)

    # OpenRouter / VLM config
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--vlm_every", type=int, default=1,
                        help="Call VLM every N steps")
    parser.add_argument("--vlm_retries", type=int, default=1)

    # PIVOT config
    parser.add_argument("--pivot_n_dirs", type=int, default=8)
    parser.add_argument("--pivot_n_mags", type=int, default=1)
    parser.add_argument("--pivot_arrow_len", type=float, default=0.6)

    # Diffusion config
    parser.add_argument("--diff_ckpt", type=str, default="hazard/new_diffusion_hazard/riskdiffusion_hazard_best03.pt")
    parser.add_argument("--fwd_ratio", type=float, default=0.5)
    parser.add_argument("--disable_diffusion", action="store_true")
    parser.add_argument("--assist_strength", type=float, default=1.0)

    # Adaptive k
    parser.add_argument("--adaptive_k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random_k", action="store_true",
                        help="Random k baseline")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=25)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.0)
    parser.add_argument("--uncertainty_scale", type=float, default=20.0)

    # Adaptive fwd_ratio
    parser.add_argument("--adaptive_fwd_ratio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fwd_ratio_near", type=float, default=0.8)
    parser.add_argument("--hazard_dist_threshold", type=float, default=0.45)

    # Logging
    parser.add_argument("--save_log", default=True)
    parser.add_argument("--log_dir", type=str, default="hazard/logs_hazard_pivot_or")
    parser.add_argument("--save_gif", default=True)
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_hazard_pivot_or")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if args.pilot_mode == "vlm" and not api_key:
        parser.error("--api_key is required for vlm mode (or set OPENROUTER_API_KEY)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Environment ---
    env_cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=env_cfg, seed=0)

    # --- Renderer ---
    renderer = HazardRenderer.from_env(env)

    if args.pilot_mode == "vlm":
        n_cand = args.pivot_n_dirs * args.pivot_n_mags
        print(f"OpenRouter VLM: {args.model}")
        print(f"PIVOT: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags = {n_cand} candidates")

    # --- Safe expert ---
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
        print(f"Diffusion loaded: cond_dim={cond_dim}, act_dim={act_dim}, k={k}/{max_k}")

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
        goal = (obs[4:6] + obs[0:2]).copy()

        if expert is not None:
            expert.reset_from_obs(obs)

        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []

        held_intent = np.zeros(2, dtype=np.float32)
        last_vlm_uncertainty = 0.5
        last_vlm_confidence = 0.5
        last_vlm_logprob = float("nan")
        last_vlm_raw = ""
        last_vlm_parsed_ok = False
        last_choice_idx = -1
        result = None

        log_rows: list[dict] = []
        effective_k = 0

        for t in range(args.max_steps):
            frame = env.render()
            if frame is None:
                frame = np.zeros((480, 480, 3), dtype=np.uint8)
            image = Image.fromarray(frame)

            if args.save_gif and t % args.gif_every == 0:
                frames.append(image.copy())

            vlm_called = False

            # --- Pilot action ---
            if args.pilot_mode == "vlm":
                if t % max(1, args.vlm_every) == 0:
                    vlm_called = True
                    result = pivot_choose_action_api(
                        api_key, args.model, image, renderer,
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
                diff = obs[4:6]
                norm = float(np.linalg.norm(diff))
                pilot = np.clip(diff / norm, -1.0, 1.0).astype(np.float32) if norm > 1e-6 else np.zeros(2, dtype=np.float32)

            elif args.pilot_mode == "noisy_expert":
                assert expert is not None
                clean = expert.act(obs)
                noise = np.random.normal(0, args.pilot_noise, size=2).astype(np.float32)
                pilot = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "random":
                pilot = np.random.uniform(-1, 1, size=2).astype(np.float32)

            else:
                raise ValueError(f"Unknown pilot_mode: {args.pilot_mode}")

            # --- Hazard proximity ---
            n_haz = env_cfg.n_hazards
            copilot_obs = np.concatenate([obs[:4], obs[6:6 + 3 * n_haz]]).astype(np.float32)
            uncertainty = last_vlm_uncertainty if args.pilot_mode == "vlm" else 0.0

            hazard_data = obs[6:6 + 3 * n_haz].reshape(n_haz, 3)
            haz_dxy = hazard_data[:, :2]
            haz_radii = hazard_data[:, 2]
            dists_to_haz = np.linalg.norm(haz_dxy, axis=1) - haz_radii
            min_haz_dist = float(np.min(dists_to_haz))

            # --- Diffusion copilot ---
            if not use_diffusion or ddpm is None:
                assist = pilot.copy()
                effective_k = 0
            else:
                if args.adaptive_fwd_ratio:
                    base_k = int(round(args.fwd_ratio_near * max_k)) if min_haz_dist < args.hazard_dist_threshold else k
                else:
                    base_k = k

                if args.random_k:
                    effective_k = int(np.random.randint(args.k_min, args.k_max + 1))
                elif args.adaptive_k and args.pilot_mode == "vlm":
                    score = float(np.clip(
                        (uncertainty - args.uncertainty_threshold) * args.uncertainty_scale, 0.0, 1.0))
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
                    "goal": goal.tolist(),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    "min_haz_dist": float(min_haz_dist),
                    "pilot": pilot.tolist(),
                    "assist": assist.tolist(),
                    "exec": exec_action.tolist(),
                    "k": effective_k if use_diffusion else 0,
                    "vlm_called": vlm_called,
                    "vlm_uncertainty": float(last_vlm_uncertainty),
                    "vlm_confidence": float(last_vlm_confidence),
                    "vlm_parsed_ok": last_vlm_parsed_ok,
                    "pivot_choice": last_choice_idx + 1,
                    "pivot_held_intent": held_intent.tolist(),
                    "reward": float(reward),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                }
                if vlm_called and result is not None:
                    row["vlm_raw"] = last_vlm_raw
                    row["vlm_reason"] = result.reason
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

        if args.save_log:
            log_path = os.path.join(run_dir, f"ep{ep:03d}.jsonl")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "type": "episode_meta", "episode": ep,
                    "pilot_mode": args.pilot_mode,
                    "model": args.model,
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

        if args.save_gif and frames:
            dur = int(1000 / max(1e-6, args.gif_fps))
            tag = "ok" if success else ("hit" if hazard_hit else "timeout")
            gif_path = os.path.join(args.gif_dir, f"ep{ep:03d}_{tag}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=dur, loop=0)

    env.close()

    # Final summary
    n = args.episodes
    succ_rate = float(np.mean(ep_successes))
    hit_rate = float(np.mean(ep_hazard_hits))
    mean_ret = float(np.mean(ep_returns))
    mean_dist = float(np.nanmean(ep_final_dist))
    timeout_rate = 1.0 - succ_rate - hit_rate

    print(f"\n{'='*60}")
    print(f"Pilot: {args.pilot_mode}  Model: {args.model}  Episodes: {n}")
    print(f"  PIVOT: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags = "
          f"{args.pivot_n_dirs * args.pivot_n_mags} candidates")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n-sum(ep_successes)-sum(ep_hazard_hits):3d}/{n}  ({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    print(f"{'='*60}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "model": args.model,
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
