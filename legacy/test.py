"""
Proximity-adaptive VLM-guided control for PointHazardEnv.

Same as shared_autonomy_hazard.py, but fwd_ratio (i.e. diffusion k) is
dynamically adapted based on the agent's distance to the nearest hazard:

    dist > danger_dist  →  fwd_ratio_far   (default 0.1)  — relaxed copilot
    dist = 0            →  fwd_ratio_near  (default 0.6)  — strong copilot

Between those extremes the ratio is linearly interpolated:

    ratio = fwd_ratio_far + (1 - dist/danger_dist) * (fwd_ratio_near - fwd_ratio_far)

The obs already provides hazards sorted by distance to agent (closest first),
so nearest-hazard distance = ||pos - h0_center|| - h0_radius.

Usage:
    python pointmaze_copilot/shared_autonomy_hazard_proximity.py \\
        --pilot_mode noisy_expert --episodes 100

    python pointmaze_copilot/shared_autonomy_hazard_proximity.py \\
        --pilot_mode noisy_expert --episodes 100 \\
        --fwd_ratio_far 0.1 --fwd_ratio_near 0.6 --danger_dist 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from hazard.env_pointhazard import PointHazardConfig, make_env
from hazard.safe_expert import SafeExpert, SafeExpertConfig
from diffusion_copilot.ddpm import ConditionalDDPM, DiffusionConfig, RunningNorm
from hazard.ddpm_hazard import HazardDDPM


# ---------------------------------------------------------------------------
# Diffusion checkpoint loading
# ---------------------------------------------------------------------------

def _load_ddpm_ckpt(path: str, *, device: torch.device) -> HazardDDPM:
    ckpt = torch.load(path, map_location="cpu")
    dd = ckpt["ddpm"]
    cfg = DiffusionConfig(**dd["cfg"])
    norm = None
    if dd.get("norm") is not None:
        norm = RunningNorm(mean=dd["norm"]["mean"].float(),
                           std=dd["norm"]["std"].float())
    model = ConditionalDDPM(cfg, device=device, norm=norm)
    model.load_state_dict(dd)
    return HazardDDPM(model)


# ---------------------------------------------------------------------------
# Proximity-adaptive fwd_ratio
# ---------------------------------------------------------------------------

def proximity_fwd_ratio(
    obs: np.ndarray,
    *,
    fwd_ratio_far: float,
    fwd_ratio_near: float,
    danger_dist: float,
) -> tuple[float, float]:
    """
    Compute fwd_ratio based on distance to nearest hazard.

    Returns:
        (fwd_ratio, dist_to_nearest_hazard)
    """
    # obs[6:8] is (dx, dy) relative to agent, obs[8] is radius
    haz_dx_dy = obs[6:8]
    haz_radius = float(obs[8])
    dist = float(np.linalg.norm(haz_dx_dy)) - haz_radius
    dist = max(0.0, dist)

    if dist >= danger_dist:
        ratio = fwd_ratio_far
    else:
        t = 1.0 - dist / danger_dist          # 0 at danger_dist, 1 at touching
        ratio = fwd_ratio_far + t * (fwd_ratio_near - fwd_ratio_far)

    return float(ratio), dist


# ---------------------------------------------------------------------------
# VLM intent extraction (copied from shared_autonomy_hazard.py)
# ---------------------------------------------------------------------------

@dataclass
class VlmIntentResult:
    intent: np.ndarray
    reason: str | None
    raw_text: str
    parsed_ok: bool
    mean_logprob: float
    confidence: float
    uncertainty: float


def _extract_intent_from_text(text: str) -> tuple[np.ndarray | None, str | None, bool]:
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not match:
        return None, None, False
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None, None, False

    direction = obj.get("direction") or obj.get("force") or obj.get("action")
    reason = obj.get("reason") or obj.get("explanation")

    if direction is None or not isinstance(direction, (list, tuple)) or len(direction) < 2:
        return None, reason, False
    try:
        fx = float(direction[0])
        fy = float(direction[1])
    except (ValueError, TypeError):
        return None, reason, False

    return np.array([fx, fy], dtype=np.float32), reason, True


def _build_hazard_prompt() -> str:
    return """You see a 2D arena from above with hazards.

Visual elements:
- Gray bordered square = arena boundary (bouncing wall)
- Red filled circles = HAZARDS (lava) — touching any of them kills you instantly
- Blue dot = YOU (the agent you control)
- Green circle with "G" = GOAL (your destination)
- Blue line = your past trajectory

You control the agent by applying a force [fx, fy], each in [-1, 1]:
- fx > 0 pushes RIGHT, fx < 0 pushes LEFT
- fy > 0 pushes UP, fy < 0 pushes DOWN

CRITICAL RULES:
1. Navigate toward the green goal "G"
2. AVOID all red hazard circles — stay far from them, do NOT cut close
3. If hazards block the direct path, go AROUND them
4. Prefer a safe detour over a risky shortcut

OUTPUT JSON only:
{"direction": [fx, fy], "reason": "brief explanation"}
"""


@torch.inference_mode()
def vlm_choose_intent(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    enable_thinking: bool = False,
) -> VlmIntentResult:
    messages = []
    if not enable_thinking:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": "/no_think"}],
        })
    messages.append({
        "role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]
    })

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    start_len = int(inputs["input_ids"].shape[1])

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(1e-6, temperature) if temperature > 0 else 1.0,
        top_p=top_p if temperature > 0 else 1.0,
        top_k=top_k if temperature > 0 else 0,
    )
    generated = model.generate(
        **inputs,
        generation_config=gen_config,
        return_dict_in_generate=True,
        output_scores=True,
    )

    gen_ids = generated.sequences[0][start_len:]
    out_text = processor.decode(gen_ids, skip_special_tokens=True)
    intent, reason, ok = _extract_intent_from_text(out_text)

    if intent is None:
        intent = np.zeros(2, dtype=np.float32)
    intent = np.clip(intent, -1.0, 1.0)

    token_logprobs = []
    scores = list(generated.scores or [])
    n = min(len(scores), int(gen_ids.numel()))
    for i in range(n):
        logits_i = scores[i][0]
        tok_id = int(gen_ids[i].item())
        lp = float(torch.log_softmax(logits_i, dim=-1)[tok_id].item())
        token_logprobs.append(lp)

    if token_logprobs:
        mean_logprob = float(np.mean(token_logprobs))
        confidence = float(np.clip(np.exp(mean_logprob), 0.0, 1.0))
    else:
        mean_logprob = float("nan")
        confidence = 0.5
    uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))

    return VlmIntentResult(
        intent=intent, reason=reason, raw_text=out_text,
        parsed_ok=ok, mean_logprob=mean_logprob,
        confidence=confidence, uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Proximity-adaptive VLM-guided control on PointHazardEnv")

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    # Pilot mode
    parser.add_argument("--pilot_mode", type=str, default="vlm",
                        choices=["vlm", "goal_error", "noisy_expert", "random"])
    parser.add_argument("--pilot_noise", type=float, default=0.5)

    # VLM config
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vlm_every", type=int, default=10)
    parser.add_argument("--vlm_retries", type=int, default=2)

    # Diffusion config
    parser.add_argument("--diff_ckpt", type=str,
                        default="hazard/diffusion_hazard/diffusion_hazard_best.pt")
    parser.add_argument("--disable_diffusion", action="store_true")
    parser.add_argument("--assist_strength", type=float, default=1.0)
    parser.add_argument("--multi_sample", type=int, default=8,
                        help="Number of diffusion samples for multi-sample refine (1=original)")

    # Proximity-adaptive fwd_ratio
    parser.add_argument("--fwd_ratio_far", type=float, default=0.1,
                        help="fwd_ratio when far from all hazards (>= danger_dist)")
    parser.add_argument("--fwd_ratio_near", type=float, default=0.4,
                        help="fwd_ratio when right next to a hazard (dist=0)")
    parser.add_argument("--danger_dist", type=float, default=0.5,
                        help="Hazard-edge-to-agent-center distance below which ratio ramps up. "
                             "agent_radius=0.3 (collision), SafeExpert clearance=0.45. "
                             "0.8 gives ~2 steps of warning at typical speed.")

    # Logging
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--log_dir", type=str, default="hazard/logs_proximity")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_proximity")
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
    n_haz = env_cfg.n_hazards

    # --- Diffusion copilot ---
    use_diffusion = (not args.disable_diffusion) and bool(args.diff_ckpt)
    ddpm = None
    max_k = 0
    if use_diffusion:
        ddpm = _load_ddpm_ckpt(args.diff_ckpt, device=device)
        cond_dim = ddpm.cfg.cond_dim
        act_dim = ddpm.cfg.x_dim - cond_dim
        max_k = ddpm.cfg.num_steps - 1
        k_far  = int(round(args.fwd_ratio_far  * max_k))
        k_near = int(round(args.fwd_ratio_near * max_k))
        print(f"Diffusion loaded: cond_dim={cond_dim}, act_dim={act_dim}, "
              f"k_far={k_far}/{max_k}, k_near={k_near}/{max_k}")

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
        print("VLM loaded.")

    # --- Safe expert ---
    expert = None
    if args.pilot_mode == "noisy_expert":
        expert = SafeExpert(env, cfg=SafeExpertConfig())

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
        goal = (obs[4:6] + obs[0:2]).copy()  # relative -> absolute

        if expert is not None:
            expert.reset_from_obs(obs)

        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []

        held_intent = np.zeros(2, dtype=np.float32)
        last_vlm_uncertainty = 0.5
        last_vlm_confidence = 0.5
        last_vlm_raw = ""
        last_vlm_parsed_ok = False
        result = None

        log_rows: list[dict] = []

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
                    prompt = _build_hazard_prompt()
                    result = None
                    for attempt in range(args.vlm_retries + 1):
                        p_text = prompt
                        if attempt > 0:
                            p_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"
                        result = vlm_choose_intent(
                            vlm_model, vlm_processor, image, p_text,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                        if result.parsed_ok:
                            break
                    assert result is not None
                    if result.parsed_ok:
                        held_intent = result.intent.copy()
                    last_vlm_uncertainty = result.uncertainty
                    last_vlm_confidence = result.confidence
                    last_vlm_raw = result.raw_text
                    last_vlm_parsed_ok = result.parsed_ok

                    if args.debug_print:
                        print(f"  [vlm t={t}] intent=({held_intent[0]:+.2f},{held_intent[1]:+.2f}) "
                              f"conf={result.confidence:.3f} ok={result.parsed_ok}")

                pilot = np.clip(held_intent, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "goal_error":
                diff = obs[4:6]  # already relative (goal - agent)
                norm = float(np.linalg.norm(diff))
                pilot = (np.clip(diff / norm, -1.0, 1.0).astype(np.float32)
                         if norm > 1e-6 else np.zeros(2, dtype=np.float32))

            elif args.pilot_mode == "noisy_expert":
                assert expert is not None
                clean = expert.act(obs)
                noise = np.random.normal(0, args.pilot_noise, size=2).astype(np.float32)
                pilot = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "random":
                pilot = np.random.uniform(-1, 1, size=2).astype(np.float32)

            else:
                raise ValueError(f"Unknown pilot_mode: {args.pilot_mode}")

            # --- Proximity-adaptive diffusion copilot ---
            # cond = [x, y, vx, vy, h1x, h1y, h1r, ...] — goal excluded by design
            copilot_obs = np.concatenate([obs[:4], obs[6:6 + 3 * n_haz]]).astype(np.float32)

            current_ratio, dist_nearest = proximity_fwd_ratio(
                obs,
                fwd_ratio_far=args.fwd_ratio_far,
                fwd_ratio_near=args.fwd_ratio_near,
                danger_dist=args.danger_dist,
            )

            if not use_diffusion or ddpm is None:
                assist = pilot.copy()
                effective_k = 0
            else:
                effective_k = int(np.clip(round(current_ratio * max_k), 0, max_k))

                if effective_k <= 0:
                    assist = pilot.copy()
                else:
                    cop = torch.from_numpy(copilot_obs).to(device).reshape(1, -1)
                    pa = torch.from_numpy(pilot).to(device).reshape(1, -1)
                    if args.multi_sample > 1:
                        a_hat = ddpm.refine_action_multi(
                            copilot_obs=cop, pilot_action=pa,
                            k=effective_k, n_samples=args.multi_sample)
                    else:
                        a_hat = ddpm.refine_action(
                            copilot_obs=cop, pilot_action=pa, k=effective_k)
                    assist = a_hat.detach().cpu().numpy().reshape(-1).astype(np.float32)

            # Blend
            assist = np.clip(assist, -1.0, 1.0)
            exec_action = pilot + args.assist_strength * (assist - pilot)
            exec_action = np.clip(exec_action, -1.0, 1.0).astype(np.float32)

            if args.debug_print:
                print(f"  [t={t}] dist_nearest={dist_nearest:.3f} "
                      f"ratio={current_ratio:.3f} k={effective_k}/{max_k}")

            # --- Step env ---
            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            ep_return += float(reward)
            last_info = dict(info or {})

            if args.save_log:
                row = {
                    "t": t, "pilot_mode": args.pilot_mode,
                    "pilot": pilot.tolist(), "assist": assist.tolist(),
                    "exec": exec_action.tolist(),
                    "k": effective_k,
                    "fwd_ratio": current_ratio,
                    "dist_nearest_hazard": dist_nearest,
                    "vlm_called": vlm_called,
                    "vlm_uncertainty": float(last_vlm_uncertainty),
                    "vlm_confidence": float(last_vlm_confidence),
                    "vlm_parsed_ok": last_vlm_parsed_ok,
                    "reward": float(reward),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                }
                if vlm_called and result is not None:
                    row["vlm_raw"] = last_vlm_raw
                    row["vlm_reason"] = result.reason
                    row["vlm_intent"] = result.intent.tolist()
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
    mean_ret = float(np.mean(ep_returns))
    succ_rate = float(np.mean(ep_successes))
    hit_rate = float(np.mean(ep_hazard_hits))
    timeout_rate = 1.0 - succ_rate - hit_rate
    mean_dist = float(np.nanmean(ep_final_dist))

    print(f"\n{'='*60}")
    print(f"Pilot: {args.pilot_mode}  Diffusion: {'ON (proximity)' if use_diffusion else 'OFF'}  "
          f"Episodes: {n}")
    print(f"  fwd_ratio_far={args.fwd_ratio_far}  fwd_ratio_near={args.fwd_ratio_near}  "
          f"danger_dist={args.danger_dist}")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n}  "
          f"({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    print(f"{'='*60}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "use_diffusion": use_diffusion,
            "fwd_ratio_far": args.fwd_ratio_far,
            "fwd_ratio_near": args.fwd_ratio_near,
            "danger_dist": args.danger_dist,
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
