"""
VLM-guided control for PointHazardEnv: safety-filtered motion-primitive PIVOT.

This is the local Qwen variant of shared_autonomy_hazard_primitive_pivot_openrouter.py:
the VLM chooses among numbered short trajectory primitives, while a local
PD controller executes the selected waypoint.  Safety is handled by a
dynamics rollout filter, not by diffusion.

Examples:
    # Smoke test without loading Qwen.
    python shared_autonomy_hazard_primitive_pivot.py --smoke_test

    # Dynamics rollout sanity check without loading Qwen.
    python shared_autonomy_hazard_primitive_pivot.py --dynamics_test_resets 100

    # Evaluate primitive PIVOT with local Qwen.
    python shared_autonomy_hazard_primitive_pivot.py \
        --model_path Qwen/Qwen3-VL-32B-Instruct --episodes 10 --vlm_every 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from env_pointhazard import PointHazardConfig, make_env
from hazard_renderer import HazardRenderer
from pivot_primitive import (
    MotionPrimitive,
    annotate_primitives,
    build_primitive_prompt,
    choose_display_primitives,
    choose_local_primitive,
    clearance_to_hazards,
    generate_waypoint_primitives,
    parse_absolute_obs,
    primitive_pd_action,
)
from pivot_vlm import _call_vlm_select


def _torch_dtype_from_arg(name: str):
    name = str(name).lower()
    if name in ("auto", "none"):
        return "auto"
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


def _load_local_vlm(args: argparse.Namespace):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading local VLM: {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    dtype = _torch_dtype_from_arg(args.torch_dtype)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model.eval()
    print("Local VLM loaded.")
    return model, processor


def _obs_clearance(obs: np.ndarray, cfg: PointHazardConfig) -> float:
    pos, _vel, _goal, hazards = parse_absolute_obs(obs, cfg)
    return clearance_to_hazards(pos, hazards, cfg)


def _build_primitives(
    obs: np.ndarray,
    cfg: PointHazardConfig,
    args: argparse.Namespace,
) -> tuple[list[MotionPrimitive], list[MotionPrimitive], int]:
    all_primitives = generate_waypoint_primitives(
        obs,
        cfg,
        n_dirs=args.primitive_n_dirs,
        radius=args.primitive_radius,
        horizon=args.primitive_horizon,
        safety_margin=args.safety_margin,
        kp=args.pd_kp,
        desired_speed=args.desired_speed,
        slow_radius=args.slow_radius,
    )
    display_primitives = choose_display_primitives(
        all_primitives,
        max_display=args.max_display_primitives,
    )
    num_safe = sum(1 for p in all_primitives if p.safe)
    return all_primitives, display_primitives, num_safe


def _choose_primitive(
    *,
    vlm_model,
    vlm_processor,
    base_image: Image.Image,
    renderer: HazardRenderer,
    obs: np.ndarray,
    cfg: PointHazardConfig,
    args: argparse.Namespace,
    allow_vlm: bool,
) -> tuple[MotionPrimitive, dict[str, Any], Image.Image]:
    """Choose a primitive using local Qwen when allowed, otherwise local fallback."""
    all_primitives, display_primitives, num_safe = _build_primitives(obs, cfg, args)
    annotated = annotate_primitives(base_image, display_primitives, renderer)

    meta: dict[str, Any] = {
        "vlm_called": False,
        "vlm_raw": "",
        "vlm_reason": None,
        "vlm_parsed_ok": False,
        "vlm_confidence": float("nan"),
        "vlm_logprob": float("nan"),
        "fallback_used": False,
        "fallback_reason": "",
        "num_safe_candidates": int(num_safe),
        "num_display_candidates": int(len(display_primitives)),
        "primitive_all_min_clearance": [float(p.min_clearance) for p in all_primitives],
        "primitive_all_safe": [bool(p.safe) for p in all_primitives],
    }

    if not display_primitives:
        raise RuntimeError("No motion primitives generated.")

    should_call_vlm = bool(allow_vlm)
    if num_safe == 0 and not args.call_vlm_when_no_safe:
        should_call_vlm = False

    selected: MotionPrimitive | None = None

    if should_call_vlm:
        if vlm_model is None or vlm_processor is None:
            raise RuntimeError("VLM was requested but local model/processor is not loaded.")

        meta["vlm_called"] = True
        prompt = build_primitive_prompt(len(display_primitives))
        raw_result = None

        for attempt in range(args.vlm_retries + 1):
            prompt_text = prompt
            if attempt > 0:
                prompt_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"

            raw_result = _call_vlm_select(
                vlm_model,
                vlm_processor,
                annotated,
                prompt_text,
                n_candidates=len(display_primitives),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                enable_thinking=args.enable_thinking,
            )
            if raw_result.parsed_ok:
                break

        assert raw_result is not None
        meta["vlm_raw"] = raw_result.raw_text
        meta["vlm_reason"] = raw_result.reason
        meta["vlm_parsed_ok"] = bool(raw_result.parsed_ok)
        meta["vlm_confidence"] = float(raw_result.confidence)
        meta["vlm_logprob"] = float(raw_result.mean_logprob)

        choice_idx = raw_result.choice_idx
        if raw_result.parsed_ok and choice_idx is not None and 0 <= choice_idx < len(display_primitives):
            selected = display_primitives[choice_idx]
            if not selected.safe:
                meta["fallback_used"] = True
                meta["fallback_reason"] = "vlm_chose_unsafe_display_candidate"
                selected = choose_local_primitive(display_primitives)
        else:
            meta["fallback_used"] = True
            meta["fallback_reason"] = "vlm_parse_failed"

    if selected is None:
        if not meta["fallback_reason"]:
            if num_safe == 0:
                meta["fallback_reason"] = "no_safe_candidates"
            elif not allow_vlm:
                meta["fallback_reason"] = "local_replan"
            else:
                meta["fallback_reason"] = "local_fallback"
        meta["fallback_used"] = True
        selected = choose_local_primitive(display_primitives)

    return selected, meta, annotated


def _primitive_to_log(prefix: str, primitive: MotionPrimitive) -> dict[str, Any]:
    return {
        f"{prefix}_choice": int(primitive.choice_id),
        f"{prefix}_source_id": int(primitive.source_id),
        f"{prefix}_target": primitive.target_xy.tolist(),
        f"{prefix}_first_action": primitive.first_action.tolist(),
        f"{prefix}_min_clearance": float(primitive.min_clearance),
        f"{prefix}_safe": bool(primitive.safe),
        f"{prefix}_goal_progress": float(primitive.goal_progress),
        f"{prefix}_final_dist_to_goal": float(primitive.final_dist_to_goal),
        f"{prefix}_score": float(primitive.score),
    }


def run_smoke_test(args: argparse.Namespace) -> None:
    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)

    obs, _info = env.reset(seed=args.seed)
    frame = env.render()
    if frame is None:
        frame = np.zeros((cfg.render_size, cfg.render_size, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    all_primitives, display_primitives, num_safe = _build_primitives(obs, cfg, args)
    annotated = annotate_primitives(image, display_primitives, renderer)

    out_dir = os.path.dirname(args.smoke_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    annotated.save(args.smoke_out)
    env.close()

    print(f"Smoke image saved: {args.smoke_out}")
    print(f"  total primitives: {len(all_primitives)}")
    print(f"  displayed: {len(display_primitives)}")
    print(f"  safe: {num_safe}")
    print(f"  min clearance: {min(p.min_clearance for p in all_primitives):.3f}")


def run_dynamics_test(args: argparse.Namespace) -> int:
    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)

    total_safe = 0
    total_primitives = 0
    all_unsafe_resets = 0
    violations = 0
    min_clearance_seen = float("inf")

    for ep in range(int(args.dynamics_test_resets)):
        obs, _info = env.reset(seed=args.seed + ep)
        all_primitives, _display_primitives, num_safe = _build_primitives(obs, cfg, args)
        total_primitives += len(all_primitives)
        total_safe += num_safe
        if num_safe == 0:
            all_unsafe_resets += 1
        for primitive in all_primitives:
            min_clearance_seen = min(min_clearance_seen, primitive.min_clearance)
            if primitive.safe and primitive.min_clearance < args.safety_margin - 1e-6:
                violations += 1

    env.close()
    print("Dynamics test complete")
    print(f"  resets: {args.dynamics_test_resets}")
    print(f"  primitives: {total_primitives}")
    print(f"  safe primitives: {total_safe}")
    print(f"  all-unsafe resets: {all_unsafe_resets}")
    print(f"  min clearance seen: {min_clearance_seen:.3f}")
    print(f"  violations: {violations}")
    return 1 if violations else 0


def evaluate(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vlm_model = None
    vlm_processor = None
    if args.pilot_mode == "vlm":
        vlm_model, vlm_processor = _load_local_vlm(args)

    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=0, with_renderer=False)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)

    print(f"Primitive PIVOT: {args.primitive_n_dirs} dirs, radius={args.primitive_radius}, "
          f"horizon={args.primitive_horizon}, vlm_every={args.vlm_every}")
    if args.pilot_mode == "vlm":
        print(f"Local VLM: {args.model_path}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ""
    if args.save_log:
        run_dir = os.path.join(args.log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
    if args.save_gif:
        os.makedirs(args.gif_dir, exist_ok=True)

    ep_returns: list[float] = []
    ep_successes: list[int] = []
    ep_hazard_hits: list[int] = []
    ep_final_dist: list[float] = []
    ep_min_clearances: list[float] = []
    ep_vlm_calls: list[int] = []
    ep_fallback_counts: list[int] = []

    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        _pos, _vel, goal, _hazards = parse_absolute_obs(obs, cfg)

        selected: MotionPrimitive | None = None
        primitive_age = 10**9
        last_choice_meta: dict[str, Any] = {}
        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []
        log_rows: list[dict[str, Any]] = []
        ep_min_clearance = float("inf")
        vlm_calls = 0
        fallback_count = 0

        for t in range(args.max_steps):
            frame = env.render()
            if frame is None:
                frame = np.zeros((cfg.render_size, cfg.render_size, 3), dtype=np.uint8)
            image = Image.fromarray(frame)
            if args.save_gif and t % args.gif_every == 0:
                frames.append(image.copy())

            current_clearance = _obs_clearance(obs, cfg)
            ep_min_clearance = min(ep_min_clearance, current_clearance)
            pos, vel, _goal, _hazards = parse_absolute_obs(obs, cfg)

            dist_to_target = float("inf")
            if selected is not None:
                dist_to_target = float(np.linalg.norm(selected.target_xy - pos))

            scheduled_vlm = (t % max(1, args.vlm_every) == 0)
            expired = (
                selected is None
                or primitive_age >= args.primitive_horizon
                or dist_to_target < args.target_reached_dist
            )
            near_unsafe = current_clearance < args.replan_clearance
            need_replan = scheduled_vlm or expired or near_unsafe

            if need_replan:
                allow_vlm = args.pilot_mode == "vlm" and scheduled_vlm
                selected, last_choice_meta, _annotated = _choose_primitive(
                    vlm_model=vlm_model,
                    vlm_processor=vlm_processor,
                    base_image=image,
                    renderer=renderer,
                    obs=obs,
                    cfg=cfg,
                    args=args,
                    allow_vlm=allow_vlm,
                )
                primitive_age = 0
                if last_choice_meta.get("vlm_called"):
                    vlm_calls += 1
                if last_choice_meta.get("fallback_used"):
                    fallback_count += 1

                if args.debug_print:
                    why = last_choice_meta.get("fallback_reason") or "vlm"
                    print(f"  [ep {ep:03d} t={t:03d}] choice={selected.choice_id} "
                          f"src={selected.source_id} safe={selected.safe} "
                          f"clear={selected.min_clearance:.2f} progress={selected.goal_progress:.2f} "
                          f"mode={why}")

            assert selected is not None
            exec_action = primitive_pd_action(
                pos,
                vel,
                selected.target_xy,
                cfg,
                kp=args.pd_kp,
                desired_speed=args.desired_speed,
                slow_radius=args.slow_radius,
            )
            exec_action = np.clip(exec_action, -1.0, 1.0).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            next_clearance = _obs_clearance(obs, cfg)
            ep_min_clearance = min(ep_min_clearance, next_clearance)
            ep_return += float(reward)
            last_info = dict(info or {})
            primitive_age += 1

            if args.save_log:
                row = {
                    "t": t,
                    "pilot_mode": args.pilot_mode,
                    "agent_pos": pos.tolist(),
                    "goal": goal.tolist(),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    "current_clearance": float(current_clearance),
                    "next_clearance": float(next_clearance),
                    "exec": exec_action.tolist(),
                    "primitive_age": int(primitive_age),
                    "scheduled_vlm": bool(scheduled_vlm),
                    "expired": bool(expired),
                    "near_unsafe": bool(near_unsafe),
                    "reward": float(reward),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                }
                row.update(_primitive_to_log("primitive", selected))
                row.update({
                    "vlm_called": bool(last_choice_meta.get("vlm_called", False)),
                    "vlm_parsed_ok": bool(last_choice_meta.get("vlm_parsed_ok", False)),
                    "vlm_confidence": float(last_choice_meta.get("vlm_confidence", float("nan"))),
                    "vlm_logprob": float(last_choice_meta.get("vlm_logprob", float("nan"))),
                    "vlm_raw": last_choice_meta.get("vlm_raw", ""),
                    "vlm_reason": last_choice_meta.get("vlm_reason"),
                    "fallback_used": bool(last_choice_meta.get("fallback_used", False)),
                    "fallback_reason": last_choice_meta.get("fallback_reason", ""),
                    "num_safe_candidates": int(last_choice_meta.get("num_safe_candidates", 0)),
                    "num_display_candidates": int(last_choice_meta.get("num_display_candidates", 0)),
                })
                log_rows.append(row)

            if terminated or truncated:
                break

        success = bool(last_info.get("goal_success", False))
        hazard_hit = bool(last_info.get("hazard_hit", False))
        dist = float(last_info.get("dist_to_goal", float("nan")))
        term_reason = last_info.get("termination_reason", "timeout")

        ep_returns.append(ep_return)
        ep_successes.append(int(success))
        ep_hazard_hits.append(int(hazard_hit))
        ep_final_dist.append(dist)
        ep_min_clearances.append(ep_min_clearance)
        ep_vlm_calls.append(vlm_calls)
        ep_fallback_counts.append(fallback_count)

        flag = "OK " if success else ("HIT" if hazard_hit else "---")
        print(f"[ep {ep:03d}] [{flag}] return={ep_return:7.1f} steps={t+1:3d} "
              f"dist={dist:.2f} min_clear={ep_min_clearance:.2f} "
              f"vlm_calls={vlm_calls} fallback={fallback_count} reason={term_reason}")

        if args.save_log:
            log_path = os.path.join(run_dir, f"ep{ep:03d}.jsonl")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "type": "episode_meta",
                    "episode": ep,
                    "pilot_mode": args.pilot_mode,
                    "model": args.model_path,
                    "seed": args.seed + ep,
                    "goal": goal.tolist(),
                    "primitive_pivot": True,
                    "local_qwen": True,
                    "args": vars(args),
                }) + "\n")
                for row in log_rows:
                    f.write(json.dumps(row) + "\n")
                f.write(json.dumps({
                    "type": "episode_end",
                    "return": ep_return,
                    "success": success,
                    "hazard_hit": hazard_hit,
                    "dist": dist,
                    "steps": t + 1,
                    "min_clearance": ep_min_clearance,
                    "vlm_calls": vlm_calls,
                    "fallback_count": fallback_count,
                }) + "\n")

        if args.save_gif and frames:
            dur = int(1000 / max(1e-6, args.gif_fps))
            tag = "ok" if success else ("hit" if hazard_hit else "timeout")
            gif_path = os.path.join(args.gif_dir, f"ep{ep:03d}_{tag}.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=dur,
                loop=0,
            )

    env.close()

    n = int(args.episodes)
    succ_rate = float(np.mean(ep_successes))
    hit_rate = float(np.mean(ep_hazard_hits))
    timeout_rate = 1.0 - succ_rate - hit_rate
    mean_ret = float(np.mean(ep_returns))
    mean_dist = float(np.nanmean(ep_final_dist))
    mean_min_clearance = float(np.nanmean(ep_min_clearances))
    mean_vlm_calls = float(np.mean(ep_vlm_calls))
    mean_fallbacks = float(np.mean(ep_fallback_counts))

    print(f"\n{'=' * 60}")
    print(f"Primitive PIVOT  Pilot: {args.pilot_mode}  Episodes: {n}")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n}  ({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    print(f"  Mean min clearance: {mean_min_clearance:.2f}")
    print(f"  VLM calls/ep: {mean_vlm_calls:.1f}")
    print(f"  Fallbacks/ep: {mean_fallbacks:.1f}")
    print(f"{'=' * 60}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "model": args.model_path,
            "primitive_pivot": True,
            "local_qwen": True,
            "episodes": n,
            "success_rate": succ_rate,
            "hazard_hit_rate": hit_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_ret,
            "mean_final_dist": mean_dist,
            "mean_min_clearance": mean_min_clearance,
            "mean_vlm_calls": mean_vlm_calls,
            "mean_fallbacks": mean_fallbacks,
            "args": vars(args),
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safety-filtered motion-primitive PIVOT on PointHazardEnv via local Qwen")

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot_mode", type=str, default="vlm", choices=["vlm", "local"])

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vlm_every", type=int, default=10)
    parser.add_argument("--vlm_retries", type=int, default=1)
    parser.add_argument("--call_vlm_when_no_safe", action="store_true")

    parser.add_argument("--primitive_n_dirs", type=int, default=8)
    parser.add_argument("--primitive_radius", type=float, default=1.2)
    parser.add_argument("--primitive_horizon", type=int, default=12)
    parser.add_argument("--max_display_primitives", type=int, default=8)
    parser.add_argument("--safety_margin", type=float, default=0.15)
    parser.add_argument("--replan_clearance", type=float, default=0.10)
    parser.add_argument("--target_reached_dist", type=float, default=0.35)

    parser.add_argument("--pd_kp", type=float, default=3.0)
    parser.add_argument("--desired_speed", type=float, default=2.4)
    parser.add_argument("--slow_radius", type=float, default=0.9)

    parser.add_argument("--save_log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_dir", type=str, default="hazard/logs_hazard_primitive_pivot_qwen")
    parser.add_argument("--save_gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_hazard_primitive_pivot_qwen")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_out", type=str, default="/tmp/primitive_pivot_qwen_smoke.png")
    parser.add_argument("--dynamics_test_resets", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = 0
    if args.smoke_test:
        run_smoke_test(args)
    if args.dynamics_test_resets > 0:
        exit_code = run_dynamics_test(args)
    if args.smoke_test or args.dynamics_test_resets > 0:
        raise SystemExit(exit_code)
    evaluate(args)


if __name__ == "__main__":
    main()
