"""
Self-iterating PIVOT VLM controller for PointHazardEnv.

This is the "let the VLM understand momentum itself" version:
  - no diffusion
  - no future-trajectory rollout shown to the VLM
  - no new environment
  - same PIVOT candidate force arrows as shared_autonomy_hazard_pivot.py

The VLM sees the current rendered frame, including the environment's velocity
arrow, and chooses a numbered force arrow.  After each real environment step,
the next prompt receives factual feedback: how speed, goal distance, and hazard
clearance changed.  Optional same-frame self-refinement lets the VLM reconsider
its own selection before execution, but without showing a computed rollout.

Examples:
    # Make one annotated PIVOT prompt image without loading a VLM.
    python shared_autonomy_hazard_self_iterating_pivot.py --smoke_test

    # Run VLM control.
    python shared_autonomy_hazard_self_iterating_pivot.py \
        --model_path Qwen/Qwen3-VL-32B-Instruct \
        --episodes 10 --vlm_every 1 --self_refine_rounds 1

    # Run a non-VLM heuristic sanity check.
    python shared_autonomy_hazard_self_iterating_pivot.py \
        --pilot_mode local --episodes 2 --no-save_log --no-save_gif
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from env_pointhazard import PointHazardConfig, make_env
from hazard_renderer import HazardRenderer
from pivot_vlm import _call_vlm_select, annotate_candidates, generate_candidates


# ---------------------------------------------------------------------------
# Local VLM loading
# ---------------------------------------------------------------------------

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
    dtype = _torch_dtype_from_arg(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype if dtype != "auto" else "auto",
    }
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model.eval()
    print("Local VLM loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

@dataclass
class SelfIteratingPivotResult:
    action: np.ndarray
    choice_idx: int
    parsed_ok: bool
    confidence: float
    uncertainty: float
    mean_logprob: float
    reason: str | None
    raw_text: str
    iterations: list[dict[str, Any]]
    fallback_used: bool
    fallback_reason: str
    annotated_image: Image.Image
    n_candidates: int


def _obs_parts(obs: np.ndarray, cfg: PointHazardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    pos = obs[0:2].astype(np.float32)
    vel = obs[2:4].astype(np.float32)
    goal = obs[4:6].astype(np.float32)
    hazards = obs[6:6 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
    return pos, vel, goal, hazards


def _clearance_to_hazards(pos: np.ndarray, hazards: np.ndarray, cfg: PointHazardConfig) -> float:
    """Current body-to-hazard clearance.  This is measured, not predicted."""
    pos = np.asarray(pos, dtype=np.float32).reshape(2)
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)
    clearances = center_d - hazards[:, 2] - float(cfg.agent_radius)
    return float(np.min(clearances))


def _nearest_hazard_vector(pos: np.ndarray, hazards: np.ndarray) -> np.ndarray:
    """Vector from agent to the nearest hazard center, for prompt feedback."""
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return np.zeros(2, dtype=np.float32)
    idx = int(np.argmin(np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)))
    return (hazards[idx, :2] - pos).astype(np.float32)


def _format_vec(v: np.ndarray) -> str:
    return f"[{float(v[0]):+.2f}, {float(v[1]):+.2f}]"


def _candidate_text(candidates: list[np.ndarray]) -> str:
    lines = []
    for i, action in enumerate(candidates, start=1):
        mag = float(np.linalg.norm(action))
        angle = math.degrees(math.atan2(float(action[1]), float(action[0])))
        lines.append(
            f"{i}: force={_format_vec(action)}, magnitude={mag:.2f}, angle_deg={angle:+.0f}"
        )
    return "\n".join(lines)


def _format_step_feedback(
    *,
    choice_idx: int,
    action: np.ndarray,
    prev_speed: float,
    next_speed: float,
    goal_progress: float,
    clearance_delta: float,
    next_clearance: float,
    hazard_hit: bool,
    goal_success: bool,
) -> str:
    status = "goal_success" if goal_success else ("hazard_hit" if hazard_hit else "running")
    choice = choice_idx + 1 if choice_idx >= 0 else 0
    return (
        f"Last chosen arrow={choice}, force={_format_vec(action)}. "
        f"Real one-step result: speed {prev_speed:.3f}->{next_speed:.3f}, "
        f"goal_progress={goal_progress:+.3f}, "
        f"hazard_clearance_change={clearance_delta:+.3f}, "
        f"current_hazard_clearance={next_clearance:.3f}, status={status}."
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_self_iterating_prompt(
    *,
    n_candidates: int,
    candidates: list[np.ndarray],
    obs: np.ndarray,
    cfg: PointHazardConfig,
    previous_feedback: str | None,
    self_reflection: str | None,
) -> str:
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    speed = float(np.linalg.norm(vel))
    goal_vec = goal - pos
    dist_to_goal = float(np.linalg.norm(goal_vec))
    nearest_hazard_vec = _nearest_hazard_vector(pos, hazards)
    clearance = _clearance_to_hazards(pos, hazards, cfg)

    feedback_block = ""
    if previous_feedback:
        feedback_block = f"""
Feedback from the previous real step:
{previous_feedback}
Use this as evidence about momentum. If an action made speed or hazard
clearance worse, adjust your next choice.
"""

    reflection_block = ""
    if self_reflection:
        reflection_block = f"""
Self-refinement:
{self_reflection}
Reconsider the same image and state. You may keep the same arrow if it is
still best, or switch to a better arrow.
"""

    return f"""You control a point-mass agent in a 2D hazard arena.

Visual elements:
- Gray bordered square = arena boundary
- Red filled circles = hazards. Touching one ends the episode.
- Blue dot = you, the controlled agent
- Green circle with "G" = goal
- Blue trail = recent path already traveled
- Blue velocity arrow from the agent = current momentum
- Orange numbered arrows (1-{n_candidates}) = candidate force choices for this step

Important: the orange arrows are forces, not predicted future paths. No future
trajectory is provided. You must infer momentum yourself from the blue velocity
arrow, the recent trail, and the numeric feedback. A force changes velocity
gradually; if the agent is already sliding toward a hazard or overshooting the
goal, choose a counter-momentum force even if it does not point directly at the
goal.

Current measured state:
- position={_format_vec(pos)}
- velocity={_format_vec(vel)}, speed={speed:.3f}
- vector_to_goal={_format_vec(goal_vec)}, dist_to_goal={dist_to_goal:.3f}
- vector_to_nearest_hazard_center={_format_vec(nearest_hazard_vec)}
- current_body_to_hazard_clearance={clearance:.3f}

Candidate forces:
{_candidate_text(candidates)}
{feedback_block}
{reflection_block}
Choose the best arrow to reach the goal while avoiding hazards. Think about
how the current velocity will continue unless corrected.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


def _iteration_to_dict(round_idx: int, raw_result) -> dict[str, Any]:
    return {
        "round": int(round_idx),
        "choice": int(raw_result.choice_idx + 1) if raw_result.choice_idx is not None else 0,
        "parsed_ok": bool(raw_result.parsed_ok),
        "reason": raw_result.reason,
        "raw_text": raw_result.raw_text,
        "confidence": float(raw_result.confidence),
        "uncertainty": float(raw_result.uncertainty),
        "mean_logprob": float(raw_result.mean_logprob),
    }


def _reflection_for_choice(raw_result, candidates: list[np.ndarray], obs: np.ndarray, cfg: PointHazardConfig) -> str:
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    speed = float(np.linalg.norm(vel))
    goal_vec = goal - pos
    hazard_vec = _nearest_hazard_vector(pos, hazards)

    if raw_result.choice_idx is None or not raw_result.parsed_ok:
        return "Your previous answer was not valid JSON. Return one valid arrow number."

    idx = int(raw_result.choice_idx)
    if idx < 0 or idx >= len(candidates):
        return "Your previous answer chose a number outside the candidate range."

    action = candidates[idx]
    vel_dot_goal = float(np.dot(vel, goal_vec))
    vel_dot_hazard = float(np.dot(vel, hazard_vec))
    action_dot_vel = float(np.dot(action, vel))
    return (
        f"You selected arrow {idx + 1} with force={_format_vec(action)}. "
        f"Current speed={speed:.3f}. "
        f"velocity_dot_goal_vector={vel_dot_goal:+.3f}; "
        f"velocity_dot_nearest_hazard_vector={vel_dot_hazard:+.3f}; "
        f"selected_force_dot_velocity={action_dot_vel:+.3f}. "
        "If velocity is already carrying you toward danger or past the goal, "
        "a force opposite the current velocity may be better."
    )


@torch.inference_mode()
def self_iterating_choose_action(
    model,
    processor,
    base_image: Image.Image,
    renderer: HazardRenderer,
    obs: np.ndarray,
    cfg: PointHazardConfig,
    *,
    n_directions: int,
    n_magnitudes: int,
    arrow_length_world: float,
    previous_feedback: str | None,
    held_action: np.ndarray,
    held_choice_idx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
    vlm_retries: int,
    self_refine_rounds: int,
) -> SelfIteratingPivotResult:
    candidates = generate_candidates(
        n_directions=int(n_directions),
        n_magnitudes=int(n_magnitudes),
    )
    annotated = annotate_candidates(
        base_image,
        candidates,
        renderer,
        agent_world_xy=np.asarray(obs[:2], dtype=np.float32),
        arrow_length_world=arrow_length_world,
    )

    iterations: list[dict[str, Any]] = []
    latest_raw = None
    latest_parsed = None
    reflection: str | None = None
    n_rounds = max(1, int(self_refine_rounds) + 1)

    for round_idx in range(n_rounds):
        prompt = build_self_iterating_prompt(
            n_candidates=len(candidates),
            candidates=candidates,
            obs=obs,
            cfg=cfg,
            previous_feedback=previous_feedback,
            self_reflection=reflection,
        )

        raw_result = None
        for attempt in range(max(0, int(vlm_retries)) + 1):
            prompt_text = prompt
            if attempt > 0:
                prompt_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"
            raw_result = _call_vlm_select(
                model,
                processor,
                annotated,
                prompt_text,
                n_candidates=len(candidates),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            if raw_result.parsed_ok:
                break

        assert raw_result is not None
        latest_raw = raw_result
        if raw_result.parsed_ok:
            latest_parsed = raw_result
        iterations.append(_iteration_to_dict(round_idx, raw_result))
        reflection = _reflection_for_choice(raw_result, candidates, obs, cfg)

    final_raw = latest_parsed if latest_parsed is not None else latest_raw
    assert final_raw is not None

    fallback_used = False
    fallback_reason = ""
    if final_raw.parsed_ok and final_raw.choice_idx is not None:
        choice_idx = int(final_raw.choice_idx)
        action = candidates[choice_idx].copy()
    elif held_choice_idx >= 0:
        fallback_used = True
        fallback_reason = "vlm_parse_failed_hold_previous_choice"
        choice_idx = int(held_choice_idx)
        action = held_action.copy()
    else:
        fallback_used = True
        fallback_reason = "vlm_parse_failed_zero_action"
        choice_idx = -1
        action = np.zeros(2, dtype=np.float32)

    return SelfIteratingPivotResult(
        action=np.clip(action, -1.0, 1.0).astype(np.float32),
        choice_idx=choice_idx,
        parsed_ok=bool(final_raw.parsed_ok),
        confidence=float(final_raw.confidence),
        uncertainty=float(final_raw.uncertainty),
        mean_logprob=float(final_raw.mean_logprob),
        reason=final_raw.reason,
        raw_text=final_raw.raw_text,
        iterations=iterations,
        fallback_used=bool(fallback_used),
        fallback_reason=fallback_reason,
        annotated_image=annotated,
        n_candidates=len(candidates),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(args: argparse.Namespace) -> None:
    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)

    obs, _info = env.reset(seed=args.seed)
    # Give the smoke image nonzero velocity, so the momentum cue is visible.
    if args.smoke_velocity != 0:
        env.vel = np.array([float(args.smoke_velocity), 0.5 * float(args.smoke_velocity)], dtype=np.float32)
        obs = env._build_obs()

    frame = env.render()
    if frame is None:
        frame = np.zeros((cfg.render_size, cfg.render_size, 3), dtype=np.uint8)
    image = Image.fromarray(frame)
    candidates = generate_candidates(args.pivot_n_dirs, args.pivot_n_mags)
    annotated = annotate_candidates(
        image,
        candidates,
        renderer,
        agent_world_xy=obs[:2],
        arrow_length_world=args.pivot_arrow_len,
    )

    out_dir = os.path.dirname(args.smoke_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    annotated.save(args.smoke_out)
    env.close()

    _pos, vel, _goal, hazards = _obs_parts(obs, cfg)
    print(f"Smoke image saved: {args.smoke_out}")
    print(f"  candidates: {len(candidates)}")
    print(f"  speed shown: {float(np.linalg.norm(vel)):.3f}")
    print(f"  current clearance: {_clearance_to_hazards(obs[:2], hazards, cfg):.3f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _local_momentum_heuristic(obs: np.ndarray, cfg: PointHazardConfig) -> np.ndarray:
    """Small non-VLM sanity check policy.  The VLM path does not use this."""
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    to_goal = goal - pos
    goal_norm = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / goal_norm if goal_norm > 1e-6 else np.zeros(2, dtype=np.float32)

    clearance = _clearance_to_hazards(pos, hazards, cfg)
    nearest = _nearest_hazard_vector(pos, hazards)
    haz_norm = float(np.linalg.norm(nearest))
    away = -nearest / haz_norm if haz_norm > 1e-6 else np.zeros(2, dtype=np.float32)

    damping = -0.25 * vel
    hazard_push = away * max(0.0, 0.8 - clearance)
    action = goal_dir + damping + hazard_push
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _save_prompt_image(result: SelfIteratingPivotResult | None, run_dir: str, ep: int, t: int) -> str:
    if result is None:
        return ""
    img_dir = os.path.join(run_dir, "prompt_images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"ep{ep:03d}_t{t:03d}.png")
    result.annotated_image.save(path)
    return path


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

    print("Self-iterating PIVOT, no diffusion, no future rollout")
    print(f"  candidates: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags")
    print(f"  vlm_every={args.vlm_every}, self_refine_rounds={args.self_refine_rounds}")
    if args.pilot_mode == "vlm":
        print(f"  VLM: {args.model_path}")

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
    ep_parse_ok: list[int] = []
    ep_parse_total: list[int] = []

    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        _pos0, _vel0, goal, _haz0 = _obs_parts(obs, cfg)

        held_action = np.zeros(2, dtype=np.float32)
        held_choice_idx = -1
        previous_feedback = ""
        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []
        log_rows: list[dict[str, Any]] = []
        ep_min_clearance = float("inf")
        vlm_calls = 0
        parse_ok_count = 0
        parse_total = 0

        for t in range(args.max_steps):
            frame = env.render()
            if frame is None:
                frame = np.zeros((cfg.render_size, cfg.render_size, 3), dtype=np.uint8)
            image = Image.fromarray(frame)
            if args.save_gif and t % args.gif_every == 0:
                frames.append(image.copy())

            pos, vel, _goal, hazards = _obs_parts(obs, cfg)
            current_clearance = _clearance_to_hazards(pos, hazards, cfg)
            prev_speed = float(np.linalg.norm(vel))
            prev_dist = float(np.linalg.norm(goal - pos))
            ep_min_clearance = min(ep_min_clearance, current_clearance)

            vlm_called = False
            result: SelfIteratingPivotResult | None = None
            prompt_image_path = ""

            if args.pilot_mode == "vlm":
                if t % max(1, args.vlm_every) == 0:
                    assert vlm_model is not None and vlm_processor is not None
                    vlm_called = True
                    result = self_iterating_choose_action(
                        vlm_model,
                        vlm_processor,
                        image,
                        renderer,
                        obs,
                        cfg,
                        n_directions=args.pivot_n_dirs,
                        n_magnitudes=args.pivot_n_mags,
                        arrow_length_world=args.pivot_arrow_len,
                        previous_feedback=previous_feedback,
                        held_action=held_action,
                        held_choice_idx=held_choice_idx,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        enable_thinking=args.enable_thinking,
                        vlm_retries=args.vlm_retries,
                        self_refine_rounds=args.self_refine_rounds,
                    )
                    held_action = result.action.copy()
                    held_choice_idx = result.choice_idx
                    vlm_calls += 1
                    parse_total += 1
                    parse_ok_count += int(result.parsed_ok)
                    if args.save_prompt_images and args.save_log:
                        prompt_image_path = _save_prompt_image(result, run_dir, ep, t)

                    if args.debug_print:
                        print(f"  [ep {ep:03d} t={t:03d}] choice={held_choice_idx + 1 if held_choice_idx >= 0 else 0} "
                              f"action={_format_vec(held_action)} parsed={result.parsed_ok} "
                              f"fallback={result.fallback_reason or 'none'}")

            elif args.pilot_mode == "local":
                held_action = _local_momentum_heuristic(obs, cfg)
                held_choice_idx = -1

            elif args.pilot_mode == "goal_error":
                diff = obs[4:6] - obs[:2]
                norm = float(np.linalg.norm(diff))
                held_action = (diff / norm).astype(np.float32) if norm > 1e-6 else np.zeros(2, dtype=np.float32)
                held_action = np.clip(held_action, -1.0, 1.0)
                held_choice_idx = -1

            elif args.pilot_mode == "random":
                held_action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                held_choice_idx = -1

            else:
                raise ValueError(f"Unknown pilot_mode: {args.pilot_mode}")

            exec_action = np.clip(held_action, -1.0, 1.0).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            next_pos, next_vel, _next_goal, next_hazards = _obs_parts(obs, cfg)
            next_clearance = _clearance_to_hazards(next_pos, next_hazards, cfg)
            next_speed = float(np.linalg.norm(next_vel))
            next_dist = float(np.linalg.norm(goal - next_pos))
            goal_progress = prev_dist - next_dist
            clearance_delta = next_clearance - current_clearance
            ep_min_clearance = min(ep_min_clearance, next_clearance)
            ep_return += float(reward)
            last_info = dict(info or {})

            previous_feedback = _format_step_feedback(
                choice_idx=held_choice_idx,
                action=exec_action,
                prev_speed=prev_speed,
                next_speed=next_speed,
                goal_progress=goal_progress,
                clearance_delta=clearance_delta,
                next_clearance=next_clearance,
                hazard_hit=bool(last_info.get("hazard_hit", False)),
                goal_success=bool(last_info.get("goal_success", False)),
            )

            if args.save_log:
                row = {
                    "t": t,
                    "pilot_mode": args.pilot_mode,
                    "agent_pos": pos.tolist(),
                    "agent_vel": vel.tolist(),
                    "goal": goal.tolist(),
                    "current_clearance": float(current_clearance),
                    "next_clearance": float(next_clearance),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    "prev_speed": float(prev_speed),
                    "next_speed": float(next_speed),
                    "goal_progress": float(goal_progress),
                    "clearance_delta": float(clearance_delta),
                    "vlm_called": bool(vlm_called),
                    "choice": int(held_choice_idx + 1) if held_choice_idx >= 0 else 0,
                    "exec": exec_action.tolist(),
                    "reward": float(reward),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                    "feedback_for_next_prompt": previous_feedback,
                    "prompt_image": prompt_image_path,
                }
                if result is not None:
                    row.update({
                        "vlm_parsed_ok": bool(result.parsed_ok),
                        "vlm_confidence": float(result.confidence),
                        "vlm_uncertainty": float(result.uncertainty),
                        "vlm_logprob": float(result.mean_logprob),
                        "vlm_reason": result.reason,
                        "vlm_raw": result.raw_text,
                        "vlm_iterations": result.iterations,
                        "fallback_used": bool(result.fallback_used),
                        "fallback_reason": result.fallback_reason,
                        "pivot_n_candidates": int(result.n_candidates),
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
        ep_parse_ok.append(parse_ok_count)
        ep_parse_total.append(parse_total)

        flag = "OK " if success else ("HIT" if hazard_hit else "---")
        print(f"[ep {ep:03d}] [{flag}] return={ep_return:7.1f} steps={t + 1:3d} "
              f"dist={dist:.2f} min_clear={ep_min_clearance:.2f} "
              f"vlm_calls={vlm_calls} reason={term_reason}")

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
                    "self_iterating_pivot": True,
                    "future_rollout_prompt": False,
                    "use_diffusion": False,
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
                    "parse_ok": parse_ok_count,
                    "parse_total": parse_total,
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
    parse_total_sum = int(np.sum(ep_parse_total))
    parse_success_rate = float(np.sum(ep_parse_ok) / parse_total_sum) if parse_total_sum else float("nan")

    print(f"\n{'=' * 60}")
    print(f"Self-iterating PIVOT  Pilot: {args.pilot_mode}  Diffusion: OFF  Future rollout prompt: OFF")
    print(f"  Episodes: {n}")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n}  ({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    print(f"  Mean min clearance: {mean_min_clearance:.2f}")
    print(f"  VLM calls/ep: {mean_vlm_calls:.1f}")
    print(f"  Parse success: {parse_success_rate:.1%}" if parse_total_sum else "  Parse success: n/a")
    print(f"{'=' * 60}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "model": args.model_path,
            "self_iterating_pivot": True,
            "future_rollout_prompt": False,
            "use_diffusion": False,
            "episodes": n,
            "success_rate": succ_rate,
            "hazard_hit_rate": hit_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_ret,
            "mean_final_dist": mean_dist,
            "mean_min_clearance": mean_min_clearance,
            "mean_vlm_calls": mean_vlm_calls,
            "parse_success_rate": parse_success_rate,
            "args": vars(args),
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-iterating PIVOT on PointHazardEnv without diffusion or future rollouts")

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot_mode", type=str, default="vlm",
                        choices=["vlm", "local", "goal_error", "random"])

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
    parser.add_argument("--vlm_every", type=int, default=1)
    parser.add_argument("--vlm_retries", type=int, default=1)
    parser.add_argument("--self_refine_rounds", type=int, default=1,
                        help="Extra same-image reconsideration rounds before executing.")

    parser.add_argument("--pivot_n_dirs", type=int, default=8)
    parser.add_argument("--pivot_n_mags", type=int, default=2)
    parser.add_argument("--pivot_arrow_len", type=float, default=0.6)

    parser.add_argument("--save_log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_dir", type=str, default="hazard/logs_hazard_self_iterating_pivot")
    parser.add_argument("--save_prompt_images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_hazard_self_iterating_pivot")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_out", type=str, default="/tmp/self_iterating_pivot_smoke.png")
    parser.add_argument("--smoke_velocity", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
