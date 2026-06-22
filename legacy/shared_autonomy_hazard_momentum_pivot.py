"""
Momentum-aware rolling PIVOT for PointHazardEnv.

This variant keeps the original PIVOT idea that the VLM chooses a numbered
candidate action, but each candidate is rendered as a short dynamics rollout
from the current position and velocity. The VLM sees the consequence of
momentum instead of only a static force arrow.

The loop is receding-horizon: observe, render rollout candidates, choose,
execute one action, then repeat with feedback from the previous step.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ddpm import ConditionalDDPM, DiffusionConfig, RunningNorm
from hazard.env_pointhazard import PointHazardConfig, PointHazardEnv, make_env
from hazard.hazard_renderer import HazardRenderer
from hazard.pivot_vlm import _call_vlm_select, generate_candidates
from hazard.safe_expert import SafeExpert, SafeExpertConfig


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
# Momentum rollout candidates
# ---------------------------------------------------------------------------

@dataclass
class MomentumCandidate:
    choice_id: int
    source_id: int
    action: np.ndarray
    trajectory_xy: np.ndarray
    final_vel: np.ndarray
    min_clearance: float
    safe: bool
    goal_progress: float
    final_dist_to_goal: float
    score: float
    label: str = ""


@dataclass
class MomentumPivotResult:
    action: np.ndarray
    confidence: float
    uncertainty: float
    choice_idx: int
    reason: str | None
    raw_text: str
    parsed_ok: bool
    mean_logprob: float
    n_candidates: int
    selected_candidate: MomentumCandidate
    all_candidates: list[MomentumCandidate]
    fallback_used: bool
    fallback_reason: str
    annotated_image: Image.Image


def _obs_parts(obs: np.ndarray, cfg: PointHazardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    pos = obs[0:2].astype(np.float32)
    vel = obs[2:4].astype(np.float32)
    goal = obs[4:6].astype(np.float32)
    hazards = obs[6:6 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
    return pos, vel, goal, hazards


def clearance_to_hazards(pos: np.ndarray, hazards: np.ndarray, cfg: PointHazardConfig) -> float:
    """Minimum edge clearance from the agent body to any hazard."""
    pos = np.asarray(pos, dtype=np.float32).reshape(2)
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)
    clearances = center_d - hazards[:, 2] - float(cfg.agent_radius)
    return float(np.min(clearances))


def hazard_edge_distance(pos: np.ndarray, hazards: np.ndarray) -> float:
    """Distance to nearest hazard edge, without subtracting agent radius."""
    pos = np.asarray(pos, dtype=np.float32).reshape(2)
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)
    return float(np.min(center_d - hazards[:, 2]))


def simulate_dynamics_step(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    cfg: PointHazardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One PointHazardEnv dynamics step, matching env_pointhazard.py."""
    pos = np.asarray(pos, dtype=np.float32).copy()
    vel = np.asarray(vel, dtype=np.float32).copy()
    action = np.asarray(action, dtype=np.float32).reshape(2)

    force = float(cfg.force_scale) * np.clip(action, -1.0, 1.0)
    new_vel = vel + (force - float(cfg.drag) * vel) * float(cfg.dt)
    speed = float(np.linalg.norm(new_vel))
    if speed > float(cfg.max_speed):
        new_vel = new_vel * (float(cfg.max_speed) / speed)

    new_pos = pos + new_vel * float(cfg.dt)

    arena_limit = float(cfg.arena_half) - float(cfg.agent_radius)
    for i in range(2):
        if new_pos[i] < -arena_limit:
            new_pos[i] = -arena_limit
            new_vel[i] = -new_vel[i] * 0.5
        elif new_pos[i] > arena_limit:
            new_pos[i] = arena_limit
            new_vel[i] = -new_vel[i] * 0.5

    return new_pos.astype(np.float32), new_vel.astype(np.float32)


def braking_action_from_velocity(vel: np.ndarray, cfg: PointHazardConfig) -> np.ndarray:
    """Action that approximately cancels current velocity in one step."""
    vel = np.asarray(vel, dtype=np.float32).reshape(2)
    if float(np.linalg.norm(vel)) < 1e-6:
        return np.zeros(2, dtype=np.float32)
    gain = (float(cfg.drag) * float(cfg.dt) - 1.0) / (float(cfg.force_scale) * float(cfg.dt))
    return np.clip(gain * vel, -1.0, 1.0).astype(np.float32)


def rollout_action_candidate(
    obs: np.ndarray,
    cfg: PointHazardConfig,
    action: np.ndarray,
    *,
    choice_id: int,
    source_id: int,
    horizon: int,
    safety_margin: float,
    clearance_weight: float,
    label: str = "",
) -> MomentumCandidate:
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

    start_dist = float(np.linalg.norm(goal - pos))
    min_clearance = clearance_to_hazards(pos, hazards, cfg)
    safe = min_clearance >= float(safety_margin)
    trajectory = [pos.copy()]

    p = pos.copy()
    v = vel.copy()
    for _ in range(max(1, int(horizon))):
        p, v = simulate_dynamics_step(p, v, action, cfg)
        trajectory.append(p.copy())
        clear = clearance_to_hazards(p, hazards, cfg)
        min_clearance = min(min_clearance, clear)
        if clear < float(safety_margin):
            safe = False

    final_dist = float(np.linalg.norm(goal - p))
    goal_progress = start_dist - final_dist
    score = goal_progress + float(clearance_weight) * min_clearance
    if not safe:
        score -= 10.0

    return MomentumCandidate(
        choice_id=int(choice_id),
        source_id=int(source_id),
        action=action.astype(np.float32),
        trajectory_xy=np.asarray(trajectory, dtype=np.float32),
        final_vel=v.astype(np.float32),
        min_clearance=float(min_clearance),
        safe=bool(safe),
        goal_progress=float(goal_progress),
        final_dist_to_goal=float(final_dist),
        score=float(score),
        label=label,
    )


def build_momentum_candidates(
    obs: np.ndarray,
    cfg: PointHazardConfig,
    *,
    n_directions: int,
    n_magnitudes: int,
    horizon: int,
    safety_margin: float,
    clearance_weight: float,
    include_brake: bool,
) -> list[MomentumCandidate]:
    pos, vel, _goal, _hazards = _obs_parts(obs, cfg)
    del pos

    actions = generate_candidates(
        n_directions=int(n_directions),
        n_magnitudes=int(n_magnitudes),
    )
    if include_brake:
        actions.append(braking_action_from_velocity(vel, cfg))

    candidates: list[MomentumCandidate] = []
    for i, action in enumerate(actions):
        label = "brake" if include_brake and i == len(actions) - 1 else ""
        candidates.append(rollout_action_candidate(
            obs,
            cfg,
            action,
            choice_id=i + 1,
            source_id=i + 1,
            horizon=horizon,
            safety_margin=safety_margin,
            clearance_weight=clearance_weight,
            label=label,
        ))
    return candidates


def choose_local_candidate(candidates: list[MomentumCandidate]) -> MomentumCandidate:
    if not candidates:
        raise ValueError("No momentum candidates available")
    safe = [c for c in candidates if c.safe]
    if safe:
        return max(safe, key=lambda c: (c.score, c.min_clearance, c.goal_progress))
    return max(candidates, key=lambda c: (c.min_clearance, c.goal_progress))


# ---------------------------------------------------------------------------
# Image annotation and prompt
# ---------------------------------------------------------------------------

_PATH_COLORS = [
    (255, 150, 0),
    (0, 170, 210),
    (160, 90, 220),
    (40, 170, 90),
    (240, 190, 30),
    (230, 100, 160),
    (80, 130, 255),
    (140, 140, 35),
    (190, 90, 40),
    (60, 150, 150),
    (130, 100, 210),
    (70, 120, 60),
]
_UNSAFE_COLOR = (150, 150, 150)
_BRAKE_COLOR = (30, 30, 30)
_LABEL_BG = (35, 35, 35)
_LABEL_FG = (255, 255, 255)


def _font(size: int = 15):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, img_size: tuple[int, int]) -> None:
    font = _font(15)
    radius = 11
    x, y = xy
    x = max(radius + 2, min(img_size[0] - radius - 2, int(x)))
    y = max(radius + 2, min(img_size[1] - radius - 2, int(y)))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=_LABEL_BG)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x - tw // 2, y - th // 2 - 1), label, fill=_LABEL_FG, font=font)


def _candidate_color(candidate: MomentumCandidate, idx: int) -> tuple[int, int, int]:
    if not candidate.safe:
        return _UNSAFE_COLOR
    if candidate.label == "brake":
        return _BRAKE_COLOR
    return _PATH_COLORS[idx % len(_PATH_COLORS)]


def annotate_momentum_candidates_overlay(
    base_image: Image.Image,
    candidates: list[MomentumCandidate],
    renderer: HazardRenderer,
    *,
    line_width: int = 4,
) -> Image.Image:
    """Draw all numbered future rollouts on one arena image."""
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_size = img.size

    for idx, candidate in enumerate(candidates):
        traj = np.asarray(candidate.trajectory_xy, dtype=np.float32)
        if traj.shape[0] < 2:
            continue

        pts = [renderer.world_to_pixel(float(p[0]), float(p[1])) for p in traj]
        color = _candidate_color(candidate, idx)

        draw.line(pts, fill=(255, 255, 255), width=line_width + 3, joint="curve")
        draw.line(pts, fill=color, width=line_width, joint="curve")

        end_x, end_y = pts[-1]
        draw.ellipse([end_x - 4, end_y - 4, end_x + 4, end_y + 4],
                     fill=color, outline=(255, 255, 255))

        start_x, start_y = pts[0]
        theta = 2.0 * math.pi * idx / max(1, len(candidates))
        label_dist_px = max(36, renderer.world_scale(0.85))
        label_x = int(start_x + label_dist_px * math.cos(theta))
        label_y = int(start_y - label_dist_px * math.sin(theta))
        _draw_label(draw, (label_x, label_y), str(candidate.choice_id), img_size)

    return img


def annotate_single_momentum_candidate(
    base_image: Image.Image,
    candidate: MomentumCandidate,
    renderer: HazardRenderer,
    *,
    color_idx: int,
    line_width: int = 6,
) -> Image.Image:
    """Draw exactly one rollout candidate, avoiding visual overlap."""
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_size = img.size
    traj = np.asarray(candidate.trajectory_xy, dtype=np.float32)
    if traj.shape[0] < 2:
        return img

    pts = [renderer.world_to_pixel(float(p[0]), float(p[1])) for p in traj]
    color = _candidate_color(candidate, color_idx)
    draw.line(pts, fill=(255, 255, 255), width=line_width + 4, joint="curve")
    draw.line(pts, fill=color, width=line_width, joint="curve")

    end_x, end_y = pts[-1]
    draw.ellipse([end_x - 5, end_y - 5, end_x + 5, end_y + 5],
                 fill=color, outline=(255, 255, 255), width=2)
    _draw_label(draw, (end_x, end_y), str(candidate.choice_id), img_size)
    return img


def annotate_momentum_candidates_grid(
    base_image: Image.Image,
    candidates: list[MomentumCandidate],
    renderer: HazardRenderer,
    *,
    panel_size: int = 360,
    columns: int = 3,
) -> Image.Image:
    """Draw candidates as isolated mini-panels so rollouts cannot overlap."""
    n = max(1, len(candidates))
    columns = max(1, min(int(columns), n))
    rows = int(math.ceil(n / columns))
    canvas = Image.new("RGB", (columns * panel_size, rows * panel_size), (238, 238, 238))
    draw_canvas = ImageDraw.Draw(canvas)
    font = _font(15)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

    for idx, candidate in enumerate(candidates):
        tile = annotate_single_momentum_candidate(
            base_image,
            candidate,
            renderer,
            color_idx=idx,
        )
        tile = tile.resize((panel_size, panel_size), resample=resample)
        x = (idx % columns) * panel_size
        y = (idx // columns) * panel_size
        canvas.paste(tile, (x, y))

        status = "SAFE" if candidate.safe else "RISK"
        if candidate.label == "brake":
            status = f"{status} BRAKE"
        caption = (
            f"{candidate.choice_id}: {status}  "
            f"clr={candidate.min_clearance:+.2f}  prog={candidate.goal_progress:+.2f}"
        )
        bbox = draw_canvas.textbbox((x + 8, y + 8), caption, font=font)
        draw_canvas.rectangle(
            [bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3],
            fill=(255, 255, 255),
            outline=(60, 60, 60),
        )
        draw_canvas.text((x + 8, y + 8), caption, fill=(0, 0, 0), font=font)
        draw_canvas.rectangle(
            [x, y, x + panel_size - 1, y + panel_size - 1],
            outline=(70, 70, 70),
            width=2,
        )

    return canvas


def annotate_momentum_candidates(
    base_image: Image.Image,
    candidates: list[MomentumCandidate],
    renderer: HazardRenderer,
    *,
    mode: str = "grid",
    line_width: int = 4,
    panel_size: int = 360,
) -> Image.Image:
    """Draw momentum candidates for VLM prompting or debugging."""
    if mode == "overlay":
        return annotate_momentum_candidates_overlay(
            base_image, candidates, renderer, line_width=line_width)
    if mode == "grid":
        return annotate_momentum_candidates_grid(
            base_image, candidates, renderer, panel_size=panel_size)
    raise ValueError(f"Unknown momentum candidate visualization mode: {mode!r}")


def build_momentum_prompt(
    n_candidates: int,
    prev_feedback: str | None = None,
    *,
    viz_mode: str = "grid",
) -> str:
    feedback_block = ""
    if prev_feedback:
        feedback_block = f"""
Previous step feedback:
{prev_feedback}
Use this feedback to correct the next choice if the last choice drifted toward
a hazard, failed to make goal progress, or needed braking.
"""

    if viz_mode == "grid":
        layout_block = f"""
The image is a grid of mini-panels. Each mini-panel repeats the same arena but
isolates exactly one candidate path, so paths do not overlap. The number in the
caption and endpoint badge is the candidate choice. Captions include SAFE/RISK,
minimum hazard clearance, and goal progress for that candidate.
"""
    else:
        layout_block = """
All candidate paths are drawn together on one arena image. Use the numbered
labels and curve colors to identify each candidate.
"""

    return f"""You see a 2D arena from above with hazards.

Visual elements:
- Gray bordered square = arena boundary
- Red filled circles = hazards. Touching one ends the episode.
- Blue dot = you, the controlled agent
- Green circle with "G" = goal
- Blue trail = recent path already traveled
- Blue velocity arrow from the agent = current momentum
- Numbered colored curves (1-{n_candidates}) = predicted future paths if you apply that candidate force now
- Gray numbered curves = predicted unsafe or too close to hazards
- The black/dark curve, when present, is a braking or counter-momentum candidate
{layout_block}

Each numbered curve already includes current velocity and drag. Judge the whole
future curve, not just the initial force direction. If you are moving fast,
prefer a curve that accounts for sliding and leaves enough clearance. Choose
the best candidate that moves toward the goal while avoiding all hazards.
Prefer a safe detour or braking over a risky shortcut.
{feedback_block}
OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


@torch.inference_mode()
def momentum_choose_action(
    model,
    processor,
    base_image: Image.Image,
    renderer: HazardRenderer,
    obs: np.ndarray,
    cfg: PointHazardConfig,
    *,
    n_directions: int,
    n_magnitudes: int,
    horizon: int,
    safety_margin: float,
    clearance_weight: float,
    include_brake: bool,
    prev_feedback: str | None,
    last_safe_action: np.ndarray | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
    retries: int,
    viz_mode: str,
) -> MomentumPivotResult:
    candidates = build_momentum_candidates(
        obs,
        cfg,
        n_directions=n_directions,
        n_magnitudes=n_magnitudes,
        horizon=horizon,
        safety_margin=safety_margin,
        clearance_weight=clearance_weight,
        include_brake=include_brake,
    )
    annotated = annotate_momentum_candidates(
        base_image, candidates, renderer, mode=viz_mode)
    prompt_text = build_momentum_prompt(
        len(candidates), prev_feedback=prev_feedback, viz_mode=viz_mode)

    raw_result = None
    for attempt in range(retries + 1):
        p_text = prompt_text
        if attempt > 0:
            p_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"
        raw_result = _call_vlm_select(
            model,
            processor,
            annotated,
            p_text,
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

    selected: MomentumCandidate | None = None
    fallback_used = False
    fallback_reason = ""
    choice_idx = raw_result.choice_idx

    if raw_result.parsed_ok and choice_idx is not None and 0 <= choice_idx < len(candidates):
        chosen = candidates[choice_idx]
        if chosen.safe:
            selected = chosen
        else:
            fallback_used = True
            fallback_reason = "vlm_chose_unsafe_candidate"
            selected = choose_local_candidate(candidates)
    else:
        fallback_used = True
        if last_safe_action is not None:
            held = rollout_action_candidate(
                obs,
                cfg,
                last_safe_action,
                choice_id=0,
                source_id=0,
                horizon=horizon,
                safety_margin=safety_margin,
                clearance_weight=clearance_weight,
                label="last_safe",
            )
            local = choose_local_candidate(candidates)
            if held.safe or not local.safe:
                fallback_reason = "vlm_parse_failed_hold_last_safe"
                selected = held
            else:
                fallback_reason = "vlm_parse_failed_last_safe_now_unsafe"
                selected = local
        else:
            fallback_reason = "vlm_parse_failed_no_last_safe"
            selected = choose_local_candidate(candidates)

    assert selected is not None
    action = np.clip(selected.action, -1.0, 1.0).astype(np.float32)

    return MomentumPivotResult(
        action=action,
        confidence=float(raw_result.confidence),
        uncertainty=float(raw_result.uncertainty),
        choice_idx=int(choice_idx) if choice_idx is not None else -1,
        reason=raw_result.reason,
        raw_text=raw_result.raw_text,
        parsed_ok=bool(raw_result.parsed_ok),
        mean_logprob=float(raw_result.mean_logprob),
        n_candidates=len(candidates),
        selected_candidate=selected,
        all_candidates=candidates,
        fallback_used=bool(fallback_used),
        fallback_reason=fallback_reason,
        annotated_image=annotated,
    )


# ---------------------------------------------------------------------------
# Testing helpers
# ---------------------------------------------------------------------------

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

    candidates = build_momentum_candidates(
        obs,
        cfg,
        n_directions=args.pivot_n_dirs,
        n_magnitudes=args.pivot_n_mags,
        horizon=args.rollout_horizon,
        safety_margin=args.safety_margin,
        clearance_weight=args.clearance_weight,
        include_brake=args.include_brake,
    )
    annotated = annotate_momentum_candidates(
        image, candidates, renderer, mode=args.viz_mode)

    out_dir = os.path.dirname(args.smoke_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    annotated.save(args.smoke_out)
    env.close()

    print(f"Smoke image saved: {args.smoke_out}")
    print(f"  candidates: {len(candidates)}")
    print(f"  safe: {sum(1 for c in candidates if c.safe)}")
    print(f"  speed: {float(np.linalg.norm(obs[2:4])):.3f}")
    print(f"  min clearance: {min(c.min_clearance for c in candidates):.3f}")


def _make_temp_env_from_state(
    cfg: PointHazardConfig,
    pos: np.ndarray,
    vel: np.ndarray,
    goal: np.ndarray,
    hazards: np.ndarray,
) -> PointHazardEnv:
    env = PointHazardEnv(cfg=cfg, seed=0)
    env.pos = np.asarray(pos, dtype=np.float32).copy()
    env.vel = np.asarray(vel, dtype=np.float32).copy()
    env.goal = np.asarray(goal, dtype=np.float32).copy()
    env.hazards = np.asarray(hazards, dtype=np.float32).copy()
    env.t = 0
    env._trail = [env.pos.copy()]
    return env


def run_dynamics_test(args: argparse.Namespace) -> int:
    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)

    total_candidates = 0
    safe_candidates = 0
    all_unsafe_resets = 0
    safety_violations = 0
    dynamics_mismatches = 0
    min_clearance_seen = float("inf")

    for ep in range(int(args.dynamics_test_resets)):
        obs, _info = env.reset(seed=args.seed + ep)
        pos, vel, goal, hazards = _obs_parts(obs, cfg)
        candidates = build_momentum_candidates(
            obs,
            cfg,
            n_directions=args.pivot_n_dirs,
            n_magnitudes=args.pivot_n_mags,
            horizon=args.rollout_horizon,
            safety_margin=args.safety_margin,
            clearance_weight=args.clearance_weight,
            include_brake=args.include_brake,
        )
        total_candidates += len(candidates)
        safe_now = sum(1 for c in candidates if c.safe)
        safe_candidates += safe_now
        if safe_now == 0:
            all_unsafe_resets += 1

        for candidate in candidates:
            min_clearance_seen = min(min_clearance_seen, candidate.min_clearance)
            if candidate.safe and candidate.min_clearance < args.safety_margin - 1e-6:
                safety_violations += 1

            sim_pos, sim_vel = simulate_dynamics_step(pos, vel, candidate.action, cfg)
            temp_env = _make_temp_env_from_state(cfg, pos, vel, goal, hazards)
            temp_env.step(candidate.action)
            if (
                not np.allclose(sim_pos, temp_env.pos, atol=1e-6)
                or not np.allclose(sim_vel, temp_env.vel, atol=1e-6)
            ):
                dynamics_mismatches += 1

    env.close()
    print("Dynamics test complete")
    print(f"  resets: {args.dynamics_test_resets}")
    print(f"  candidates: {total_candidates}")
    print(f"  safe candidates: {safe_candidates}")
    print(f"  all-unsafe resets: {all_unsafe_resets}")
    print(f"  min clearance seen: {min_clearance_seen:.3f}")
    print(f"  safety violations: {safety_violations}")
    print(f"  dynamics mismatches: {dynamics_mismatches}")
    return 1 if safety_violations or dynamics_mismatches else 0


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _candidate_log(candidate: MomentumCandidate | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "candidate_choice": 0,
            "candidate_source_id": 0,
            "candidate_action": [0.0, 0.0],
            "candidate_min_clearance": float("nan"),
            "candidate_goal_progress": float("nan"),
            "candidate_final_vel": [float("nan"), float("nan")],
            "candidate_final_dist_to_goal": float("nan"),
            "candidate_score": float("nan"),
            "chosen_candidate_safe": False,
        }
    return {
        "candidate_choice": int(candidate.choice_id),
        "candidate_source_id": int(candidate.source_id),
        "candidate_action": candidate.action.tolist(),
        "candidate_min_clearance": float(candidate.min_clearance),
        "candidate_goal_progress": float(candidate.goal_progress),
        "candidate_final_vel": candidate.final_vel.tolist(),
        "candidate_final_dist_to_goal": float(candidate.final_dist_to_goal),
        "candidate_score": float(candidate.score),
        "chosen_candidate_safe": bool(candidate.safe),
    }


def _format_feedback(
    *,
    choice_id: int,
    action: np.ndarray,
    goal_progress: float,
    clearance_delta: float,
    next_clearance: float,
    safety_margin: float,
    hazard_hit: bool,
    goal_success: bool,
) -> str:
    status = "goal_success" if goal_success else ("hazard_hit" if hazard_hit else "running")
    return (
        f"Last selected choice {choice_id} with action "
        f"[{float(action[0]):+.2f}, {float(action[1]):+.2f}]. "
        f"Executed one step: goal_progress={goal_progress:+.3f}, "
        f"clearance_change={clearance_delta:+.3f}, "
        f"next_clearance={next_clearance:.3f}, "
        f"near_hazard={next_clearance < safety_margin}, status={status}."
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=0)
    renderer = HazardRenderer.from_env(env)

    vlm_model = None
    vlm_processor = None
    if args.pilot_mode == "vlm":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading VLM: {args.model_path} ...")
        vlm_processor = AutoProcessor.from_pretrained(args.model_path)
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        vlm_model.eval()
        n_cand = args.pivot_n_dirs * args.pivot_n_mags + int(args.include_brake)
        print(f"VLM loaded. Momentum PIVOT candidates: {n_cand}, "
              f"horizon={args.rollout_horizon}, every={args.vlm_every}")

    expert = None
    if args.pilot_mode == "noisy_expert":
        expert = SafeExpert(env, cfg=SafeExpertConfig())

    use_diffusion = (not args.disable_diffusion) and bool(args.diff_ckpt) and args.fwd_ratio > 0
    ddpm = None
    max_k = 0
    base_k_default = 0
    if use_diffusion:
        ddpm = _load_ddpm_ckpt(args.diff_ckpt, device=device)
        cond_dim = ddpm.cfg.cond_dim
        act_dim = ddpm.cfg.x_dim - cond_dim
        max_k = ddpm.cfg.num_steps - 1
        base_k_default = int(round(args.fwd_ratio * max_k))
        print(f"Diffusion loaded: cond_dim={cond_dim}, act_dim={act_dim}, "
              f"k={base_k_default}/{max_k}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.save_log:
        run_dir = os.path.join(args.log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = ""
    if args.save_gif:
        os.makedirs(args.gif_dir, exist_ok=True)

    ep_returns: list[float] = []
    ep_successes: list[int] = []
    ep_hazard_hits: list[int] = []
    ep_final_dist: list[float] = []
    ep_min_clearances: list[float] = []
    ep_vlm_calls: list[int] = []
    ep_unsafe_fallbacks: list[int] = []
    ep_parse_ok: list[int] = []
    ep_parse_total: list[int] = []

    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        _pos0, _vel0, goal, _haz0 = _obs_parts(obs, cfg)

        if expert is not None:
            expert.reset_from_obs(obs)

        held_intent = np.zeros(2, dtype=np.float32)
        last_safe_action: np.ndarray | None = None
        last_selected_candidate: MomentumCandidate | None = None
        last_choice_idx = -1
        last_vlm_uncertainty = 0.5
        last_vlm_confidence = 0.5
        last_vlm_logprob = float("nan")
        last_vlm_raw = ""
        last_vlm_reason: str | None = None
        last_vlm_parsed_ok = False
        last_fallback_used = False
        last_fallback_reason = ""
        prev_choice_feedback = ""
        intent_age = 10**9

        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []
        log_rows: list[dict[str, Any]] = []
        ep_min_clearance = float("inf")
        vlm_calls = 0
        unsafe_fallbacks = 0
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
            current_clearance = clearance_to_hazards(pos, hazards, cfg)
            min_haz_dist = hazard_edge_distance(pos, hazards)
            momentum_speed = float(np.linalg.norm(vel))
            ep_min_clearance = min(ep_min_clearance, current_clearance)
            feedback_used_for_prompt = prev_choice_feedback

            vlm_called = False
            result: MomentumPivotResult | None = None

            if args.pilot_mode == "vlm":
                if t % max(1, args.vlm_every) == 0:
                    assert vlm_model is not None and vlm_processor is not None
                    vlm_called = True
                    result = momentum_choose_action(
                        vlm_model,
                        vlm_processor,
                        image,
                        renderer,
                        obs,
                        cfg,
                        n_directions=args.pivot_n_dirs,
                        n_magnitudes=args.pivot_n_mags,
                        horizon=args.rollout_horizon,
                        safety_margin=args.safety_margin,
                        clearance_weight=args.clearance_weight,
                        include_brake=args.include_brake,
                        prev_feedback=feedback_used_for_prompt,
                        last_safe_action=last_safe_action,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        enable_thinking=args.enable_thinking,
                        retries=args.vlm_retries,
                        viz_mode=args.viz_mode,
                    )

                    held_intent = result.action.copy()
                    last_selected_candidate = result.selected_candidate
                    if result.selected_candidate.safe:
                        last_safe_action = result.action.copy()
                    last_vlm_uncertainty = result.uncertainty
                    last_vlm_confidence = result.confidence
                    last_vlm_logprob = result.mean_logprob
                    last_vlm_raw = result.raw_text
                    last_vlm_reason = result.reason
                    last_vlm_parsed_ok = result.parsed_ok
                    last_choice_idx = result.choice_idx
                    last_fallback_used = result.fallback_used
                    last_fallback_reason = result.fallback_reason
                    intent_age = 0

                    vlm_calls += 1
                    parse_total += 1
                    parse_ok_count += int(result.parsed_ok)
                    if result.fallback_used:
                        unsafe_fallbacks += int(result.fallback_reason == "vlm_chose_unsafe_candidate")

                    if args.debug_print:
                        reason_str = (result.reason or "")[:70]
                        print(f"  [mom t={t}] choice={result.selected_candidate.choice_id} "
                              f"raw={result.choice_idx + 1 if result.choice_idx >= 0 else 0} "
                              f"safe={result.selected_candidate.safe} "
                              f"clear={result.selected_candidate.min_clearance:.2f} "
                              f"progress={result.selected_candidate.goal_progress:.2f} "
                              f"fallback={result.fallback_reason or 'none'} "
                              f"reason={reason_str}")

                pilot = np.clip(held_intent, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "goal_error":
                diff = obs[4:6] - obs[:2]
                norm = float(np.linalg.norm(diff))
                pilot = (diff / norm).astype(np.float32) if norm > 1e-6 else np.zeros(2, dtype=np.float32)
                pilot = np.clip(pilot, -1.0, 1.0)

            elif args.pilot_mode == "noisy_expert":
                assert expert is not None
                clean = expert.act(obs)
                noise = np.random.normal(0, args.pilot_noise, size=2).astype(np.float32)
                pilot = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)

            elif args.pilot_mode == "random":
                pilot = np.random.uniform(-1, 1, size=2).astype(np.float32)

            else:
                raise ValueError(f"Unknown pilot_mode: {args.pilot_mode}")

            candidate_for_log = rollout_action_candidate(
                obs,
                cfg,
                pilot,
                choice_id=(last_selected_candidate.choice_id if last_selected_candidate is not None else 0),
                source_id=(last_selected_candidate.source_id if last_selected_candidate is not None else 0),
                horizon=args.rollout_horizon,
                safety_margin=args.safety_margin,
                clearance_weight=args.clearance_weight,
                label="held",
            )

            n_haz = cfg.n_hazards
            copilot_obs = np.concatenate([obs[:4], obs[6:6 + 3 * n_haz]]).astype(np.float32)
            uncertainty = last_vlm_uncertainty if args.pilot_mode == "vlm" else 0.0

            if not use_diffusion or ddpm is None:
                assist = pilot.copy()
                effective_k = 0
            else:
                if args.adaptive_fwd_ratio:
                    if min_haz_dist < args.hazard_dist_threshold:
                        base_k = int(round(args.fwd_ratio_near * max_k))
                    else:
                        base_k = base_k_default
                else:
                    base_k = base_k_default

                if args.random_k:
                    effective_k = int(np.random.randint(args.k_min, args.k_max + 1))
                elif args.adaptive_k and args.pilot_mode == "vlm":
                    score = float(np.clip(
                        (uncertainty - args.uncertainty_threshold) * args.uncertainty_scale,
                        0.0,
                        1.0,
                    ))
                    effective_k = int(args.k_min + (args.k_max - args.k_min) * score)
                else:
                    effective_k = base_k
                effective_k = int(np.clip(effective_k, 0, max_k))

                if effective_k <= 0:
                    assist = pilot.copy()
                else:
                    cop = torch.from_numpy(copilot_obs).to(device).reshape(1, -1)
                    pa = torch.from_numpy(pilot).to(device).reshape(1, -1)
                    a_hat = ddpm.refine_action(copilot_obs=cop, pilot_action=pa, k=effective_k)
                    assist = a_hat.detach().cpu().numpy().reshape(-1).astype(np.float32)

            assist = np.clip(assist, -1.0, 1.0)
            alpha = float(args.assist_strength)
            exec_action = pilot + alpha * (assist - pilot)
            exec_action = np.clip(exec_action, -1.0, 1.0).astype(np.float32)

            prev_dist = float(np.linalg.norm(goal - pos))
            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            next_pos, _next_vel, _next_goal, next_hazards = _obs_parts(obs, cfg)
            next_clearance = clearance_to_hazards(next_pos, next_hazards, cfg)
            ep_min_clearance = min(ep_min_clearance, next_clearance)
            next_dist = float(np.linalg.norm(goal - next_pos))
            step_goal_progress = prev_dist - next_dist
            clearance_delta = next_clearance - current_clearance
            ep_return += float(reward)
            last_info = dict(info or {})

            prev_choice_feedback = _format_feedback(
                choice_id=int(candidate_for_log.choice_id),
                action=exec_action,
                goal_progress=step_goal_progress,
                clearance_delta=clearance_delta,
                next_clearance=next_clearance,
                safety_margin=args.safety_margin,
                hazard_hit=bool(last_info.get("hazard_hit", False)),
                goal_success=bool(last_info.get("goal_success", False)),
            )
            intent_age += 1

            if args.save_log:
                row = {
                    "t": t,
                    "pilot_mode": args.pilot_mode,
                    "agent_pos": pos.tolist(),
                    "agent_vel": vel.tolist(),
                    "goal": goal.tolist(),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    "current_clearance": float(current_clearance),
                    "next_clearance": float(next_clearance),
                    "min_haz_dist": float(min_haz_dist),
                    "adaptive_fwd_ratio_active": bool(
                        args.adaptive_fwd_ratio and min_haz_dist < args.hazard_dist_threshold
                    ),
                    "momentum_speed": float(momentum_speed),
                    "pilot": pilot.tolist(),
                    "assist": assist.tolist(),
                    "exec": exec_action.tolist(),
                    "pilot_assist_diff": float(np.linalg.norm(assist - pilot)),
                    "k": effective_k if use_diffusion else 0,
                    "fwd_ratio_effective": (effective_k / max_k) if (use_diffusion and max_k > 0) else 0.0,
                    "vlm_called": bool(vlm_called),
                    "vlm_uncertainty": float(last_vlm_uncertainty),
                    "vlm_confidence": float(last_vlm_confidence),
                    "vlm_logprob": float(last_vlm_logprob),
                    "vlm_parsed_ok": bool(last_vlm_parsed_ok),
                    "vlm_reason": last_vlm_reason,
                    "fallback_used": bool(last_fallback_used),
                    "fallback_reason": last_fallback_reason,
                    "prev_choice_feedback": feedback_used_for_prompt,
                    "next_choice_feedback": prev_choice_feedback,
                    "intent_age": int(intent_age),
                    "stale_intent": bool(args.pilot_mode == "vlm" and intent_age > 1),
                    "reward": float(reward),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                }
                row.update(_candidate_log(candidate_for_log))
                if result is not None:
                    row.update({
                        "vlm_raw": last_vlm_raw,
                        "momentum_n_candidates": int(result.n_candidates),
                        "candidate_all_min_clearance": [float(c.min_clearance) for c in result.all_candidates],
                        "candidate_all_goal_progress": [float(c.goal_progress) for c in result.all_candidates],
                        "candidate_all_safe": [bool(c.safe) for c in result.all_candidates],
                        "candidate_all_final_vel": [c.final_vel.tolist() for c in result.all_candidates],
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
        ep_unsafe_fallbacks.append(unsafe_fallbacks)
        ep_parse_ok.append(parse_ok_count)
        ep_parse_total.append(parse_total)

        flag = "OK " if success else ("HIT" if hazard_hit else "---")
        print(f"[ep {ep:03d}] [{flag}] return={ep_return:7.1f} steps={t+1:3d} "
              f"dist={dist:.2f} min_clear={ep_min_clearance:.2f} "
              f"vlm_calls={vlm_calls} unsafe_fallback={unsafe_fallbacks} "
              f"reason={term_reason}")

        if args.save_log:
            log_path = os.path.join(run_dir, f"ep{ep:03d}.jsonl")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "type": "episode_meta",
                    "episode": ep,
                    "pilot_mode": args.pilot_mode,
                    "momentum_pivot": True,
                    "use_diffusion": use_diffusion,
                    "seed": args.seed + ep,
                    "goal": goal.tolist(),
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
                    "unsafe_fallbacks": unsafe_fallbacks,
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
    mean_unsafe_fallbacks = float(np.mean(ep_unsafe_fallbacks))
    parse_total_sum = int(np.sum(ep_parse_total))
    parse_success_rate = float(np.sum(ep_parse_ok) / parse_total_sum) if parse_total_sum else float("nan")

    print(f"\n{'=' * 60}")
    print(f"Pilot: {args.pilot_mode} (Momentum PIVOT)  "
          f"Diffusion: {'ON' if use_diffusion else 'OFF'}  Episodes: {n}")
    print(f"  Candidates: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags "
          f"+ brake={args.include_brake}  horizon={args.rollout_horizon}")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n}  ({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    print(f"  Mean min clearance: {mean_min_clearance:.2f}")
    print(f"  VLM calls/ep: {mean_vlm_calls:.1f}")
    print(f"  Unsafe fallbacks/ep: {mean_unsafe_fallbacks:.1f}")
    print(f"  Parse success: {parse_success_rate:.1%}" if parse_total_sum else "  Parse success: n/a")
    if args.pilot_mode == "vlm":
        print(f"  VLM: {args.model_path}  every={args.vlm_every} temp={args.temperature}")
    print(f"{'=' * 60}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "momentum_pivot": True,
            "use_diffusion": use_diffusion,
            "episodes": n,
            "success_rate": succ_rate,
            "hazard_hit_rate": hit_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_ret,
            "mean_final_dist": mean_dist,
            "mean_min_clearance": mean_min_clearance,
            "mean_vlm_calls": mean_vlm_calls,
            "mean_unsafe_fallbacks": mean_unsafe_fallbacks,
            "parse_success_rate": parse_success_rate,
            "args": vars(args),
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Momentum-aware rolling PIVOT on PointHazardEnv")

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pilot_mode", type=str, default="vlm",
                        choices=["vlm", "goal_error", "noisy_expert", "random"])
    parser.add_argument("--pilot_noise", type=float, default=0.3)

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vlm_every", type=int, default=1,
                        help="Call VLM every N steps; held intent is logged as stale between calls")
    parser.add_argument("--vlm_retries", type=int, default=1)

    parser.add_argument("--pivot_n_dirs", type=int, default=8)
    parser.add_argument("--pivot_n_mags", type=int, default=1)
    parser.add_argument("--rollout_horizon", type=int, default=12)
    parser.add_argument("--safety_margin", type=float, default=0.15)
    parser.add_argument("--clearance_weight", type=float, default=0.25)
    parser.add_argument("--include_brake", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viz_mode", type=str, default="grid", choices=["grid", "overlay"],
                        help="How to render VLM candidates; grid isolates each rollout to avoid overlap")

    parser.add_argument("--diff_ckpt", type=str,
                        default="hazard/new_diffusion_hazard/riskdiffusion_hazard_best03.pt")
    parser.add_argument("--fwd_ratio", type=float, default=0.0)
    parser.add_argument("--disable_diffusion", action="store_true")
    parser.add_argument("--assist_strength", type=float, default=1.0)

    parser.add_argument("--adaptive_k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random_k", action="store_true")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=25)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.0)
    parser.add_argument("--uncertainty_scale", type=float, default=20.0)

    parser.add_argument("--adaptive_fwd_ratio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fwd_ratio_near", type=float, default=0.8)
    parser.add_argument("--hazard_dist_threshold", type=float, default=0.45)

    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--log_dir", type=str, default="hazard/logs_hazard_momentum_pivot")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_hazard_momentum_pivot")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_out", type=str, default="/tmp/momentum_pivot_smoke.png")
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
