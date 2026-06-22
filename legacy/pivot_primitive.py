"""
Safety-filtered motion-primitive PIVOT for PointHazardEnv.

The old PIVOT prompt asks the VLM to choose one force arrow.  This module
instead builds short waypoint-following motion primitives, rolls them out
with the same point-mass dynamics as the environment, filters by predicted
hazard clearance, and draws numbered trajectory candidates for the VLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class MotionPrimitive:
    """A short closed-loop waypoint primitive shown to the VLM."""

    choice_id: int
    source_id: int
    target_xy: np.ndarray
    trajectory_xy: np.ndarray
    first_action: np.ndarray
    min_clearance: float
    safe: bool
    goal_progress: float
    final_dist_to_goal: float
    score: float


def parse_absolute_obs(obs: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pos, vel, absolute goal, absolute hazards from PointHazardEnv obs."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    pos = obs[0:2].astype(np.float32)
    vel = obs[2:4].astype(np.float32)
    goal = obs[4:6].astype(np.float32)
    hazards = obs[6 : 6 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
    return pos, vel, goal, hazards


def clearance_to_hazards(pos: np.ndarray, hazards_abs: np.ndarray, cfg) -> float:
    """Minimum edge clearance from the agent body to any hazard."""
    pos = np.asarray(pos, dtype=np.float32)
    hazards_abs = np.asarray(hazards_abs, dtype=np.float32)
    if hazards_abs.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards_abs[:, :2] - pos[None, :], axis=1)
    clearances = center_d - hazards_abs[:, 2] - float(cfg.agent_radius)
    return float(np.min(clearances))


def primitive_pd_action(
    pos: np.ndarray,
    vel: np.ndarray,
    target_xy: np.ndarray,
    cfg,
    *,
    kp: float = 3.0,
    desired_speed: float = 2.4,
    slow_radius: float = 0.9,
) -> np.ndarray:
    """PD action that tracks a waypoint while respecting the env action range."""
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)

    delta = target_xy - pos
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        desired_vel = np.zeros(2, dtype=np.float32)
    else:
        speed = float(desired_speed) * min(1.0, dist / max(1e-6, float(slow_radius)))
        desired_vel = (delta / dist * speed).astype(np.float32)

    force = float(kp) * (desired_vel - vel)
    force = force + float(cfg.drag) * desired_vel
    action = force / float(cfg.force_scale)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def simulate_step(pos: np.ndarray, vel: np.ndarray, action: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
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


def rollout_primitive(
    obs: np.ndarray,
    cfg,
    target_xy: np.ndarray,
    *,
    source_id: int = 0,
    choice_id: int = 0,
    horizon: int = 12,
    safety_margin: float = 0.15,
    kp: float = 3.0,
    desired_speed: float = 2.4,
    slow_radius: float = 0.9,
    clearance_weight: float = 0.25,
) -> MotionPrimitive:
    """Roll out a closed-loop waypoint primitive and score its safety/progress."""
    pos, vel, goal, hazards = parse_absolute_obs(obs, cfg)
    target_xy = np.asarray(target_xy, dtype=np.float32).reshape(2)

    start_dist = float(np.linalg.norm(goal - pos))
    min_clearance = clearance_to_hazards(pos, hazards, cfg)
    safe = min_clearance >= float(safety_margin)
    trajectory = [pos.copy()]
    first_action = np.zeros(2, dtype=np.float32)

    p = pos.copy()
    v = vel.copy()
    for t in range(max(1, int(horizon))):
        action = primitive_pd_action(
            p, v, target_xy, cfg, kp=kp,
            desired_speed=desired_speed, slow_radius=slow_radius)
        if t == 0:
            first_action = action.copy()
        p, v = simulate_step(p, v, action, cfg)
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

    return MotionPrimitive(
        choice_id=int(choice_id),
        source_id=int(source_id),
        target_xy=target_xy.astype(np.float32),
        trajectory_xy=np.asarray(trajectory, dtype=np.float32),
        first_action=first_action.astype(np.float32),
        min_clearance=float(min_clearance),
        safe=bool(safe),
        goal_progress=float(goal_progress),
        final_dist_to_goal=float(final_dist),
        score=float(score),
    )


def generate_waypoint_primitives(
    obs: np.ndarray,
    cfg,
    *,
    n_dirs: int = 8,
    radius: float = 1.2,
    horizon: int = 12,
    safety_margin: float = 0.15,
    kp: float = 3.0,
    desired_speed: float = 2.4,
    slow_radius: float = 0.9,
) -> list[MotionPrimitive]:
    """Generate evenly spaced waypoint primitives around the current agent."""
    pos, _vel, _goal, _hazards = parse_absolute_obs(obs, cfg)
    arena_limit = float(cfg.arena_half) - float(cfg.agent_radius)
    primitives: list[MotionPrimitive] = []

    for i in range(int(n_dirs)):
        theta = 2.0 * math.pi * i / max(1, int(n_dirs))
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        target = pos + float(radius) * direction
        target = np.clip(target, -arena_limit, arena_limit).astype(np.float32)
        primitive = rollout_primitive(
            obs, cfg, target,
            source_id=i + 1,
            choice_id=i + 1,
            horizon=horizon,
            safety_margin=safety_margin,
            kp=kp,
            desired_speed=desired_speed,
            slow_radius=slow_radius,
        )
        primitives.append(primitive)

    return primitives


def choose_display_primitives(
    primitives: list[MotionPrimitive],
    *,
    max_display: int = 8,
) -> list[MotionPrimitive]:
    """Return safe primitives for prompting, or top-clearance primitives if none are safe."""
    safe = [p for p in primitives if p.safe]
    if safe:
        chosen = safe[: int(max_display)]
    else:
        chosen = sorted(
            primitives,
            key=lambda p: (p.min_clearance, p.goal_progress),
            reverse=True,
        )[: int(max_display)]

    return [
        replace(p, choice_id=i + 1)
        for i, p in enumerate(chosen)
    ]


def choose_local_primitive(primitives: list[MotionPrimitive]) -> MotionPrimitive:
    """Local fallback: prefer safe progress, otherwise maximum clearance."""
    if not primitives:
        raise ValueError("No primitives available")

    safe = [p for p in primitives if p.safe]
    if safe:
        return max(safe, key=lambda p: (p.score, p.min_clearance, p.goal_progress))
    return max(primitives, key=lambda p: (p.min_clearance, p.goal_progress))


def build_primitive_prompt(n_candidates: int) -> str:
    return f"""You see a 2D arena from above with hazards.

Visual elements:
- Gray bordered square = arena boundary
- Red filled circles = HAZARDS. Touching one ends the episode.
- Blue dot = YOU, the agent being controlled
- Green circle with "G" = GOAL
- Blue line = past trajectory
- Numbered colored curves (1-{n_candidates}) = short motion primitives the low-level controller can execute

Each numbered curve shows the predicted path for the next few control steps.
These candidates have already been checked by a local dynamics safety filter
when possible. Your job is to choose the best curve that moves toward the
green goal while keeping clear of all red hazards.

Prefer a safe detour over a risky shortcut. Do not choose a path that cuts
close to red hazards if another candidate makes reasonable goal progress.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


_PATH_COLORS = [
    (255, 150, 0),
    (0, 170, 210),
    (160, 90, 220),
    (40, 170, 90),
    (240, 190, 30),
    (230, 100, 160),
    (80, 130, 255),
    (140, 140, 35),
]
_UNSAFE_COLOR = (150, 150, 150)
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


def annotate_primitives(
    base_image: Image.Image,
    primitives: list[MotionPrimitive],
    renderer,
    *,
    line_width: int = 4,
) -> Image.Image:
    """Draw numbered primitive trajectories on a rendered environment frame."""
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_size = img.size
    n_source = max([p.source_id for p in primitives] + [1])

    for idx, primitive in enumerate(primitives):
        traj = np.asarray(primitive.trajectory_xy, dtype=np.float32)
        if traj.shape[0] < 2:
            continue

        pts = [renderer.world_to_pixel(float(p[0]), float(p[1])) for p in traj]
        color = _PATH_COLORS[idx % len(_PATH_COLORS)] if primitive.safe else _UNSAFE_COLOR

        # White underlay makes the curve visible over grid, hazards, and trail.
        draw.line(pts, fill=(255, 255, 255), width=line_width + 3, joint="curve")
        draw.line(pts, fill=color, width=line_width, joint="curve")

        tx, ty = renderer.world_to_pixel(float(primitive.target_xy[0]), float(primitive.target_xy[1]))
        draw.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=color, outline=(255, 255, 255))

        # Keep labels legible even when target points get clipped at arena edges.
        theta = 2.0 * math.pi * (primitive.source_id - 1) / max(1, n_source)
        label_dist_px = max(30, renderer.world_scale(0.8))
        label_xy = (
            int(pts[0][0] + label_dist_px * math.cos(theta)),
            int(pts[0][1] - label_dist_px * math.sin(theta)),
        )
        _draw_label(draw, label_xy, str(primitive.choice_id), img_size)

    return img
