"""
Deterministic physics validation for PointPushHazardEnv.

Writes visual strips under hazard/pointpush_validation/:
  - 01_push_success_strip.png
  - 02_box_hazard_strip.png
  - 03_agent_hazard_strip.png
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env_pointpushhazard import PointPushHazardConfig, PointPushHazardEnv, make_env


OUT_DIR = os.path.join("hazard", "pointpush_validation")


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _base_cfg(**kwargs) -> PointPushHazardConfig:
    cfg = PointPushHazardConfig(
        n_hazards=2,
        arena_half=4.0,
        render_size=420,
        max_episode_steps=260,
        box_progress_reward_scale=0.5,
        max_agent_speed=3.2,
        max_box_speed=2.4,
    )
    return replace(cfg, **kwargs)


def _manual_env(
    *,
    cfg: PointPushHazardConfig,
    agent: tuple[float, float],
    box: tuple[float, float],
    goal: tuple[float, float],
    hazards: list[tuple[float, float, float]],
) -> PointPushHazardEnv:
    env = make_env(cfg=cfg, seed=0, with_renderer=True)
    env.reset(seed=0)
    env.hazards = np.asarray(hazards, dtype=np.float32)
    env.pos = np.asarray(agent, dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)
    env.box_pos = np.asarray(box, dtype=np.float32)
    env.box_vel = np.zeros(2, dtype=np.float32)
    env.goal = np.asarray(goal, dtype=np.float32)
    env.t = 0
    env._agent_trail = [env.pos.copy()]
    env._box_trail = [env.box_pos.copy()]
    assert not env._agent_hazard_collision()
    assert not env._box_hazard_collision()
    assert not env._agent_box_overlaps_at(env.pos, env.box_pos)
    return env


def _pd_action(env: PointPushHazardEnv, target: np.ndarray, *, speed: float = 2.3) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float32) - env.pos
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        desired_vel = np.zeros(2, dtype=np.float32)
    else:
        desired_vel = delta / dist * min(speed, dist / max(1e-6, env.cfg.dt))
    force = 3.0 * (desired_vel - env.vel) + float(env.cfg.agent_drag) * desired_vel
    return np.clip(force / float(env.cfg.force_scale), -1.0, 1.0).astype(np.float32)


def _push_right_action(env: PointPushHazardEnv) -> np.ndarray:
    target = env.box_pos + np.array([-0.78, 0.0], dtype=np.float32)
    if env.pos[0] < target[0] - 0.08 or abs(float(env.pos[1] - target[1])) > 0.08:
        return _pd_action(env, target, speed=2.4)
    return np.array([1.0, 0.0], dtype=np.float32)


def _render_frame(env: PointPushHazardEnv, label: str, info: dict | None = None) -> Image.Image:
    reason = "" if info is None else str(info.get("termination_reason", "running"))
    text = (
        f"{label}\n"
        f"t={env.t}  reason={reason}\n"
        f"agent=({env.pos[0]:+.2f},{env.pos[1]:+.2f})\n"
        f"box=({env.box_pos[0]:+.2f},{env.box_pos[1]:+.2f})"
    )
    arr = env.render(info_text=text)
    return Image.fromarray(arr)


def _save_strip(frames: list[Image.Image], labels: list[str], path: str) -> None:
    assert len(frames) == len(labels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = frames[0].size
    title_h = 34
    canvas = Image.new("RGB", (w * len(frames), h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x = i * w
        canvas.paste(frame, (x, title_h))
        draw.rectangle([x, 0, x + w, title_h], fill=(35, 35, 35))
        draw.text((x + 10, 8), label, fill=(255, 255, 255), font=font)
    canvas.save(path)


def run_push_success() -> dict[str, object]:
    env = _manual_env(
        cfg=_base_cfg(max_episode_steps=220),
        agent=(-2.6, 0.0),
        box=(-1.4, 0.0),
        goal=(2.35, 0.0),
        hazards=[(0.0, 1.65, 0.45), (0.25, -1.65, 0.45)],
    )
    frames = [_render_frame(env, "start")]
    labels = ["start"]
    capture_steps = {8: "contact", 16: "box sliding"}
    info: dict | None = None
    contact_seen = False
    initial_box_x = float(env.box_pos[0])
    for _ in range(env.cfg.max_episode_steps):
        _obs, _reward, terminated, truncated, info = env.step(_push_right_action(env))
        contact_seen = contact_seen or bool(info.get("agent_box_contact", False))
        if env.t in capture_steps:
            frames.append(_render_frame(env, capture_steps[env.t], info))
            labels.append(capture_steps[env.t])
        if terminated or truncated:
            break
    frames.append(_render_frame(env, "finish", info))
    labels.append("finish")

    assert info is not None
    assert bool(info.get("goal_success", False)), info
    assert contact_seen
    assert float(env.box_pos[0]) - initial_box_x > 2.5
    assert abs(float(env.box_pos[1])) < 0.2
    assert not bool(info.get("hazard_hit", False))

    path = os.path.join(OUT_DIR, "01_push_success_strip.png")
    _save_strip(frames, labels, path)
    return {
        "name": "push_success",
        "path": path,
        "steps": env.t,
        "box_start_x": initial_box_x,
        "box_final_x": float(env.box_pos[0]),
        "termination_reason": info.get("termination_reason"),
    }


def run_box_hazard() -> dict[str, object]:
    env = _manual_env(
        cfg=_base_cfg(max_episode_steps=180),
        agent=(-2.25, 0.0),
        box=(-1.0, 0.0),
        goal=(3.0, 0.0),
        hazards=[(0.55, 0.0, 0.42), (0.2, 1.8, 0.35)],
    )
    frames = [_render_frame(env, "start")]
    labels = ["start"]
    info: dict | None = None
    for _ in range(env.cfg.max_episode_steps):
        _obs, _reward, terminated, truncated, info = env.step(_push_right_action(env))
        if env.t in (5, 10):
            frames.append(_render_frame(env, "approach", info))
            labels.append("approach")
        if terminated or truncated:
            break
    frames.append(_render_frame(env, "box hazard", info))
    labels.append("box hazard")

    assert info is not None
    assert info.get("termination_reason") == "box_hazard", info
    assert bool(info.get("box_hazard_hit", False))
    assert not bool(info.get("agent_hazard_hit", False))

    path = os.path.join(OUT_DIR, "02_box_hazard_strip.png")
    _save_strip(frames, labels, path)
    return {
        "name": "box_hazard",
        "path": path,
        "steps": env.t,
        "termination_reason": info.get("termination_reason"),
        "min_box_hazard_clearance": float(info["min_box_hazard_clearance"]),
    }


def run_agent_hazard() -> dict[str, object]:
    env = _manual_env(
        cfg=_base_cfg(max_episode_steps=120),
        agent=(-2.6, 0.0),
        box=(2.6, 1.0),
        goal=(3.25, 1.0),
        hazards=[(-1.25, 0.0, 0.38), (0.0, 1.8, 0.35)],
    )
    frames = [_render_frame(env, "start")]
    labels = ["start"]
    info: dict | None = None
    for _ in range(env.cfg.max_episode_steps):
        _obs, _reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if env.t == 3:
            frames.append(_render_frame(env, "approach", info))
            labels.append("approach")
        if terminated or truncated:
            break
    frames.append(_render_frame(env, "agent hazard", info))
    labels.append("agent hazard")

    assert info is not None
    assert info.get("termination_reason") == "agent_hazard", info
    assert bool(info.get("agent_hazard_hit", False))
    assert not bool(info.get("box_hazard_hit", False))

    path = os.path.join(OUT_DIR, "03_agent_hazard_strip.png")
    _save_strip(frames, labels, path)
    return {
        "name": "agent_hazard",
        "path": path,
        "steps": env.t,
        "termination_reason": info.get("termination_reason"),
        "min_agent_hazard_clearance": float(info["min_agent_hazard_clearance"]),
    }


def main() -> None:
    results = [run_push_success(), run_box_hazard(), run_agent_hazard()]
    for item in results:
        print(item)


if __name__ == "__main__":
    main()
