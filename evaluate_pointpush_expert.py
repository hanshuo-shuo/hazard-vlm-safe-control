"""
Run the god-view expert on PointPushHazardEnv and save visual diagnostics.

Examples:
    python evaluate_pointpush_expert.py --episodes 20 --save_gif
    python evaluate_pointpush_expert.py --episodes 50 --no-save_gif
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env_pointpushhazard import PointPushHazardConfig, make_env
from pointpush_expert import (
    PointPushExpert,
    PointPushExpertConfig,
    min_agent_hazard_clearance,
    min_box_hazard_clearance,
    parse_pointpush_obs,
)


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _annotate_expert_overlay(image: Image.Image, env, expert: PointPushExpert) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    renderer = env._renderer
    if renderer is None:
        return img

    if expert.path is not None and len(expert.path) >= 2:
        pts = [renderer.world_to_pixel(float(p[0]), float(p[1])) for p in expert.path]
        draw.line(pts, fill=(20, 150, 95), width=3, joint="curve")
        for p in pts[1:-1]:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(20, 150, 95))

    pose = expert.last_push_pose
    px, py = renderer.world_to_pixel(float(pose[0]), float(pose[1]))
    draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(120, 50, 190), outline=(255, 255, 255))
    bx, by = renderer.world_to_pixel(float(env.box_pos[0]), float(env.box_pos[1]))
    ex = env.box_pos + expert.last_push_dir * 0.9
    epx, epy = renderer.world_to_pixel(float(ex[0]), float(ex[1]))
    draw.line([(bx, by), (epx, epy)], fill=(120, 50, 190), width=3)
    return img


def _save_episode_strip(frames: list[Image.Image], labels: list[str], path: str) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = frames[0].size
    title_h = 34
    canvas = Image.new("RGB", (w * len(frames), h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x = i * w
        canvas.paste(frame, (x, title_h))
        draw.rectangle([x, 0, x + w, title_h], fill=(35, 35, 35))
        draw.text((x + 8, 8), label, fill=(255, 255, 255), font=font)
    canvas.save(path)


def evaluate(args: argparse.Namespace) -> None:
    cfg = PointPushHazardConfig(
        max_episode_steps=args.max_steps,
        n_hazards=args.n_hazards,
        render_size=args.render_size,
    )
    expert_cfg = PointPushExpertConfig(
        grid_res=args.grid_res,
        safety_margin=args.safety_margin,
        agent_safety_margin=args.agent_safety_margin,
    )
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=True)
    expert = PointPushExpert(env, expert_cfg)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    gif_dir = os.path.join(run_dir, "gifs")
    strip_dir = os.path.join(run_dir, "strips")
    if args.save_gif:
        os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(strip_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        obs, _info = env.reset(seed=args.seed + ep)
        expert.path = None
        expert.path_idx = 0
        expert.agent_path = None
        expert.agent_path_idx = 0
        expert.last_plan_ok = False
        expert.last_agent_plan_ok = False

        frames: list[Image.Image] = []
        strip_frames: list[Image.Image] = []
        strip_labels: list[str] = []
        min_agent_clear = float("inf")
        min_box_clear = float("inf")
        contacts = 0
        plan_fail_steps = 0
        ep_return = 0.0
        last_info: dict[str, Any] = {}
        capture = {0: "start", args.max_steps // 4: "early", args.max_steps // 2: "mid"}

        for t in range(int(args.max_steps)):
            agent_pos, _agent_vel, box_pos, _box_vel, _goal, hazards = parse_pointpush_obs(obs, cfg)
            min_agent_clear = min(min_agent_clear, min_agent_hazard_clearance(agent_pos, hazards, cfg))
            min_box_clear = min(min_box_clear, min_box_hazard_clearance(box_pos, hazards, cfg))
            action = expert.act(obs)
            plan_fail_steps += int(not expert.last_plan_ok)

            frame = Image.fromarray(env.render(info_text=f"ep={ep} t={t}\nplan_ok={expert.last_plan_ok}"))
            frame = _annotate_expert_overlay(frame, env, expert)
            if args.save_gif and t % max(1, int(args.gif_every)) == 0:
                frames.append(frame.copy())
            if t in capture:
                strip_frames.append(frame.copy())
                strip_labels.append(capture[t])

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            contacts += int(bool(info.get("agent_box_contact", False)))
            last_info = dict(info)
            if terminated or truncated:
                break

        final_frame = Image.fromarray(env.render(info_text=f"ep={ep} final\n{last_info.get('termination_reason', 'timeout')}"))
        final_frame = _annotate_expert_overlay(final_frame, env, expert)
        strip_frames.append(final_frame)
        strip_labels.append(str(last_info.get("termination_reason", "timeout")))

        success = bool(last_info.get("goal_success", False))
        hazard_hit = bool(last_info.get("hazard_hit", False))
        reason = str(last_info.get("termination_reason", "timeout"))
        row = {
            "episode": ep,
            "seed": args.seed + ep,
            "success": success,
            "hazard_hit": hazard_hit,
            "termination_reason": reason,
            "return": ep_return,
            "steps": int(env.t),
            "dist_box_to_goal": float(last_info.get("dist_box_to_goal", float("nan"))),
            "min_agent_hazard_clearance": min_agent_clear,
            "min_box_hazard_clearance": min_box_clear,
            "contact_steps": contacts,
            "plan_fail_steps": plan_fail_steps,
        }
        rows.append(row)
        print(
            f"[ep {ep:03d}] success={success} hazard={hazard_hit} "
            f"steps={env.t:3d} dist={row['dist_box_to_goal']:.2f} "
            f"agent_clear={min_agent_clear:.2f} box_clear={min_box_clear:.2f} "
            f"reason={reason}"
        )

        tag = "ok" if success else ("hit" if hazard_hit else "timeout")
        if args.save_gif and frames:
            gif_path = os.path.join(gif_dir, f"ep{ep:03d}_{tag}.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / max(1e-6, float(args.gif_fps))),
                loop=0,
            )
        _save_episode_strip(
            strip_frames,
            strip_labels,
            os.path.join(strip_dir, f"ep{ep:03d}_{tag}.png"),
        )

    env.close()
    n = max(1, len(rows))
    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "hazard_hit_rate": float(np.mean([r["hazard_hit"] for r in rows])),
        "timeout_rate": float(np.mean([
            not r["success"] and not r["hazard_hit"] for r in rows
        ])),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_final_box_goal_dist": float(np.nanmean([r["dist_box_to_goal"] for r in rows])),
        "min_agent_hazard_clearance": float(np.min([r["min_agent_hazard_clearance"] for r in rows])),
        "min_box_hazard_clearance": float(np.min([r["min_box_hazard_clearance"] for r in rows])),
        "total_plan_fail_steps": int(np.sum([r["plan_fail_steps"] for r in rows])),
        "run_dir": run_dir,
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "episodes.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExpert validation summary")
    print(f"  Success:    {sum(r['success'] for r in rows):3d}/{n} ({summary['success_rate']:.1%})")
    print(f"  Hazard hit: {sum(r['hazard_hit'] for r in rows):3d}/{n} ({summary['hazard_hit_rate']:.1%})")
    print(f"  Timeout:    {sum((not r['success'] and not r['hazard_hit']) for r in rows):3d}/{n} ({summary['timeout_rate']:.1%})")
    print(f"  Min agent clearance: {summary['min_agent_hazard_clearance']:.3f}")
    print(f"  Min box clearance:   {summary['min_box_hazard_clearance']:.3f}")
    print(f"  Output: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="God-view PointPushHazard expert validation")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_hazards", type=int, default=8)
    parser.add_argument("--render_size", type=int, default=360)
    parser.add_argument("--grid_res", type=int, default=90)
    parser.add_argument("--safety_margin", type=float, default=0.22)
    parser.add_argument("--agent_safety_margin", type=float, default=0.18)
    parser.add_argument("--out_dir", type=str, default="hazard/pointpush_expert_eval")
    parser.add_argument("--save_gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif_every", type=int, default=3)
    parser.add_argument("--gif_fps", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
