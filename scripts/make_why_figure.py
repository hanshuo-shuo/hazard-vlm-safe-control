#!/usr/bin/env python3
"""Generate the opener figure for docs/WHY_VLM.md.

Same scene, two panels (matched seed): (1) geometry only — a classical planner
reaches the goal cleanly; (2) the same scene + a water patch the robot cannot
measure — the SAME classical path now drives straight through it. No API / VLM
needed (the driver is pure classical MPC); this is deterministic.

    python scripts/make_why_figure.py [--seed 44] [--outdir outputs]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subgoal_pivot_hazard as S  # noqa: E402
from env_pointhazard import PointHazardConfig, PointHazardEnv  # noqa: E402
from hazard_renderer import HazardRenderer  # noqa: E402


def _font(sz: int):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, sz)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _titled(img: Image.Image, title: str, sub: str) -> Image.Image:
    pad = 46
    canvas = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title, fill=(0, 0, 0), font=_font(17))
    d.text((8, 27), sub, fill=(90, 90, 90), font=_font(12))
    canvas.paste(img, (0, pad))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    cfg = PointHazardConfig(n_hazards=8, n_semantic_zones=1, max_episode_steps=300)
    env = PointHazardEnv(cfg=cfg)
    renderer = HazardRenderer.from_env(env, semantic_style="water")
    env.attach_renderer(renderer)

    # Pure classical MPC (C1) — blind to the water, so its path is identical
    # whether or not the zone is drawn. Seeding from the episode seed makes it
    # reproducible (same path as the run-of-record).
    policy = S.ControllerOnlyPolicy(cfg, "mpc", 0.15)
    obs, info = env.reset(seed=args.seed)
    policy.reset(obs, info, seed=args.seed)
    for _ in range(cfg.max_episode_steps):
        obs, _r, term, trunc, info = env.step(policy.act(obs, env))
        if term or trunc:
            break

    right = Image.fromarray(env.render())                 # with the water drawn
    zones = env.semantic_zones.copy()
    env.semantic_zones = np.zeros((0, 3), dtype=np.float32)
    left = Image.fromarray(env.render())                  # same world, zone hidden
    env.semantic_zones = zones

    left = _titled(
        left, "1. Geometry task (red hazards only)",
        "Classical planner is near-perfect here. A VLM adds nothing.")
    right = _titled(
        right, "2. Add a water patch the robot cannot measure",
        "The SAME classical planner drives straight through it (blind).")

    gap = 14
    canvas = Image.new("RGB", (left.width + right.width + gap,
                               max(left.height, right.height)), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "fig_why_vlm.png")
    canvas.save(path)
    print(f"wrote {path}  (classical in-zone steps: {info.get('semantic_steps')})")


if __name__ == "__main__":
    main()
