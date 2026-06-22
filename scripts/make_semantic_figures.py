"""Generate illustrative figures for the semantic keep-out experiment.

Produces, into outputs/:
  - fig_vlm_view_seed<seed>.png   : the exact image the subgoal-VLM is shown
                                    (arena + amber keep-out zone + numbered
                                    purple candidate waypoints). "What the VLM
                                    sees" — no clearance / label / score on it.
  - fig_traj_seed<seed>.png       : 3-panel trajectory contrast on one seed,
                                    C1 (geometry-blind) | C2 (oracle) | B (VLM),
                                    each showing the blue trail through/around
                                    the amber zone.

C1/C2 are deterministic (no API). B re-runs the VLM; with temperature=0 it
reproduces the choices from the recorded run (outputs/semantic_pilot.json), so
the figure matches the table. Run from the repo root:

    set -a; . ./.env; set +a
    python scripts/make_semantic_figures.py --seeds 43 47
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_pointhazard import PointHazardConfig, PointHazardEnv
from hazard_renderer import HazardRenderer
import subgoal_pivot_hazard as S

MODEL = "google/gemini-3-flash-preview"
SAFETY_MARGIN = 0.15


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _titled(img: Image.Image, title: str, subtitle: str) -> Image.Image:
    """Add a two-line header strip above a panel."""
    bar_h = 46
    w = img.width
    out = Image.new("RGB", (w, img.height + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((8, 5), title, fill=(0, 0, 0), font=_font(18))
    d.text((8, 27), subtitle, fill=(90, 90, 90), font=_font(13))
    return out


def build_policy(name: str, cfg: PointHazardConfig, renderer, vlm_fn):
    if name == "mpc":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN)
    if name == "mpc_oracle":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN, semantic_aware=True)
    if name == "subgoal":
        return S.SubgoalPivotPolicy(
            cfg, renderer, pilot_mode="vlm", vlm_fn=vlm_fn, low_level="mpc",
            safety_margin=SAFETY_MARGIN, n_dirs=8, subgoal_radius=2.5,
            subgoal_horizon=15, subgoal_reach=0.6, fallback="hold", semantic=True,
        )
    raise ValueError(name)


def run_and_render(env: PointHazardEnv, policy, seed: int) -> tuple[Image.Image, int]:
    """Run one episode to completion, return (final frame with full trail, sem_steps).

    The MPC controller's CEM RNG is unseeded in the experiment (a known
    reproducibility gap, see RESULTS_SEMANTIC.md), so for a DETERMINISTIC figure
    we seed it here from the episode seed. The figure's in-zone counts are thus
    illustrative of this seed; the paper table is the run of record.
    """
    obs, info = env.reset(seed=seed)
    policy.reset(obs, info)
    if hasattr(policy, "expert") and hasattr(policy.expert, "rng"):
        policy.expert.rng = np.random.default_rng(seed)
    sem_steps = 0
    for _ in range(env.cfg.max_episode_steps):
        action = policy.act(obs, env)
        obs, _r, term, trunc, info = env.step(action)
        sem_steps = int(info.get("semantic_steps", sem_steps))
        if term or trunc:
            break
    return Image.fromarray(env.render()), sem_steps


def vlm_view(env: PointHazardEnv, renderer, seed: int) -> Image.Image:
    """The exact annotated image the subgoal-VLM receives on step 0."""
    obs, _info = env.reset(seed=seed)
    pos, _, _, _ = S._obs_parts(obs, env.cfg)
    base = Image.fromarray(env.render())
    subs = S.generate_subgoals(pos, 8, 2.5, env.cfg.arena_half)
    return S.annotate_subgoals(base, subs, renderer, pos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 47])
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    cfg = PointHazardConfig(n_semantic_zones=1, max_episode_steps=300)
    env = PointHazardEnv(cfg=cfg)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("set OPENROUTER_API_KEY (source ./.env) — B needs the VLM.")
    vlm_fn = S.make_openrouter_vlm(key, MODEL, temperature=0.0, retries=1)

    panels = [
        ("mpc", "C1  geometry-only (blind)", "pure classical planner"),
        ("mpc_oracle", "C2  oracle", "zone hand-coded as obstacle"),
        ("subgoal", "B  VLM subgoal", "reads zone from the image"),
    ]

    os.makedirs(args.outdir, exist_ok=True)
    for seed in args.seeds:
        # What the VLM sees.
        view = vlm_view(env, renderer, seed)
        vpath = os.path.join(args.outdir, f"fig_vlm_view_seed{seed}.png")
        _titled(view, f"What the subgoal-VLM sees (seed {seed})",
                "amber X = keep-out zone · purple 1-8 = candidate waypoints").save(vpath)
        print("wrote", vpath)

        # Trajectory contrast.
        rendered = []
        for name, title, sub in panels:
            frame, sem_steps = run_and_render(env, build_policy(name, cfg, renderer, vlm_fn), seed)
            sub2 = f"{sub} · in-zone steps: {sem_steps}"
            rendered.append(_titled(frame, title, sub2))
            print(f"  seed {seed} {name}: in-zone steps = {sem_steps}")
        gap = 10
        w = sum(p.width for p in rendered) + gap * (len(rendered) - 1)
        h = max(p.height for p in rendered)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for p in rendered:
            canvas.paste(p, (x, 0))
            x += p.width + gap
        tpath = os.path.join(args.outdir, f"fig_traj_seed{seed}.png")
        canvas.save(tpath)
        print("wrote", tpath)


if __name__ == "__main__":
    main()
