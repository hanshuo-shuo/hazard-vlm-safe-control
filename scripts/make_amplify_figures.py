"""Image-rich figures for the "amplify the VLM advantage" results (lever 1 = B+
perception-feeding, lever 2 = heterogeneous terrains).

ZERO new API calls: B / B+ trajectories are reproduced by REPLAYING the VLM
choices already saved in the run-of-record transcripts. Because the controllers
are seeded per-episode and the dynamics are deterministic, replaying the recorded
(choice, raw) sequence reproduces the exact trajectory behind the results table.
C1 / C2 are deterministic anyway (no VLM).

Produces, into outputs/:
  - fig_bplus_traj_implicit_seed<seed>.png  : 4-panel C1|C2|B|B+ on the single
        implicit (water) keep-out — shows B+ closing B's corner-cut to the oracle.
  - fig_hetero_view_seed<seed>.png          : what the VLM sees on the 3-terrain
        scene (water+mud+grass + numbered candidate waypoints, no labels).
  - fig_hetero_traj_seed<seed>.png          : 4-panel C1|C2|B|B+ on the 3-terrain
        scene — C1 through everything, B+ around all three.

Run from repo root (no key needed):
    python scripts/make_amplify_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_pointhazard import PointHazardConfig, PointHazardEnv
from hazard_renderer import HazardRenderer
import subgoal_pivot_hazard as S

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
    bar_h = 46
    out = Image.new("RGB", (img.width, img.height + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((8, 5), title, fill=(0, 0, 0), font=_font(18))
    d.text((8, 27), subtitle, fill=(90, 90, 90), font=_font(13))
    return out


def _hstack(panels: list[Image.Image], gap: int = 10) -> Image.Image:
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


class ReplayVlm:
    """A vlm_fn that returns recorded calls in order — no network, exact replay."""

    def __init__(self, calls: list[dict]):
        self.calls = calls
        self.i = 0

    def __call__(self, image, prompt, n_candidates):
        t = self.calls[self.i] if self.i < len(self.calls) else self.calls[-1]
        self.i += 1
        return S.VlmCall(
            choice=t.get("choice"), reason=t.get("reason"), raw=t.get("raw", ""),
            parsed_ok=t.get("parsed_ok", True), api_error=t.get("api_error", False),
            attempts=t.get("attempts", 1),
        )


def _calls_for(transcripts: list[dict], policy: str, seed: int) -> list[dict]:
    for ep in transcripts:
        if ep["policy"] == policy and ep["seed"] == seed:
            return ep["calls"]
    return []


def build_policy(name: str, cfg, renderer, transcripts, seed: int):
    if name == "mpc":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN)
    if name == "mpc_oracle":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN, semantic_aware=True)
    if name in ("subgoal", "subgoal_perceive"):
        vlm_fn = ReplayVlm(_calls_for(transcripts, name, seed))
        return S.SubgoalPivotPolicy(
            cfg, renderer, pilot_mode="vlm", vlm_fn=vlm_fn, low_level="mpc",
            safety_margin=SAFETY_MARGIN, n_dirs=8, subgoal_radius=2.5,
            subgoal_horizon=15, subgoal_reach=0.6, fallback="hold", semantic=True,
            zone_mode="implicit", perceive_zones=(name == "subgoal_perceive"),
        )
    raise ValueError(name)


def run_and_render(env, policy, seed: int):
    obs, info = env.reset(seed=seed)
    policy.reset(obs, info, seed=seed)
    sem_steps = 0
    for _ in range(env.cfg.max_episode_steps):
        action = policy.act(obs, env)
        obs, _r, term, trunc, info = env.step(action)
        sem_steps = int(info.get("semantic_steps", sem_steps))
        if term or trunc:
            break
    outcome = info.get("termination_reason", "timeout")
    return Image.fromarray(env.render()), sem_steps, outcome


def vlm_view(env, renderer, seed: int) -> Image.Image:
    obs, _info = env.reset(seed=seed)
    pos, _, _, _ = S._obs_parts(obs, env.cfg)
    base = Image.fromarray(env.render())
    subs = S.generate_subgoals(pos, 8, 2.5, env.cfg.arena_half)
    return S.annotate_subgoals(base, subs, renderer, pos)


PANELS = [
    ("mpc", "C1  geometry-only (blind)", "pure classical — drives through"),
    ("mpc_oracle", "C2  oracle", "every zone hand-coded as obstacle"),
    ("subgoal", "B  VLM subgoal", "VLM picks waypoints; controller zone-blind"),
    ("subgoal_perceive", "B+  VLM subgoal + perception", "VLM's keep-out fed to controller"),
]


def make_set(tag, cfg, renderer_style, styles, transcripts, seeds, outdir, want_view):
    env = PointHazardEnv(cfg=cfg)
    renderer = HazardRenderer.from_env(env, semantic_style=renderer_style)
    env.attach_renderer(renderer)
    for seed in seeds:
        if want_view:
            view = vlm_view(env, renderer, seed)
            vpath = os.path.join(outdir, f"fig_{tag}_view_seed{seed}.png")
            _titled(view, f"What the VLM sees — {tag} (seed {seed})",
                    "distinct unsafe terrains, NO labels · purple 1-8 = candidate waypoints"
                    ).save(vpath)
            print("wrote", vpath)
        rendered = []
        for name, title, sub in PANELS:
            frame, sem_steps, outcome = run_and_render(
                env, build_policy(name, cfg, renderer, transcripts, seed), seed)
            rendered.append(_titled(frame, title, f"{sub} · in-zone steps: {sem_steps} · {outcome}"))
            print(f"  {tag} seed {seed} {name}: in-zone={sem_steps} {outcome}")
        tpath = os.path.join(outdir, f"fig_{tag}_traj_seed{seed}.png")
        _hstack(rendered).save(tpath)
        print("wrote", tpath)


def main():
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # Lever 1: single implicit (water) zone — B+ closes B's corner-cut.
    tr1 = json.load(open(os.path.join(outdir, "semantic_bplus_implicit.transcripts.json")))
    cfg1 = PointHazardConfig(n_semantic_zones=1, max_episode_steps=300)
    make_set("bplus_implicit", cfg1, "water", (), tr1, [47, 44], outdir, want_view=False)

    # Lever 2: heterogeneous water/mud/grass.
    tr2 = json.load(open(os.path.join(outdir, "semantic_hetero.transcripts.json")))
    cfg2 = PointHazardConfig(
        n_semantic_zones=3, max_episode_steps=300,
        semantic_styles=("water", "mud", "grass"),
    )
    make_set("hetero", cfg2, "water", ("water", "mud", "grass"), tr2, [46, 44], outdir, want_view=True)


if __name__ == "__main__":
    main()
