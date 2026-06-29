"""Figures for the kill-shot fixes (fair soft oracle, CV detector baseline, L2).

Reads the 5-seed VLM runs of record:
  outputs/vlm5_single_l1.json (+ .transcripts.json)   single-zone, prompt level L1
  outputs/vlm5_l2.json        (+ .transcripts.json)   single-zone, prompt level L2

Produces, in outputs/:
  fig_ks_bars_l1.png        grouped bars: sem_viol + success per arm (Wilson CI)
  fig_ks_traj_seed44.png    trajectory panel: C1 / C2-hard / C2-soft / detector / B / B+
  fig_ks_detector.png       CV detector accuracy: detected (dashed) vs true (solid) zones
  fig_ks_l2.png             L2 commonsense: bars + what-the-VLM-saw (no labels)

VLM arms (B/B+) replay the recorded choices; controllers are seeded per episode and
the dynamics are deterministic, so the replay reproduces the table exactly. The
VLM-free arms (C1/C2-hard/C2-soft/detector) are re-run directly.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_pointhazard import PointHazardConfig, PointHazardEnv
from hazard_renderer import HazardRenderer
from zone_detector import estimate_zones_from_image
import subgoal_pivot_hazard as S

SAFETY_MARGIN = 0.15
OUT = "outputs"

# Display names + colors per arm.
ARM_INFO = {
    "mpc": ("C1 blind", "#9e9e9e"),
    "mpc_oracle": ("C2 hard\n(true)", "#1f77b4"),
    "mpc_oracle_soft": ("C2 soft\n(true, FAIR)", "#2ca02c"),
    "mpc_detector": ("CV detector\n(estimated)", "#8c564b"),
    "subgoal": ("B VLM", "#ff7f0e"),
    "subgoal_perceive": ("B+ VLM\n+perception", "#d62728"),
}


def _font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _titled(img, title, subtitle=""):
    bar_h = 46 if subtitle else 28
    out = Image.new("RGB", (img.width, img.height + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((8, 5), title, fill=(0, 0, 0), font=_font(18))
    if subtitle:
        d.text((8, 27), subtitle, fill=(90, 90, 90), font=_font(13))
    return out


def _hstack(panels, gap=10):
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


class ReplayVlm:
    """vlm_fn that returns recorded calls in order — no network, exact replay."""

    def __init__(self, calls):
        self.calls = calls
        self.i = 0

    def __call__(self, image, prompt, n_candidates):
        t = self.calls[self.i] if self.i < len(self.calls) else (self.calls[-1] if self.calls else {})
        self.i += 1
        return S.VlmCall(
            choice=t.get("choice"), reason=t.get("reason"), raw=t.get("raw", ""),
            parsed_ok=t.get("parsed_ok", True), api_error=t.get("api_error", False),
            attempts=t.get("attempts", 1),
        )


def _calls_for(transcripts, policy, seed):
    for ep in transcripts:
        if ep["policy"] == policy and ep["seed"] == seed:
            return ep["calls"]
    return []


def build_policy(name, cfg, renderer, transcripts, seed, *, prompt_level="L1", styles=()):
    sem_prompt = prompt_level != "L2"
    zmode = "explicit" if prompt_level == "L0" else "implicit"
    det_styles = tuple(styles) or (renderer.semantic_style,)
    if name == "mpc":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN)
    if name == "mpc_oracle":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN, semantic_aware=True)
    if name == "mpc_oracle_soft":
        return S.ControllerOnlyPolicy(cfg, "mpc", SAFETY_MARGIN, soft_oracle=True,
                                      oracle_soft_mode="bplus")
    if name == "mpc_detector":
        return S.DetectorControllerPolicy(cfg, renderer, "mpc", SAFETY_MARGIN,
                                          oracle_soft_mode="bplus", detect_styles=det_styles)
    if name in ("subgoal", "subgoal_perceive"):
        vlm_fn = ReplayVlm(_calls_for(transcripts, name, seed))
        return S.SubgoalPivotPolicy(
            cfg, renderer, pilot_mode="vlm", vlm_fn=vlm_fn, low_level="mpc",
            safety_margin=SAFETY_MARGIN, n_dirs=8, subgoal_radius=2.5,
            subgoal_horizon=15, subgoal_reach=0.6, fallback="hold", semantic=sem_prompt,
            zone_mode=zmode, perceive_zones=(name == "subgoal_perceive"),
        )
    raise ValueError(name)


def run_and_render(env, policy, seed):
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


def vlm_view(env, renderer, seed):
    obs, _info = env.reset(seed=seed)
    pos, _, _, _ = S._obs_parts(obs, env.cfg)
    base = Image.fromarray(env.render())
    subs = S.generate_subgoals(pos, 8, 2.5, env.cfg.arena_half)
    return S.annotate_subgoals(base, subs, renderer, pos)


# ---------------------------------------------------------------------------
# Bar chart of per-arm sem_viol + success with Wilson CI.
# ---------------------------------------------------------------------------
def bars(summary, arms, title, path, *, ncol_note=""):
    labels = [ARM_INFO[a][0] for a in arms]
    colors = [ARM_INFO[a][1] for a in arms]
    viol = [100 * summary[a]["semantic_violation_rate"] for a in arms]
    viol_ci = [summary[a]["semantic_violation_ci95"] for a in arms]
    succ = [100 * summary[a]["success_rate"] for a in arms]
    succ_ci = [summary[a]["success_ci95"] for a in arms]

    def err(vals, cis):
        lo = [max(0, v - 100 * c[0]) for v, c in zip(vals, cis)]
        hi = [max(0, 100 * c[1] - v) for v, c in zip(vals, cis)]
        return [lo, hi]

    x = np.arange(len(arms))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax1.bar(x, viol, color=colors, yerr=err(viol, viol_ci), capsize=4, edgecolor="k", linewidth=0.6)
    ax1.set_title("Semantic-violation rate  (lower = better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("% episodes entering a keep-out zone")
    ax2.bar(x, succ, color=colors, yerr=err(succ, succ_ci), capsize=4, edgecolor="k", linewidth=0.6)
    ax2.set_title("Goal-success rate  (higher = better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("% episodes reaching the goal")
    for ax, vals in ((ax1, viol), (ax2, succ)):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 109)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if ncol_note:
        fig.text(0.5, 0.005, ncol_note, ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Detector-accuracy overlay: render a scene, draw true (solid) vs detected (dashed).
# ---------------------------------------------------------------------------
def detector_overlay(cfg, renderer_style, styles, seeds, path):
    env = PointHazardEnv(cfg=cfg)
    rnd = HazardRenderer.from_env(env, semantic_style=renderer_style)
    env.attach_renderer(rnd)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.4 * len(seeds), 4.6))
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        env.reset(seed=seed)
        img = env.render()
        true_z = np.asarray(env.semantic_zones, np.float32).reshape(-1, 3)
        det_z = estimate_zones_from_image(img, rnd, styles=tuple(styles) or None)
        ax.imshow(img)
        # world->pixel for circle overlays
        def wp(x, y):
            return rnd.world_to_pixel(float(x), float(y))
        for z in true_z:
            cx, cy = wp(z[0], z[1]); pr = rnd.world_scale(float(z[2]))
            ax.add_patch(plt.Circle((cx, cy), pr, fill=False, color="lime", lw=2.2, label="_t"))
        for z in det_z:
            cx, cy = wp(z[0], z[1]); pr = rnd.world_scale(float(z[2]))
            ax.add_patch(plt.Circle((cx, cy), pr, fill=False, color="red", lw=2.0, ls=(0, (4, 3))))
        ax.set_title(f"seed {seed}: true={len(true_z)} detected={len(det_z)}", fontsize=11)
        ax.axis("off")
    fig.suptitle("CV detector accuracy — green = true zone, red dashed = detected from pixels",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def main():
    # ---- single-zone L1: bars + trajectory panel + detector overlay ----------
    d1 = json.load(open(os.path.join(OUT, "vlm5_single_l1.json")))
    tr1 = json.load(open(os.path.join(OUT, "vlm5_single_l1.transcripts.json")))
    arms1 = list(d1["summary"].keys())
    n = d1["config"]["episodes"]
    bars(d1["summary"], arms1,
         f"Single implicit (water) keep-out — {n}-seed VLM (gemini-3-flash), with the FAIR soft oracle + CV detector",
         os.path.join(OUT, "fig_ks_bars_l1.png"),
         ncol_note="C2-soft (true geometry, soft) and the CV detector are the honest baselines B+ must be read against.")

    cfg1 = PointHazardConfig(n_semantic_zones=1, max_episode_steps=300)
    env = PointHazardEnv(cfg=cfg1)
    rnd = HazardRenderer.from_env(env, semantic_style="water")
    env.attach_renderer(rnd)
    seed = 44
    panels = []
    for name in arms1:
        frame, sem, outc = run_and_render(env, build_policy(name, cfg1, rnd, tr1, seed), seed)
        panels.append(_titled(frame, ARM_INFO[name][0].replace("\n", " "),
                              f"in-zone steps: {sem} · {outc}"))
    _hstack(panels).save(os.path.join(OUT, f"fig_ks_traj_seed{seed}.png"))
    print("wrote", os.path.join(OUT, f"fig_ks_traj_seed{seed}.png"))

    detector_overlay(cfg1, "water", (), [44, 47],
                     os.path.join(OUT, "fig_ks_detector.png"))

    # ---- L2 commonsense: bars + what-the-VLM-saw (no labels) -----------------
    if os.path.exists(os.path.join(OUT, "vlm5_l2.json")):
        d2 = json.load(open(os.path.join(OUT, "vlm5_l2.json")))
        arms2 = list(d2["summary"].keys())
        bars(d2["summary"], arms2,
             f"L2 'no-hint' — prompt NEVER mentions terrain; does the VLM avoid water unprompted? ({n}-seed VLM)",
             os.path.join(OUT, "fig_ks_l2_bars.png"),
             ncol_note="B here is told only 'reach the goal, avoid red hazards' — any avoidance is pure commonsense.")
        env2 = PointHazardEnv(cfg=cfg1)
        rnd2 = HazardRenderer.from_env(env2, semantic_style="water")
        env2.attach_renderer(rnd2)
        view = _titled(vlm_view(env2, rnd2, 44),
                       "What the VLM sees under L2 (seed 44)",
                       "rendered water terrain, purple 1-8 = candidates · prompt says NOTHING about terrain")
        view.save(os.path.join(OUT, "fig_ks_l2_view.png"))
        print("wrote", os.path.join(OUT, "fig_ks_l2_view.png"))


if __name__ == "__main__":
    main()
