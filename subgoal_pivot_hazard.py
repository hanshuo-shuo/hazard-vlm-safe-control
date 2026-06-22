"""
subgoal_pivot_hazard.py — Month-1 "confidence experiment" for Path B.

Three-way, matched-seed comparison on the *pure-geometry* PointHazard task:

    A. direct      — VLM picks a low-level force directly (de-leaked PIVOT).
    B. subgoal     — VLM picks a discrete high-level subgoal, then the
                     provably-safe A*+PD controller (SafeExpert) executes it.
    C. safe_expert — pure SafeExpert (no VLM at all), the classical planner.

The point of this experiment is *deliberately* to show that on a fully
observable geometric toy the classical planner (C) is already near-100% safe,
that routing it with a VLM high-level (B) keeps that safety but adds nothing,
and that using the VLM as the low-level controller (A) is the unreliable one.
That negative result is the first figure of the Path B story (see
docs/PROJECT_PLAN.md §1–§2 and docs/FAILURE_MODE_ANALYSIS.md).

ANTI-LEAKAGE CONTRACT
---------------------
The VLM prompts here MUST NOT contain any safety oracle: no per-candidate
clearance, no safe/unsafe label, no score, no "this one is dangerous" hint.
The VLM gets ONLY the rendered image (hazards drawn as red circles + numbered
candidates) and a generic task description. Everything the VLM is told is
built by `build_direct_prompt` / `build_subgoal_prompt` below — keep them clean.

USAGE
-----
    # Offline smoke test (no API key; uses a geometric heuristic stand-in pilot)
    python subgoal_pivot_hazard.py --pilot_mode heuristic --episodes 5

    # Real run against an OpenRouter VLM
    OPENROUTER_API_KEY=sk-... python subgoal_pivot_hazard.py \
        --pilot_mode vlm --model google/gemini-3-flash-preview --episodes 30

Outputs a comparison table (success / hazard / timeout / mean-min-clearance /
mean VLM calls) and, with --out, a JSON dump of per-episode records.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env_pointhazard import PointHazardConfig, PointHazardEnv
from hazard_renderer import HazardRenderer
from mpc_expert import MPCConfig, MPCExpert
from pivot_vlm import _parse_pivot_selection, annotate_candidates, generate_candidates
from safe_expert import SafeExpert, SafeExpertConfig


# ---------------------------------------------------------------------------
# Obs helpers (PointHazard obs layout, see env_pointhazard.py docstring)
# ---------------------------------------------------------------------------

def _obs_parts(
    obs: np.ndarray, cfg: PointHazardConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split an obs vector into (pos, vel, goal_abs, hazards_abs)."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n = int(cfg.n_hazards)
    pos = obs[0:2].copy()
    vel = obs[2:4].copy()
    goal = obs[4:6].copy()
    hazards = obs[6 : 6 + 3 * n].reshape(n, 3).copy()
    return pos, vel, goal, hazards


def _min_clearance(pos: np.ndarray, hazards: np.ndarray, agent_radius: float) -> float:
    """Edge-to-body clearance to the nearest hazard (negative => overlap)."""
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)
    return float(np.min(center_d - hazards[:, 2] - agent_radius))


# ---------------------------------------------------------------------------
# OpenRouter VLM call (mirrors the existing *_openrouter.py scripts)
# ---------------------------------------------------------------------------

def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class VlmCall:
    """One VLM query, with everything needed to audit it later.

    A clean paper run must be able to (a) prove no leakage by showing the exact
    raw response, and (b) distinguish a genuine VLM choice from a parse failure
    or an API/network error — because a silent fallback would otherwise inject
    the classical pilot's competence into a "VLM" row and confound the table.
    """

    choice: int | None       # 0-indexed candidate, or None if not usable
    reason: str | None
    raw: str                 # raw VLM text (or the exception string)
    parsed_ok: bool          # the response parsed into a valid choice
    api_error: bool          # the API/network call itself failed
    attempts: int            # how many requests were issued (incl. retries)


def make_openrouter_vlm(
    api_key: str,
    model: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    retries: int = 1,
) -> Callable[[Image.Image, str, int], VlmCall]:
    """Return a `vlm_fn(image, prompt, n_candidates) -> VlmCall`.

    Retries on *parse* failure (re-asks for JSON); on an API exception it stops
    and reports `api_error=True` rather than silently degrading.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def vlm_fn(image: Image.Image, prompt: str, n_candidates: int) -> VlmCall:
        b64 = _image_to_base64(image)
        last_raw = ""
        last_reason: str | None = None
        last_choice: int | None = None
        for attempt in range(max(1, retries + 1)):
            p_text = prompt
            if attempt > 0:
                p_text += "\nREMINDER: Output ONLY the JSON object, starting with '{'.\n"
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                                {"type": "text", "text": p_text},
                            ],
                        }
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as exc:  # network / rate-limit / provider error
                return VlmCall(None, None, f"<api_error: {exc!r}>", False, True, attempt + 1)
            last_raw = response.choices[0].message.content or ""
            last_choice, last_reason, ok = _parse_pivot_selection(last_raw, n_candidates)
            if ok:
                return VlmCall(last_choice, last_reason, last_raw, True, False, attempt + 1)
        return VlmCall(last_choice, last_reason, last_raw, False, False, max(1, retries + 1))

    return vlm_fn


# ---------------------------------------------------------------------------
# Prompts — IMAGE + generic task only.  No clearance / label / score. (no leak)
# ---------------------------------------------------------------------------

def build_direct_prompt(n_candidates: int) -> str:
    return f"""You see a 2D arena from above.

Visual elements:
- Gray bordered square = arena boundary
- Red filled circles = HAZARDS — touching any of them ends the episode
- Blue dot = YOU (the agent you control)
- Green circle with "G" = GOAL (your destination)
- Blue line = your past trajectory
- Orange numbered arrows (1-{n_candidates}) = candidate force directions

Each arrow is a force you could apply this step; longer = stronger.

YOUR TASK: pick the single numbered arrow that best moves you toward the
green goal "G" without running into any red hazard circle. If the straight
line to the goal passes through a hazard, pick an arrow that routes around it.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


def build_subgoal_prompt(n_candidates: int) -> str:
    return f"""You see a 2D arena from above.

Visual elements:
- Gray bordered square = arena boundary
- Red filled circles = HAZARDS — the route must not cross them
- Blue dot = YOU (the agent)
- Green circle with "G" = GOAL (your final destination)
- Blue line = your past trajectory
- Purple numbered markers (1-{n_candidates}) = candidate WAYPOINTS to aim for next

A low-level controller will drive you to whichever waypoint you choose,
following a collision-free path. You only decide the general DIRECTION of travel.

YOUR TASK: pick the single numbered waypoint that makes the best next step of
a route from you to the green goal "G" while keeping clear of the red hazards.
Choose the waypoint that heads toward the goal, detouring around hazards if the
direct heading is blocked.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


# ---------------------------------------------------------------------------
# Subgoal candidate generation + annotation
# ---------------------------------------------------------------------------

def generate_subgoals(
    pos: np.ndarray,
    n_directions: int,
    radius: float,
    arena_half: float,
) -> list[np.ndarray]:
    """Ring of `n_directions` candidate waypoints at `radius` around the agent."""
    out: list[np.ndarray] = []
    lim = arena_half - 0.1
    for i in range(n_directions):
        theta = 2.0 * math.pi * i / n_directions
        p = pos + radius * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        out.append(np.clip(p, -lim, lim).astype(np.float32))
    return out


def annotate_subgoals(
    base_image: Image.Image,
    subgoals: list[np.ndarray],
    renderer: HazardRenderer,
    agent_world_xy: np.ndarray,
) -> Image.Image:
    """Draw numbered purple waypoint markers (with faint connector lines)."""
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    apx, apy = renderer.world_to_pixel(float(agent_world_xy[0]), float(agent_world_xy[1]))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    marker_color = (150, 60, 200)   # purple
    label_bg = (60, 20, 90)
    label_fg = (255, 255, 255)
    r = 11

    for idx, sg in enumerate(subgoals):
        px, py = renderer.world_to_pixel(float(sg[0]), float(sg[1]))
        draw.line([(apx, apy), (px, py)], fill=marker_color, width=1)
        lx = max(r + 2, min(img_w - r - 2, px))
        ly = max(r + 2, min(img_h - r - 2, py))
        draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=label_bg, outline=marker_color, width=2)
        label = str(idx + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((lx - tw // 2, ly - th // 2 - 1), label, fill=label_fg, font=font)

    return img


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

VlmFn = Callable[[Image.Image, str, int], VlmCall]


class _BasePolicy:
    name = "base"

    def reset(self, obs: np.ndarray, info: dict) -> None:
        self.vlm_calls = 0          # VLM queries issued
        self.vlm_parse_fail = 0     # queries that returned but failed to parse
        self.vlm_api_fail = 0       # queries where the API/network call failed
        self.fallback_used = 0      # decisions that fell back (not a VLM choice)
        self.transcripts: list[dict] = []  # per-call audit log

    def act(self, obs: np.ndarray, env: PointHazardEnv) -> np.ndarray:
        raise NotImplementedError

    def _log_call(self, kind: str, call: VlmCall, *, used_fallback: bool) -> None:
        """Record one VLM query for later auditing (leakage proof + failure modes)."""
        self.vlm_calls += 1
        self.vlm_parse_fail += int((not call.parsed_ok) and (not call.api_error))
        self.vlm_api_fail += int(call.api_error)
        self.fallback_used += int(used_fallback)
        self.transcripts.append(
            {
                "kind": kind,
                "choice": call.choice,
                "reason": call.reason,
                "parsed_ok": call.parsed_ok,
                "api_error": call.api_error,
                "attempts": call.attempts,
                "used_fallback": used_fallback,
                "raw": call.raw,
            }
        )


def make_controller(cfg: PointHazardConfig, low_level: str, safety_margin: float):
    """Build a low-level safe controller (drop-in API: plan/reset_from_obs/act)."""
    dummy = _DummyEnv(cfg)
    if low_level == "mpc":
        return MPCExpert(dummy, cfg=MPCConfig(safety_margin=safety_margin))
    if low_level == "safe_expert":
        return SafeExpert(dummy, cfg=SafeExpertConfig(safety_margin=safety_margin))
    raise ValueError(f"unknown low_level controller '{low_level}'")


class ControllerOnlyPolicy(_BasePolicy):
    """C — pure low-level controller driving straight to the goal, no VLM."""

    def __init__(self, cfg: PointHazardConfig, low_level: str, safety_margin: float):
        self.name = low_level
        self.expert = make_controller(cfg, low_level, safety_margin)

    def reset(self, obs: np.ndarray, info: dict) -> None:
        self.vlm_calls = 0
        self.expert.reset_from_obs(obs)

    def act(self, obs: np.ndarray, env: PointHazardEnv) -> np.ndarray:
        return self.expert.act(obs)


class DirectPivotPolicy(_BasePolicy):
    """A — VLM (or heuristic stand-in) selects a low-level force directly."""

    name = "direct"

    def __init__(
        self,
        cfg: PointHazardConfig,
        renderer: HazardRenderer,
        *,
        pilot_mode: str,
        vlm_fn: VlmFn | None,
        n_dirs: int,
        n_mags: int,
        arrow_len: float,
        vlm_every: int,
        fallback: str = "heuristic",
    ):
        self.cfg = cfg
        self.renderer = renderer
        self.pilot_mode = pilot_mode
        self.vlm_fn = vlm_fn
        self.candidates = generate_candidates(n_dirs, n_mags)
        self.arrow_len = arrow_len
        self.vlm_every = max(1, vlm_every)
        self.fallback = fallback

    def reset(self, obs: np.ndarray, info: dict) -> None:
        super().reset(obs, info)
        self._step = 0
        self._action = np.zeros(2, dtype=np.float32)

    def _heuristic_idx(self, pos: np.ndarray, goal: np.ndarray) -> int:
        gdir = goal - pos
        gn = float(np.linalg.norm(gdir))
        if gn < 1e-6:
            return 0
        gdir = gdir / gn
        scores = [float(np.dot(c / (np.linalg.norm(c) + 1e-9), gdir)) for c in self.candidates]
        return int(np.argmax(scores))

    def _choose(self, obs: np.ndarray, env: PointHazardEnv) -> np.ndarray:
        """Return the force action for this decision (and log any VLM call)."""
        pos, _, goal, _ = _obs_parts(obs, self.cfg)
        if self.pilot_mode == "vlm" and self.vlm_fn is not None:
            base = Image.fromarray(env.render())
            ann = annotate_candidates(base, self.candidates, self.renderer, pos, self.arrow_len)
            call = self.vlm_fn(ann, build_direct_prompt(len(self.candidates)), len(self.candidates))
            if call.parsed_ok and call.choice is not None:
                self._log_call("direct", call, used_fallback=False)
                return self.candidates[call.choice].copy()
            # VLM unusable -> explicit, logged fallback (never a silent one).
            self._log_call("direct", call, used_fallback=True)
            if self.fallback == "hold":
                return np.zeros(2, dtype=np.float32)  # neutral: no force
            return self.candidates[self._heuristic_idx(pos, goal)].copy()
        # Offline heuristic stand-in pilot (no VLM in the loop at all).
        return self.candidates[self._heuristic_idx(pos, goal)].copy()

    def act(self, obs: np.ndarray, env: PointHazardEnv) -> np.ndarray:
        if self._step % self.vlm_every == 0:
            self._action = self._choose(obs, env)
        self._step += 1
        return np.clip(self._action, -1.0, 1.0).astype(np.float32)


class SubgoalPivotPolicy(_BasePolicy):
    """B — VLM (or heuristic) picks a discrete subgoal; SafeExpert executes it."""

    name = "subgoal"

    def __init__(
        self,
        cfg: PointHazardConfig,
        renderer: HazardRenderer,
        *,
        pilot_mode: str,
        vlm_fn: VlmFn | None,
        low_level: str,
        safety_margin: float,
        n_dirs: int,
        subgoal_radius: float,
        subgoal_horizon: int,
        subgoal_reach: float,
        fallback: str = "heuristic",
    ):
        self.cfg = cfg
        self.renderer = renderer
        self.pilot_mode = pilot_mode
        self.vlm_fn = vlm_fn
        self.expert = make_controller(cfg, low_level, safety_margin)
        self.n_dirs = n_dirs
        self.subgoal_radius = subgoal_radius
        self.subgoal_horizon = max(1, subgoal_horizon)
        self.subgoal_reach = subgoal_reach
        self.fallback = fallback

    def reset(self, obs: np.ndarray, info: dict) -> None:
        super().reset(obs, info)
        self._steps_on_subgoal = self.subgoal_horizon  # force a query on step 0
        self._subgoal: np.ndarray | None = None

    def _heuristic_idx(self, pos: np.ndarray, goal: np.ndarray, subgoals: list[np.ndarray]) -> int:
        gdir = goal - pos
        gdir = gdir / (float(np.linalg.norm(gdir)) + 1e-9)
        scores = [
            float(np.dot((sg - pos) / (np.linalg.norm(sg - pos) + 1e-9), gdir))
            for sg in subgoals
        ]
        return int(np.argmax(scores))

    def _choose_subgoal(self, obs: np.ndarray, env: PointHazardEnv) -> np.ndarray:
        pos, _, goal, _ = _obs_parts(obs, self.cfg)
        subgoals = generate_subgoals(pos, self.n_dirs, self.subgoal_radius, self.cfg.arena_half)
        # If the real goal is within reach, just aim straight at it (no VLM call).
        if float(np.linalg.norm(goal - pos)) <= self.subgoal_radius:
            return goal.astype(np.float32)

        if self.pilot_mode == "vlm" and self.vlm_fn is not None:
            base = Image.fromarray(env.render())
            ann = annotate_subgoals(base, subgoals, self.renderer, pos)
            call = self.vlm_fn(ann, build_subgoal_prompt(len(subgoals)), len(subgoals))
            if call.parsed_ok and call.choice is not None:
                self._log_call("subgoal", call, used_fallback=False)
                return subgoals[call.choice]
            # VLM unusable -> explicit, logged fallback (never a silent one).
            self._log_call("subgoal", call, used_fallback=True)
            if self.fallback == "hold" and self._subgoal is not None:
                return self._subgoal  # keep the current subgoal, gain no new info
            return subgoals[self._heuristic_idx(pos, goal, subgoals)]
        # Offline heuristic stand-in pilot (no VLM in the loop at all).
        return subgoals[self._heuristic_idx(pos, goal, subgoals)]

    def act(self, obs: np.ndarray, env: PointHazardEnv) -> np.ndarray:
        pos, _, goal, hazards = _obs_parts(obs, self.cfg)
        need_new = (
            self._subgoal is None
            or self._steps_on_subgoal >= self.subgoal_horizon
            or float(np.linalg.norm(self._subgoal - pos)) <= self.subgoal_reach
        )
        if need_new:
            self._subgoal = self._choose_subgoal(obs, env)
            # Plan a collision-free path to the chosen subgoal; if that subgoal
            # is unreachable, fall back to planning straight to the real goal.
            if not self.expert.plan(pos, self._subgoal, hazards):
                self.expert.plan(pos, goal, hazards)
            self._steps_on_subgoal = 0
        self._steps_on_subgoal += 1
        return self.expert.act(obs)


class _DummyEnv:
    """SafeExpert only reads `env.cfg`; give it one without a renderer."""

    def __init__(self, cfg: PointHazardConfig):
        self.cfg = cfg


# ---------------------------------------------------------------------------
# Episode runner + metrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    policy: str
    seed: int
    outcome: str            # "goal" | "hazard" | "timeout"
    steps: int
    min_clearance: float
    vlm_calls: int
    vlm_parse_fail: int = 0
    vlm_api_fail: int = 0
    fallback_used: int = 0


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (fractions in [0, 1])."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class Aggregate:
    policy: str
    n: int = 0
    success: int = 0
    hazard: int = 0
    timeout: int = 0
    min_clearances: list[float] = field(default_factory=list)
    vlm_calls: list[int] = field(default_factory=list)
    parse_fail: int = 0
    api_fail: int = 0
    fallback_used: int = 0

    def add(self, r: EpisodeResult) -> None:
        self.n += 1
        self.success += int(r.outcome == "goal")
        self.hazard += int(r.outcome == "hazard")
        self.timeout += int(r.outcome == "timeout")
        self.min_clearances.append(r.min_clearance)
        self.vlm_calls.append(r.vlm_calls)
        self.parse_fail += r.vlm_parse_fail
        self.api_fail += r.vlm_api_fail
        self.fallback_used += r.fallback_used

    @property
    def total_calls(self) -> int:
        return int(np.sum(self.vlm_calls)) if self.vlm_calls else 0

    def row(self) -> str:
        def pct_ci(k: int) -> str:
            lo, hi = _wilson_ci(k, self.n)
            return f"{100.0 * k / max(1, self.n):5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]"

        mc = float(np.mean(self.min_clearances)) if self.min_clearances else float("nan")
        vc = float(np.mean(self.vlm_calls)) if self.vlm_calls else 0.0
        fb = self.fallback_used
        calls = self.total_calls
        fb_str = f"{100.0 * fb / calls:4.1f}%" if calls else "  -  "
        return (
            f"{self.policy:<12} {pct_ci(self.success)}  {pct_ci(self.hazard)}  "
            f"{mc:+7.3f}  {vc:6.1f}  {fb_str}"
        )


def run_episode(
    env: PointHazardEnv,
    policy: _BasePolicy,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    policy.reset(obs, info)
    agent_radius = env.cfg.agent_radius

    pos, _, _, hazards = _obs_parts(obs, env.cfg)
    min_clear = _min_clearance(pos, hazards, agent_radius)

    outcome = "timeout"
    steps = 0
    for t in range(max_steps):
        action = policy.act(obs, env)
        obs, _r, terminated, truncated, info = env.step(action)
        steps = t + 1
        pos, _, _, hazards = _obs_parts(obs, env.cfg)
        min_clear = min(min_clear, _min_clearance(pos, hazards, agent_radius))
        if terminated or truncated:
            outcome = info.get("termination_reason", "timeout")
            break

    return EpisodeResult(
        policy=policy.name,
        seed=seed,
        outcome=outcome,
        steps=steps,
        min_clearance=min_clear,
        vlm_calls=int(getattr(policy, "vlm_calls", 0)),
        vlm_parse_fail=int(getattr(policy, "vlm_parse_fail", 0)),
        vlm_api_fail=int(getattr(policy, "vlm_api_fail", 0)),
        fallback_used=int(getattr(policy, "fallback_used", 0)),
    )


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    cfg = PointHazardConfig(
        n_hazards=args.n_hazards,
        max_episode_steps=args.max_steps,
    )

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    vlm_fn: VlmFn | None = None
    if args.pilot_mode == "vlm":
        if not api_key:
            sys.exit(
                "pilot_mode=vlm requires an API key: pass --api_key or set "
                "OPENROUTER_API_KEY (or use --pilot_mode heuristic for an "
                "offline geometric stand-in pilot)."
            )
        vlm_fn = make_openrouter_vlm(
            api_key, args.model,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            retries=args.vlm_retries,
        )

    # One env + renderer, reused across policies (each reset re-seeds layout).
    env = PointHazardEnv(cfg=cfg)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)

    # Policy "C" is the pure low-level controller; it is named after --low_level.
    which = args.policies.split(",") if args.policies else [args.low_level, "subgoal", "direct"]

    def build(name: str) -> _BasePolicy:
        if name in ("mpc", "safe_expert"):
            return ControllerOnlyPolicy(cfg, low_level=name, safety_margin=args.safety_margin)
        if name == "subgoal":
            return SubgoalPivotPolicy(
                cfg, renderer, pilot_mode=args.pilot_mode, vlm_fn=vlm_fn,
                low_level=args.low_level, safety_margin=args.safety_margin,
                n_dirs=args.subgoal_n_dirs, subgoal_radius=args.subgoal_radius,
                subgoal_horizon=args.subgoal_horizon, subgoal_reach=args.subgoal_reach,
                fallback=args.vlm_fallback,
            )
        if name == "direct":
            return DirectPivotPolicy(
                cfg, renderer, pilot_mode=args.pilot_mode, vlm_fn=vlm_fn,
                n_dirs=args.pivot_n_dirs, n_mags=args.pivot_n_mags,
                arrow_len=args.pivot_arrow_len, vlm_every=args.vlm_every,
                fallback=args.vlm_fallback,
            )
        raise ValueError(f"unknown policy '{name}'")

    seeds = [args.seed + i for i in range(args.episodes)]
    aggregates = {name: Aggregate(policy=name) for name in which}
    records: list[EpisodeResult] = []
    transcripts: list[dict] = []  # per-episode VLM audit logs (when requested)

    for name in which:
        policy = build(name)
        for seed in seeds:
            r = run_episode(env, policy, seed, args.max_steps)
            aggregates[name].add(r)
            records.append(r)
            if args.log_transcripts and getattr(policy, "transcripts", None):
                transcripts.append(
                    {"policy": name, "seed": seed, "calls": list(policy.transcripts)}
                )
            if args.verbose:
                print(
                    f"  [{name:<11} seed={seed}] {r.outcome:<7} "
                    f"steps={r.steps:<3} min_clear={r.min_clearance:+.3f} "
                    f"vlm={r.vlm_calls}"
                )

    # ---- table -----------------------------------------------------------
    print()
    print(f"PointHazard matched-seed comparison  "
          f"(episodes={args.episodes}, seed0={args.seed}, pilot={args.pilot_mode}, "
          f"model={args.model if args.pilot_mode == 'vlm' else '-'})")
    print("-" * 78)
    print(f"{'policy':<12} {'success [95% CI]':>22}  {'hazard [95% CI]':>22}  "
          f"{'min_clr':>7}  {'vlm/ep':>6}  {'fb%':>5}")
    print("-" * 78)
    for name in which:
        print(aggregates[name].row())
    print("-" * 78)
    print("success/hazard = % of episodes, with Wilson 95% CI.")
    print("min_clr = mean per-episode min edge-to-body clearance (negative => collision).")
    print("vlm/ep = mean VLM queries per episode; fb% = fraction of those queries that")
    print("         fell back (parse/API failure) — a clean run keeps this near 0.")

    if args.out:
        payload = {
            "config": {
                "episodes": args.episodes, "seed0": args.seed,
                "pilot_mode": args.pilot_mode, "model": args.model,
                "n_hazards": args.n_hazards, "max_steps": args.max_steps,
                "low_level": args.low_level, "temperature": args.temperature,
                "vlm_fallback": args.vlm_fallback, "vlm_retries": args.vlm_retries,
            },
            "summary": {
                name: {
                    "n": agg.n, "success": agg.success, "hazard": agg.hazard,
                    "timeout": agg.timeout,
                    "success_rate": agg.success / max(1, agg.n),
                    "success_ci95": _wilson_ci(agg.success, agg.n),
                    "hazard_rate": agg.hazard / max(1, agg.n),
                    "hazard_ci95": _wilson_ci(agg.hazard, agg.n),
                    "mean_min_clearance": float(np.mean(agg.min_clearances)),
                    "mean_vlm_calls": float(np.mean(agg.vlm_calls)),
                    "total_vlm_calls": agg.total_calls,
                    "parse_fail": agg.parse_fail, "api_fail": agg.api_fail,
                    "fallback_used": agg.fallback_used,
                }
                for name, agg in aggregates.items()
            },
            "episodes": [vars(r) for r in records],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote per-episode records to {args.out}")
        if args.log_transcripts and transcripts:
            tpath = os.path.splitext(args.out)[0] + ".transcripts.json"
            with open(tpath, "w") as f:
                json.dump(transcripts, f, indent=2)
            print(f"Wrote {sum(len(t['calls']) for t in transcripts)} VLM "
                  f"transcripts to {tpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--n_hazards", type=int, default=8)
    p.add_argument("--policies", type=str, default="",
                   help="comma list of {mpc|safe_expert},subgoal,direct (default: all three)")
    p.add_argument("--low_level", type=str, default="mpc", choices=["mpc", "safe_expert"],
                   help="safe low-level controller used by policy C and inside subgoal (B)")

    p.add_argument("--pilot_mode", type=str, default="heuristic",
                   choices=["vlm", "heuristic"],
                   help="'vlm' = OpenRouter VLM; 'heuristic' = offline geometric stand-in")
    p.add_argument("--api_key", type=str, default="",
                   help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    p.add_argument("--model", type=str, default="google/gemini-3-flash-preview")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="VLM sampling temperature (0 = greedy, for a reproducible run)")
    p.add_argument("--vlm_retries", type=int, default=1,
                   help="re-ask the VLM this many times on a parse failure")
    p.add_argument("--vlm_fallback", type=str, default="heuristic",
                   choices=["heuristic", "hold"],
                   help="what a VLM policy does when the VLM is unusable: 'heuristic' "
                        "(goal-greedy pilot) or 'hold' (neutral: no force / keep subgoal). "
                        "Always logged via fb%; use 'hold' for the most conservative claim.")
    p.add_argument("--log_transcripts", action="store_true",
                   help="dump every raw VLM response next to --out (leakage audit trail)")

    # Shared low-level safety
    p.add_argument("--safety_margin", type=float, default=0.15)

    # Direct PIVOT (force candidates)
    p.add_argument("--pivot_n_dirs", type=int, default=8)
    p.add_argument("--pivot_n_mags", type=int, default=1)
    p.add_argument("--pivot_arrow_len", type=float, default=0.6)
    p.add_argument("--vlm_every", type=int, default=1,
                   help="re-query the direct pilot every N steps")

    # Subgoal PIVOT (waypoint candidates)
    p.add_argument("--subgoal_n_dirs", type=int, default=8)
    p.add_argument("--subgoal_radius", type=float, default=2.5)
    p.add_argument("--subgoal_horizon", type=int, default=15,
                   help="re-query the subgoal pilot at most every N steps")
    p.add_argument("--subgoal_reach", type=float, default=0.6,
                   help="treat a subgoal as reached within this distance")

    p.add_argument("--out", type=str, default="",
                   help="optional path to write a JSON dump of per-episode records")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
