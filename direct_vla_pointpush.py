"""
Qwen-backed direct VLA baseline for PointPushHazardEnv.

This is the PointPush counterpart of direct_vla_hazard.py.  It reuses the
shared Qwen backbone, action-head, LoRA, and training-loop code via imports,
and only re-implements the environment-specific parts:

    obs parsing  |  clearance / physics sim  |  rendering  |  VLA prompt
    expert collection  |  evaluation loop

The policy is identical in structure:
    Qwen-VL(image + task/state prompt) -> hidden state -> action head -> [fx, fy]

Examples:
    python direct_vla_pointpush.py --mode collect_expert \
        --episodes 1000 \
        --seed 42 \
        --out hazard/direct_vla_pointpush_expert_demos.npz

    python direct_vla_pointpush.py --mode train \
        --dataset hazard/direct_vla_pointpush_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_pointpush_head.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_pointpush.py --mode train \
        --dataset hazard/direct_vla_pointpush_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_pointpush_diffusion_head.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --action_head_type diffusion \
        --train_lora \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_pointpush.py --mode train \
        --dataset hazard/direct_vla_pointpush_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_pointpush_flow_head.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --action_head_type flow \
        --train_lora \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_pointpush.py --mode train \
        --dataset hazard/direct_vla_pointpush_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_pointpush_lora_physics_score.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --train_lora \
        --train_target physics_score \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_pointpush.py --mode eval \
        --ckpt hazard/direct_qwen_vla_pointpush_head.pt \
        --episodes 100 \
        --seed 1000 \
        --summary hazard/direct_qwen_vla_pointpush_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from env_pointpushhazard import PointPushHazardConfig, make_env
from pointpush_hazard_renderer import PointPushHazardRenderer
from pointpush_expert import PointPushExpert, PointPushExpertConfig

# ---------------------------------------------------------------------------
# Reuse generic Qwen / action-head / LoRA / training code from the Hazard VLA
# ---------------------------------------------------------------------------
from direct_vla_hazard import (
    # utilities
    set_seed,
    get_device,
    torch_dtype_from_arg,
    module_device,
    move_batch_to_device,
    finite_mean,
    format_vec,
    # dataset helpers
    qwen_vla_collate,
    load_npz_dataset,
    split_dataset,
    build_processor_inputs,
    # Qwen model
    load_qwen_backbone,
    infer_hidden_size,
    ActionHead,
    GenerativeActionMixin,
    DiffusionActionHead,
    FlowActionHead,
    action_head_type,
    make_action_head,
    make_action_head_from_ckpt,
    action_head_config,
    pool_last_token,
    qwen_forward_pooled,
    qwen_forward_action,
    # LoRA
    is_lora_parameter,
    set_qwen_trainability,
    lora_target_modules_from_arg,
    lora_config_from_args,
    apply_lora_adapter,
    extract_lora_state_dict,
    # training loop
    run_epoch,
)

TASK_TEXT = "Push the orange box into the green goal while avoiding red circular hazards."


# ---------------------------------------------------------------------------
# Physics-score candidate helpers (inlined to avoid import issues)
# ---------------------------------------------------------------------------

def parse_action_magnitudes(value: str | None, n_magnitudes: int) -> list[float]:
    if value is not None and str(value).strip():
        mags = [float(part.strip()) for part in str(value).split(",") if part.strip()]
        if not mags:
            raise ValueError("--physics_action_magnitudes was provided but parsed to an empty list.")
        return mags
    if int(n_magnitudes) == 1:
        return [1.0]
    if int(n_magnitudes) == 2:
        return [0.5, 1.0]
    return np.linspace(0.3, 1.0, int(n_magnitudes)).astype(np.float32).tolist()


def generate_physics_score_candidates(
    *,
    n_directions: int,
    n_magnitudes: int,
    magnitudes: str | None,
    include_brake: bool,
) -> list[np.ndarray]:
    mags = parse_action_magnitudes(magnitudes, n_magnitudes)
    candidates: list[np.ndarray] = []
    for i in range(max(1, int(n_directions))):
        theta = 2.0 * math.pi * i / max(1, int(n_directions))
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        for mag in mags:
            candidates.append((float(mag) * direction).astype(np.float32))
    if include_brake:
        candidates.append(np.zeros(2, dtype=np.float32))
    return candidates


# ---------------------------------------------------------------------------
# Observation, clearance, and physics helpers
# ---------------------------------------------------------------------------

def obs_parts(
    obs: np.ndarray, cfg: PointPushHazardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PointPushHazard obs -> (agent_pos, agent_vel, box_pos, box_vel, goal, hazards)."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    agent_pos = obs[0:2].astype(np.float32)
    agent_vel = obs[2:4].astype(np.float32)
    box_pos = obs[4:6].astype(np.float32)
    box_vel = obs[6:8].astype(np.float32)
    goal = obs[8:10].astype(np.float32)
    hazards = obs[10:10 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
    return agent_pos, agent_vel, box_pos, box_vel, goal, hazards


def _square_circle_clearance(
    square_xy: np.ndarray,
    half_size: float,
    circle_xy: np.ndarray,
    circle_radius: float,
) -> float:
    delta = np.asarray(circle_xy, dtype=np.float32) - np.asarray(square_xy, dtype=np.float32)
    closest = np.clip(delta, -float(half_size), float(half_size))
    outside = delta - closest
    return float(np.linalg.norm(outside) - float(circle_radius))


def agent_clearance(agent_pos: np.ndarray, hazards: np.ndarray, cfg: PointPushHazardConfig) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - agent_pos[None, :], axis=1)
    return float(np.min(center_d - hazards[:, 2] - float(cfg.agent_radius)))


def box_clearance(box_pos: np.ndarray, hazards: np.ndarray, cfg: PointPushHazardConfig) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    vals = [
        _square_circle_clearance(box_pos, cfg.box_half_size, h[:2], float(h[2]))
        for h in hazards
    ]
    return float(np.min(vals))


def obs_clearance(obs: np.ndarray, cfg: PointPushHazardConfig) -> float:
    agent_pos, _, box_pos, _, _, hazards = obs_parts(obs, cfg)
    return min(agent_clearance(agent_pos, hazards, cfg), box_clearance(box_pos, hazards, cfg))


def simulate_pointpush_step(
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    box_pos: np.ndarray,
    box_vel: np.ndarray,
    action: np.ndarray,
    cfg: PointPushHazardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One PointPushHazard dynamics step (simplified single-step for scoring)."""
    agent_pos = np.asarray(agent_pos, dtype=np.float32).reshape(2).copy()
    agent_vel = np.asarray(agent_vel, dtype=np.float32).reshape(2).copy()
    box_pos = np.asarray(box_pos, dtype=np.float32).reshape(2).copy()
    box_vel = np.asarray(box_vel, dtype=np.float32).reshape(2).copy()
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

    dt = float(cfg.dt)
    force = float(cfg.force_scale) * action

    # Agent dynamics
    agent_vel = agent_vel + (force - float(cfg.agent_drag) * agent_vel) * dt
    speed = float(np.linalg.norm(agent_vel))
    if speed > float(cfg.max_agent_speed):
        agent_vel = agent_vel * (float(cfg.max_agent_speed) / speed)

    # Box drag only (box is passive until pushed)
    box_vel = box_vel + (-float(cfg.box_drag) * box_vel) * dt
    box_speed = float(np.linalg.norm(box_vel))
    if box_speed > float(cfg.max_box_speed):
        box_vel = box_vel * (float(cfg.max_box_speed) / box_speed)

    # Position update
    agent_pos = agent_pos + agent_vel * dt
    box_pos = box_pos + box_vel * dt

    # Wall bouncing — agent
    agent_limit = float(cfg.arena_half) - float(cfg.agent_radius)
    for i in range(2):
        if agent_pos[i] < -agent_limit:
            agent_pos[i] = -agent_limit
            agent_vel[i] = -agent_vel[i] * float(cfg.wall_bounce)
        elif agent_pos[i] > agent_limit:
            agent_pos[i] = agent_limit
            agent_vel[i] = -agent_vel[i] * float(cfg.wall_bounce)

    # Wall bouncing — box
    box_limit = float(cfg.arena_half) - float(cfg.box_half_size)
    for i in range(2):
        if box_pos[i] < -box_limit:
            box_pos[i] = -box_limit
            box_vel[i] = -box_vel[i] * float(cfg.wall_bounce)
        elif box_pos[i] > box_limit:
            box_pos[i] = box_limit
            box_vel[i] = -box_vel[i] * float(cfg.wall_bounce)

    # Simplified agent–box contact (push impulse)
    rel = agent_pos - box_pos
    half = float(cfg.box_half_size)
    closest_rel = np.clip(rel, -half, half)
    delta = rel - closest_rel
    dist = float(np.linalg.norm(delta))
    if 0.0 < dist < float(cfg.agent_radius):
        normal = delta / dist
        penetration = float(cfg.agent_radius) - dist
        inv_agent = 1.0 / max(1e-6, float(cfg.agent_mass))
        inv_box = 1.0 / max(1e-6, float(cfg.box_mass))
        inv_sum = inv_agent + inv_box
        correction = (penetration / inv_sum * float(cfg.contact_percent)) * normal
        agent_pos = agent_pos + correction * inv_agent
        box_pos = box_pos - correction * inv_box
        rel_vel = agent_vel - box_vel
        vel_along = float(np.dot(rel_vel, normal))
        if vel_along < 0.0:
            impulse = (-(1.0 + float(cfg.contact_restitution)) * vel_along / inv_sum) * normal
            agent_vel = agent_vel + impulse * inv_agent
            box_vel = box_vel - impulse * inv_box

    return (
        agent_pos.astype(np.float32),
        agent_vel.astype(np.float32),
        box_pos.astype(np.float32),
        box_vel.astype(np.float32),
    )


def score_constant_action_rollout(
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
    action: np.ndarray,
    *,
    horizon: int,
    safety_margin: float,
    clearance_weight: float,
    unsafe_penalty: float,
) -> tuple[float, float, float, bool]:
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = obs_parts(obs, cfg)
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

    start_dist = float(np.linalg.norm(goal - box_pos))
    min_clear = min(agent_clearance(agent_pos, hazards, cfg), box_clearance(box_pos, hazards, cfg))
    safe = min_clear >= float(safety_margin)
    ap, av, bp, bv = agent_pos.copy(), agent_vel.copy(), box_pos.copy(), box_vel.copy()

    for _ in range(max(1, int(horizon))):
        ap, av, bp, bv = simulate_pointpush_step(ap, av, bp, bv, action, cfg)
        c = min(agent_clearance(ap, hazards, cfg), box_clearance(bp, hazards, cfg))
        min_clear = min(min_clear, c)
        if c < float(safety_margin):
            safe = False

    final_dist = float(np.linalg.norm(goal - bp))
    goal_progress = start_dist - final_dist
    score = goal_progress + float(clearance_weight) * min_clear
    if not safe:
        score -= float(unsafe_penalty)
    return float(score), float(min_clear), float(goal_progress), bool(safe)


def relabel_actions_with_physics_score(
    obs: np.ndarray,
    actions: np.ndarray,
    cfg: PointPushHazardConfig,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = generate_physics_score_candidates(
        n_directions=args.physics_n_dirs,
        n_magnitudes=args.physics_n_mags,
        magnitudes=args.physics_action_magnitudes,
        include_brake=bool(args.physics_include_brake),
    )
    if not candidates:
        raise ValueError("Physics-score relabeling needs at least one candidate action.")

    target_actions = np.empty((obs.shape[0], 2), dtype=np.float32)
    best_scores: list[float] = []
    best_clearances: list[float] = []
    best_progresses: list[float] = []
    best_safe: list[float] = []
    best_indices: list[int] = []

    t0 = time.time()
    log_every = max(1, int(args.physics_target_log_every))
    for i, ob in enumerate(obs):
        pool = candidates
        if bool(args.physics_include_dataset_action):
            pool = [np.asarray(actions[i], dtype=np.float32).reshape(2)] + candidates

        scored = [
            score_constant_action_rollout(
                ob, cfg, candidate,
                horizon=args.physics_rollout_horizon,
                safety_margin=args.physics_safety_margin,
                clearance_weight=args.physics_clearance_weight,
                unsafe_penalty=args.physics_unsafe_penalty,
            )
            for candidate in pool
        ]
        best_idx = max(
            range(len(pool)),
            key=lambda j: (scored[j][0], scored[j][1], scored[j][2]),
        )
        sc, clearance, progress, s = scored[best_idx]
        target_actions[i] = np.asarray(pool[best_idx], dtype=np.float32)
        best_scores.append(sc)
        best_clearances.append(clearance)
        best_progresses.append(progress)
        best_safe.append(float(s))
        best_indices.append(int(best_idx))

        if (i + 1) % log_every == 0 or i == int(obs.shape[0]) - 1:
            elapsed = time.time() - t0
            print(
                f"[physics_target] {i + 1}/{obs.shape[0]} "
                f"safe={finite_mean(best_safe):.1%} "
                f"score={finite_mean(best_scores):+.3f} "
                f"elapsed={elapsed:.0f}s"
            )

    deltas = np.linalg.norm(target_actions - actions.astype(np.float32), axis=1)
    summary = {
        "target_source": "physics_score",
        "n_obs": int(obs.shape[0]),
        "n_base_candidates": int(len(candidates)),
        "n_scored_candidates_per_obs": int(len(candidates) + int(bool(args.physics_include_dataset_action))),
        "include_dataset_action": bool(args.physics_include_dataset_action),
        "n_dirs": int(args.physics_n_dirs),
        "n_mags": int(args.physics_n_mags),
        "action_magnitudes": parse_action_magnitudes(args.physics_action_magnitudes, args.physics_n_mags),
        "include_brake": bool(args.physics_include_brake),
        "rollout_horizon": int(args.physics_rollout_horizon),
        "safety_margin": float(args.physics_safety_margin),
        "clearance_weight": float(args.physics_clearance_weight),
        "unsafe_penalty": float(args.physics_unsafe_penalty),
        "mean_best_score": finite_mean(best_scores),
        "mean_best_min_clearance": finite_mean(best_clearances),
        "mean_best_goal_progress": finite_mean(best_progresses),
        "safe_target_rate": finite_mean(best_safe),
        "mean_l2_delta_from_dataset_action": float(np.nanmean(deltas)) if len(deltas) else float("nan"),
        "median_l2_delta_from_dataset_action": float(np.nanmedian(deltas)) if len(deltas) else float("nan"),
        "best_candidate_histogram": {
            str(k): int(v)
            for k, v in zip(*np.unique(np.asarray(best_indices, dtype=np.int64), return_counts=True))
        },
    }
    return target_actions, summary


# ---------------------------------------------------------------------------
# Rendering and prompt
# ---------------------------------------------------------------------------

def make_renderer(cfg: PointPushHazardConfig, render_size: int) -> PointPushHazardRenderer:
    return PointPushHazardRenderer(
        arena_half=cfg.arena_half,
        agent_radius=cfg.agent_radius,
        box_half_size=cfg.box_half_size,
        goal_radius=cfg.goal_radius,
        img_size=int(render_size),
    )


def render_obs_pil(obs: np.ndarray, cfg: PointPushHazardConfig, renderer: PointPushHazardRenderer) -> Image.Image:
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = obs_parts(obs, cfg)
    arr = renderer.render(
        agent_xy=agent_pos,
        box_xy=box_pos,
        goal_xy=goal,
        hazards=hazards,
        agent_vel_xy=agent_vel,
        box_vel_xy=box_vel,
        agent_trail=None,
        box_trail=None,
        info_text=None,
    )
    return Image.fromarray(arr)


def build_vla_prompt(obs: np.ndarray, cfg: PointPushHazardConfig, prompt_mode: str = "image_state") -> str:
    prompt_mode = str(prompt_mode)
    if prompt_mode == "image_only":
        return f"""You are a direct vision-language-action controller for a 2D point-push robot.

Task: {TASK_TEXT}

Visual elements:
- Red filled circles are hazards. If either the agent or the box touches one, the episode fails.
- Blue circle is the controlled point agent.
- Orange square is the pushable box.
- Green circle marked G is the box goal.
- Blue/orange arrows show current velocities.

Predict the next continuous action [fx, fy], where each component is in [-1, 1].
The action controls the force on the blue agent. Push the box toward the goal while avoiding hazards.
"""
    if prompt_mode != "image_state":
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = obs_parts(obs, cfg)
    box_goal_vec = goal - box_pos
    agent_box_vec = box_pos - agent_pos
    dist_box_goal = float(np.linalg.norm(box_goal_vec))
    dist_agent_box = float(np.linalg.norm(agent_box_vec))
    agent_speed = float(np.linalg.norm(agent_vel))
    box_speed = float(np.linalg.norm(box_vel))
    a_clear = agent_clearance(agent_pos, hazards, cfg)
    b_clear = box_clearance(box_pos, hazards, cfg)

    nearest_idx = int(np.argmin(np.linalg.norm(hazards[:, :2] - agent_pos[None, :], axis=1)))
    nearest = hazards[nearest_idx]
    nearest_vec = nearest[:2] - agent_pos

    return f"""You are a direct vision-language-action controller for a 2D point-push robot.

Task: {TASK_TEXT}

Visual elements:
- Red filled circles are hazards. If either the agent or the box touches one, the episode fails.
- Blue circle is the controlled point agent.
- Orange square is the pushable box.
- Green circle marked G is the box goal.
- Blue/orange arrows show current velocities.

Current measured state:
- agent_position={format_vec(agent_pos)}, agent_velocity={format_vec(agent_vel)}, agent_speed={agent_speed:.3f}
- box_position={format_vec(box_pos)}, box_velocity={format_vec(box_vel)}, box_speed={box_speed:.3f}
- goal_position={format_vec(goal)}
- vector_box_to_goal={format_vec(box_goal_vec)}, dist_box_to_goal={dist_box_goal:.3f}
- vector_agent_to_box={format_vec(agent_box_vec)}, dist_agent_to_box={dist_agent_box:.3f}
- nearest_hazard_center={format_vec(nearest[:2])}, radius={float(nearest[2]):.3f}
- vector_to_nearest_hazard={format_vec(nearest_vec)}
- agent_hazard_clearance={a_clear:.3f}, box_hazard_clearance={b_clear:.3f}

Predict the next continuous action [fx, fy], where each component is in [-1, 1].
The action controls the force on the blue agent. Push the box toward the goal while avoiding hazards.
"""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QwenVLADataset(Dataset):
    """PointPush version — renders box + agent and uses push-task prompt."""

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        *,
        cfg: PointPushHazardConfig,
        render_size: int,
        prompt_mode: str,
    ):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.cfg = cfg
        self.render_size = int(render_size)
        self.prompt_mode = str(prompt_mode)
        self._renderer: PointPushHazardRenderer | None = None

        if self.obs.ndim != 2:
            raise ValueError(f"Expected obs shape (N, D), got {self.obs.shape}")
        if self.actions.ndim != 2 or self.actions.shape[1] != 2:
            raise ValueError(f"Expected actions shape (N, 2), got {self.actions.shape}")
        if self.obs.shape[0] != self.actions.shape[0]:
            raise ValueError("obs and actions must have the same number of rows")

    @property
    def renderer(self) -> PointPushHazardRenderer:
        if self._renderer is None:
            self._renderer = make_renderer(self.cfg, self.render_size)
        return self._renderer

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ob = self.obs[int(idx)]
        return {
            "image": render_obs_pil(ob, self.cfg, self.renderer),
            "prompt": build_vla_prompt(ob, self.cfg, self.prompt_mode),
            "action": self.actions[int(idx)].astype(np.float32),
        }


def cfg_from_dataset_meta(meta: dict[str, Any]) -> PointPushHazardConfig:
    raw = meta.get("env_config_json")
    if not raw:
        return PointPushHazardConfig()
    try:
        data = json.loads(str(raw))
        known = {f.name for f in PointPushHazardConfig.__dataclass_fields__.values()}
        return PointPushHazardConfig(**{k: v for k, v in data.items() if k in known})
    except (TypeError, json.JSONDecodeError):
        return PointPushHazardConfig()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def collect_expert(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    cfg = PointPushHazardConfig()
    cfg.max_episode_steps = int(args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)
    expert_cfg = PointPushExpertConfig(
        grid_res=args.expert_grid_res,
        safety_margin=args.expert_safety_margin,
        agent_safety_margin=args.expert_agent_safety_margin,
        waypoint_advance_dist=args.expert_waypoint_advance_dist,
        push_speed=args.expert_push_speed,
        approach_speed=args.expert_approach_speed,
        kp=args.expert_kp,
        lateral_gain=args.expert_lateral_gain,
        drag_feedforward=not args.no_expert_drag_feedforward,
    )
    expert = PointPushExpert(env, cfg=expert_cfg)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    ep_final_dists: list[float] = []
    ep_min_clearances: list[float] = []

    n_success = 0
    n_agent_hazard = 0
    n_box_hazard = 0
    n_timeout = 0
    n_plan_fail = 0
    n_layout_fail = 0
    n_discarded_transitions = 0
    t0 = time.time()

    for ep in range(int(args.episodes)):
        try:
            obs, info = env.reset(seed=int(args.seed) + ep)
        except RuntimeError:
            n_layout_fail += 1
            continue
        obs = np.asarray(obs, dtype=np.float32)
        plan_ok = expert.plan_box_path(env.box_pos.copy(), env.goal.copy(), env.hazards.copy())
        if not plan_ok:
            n_plan_fail += 1

        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_return = 0.0
        ep_min_clearance = obs_clearance(obs, cfg)
        last_info: dict[str, Any] = dict(info)
        hit_hazard = False
        success = False

        if plan_ok:
            for _step in range(int(args.max_steps)):
                action = expert.act(obs)
                if args.expert_action_noise > 0:
                    noise = np.random.normal(0.0, float(args.expert_action_noise), size=action.shape)
                    action = np.clip(action + noise, -1.0, 1.0).astype(np.float32)

                ep_obs.append(obs.copy())
                ep_actions.append(action.copy())

                obs, reward, terminated, truncated, info = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                last_info = dict(info)
                ep_return += float(reward)
                ep_min_clearance = min(ep_min_clearance, obs_clearance(obs, cfg))

                if terminated:
                    reason = info.get("termination_reason", "")
                    if reason == "goal":
                        success = True
                    elif reason in ("agent_hazard", "box_hazard"):
                        hit_hazard = True
                    break
                if truncated:
                    break

        if hit_hazard:
            reason = last_info.get("termination_reason", "hazard")
            if reason == "agent_hazard":
                n_agent_hazard += 1
            else:
                n_box_hazard += 1
        elif success:
            n_success += 1
        else:
            n_timeout += 1

        if plan_ok and (args.keep_hazard_episodes or not hit_hazard):
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
        else:
            n_discarded_transitions += len(ep_obs)

        ep_returns.append(float(ep_return))
        ep_steps.append(len(ep_obs))
        ep_final_dists.append(float(last_info.get("dist_box_to_goal", np.nan)))
        ep_min_clearances.append(float(ep_min_clearance))

        if (ep + 1) % int(args.log_every) == 0 or ep == int(args.episodes) - 1:
            elapsed = time.time() - t0
            n_hazard_total = n_agent_hazard + n_box_hazard
            print(
                f"[collect_expert] ep {ep + 1}/{args.episodes} "
                f"success={n_success} agent_hazard={n_agent_hazard} "
                f"box_hazard={n_box_hazard} timeout={n_timeout} "
                f"plan_fail={n_plan_fail} layout_fail={n_layout_fail} "
                f"kept_transitions={len(all_obs)} elapsed={elapsed:.0f}s"
            )

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    action_arr = np.asarray(all_actions, dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_hazard_total = n_agent_hazard + n_box_hazard
    metadata = np.array(
        [
            int(args.episodes),
            n_success,
            n_hazard_total,
            n_timeout,
            n_plan_fail,
            int(len(all_obs)),
            int(n_discarded_transitions),
            int(args.max_steps),
        ],
        dtype=np.int64,
    )
    np.savez_compressed(
        out_path,
        obs=obs_arr,
        actions=action_arr,
        metadata=metadata,
        env_config_json=np.array(json.dumps(asdict(cfg), sort_keys=True)),
        expert_config_json=np.array(json.dumps(asdict(expert_cfg), sort_keys=True)),
        task_text=np.array(TASK_TEXT),
    )

    summary = {
        "mode": "collect_expert",
        "out": str(out_path),
        "task_text": TASK_TEXT,
        "episodes": int(args.episodes),
        "success_rate": n_success / max(1, int(args.episodes)),
        "agent_hazard_hit_rate": n_agent_hazard / max(1, int(args.episodes)),
        "box_hazard_hit_rate": n_box_hazard / max(1, int(args.episodes)),
        "hazard_hit_rate": n_hazard_total / max(1, int(args.episodes)),
        "timeout_rate": n_timeout / max(1, int(args.episodes)),
        "plan_fail_rate": n_plan_fail / max(1, int(args.episodes)),
        "layout_fail": int(n_layout_fail),
        "kept_transitions": int(len(all_obs)),
        "discarded_transitions": int(n_discarded_transitions),
        "mean_return": finite_mean(ep_returns),
        "mean_steps": finite_mean([float(x) for x in ep_steps]),
        "mean_final_dist": finite_mean(ep_final_dists),
        "mean_min_clearance": finite_mean(ep_min_clearances),
        "env_config": asdict(cfg),
        "expert_config": asdict(expert_cfg),
        "args": vars(args),
    }
    summary_path = Path(args.collect_summary) if args.collect_summary else out_path.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPointPush expert dataset saved: {out_path}")
    print(f"Summary saved: {summary_path}")
    print(f"  obs={obs_arr.shape} actions={action_arr.shape}")
    print(
        f"  success={summary['success_rate']:.1%} "
        f"hazard={summary['hazard_hit_rate']:.1%} "
        f"timeout={summary['timeout_rate']:.1%} "
        f"transitions={len(all_obs)}"
    )


def save_checkpoint(
    *,
    path: Path,
    qwen,
    action_head: nn.Module,
    cfg: PointPushHazardConfig,
    args: argparse.Namespace,
    hidden_size: int,
    dataset_meta: dict[str, Any],
    history: list[dict[str, float | int]],
    best_epoch: int,
    best_loss: float,
    physics_target_summary: dict[str, Any] | None,
) -> None:
    payload = {
        "action_head_state": action_head.state_dict(),
        "action_head_type": action_head_type(action_head),
        "action_head_config": action_head_config(action_head, args, hidden_size),
        "model_path": args.model_path,
        "trust_remote_code": bool(args.trust_remote_code),
        "torch_dtype": args.torch_dtype,
        "env_config": asdict(cfg),
        "env_type": "PointPushHazard",
        "render_size": int(args.render_size),
        "task_text": TASK_TEXT,
        "prompt_mode": str(args.prompt_mode),
        "uses_pivot": False,
        "uses_expert_training": str(args.train_target) == "expert",
        "uses_learned_physics": False,
        "uses_physics_score_training": str(args.train_target) == "physics_score",
        "train_target": str(args.train_target),
        "physics_target_summary": physics_target_summary,
        "qwen_backbone": True,
        "train_vlm": bool(args.train_vlm),
        "train_lora": bool(args.train_lora),
        "lora_config": lora_config_from_args(args) if args.train_lora else None,
        "qwen_lora_state": extract_lora_state_dict(qwen) if args.train_lora else None,
        "dataset": args.dataset,
        "dataset_meta": dataset_meta,
        "train_args": vars(args),
        "history": history,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
    }
    torch.save(payload, path)


def train_policy(args: argparse.Namespace) -> None:
    if bool(args.train_vlm) and bool(args.train_lora):
        raise ValueError("--train_vlm and --train_lora are mutually exclusive.")

    set_seed(args.seed)
    obs, actions, dataset_meta = load_npz_dataset(args.dataset)
    cfg = cfg_from_dataset_meta(dataset_meta)

    expected_obs_dim = 4 + 4 + 2 + 3 * int(cfg.n_hazards)
    if obs.shape[1] != expected_obs_dim:
        raise ValueError(f"Expected obs_dim={expected_obs_dim}, got {obs.shape[1]}")

    physics_target_summary: dict[str, Any] | None = None
    if str(args.train_target) == "physics_score":
        print("Relabeling training actions with physics-score teacher")
        actions, physics_target_summary = relabel_actions_with_physics_score(obs, actions, cfg, args)

    train_obs, train_actions, val_obs, val_actions = split_dataset(
        obs,
        actions,
        val_frac=args.val_frac,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
    )

    train_ds = QwenVLADataset(
        train_obs, train_actions, cfg=cfg, render_size=args.render_size, prompt_mode=args.prompt_mode,
    )
    val_ds = QwenVLADataset(
        val_obs, val_actions, cfg=cfg, render_size=args.render_size, prompt_mode=args.prompt_mode,
    ) if len(val_obs) > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=qwen_vla_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=qwen_vla_collate, drop_last=False,
    ) if val_ds is not None else None

    qwen, processor = load_qwen_backbone(args)
    hidden_size = infer_hidden_size(qwen)
    if args.train_lora:
        qwen = apply_lora_adapter(qwen, args, training=True)
    action_head = make_action_head(args, hidden_size).to(module_device(qwen))
    set_qwen_trainability(qwen, train_vlm=bool(args.train_vlm), train_lora=bool(args.train_lora))

    param_groups: list[dict[str, Any]] = [
        {"params": list(action_head.parameters()), "lr": float(args.lr)}
    ]
    qwen_trainable = [p for p in qwen.parameters() if p.requires_grad]
    if args.train_lora and not qwen_trainable:
        raise RuntimeError(
            "LoRA was requested but no LoRA parameters are trainable. "
            "Check --lora_target_modules for this Qwen model."
        )
    if qwen_trainable:
        qwen_lr = float(args.lora_lr if args.train_lora else args.lr)
        param_groups.append({"params": qwen_trainable, "lr": qwen_lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    print("Training Qwen direct VLA (PointPush)")
    print(f"  model_path={args.model_path}")
    print(f"  action_head_type={action_head_type(action_head)}")
    if isinstance(action_head, DiffusionActionHead):
        print(
            f"  diffusion_steps={action_head.num_steps} "
            f"beta={action_head.beta_schedule}[{action_head.beta_min:g},{action_head.beta_max:g}] "
            f"cond_dim={action_head.cond_dim}"
        )
    elif isinstance(action_head, FlowActionHead):
        print(f"  flow_sample_steps={action_head.sample_steps} cond_dim={action_head.cond_dim}")
    print(f"  train_vlm={bool(args.train_vlm)}")
    print(f"  train_lora={bool(args.train_lora)}")
    if args.train_lora:
        print(
            f"  lora_rank={args.lora_rank} lora_alpha={args.lora_alpha} "
            f"lora_lr={args.lora_lr} targets={args.lora_target_modules}"
        )
    print(f"  train_target={args.train_target}")
    if physics_target_summary is not None:
        print(
            "  physics_score_target="
            f"horizon={physics_target_summary['rollout_horizon']} "
            f"clearance_weight={physics_target_summary['clearance_weight']} "
            f"safe_rate={physics_target_summary['safe_target_rate']:.1%} "
            f"mean_score={physics_target_summary['mean_best_score']:+.3f} "
            f"mean_delta_from_dataset_action={physics_target_summary['mean_l2_delta_from_dataset_action']:.3f}"
        )
    print(f"  dataset={args.dataset}")
    print(f"  train={len(train_ds)} val={0 if val_ds is None else len(val_ds)}")
    print(f"  batch_size={args.batch_size} epochs={args.epochs} render_size={args.render_size}")
    print(f"  prompt_mode={args.prompt_mode}")

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        train_loss = run_epoch(
            qwen=qwen, action_head=action_head, processor=processor,
            loader=train_loader, optimizer=optimizer,
            train_vlm=bool(args.train_vlm), train_lora=bool(args.train_lora),
            grad_clip=args.grad_clip,
        )
        val_loss = float("nan")
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(
                    qwen=qwen, action_head=action_head, processor=processor,
                    loader=val_loader, optimizer=None,
                    train_vlm=False, train_lora=False, grad_clip=0.0,
                )
        monitor = val_loss if math.isfinite(val_loss) else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if monitor < best_loss:
            best_loss = monitor
            best_epoch = epoch
            save_checkpoint(
                path=ckpt_path, qwen=qwen, action_head=action_head,
                cfg=cfg, args=args, hidden_size=hidden_size,
                dataset_meta=dataset_meta, history=history,
                best_epoch=best_epoch, best_loss=best_loss,
                physics_target_summary=physics_target_summary,
            )

        elapsed = time.time() - t0
        print(
            f"[train] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_epoch={best_epoch} elapsed={elapsed:.1f}s"
        )

    print(f"Best Qwen VLA action-head checkpoint saved: {ckpt_path}")


def load_qwen_vla_from_ckpt(args: argparse.Namespace):
    device = get_device(args.device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)

    if not args.model_path:
        args.model_path = ckpt.get("model_path", "Qwen/Qwen2-VL-7B-Instruct")
    if args.torch_dtype == "from_ckpt":
        args.torch_dtype = ckpt.get("torch_dtype", "bfloat16")
    if args.trust_remote_code is False:
        args.trust_remote_code = bool(ckpt.get("trust_remote_code", False))
    if args.prompt_mode == "from_ckpt":
        args.prompt_mode = ckpt.get("prompt_mode", "image_state")

    qwen, processor = load_qwen_backbone(args)
    if bool(ckpt.get("train_lora", False)):
        qwen = apply_lora_adapter(
            qwen, args, training=False, lora_config=ckpt.get("lora_config") or None,
        )
        lora_state = ckpt.get("qwen_lora_state", None)
        if not lora_state:
            raise RuntimeError("Checkpoint is marked train_lora=True but has no qwen_lora_state.")
        incompatible = qwen.load_state_dict(lora_state, strict=False)
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if unexpected:
            print(f"Warning: unexpected LoRA checkpoint keys: {unexpected[:8]}")

    action_head = make_action_head_from_ckpt(ckpt).to(module_device(qwen))
    action_head.load_state_dict(ckpt["action_head_state"])
    action_head.eval()
    set_qwen_trainability(qwen, train_vlm=False, train_lora=False)
    return qwen, processor, action_head, ckpt


@torch.no_grad()
def predict_action(
    *,
    qwen,
    processor,
    action_head: nn.Module,
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
    renderer: PointPushHazardRenderer,
    prompt_mode: str,
    action_samples: int = 1,
) -> np.ndarray:
    image = render_obs_pil(obs, cfg, renderer)
    prompt = build_vla_prompt(obs, cfg, prompt_mode)
    pred = qwen_forward_action(
        qwen=qwen, action_head=action_head, processor=processor,
        images=[image], prompts=[prompt],
        train_vlm=False, action_samples=action_samples,
    )
    action = pred.squeeze(0).detach().cpu().numpy()
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def eval_policy(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    qwen, processor, action_head, ckpt = load_qwen_vla_from_ckpt(args)

    cfg_data = ckpt.get("env_config", asdict(PointPushHazardConfig()))
    cfg = cfg_from_dataset_meta({"env_config_json": json.dumps(cfg_data)}) if cfg_data else PointPushHazardConfig()
    cfg.max_episode_steps = int(args.max_steps)
    render_size = int(args.render_size if args.render_size > 0 else ckpt.get("render_size", 128))
    renderer = make_renderer(cfg, render_size)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)

    n_success = 0
    n_agent_hazard = 0
    n_box_hazard = 0
    n_timeout = 0
    ep_returns: list[float] = []
    ep_steps: list[float] = []
    ep_final_dists: list[float] = []
    ep_min_clearances: list[float] = []
    ep_action_norms: list[float] = []
    episodes_log: list[dict[str, Any]] = []

    n_layout_fail = 0
    t0 = time.time()
    for ep in range(int(args.episodes)):
        try:
            obs, info = env.reset(seed=int(args.seed) + ep)
        except RuntimeError:
            n_layout_fail += 1
            continue
        obs = np.asarray(obs, dtype=np.float32)
        ep_return = 0.0
        ep_min_clearance = obs_clearance(obs, cfg)
        action_norms: list[float] = []
        last_info: dict[str, Any] = dict(info)
        outcome = "timeout"
        steps = 0

        for t in range(int(args.max_steps)):
            action = predict_action(
                qwen=qwen, processor=processor, action_head=action_head,
                obs=obs, cfg=cfg, renderer=renderer,
                prompt_mode=args.prompt_mode, action_samples=args.action_samples,
            )
            action_norms.append(float(np.linalg.norm(action)))
            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            last_info = dict(info)
            ep_return += float(reward)
            ep_min_clearance = min(ep_min_clearance, obs_clearance(obs, cfg))
            steps = t + 1

            if terminated:
                reason = info.get("termination_reason", "")
                if reason == "goal":
                    outcome = "success"
                    n_success += 1
                elif reason == "agent_hazard":
                    outcome = "agent_hazard"
                    n_agent_hazard += 1
                elif reason == "box_hazard":
                    outcome = "box_hazard"
                    n_box_hazard += 1
                break
            if truncated:
                outcome = "timeout"
                n_timeout += 1
                break
        else:
            outcome = "timeout"
            n_timeout += 1

        final_dist = float(last_info.get("dist_box_to_goal", np.nan))
        mean_action_norm = finite_mean(action_norms)
        ep_returns.append(float(ep_return))
        ep_steps.append(float(steps))
        ep_final_dists.append(final_dist)
        ep_min_clearances.append(float(ep_min_clearance))
        ep_action_norms.append(float(mean_action_norm))

        record = {
            "episode": ep + 1,
            "seed": int(args.seed) + ep,
            "outcome": outcome,
            "return": float(ep_return),
            "steps": int(steps),
            "final_dist": final_dist,
            "min_clearance": float(ep_min_clearance),
            "mean_action_norm": float(mean_action_norm),
        }
        episodes_log.append(record)

        if args.debug_print or (ep + 1) % int(args.log_every) == 0 or ep == int(args.episodes) - 1:
            elapsed = time.time() - t0
            print(
                f"[eval] ep {ep + 1}/{args.episodes} outcome={outcome} "
                f"return={ep_return:.2f} steps={steps} dist={final_dist:.2f} "
                f"min_clear={ep_min_clearance:.2f} elapsed={elapsed:.0f}s"
            )

    n_hazard_total = n_agent_hazard + n_box_hazard
    summary = {
        "method": "direct_qwen_vla_pointpush",
        "action_head_type": str(ckpt.get("action_head_type", "regression")),
        "action_samples": int(args.action_samples),
        "model_path": ckpt.get("model_path", args.model_path),
        "qwen_backbone": True,
        "task_text": ckpt.get("task_text", TASK_TEXT),
        "prompt_mode": str(args.prompt_mode),
        "uses_pivot": False,
        "uses_expert_training": bool(ckpt.get("uses_expert_training", True)),
        "uses_learned_physics": False,
        "uses_physics_score_training": bool(ckpt.get("uses_physics_score_training", False)),
        "train_target": ckpt.get("train_target", "expert"),
        "physics_target_summary": ckpt.get("physics_target_summary", None),
        "train_vlm": bool(ckpt.get("train_vlm", False)),
        "train_lora": bool(ckpt.get("train_lora", False)),
        "lora_config": ckpt.get("lora_config", None),
        "ckpt": str(args.ckpt),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "render_size": int(render_size),
        "success_rate": n_success / max(1, int(args.episodes)),
        "agent_hazard_hit_rate": n_agent_hazard / max(1, int(args.episodes)),
        "box_hazard_hit_rate": n_box_hazard / max(1, int(args.episodes)),
        "hazard_hit_rate": n_hazard_total / max(1, int(args.episodes)),
        "timeout_rate": n_timeout / max(1, int(args.episodes)),
        "layout_fail": int(n_layout_fail),
        "mean_return": finite_mean(ep_returns),
        "mean_steps": finite_mean(ep_steps),
        "mean_final_dist": finite_mean(ep_final_dists),
        "mean_min_clearance": finite_mean(ep_min_clearances),
        "mean_action_norm": finite_mean(ep_action_norms),
        "train_dataset": ckpt.get("dataset", ""),
        "train_dataset_meta": ckpt.get("dataset_meta", {}),
        "best_epoch": ckpt.get("best_epoch", None),
        "best_loss": ckpt.get("best_loss", None),
        "env_config": asdict(cfg),
        "episodes_log": episodes_log,
        "args": vars(args),
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if args.episode_log:
        log_path = Path(args.episode_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            for record in episodes_log:
                f.write(json.dumps(record) + "\n")

    print("\nDirect Qwen VLA (PointPush) eval complete")
    print(f"Summary saved: {summary_path}")
    print(f"  success={summary['success_rate']:.1%}")
    print(f"  agent_hazard={summary['agent_hazard_hit_rate']:.1%}")
    print(f"  box_hazard={summary['box_hazard_hit_rate']:.1%}")
    print(f"  timeout={summary['timeout_rate']:.1%}")
    print(f"  return={summary['mean_return']:.2f}")
    print(f"  final_dist={summary['mean_final_dist']:.2f}")
    print(f"  min_clearance={summary['mean_min_clearance']:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen-backed direct VLA baseline for PointPushHazardEnv")
    parser.add_argument("--mode", type=str, required=True, choices=["collect_expert", "train", "eval"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=350)
    parser.add_argument("--render_size", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=20)

    # Qwen backbone.
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--prompt_mode", type=str, default="image_state",
        choices=["image_only", "image_state", "from_ckpt"],
    )

    # collect_expert.
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out", type=str, default="hazard/direct_vla_pointpush_expert_demos.npz")
    parser.add_argument("--collect_summary", type=str, default="")
    parser.add_argument("--keep_hazard_episodes", action="store_true")
    parser.add_argument("--expert_action_noise", type=float, default=0.0)
    parser.add_argument("--expert_grid_res", type=int, default=90)
    parser.add_argument("--expert_safety_margin", type=float, default=0.22)
    parser.add_argument("--expert_agent_safety_margin", type=float, default=0.18)
    parser.add_argument("--expert_waypoint_advance_dist", type=float, default=0.45)
    parser.add_argument("--expert_push_speed", type=float, default=1.8)
    parser.add_argument("--expert_approach_speed", type=float, default=2.1)
    parser.add_argument("--expert_kp", type=float, default=3.2)
    parser.add_argument("--expert_lateral_gain", type=float, default=1.5)
    parser.add_argument("--no_expert_drag_feedforward", action="store_true")

    # train.
    parser.add_argument("--dataset", type=str, default="hazard/direct_vla_pointpush_expert_demos.npz")
    parser.add_argument("--ckpt", type=str, default="hazard/direct_qwen_vla_pointpush_head.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--train_target", type=str, default="expert",
        choices=["expert", "physics_score"],
    )
    parser.add_argument("--physics_rollout_horizon", type=int, default=12)
    parser.add_argument("--physics_clearance_weight", type=float, default=0.25)
    parser.add_argument("--physics_safety_margin", type=float, default=0.15)
    parser.add_argument("--physics_unsafe_penalty", type=float, default=10.0)
    parser.add_argument("--physics_n_dirs", type=int, default=8)
    parser.add_argument("--physics_n_mags", type=int, default=2)
    parser.add_argument("--physics_action_magnitudes", type=str, default="")
    parser.add_argument("--physics_include_brake", action="store_true")
    parser.add_argument("--physics_include_dataset_action", action="store_true")
    parser.add_argument("--physics_target_log_every", type=int, default=5000)
    parser.add_argument(
        "--action_head_type", type=str, default="regression",
        choices=["regression", "diffusion", "flow"],
    )
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dropout", type=float, default=0.05)
    parser.add_argument("--generative_cond_dim", type=int, default=256)
    parser.add_argument("--generative_time_dim", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--diffusion_beta_schedule", type=str, default="sigmoid", choices=["sigmoid", "linear"])
    parser.add_argument("--diffusion_beta_min", type=float, default=1e-4)
    parser.add_argument("--diffusion_beta_max", type=float, default=0.02)
    parser.add_argument("--flow_sample_steps", type=int, default=16)
    parser.add_argument("--train_vlm", action="store_true")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)

    # eval.
    parser.add_argument("--action_samples", type=int, default=1)
    parser.add_argument("--summary", type=str, default="hazard/direct_qwen_vla_pointpush_eval_summary.json")
    parser.add_argument("--episode_log", type=str, default="")
    parser.add_argument("--debug_print", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_train_samples <= 0:
        args.max_train_samples = None
    args.action_samples = max(1, int(args.action_samples))
    if args.train_vlm and args.train_lora:
        raise ValueError("--train_vlm and --train_lora are mutually exclusive.")

    if args.mode == "collect_expert":
        collect_expert(args)
    elif args.mode == "train":
        train_policy(args)
    elif args.mode == "eval":
        eval_policy(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
