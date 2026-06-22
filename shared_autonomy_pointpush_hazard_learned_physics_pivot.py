"""
Learned-physics PIVOT controller for PointPushHazardEnv.

This is the push-task sibling of shared_autonomy_hazard_learned_physics_pivot.py:
  - No diffusion.
  - No oracle future rollouts in the PIVOT path.
  - A small online torch model learns one-step agent+box physics from real
    executed transitions: (obs_t, action_t) -> delta(agent, box).
  - Candidate force actions are recursively rolled out through the learned
    model and drawn over the image for local/VLM selection.

Examples:
    python shared_autonomy_pointpush_hazard_learned_physics_pivot.py --smoke_test

    python shared_autonomy_pointpush_hazard_learned_physics_pivot.py \\
        --pilot_mode local --episodes 5 --save_gif

    python shared_autonomy_pointpush_hazard_learned_physics_pivot.py \\
        --pilot_mode vlm --model_path Qwen/Qwen3-VL-32B-Instruct --episodes 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from env_pointpushhazard import PointPushHazardConfig, make_env
from pivot_vlm import _call_vlm_select, annotate_candidates, generate_candidates
from pointpush_hazard_renderer import PointPushHazardRenderer


# ---------------------------------------------------------------------------
# VLM loading
# ---------------------------------------------------------------------------

def _torch_dtype_from_arg(name: str):
    name = str(name).lower()
    if name in ("auto", "none"):
        return "auto"
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


def _load_local_vlm(args: argparse.Namespace):
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "VLM mode requires transformers. Use --pilot_mode local for local smoke/eval."
        ) from exc

    print(f"Loading local VLM: {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    dtype = _torch_dtype_from_arg(args.torch_dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype if dtype != "auto" else "auto",
    )
    model.eval()
    print("Local VLM loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _format_vec(v: np.ndarray) -> str:
    return f"[{float(v[0]):+.2f}, {float(v[1]):+.2f}]"


def _obs_parts(
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    agent_pos = obs[0:2].astype(np.float32)
    agent_vel = obs[2:4].astype(np.float32)
    box_pos = obs[4:6].astype(np.float32)
    box_vel = obs[6:8].astype(np.float32)
    goal = obs[8:10].astype(np.float32)
    hazards = obs[10 : 10 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
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


def _agent_clearance(agent_pos: np.ndarray, hazards: np.ndarray, cfg: PointPushHazardConfig) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - agent_pos[None, :], axis=1)
    return float(np.min(center_d - hazards[:, 2] - float(cfg.agent_radius)))


def _box_clearance(box_pos: np.ndarray, hazards: np.ndarray, cfg: PointPushHazardConfig) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    vals = [
        _square_circle_clearance(box_pos, cfg.box_half_size, h[:2], float(h[2]))
        for h in hazards
    ]
    return float(np.min(vals))


def _system_clearance(
    agent_pos: np.ndarray,
    box_pos: np.ndarray,
    hazards: np.ndarray,
    cfg: PointPushHazardConfig,
) -> float:
    return min(_agent_clearance(agent_pos, hazards, cfg), _box_clearance(box_pos, hazards, cfg))


def _push_pose_for_box(
    box_pos: np.ndarray,
    goal: np.ndarray,
    cfg: PointPushHazardConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    to_goal = np.asarray(goal, dtype=np.float32) - np.asarray(box_pos, dtype=np.float32)
    goal_dist = float(np.linalg.norm(to_goal))
    if goal_dist > 1e-6:
        push_dir = (to_goal / goal_dist).astype(np.float32)
    else:
        push_dir = np.zeros(2, dtype=np.float32)
    push_gap = float(cfg.box_half_size) + float(cfg.agent_radius) + 0.18
    push_pose = np.asarray(box_pos, dtype=np.float32) - push_dir * push_gap
    arena_limit = float(cfg.arena_half) - float(cfg.agent_radius)
    push_pose = np.clip(push_pose, -arena_limit, arena_limit).astype(np.float32)
    return push_pose, push_dir, push_gap


def _push_pose_metrics(
    agent_pos: np.ndarray,
    box_pos: np.ndarray,
    goal: np.ndarray,
    cfg: PointPushHazardConfig,
) -> tuple[float, float, float, float]:
    push_pose, push_dir, push_gap = _push_pose_for_box(box_pos, goal, cfg)
    dist_to_push_pose = float(np.linalg.norm(np.asarray(agent_pos, dtype=np.float32) - push_pose))
    if float(np.linalg.norm(push_dir)) < 1e-6:
        return dist_to_push_pose, 0.0, 0.0, 0.0

    agent_to_box = np.asarray(box_pos, dtype=np.float32) - np.asarray(agent_pos, dtype=np.float32)
    behind = float(np.dot(agent_to_box, push_dir))
    lateral = agent_to_box - behind * push_dir
    lateral_err = float(np.linalg.norm(lateral))

    slot_dist_score = max(0.0, 1.0 - dist_to_push_pose / max(0.75, push_gap))
    behind_score = max(0.0, 1.0 - abs(behind - push_gap) / max(1e-6, push_gap))
    lateral_score = max(0.0, 1.0 - lateral_err / 0.45)
    slot_score = slot_dist_score * (0.5 + 0.25 * behind_score + 0.25 * lateral_score)
    return dist_to_push_pose, behind, lateral_err, float(slot_score)


def _sort_hazards_for_obs(
    agent_pos: np.ndarray,
    box_pos: np.ndarray,
    hazards: np.ndarray,
    cfg: PointPushHazardConfig,
) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return hazards
    agent_d = np.linalg.norm(hazards[:, :2] - agent_pos[None, :], axis=1)
    agent_clear = agent_d - hazards[:, 2] - float(cfg.agent_radius)
    box_clear = np.asarray([
        _square_circle_clearance(box_pos, cfg.box_half_size, h[:2], float(h[2]))
        for h in hazards
    ], dtype=np.float32)
    order = np.argsort(np.minimum(agent_clear, box_clear))
    return hazards[order].astype(np.float32)


def _build_obs_from_parts(
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    box_pos: np.ndarray,
    box_vel: np.ndarray,
    goal: np.ndarray,
    hazards: np.ndarray,
    cfg: PointPushHazardConfig,
) -> np.ndarray:
    haz_sorted = _sort_hazards_for_obs(agent_pos, box_pos, hazards, cfg)
    obs = np.empty(10 + 3 * int(cfg.n_hazards), dtype=np.float32)
    obs[0:2] = np.asarray(agent_pos, dtype=np.float32)
    obs[2:4] = np.asarray(agent_vel, dtype=np.float32)
    obs[4:6] = np.asarray(box_pos, dtype=np.float32)
    obs[6:8] = np.asarray(box_vel, dtype=np.float32)
    obs[8:10] = np.asarray(goal, dtype=np.float32)
    obs[10:] = haz_sorted.reshape(-1)
    return obs


def _local_push_heuristic(
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
    candidates: list[np.ndarray],
) -> int:
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)
    to_goal = goal - box_pos
    goal_dist = float(np.linalg.norm(to_goal))
    push_dir = to_goal / goal_dist if goal_dist > 1e-6 else np.zeros(2, dtype=np.float32)
    push_pose_dist = float(cfg.box_half_size) + float(cfg.agent_radius) + 0.18
    push_pose = box_pos - push_dir * push_pose_dist
    to_push_pose = push_pose - agent_pos
    agent_to_box = box_pos - agent_pos
    behind = float(np.dot(agent_to_box, push_dir))
    lateral = agent_to_box - behind * push_dir

    if float(np.linalg.norm(to_push_pose)) > 0.20 or behind < 0.25 or float(np.linalg.norm(lateral)) > 0.35:
        desired = to_push_pose - 0.25 * agent_vel
    else:
        desired = 1.25 * push_dir + 1.2 * lateral + 0.15 * box_vel - 0.15 * agent_vel

    # Small repulsion keeps the warmup heuristic from scraping hazards.
    for hx, hy, hr in hazards:
        away = agent_pos - np.array([hx, hy], dtype=np.float32)
        dist = float(np.linalg.norm(away))
        if dist > 1e-6:
            clearance = dist - float(hr) - float(cfg.agent_radius)
            if clearance < 0.55:
                desired += away / dist * (0.55 - clearance) * 2.0

    if float(np.linalg.norm(desired)) < 1e-6:
        return 0
    desired_unit = desired / float(np.linalg.norm(desired))
    scores = []
    for cand in candidates:
        c = np.asarray(cand, dtype=np.float32)
        cn = float(np.linalg.norm(c))
        scores.append(float(np.dot(c / max(1e-6, cn), desired_unit)))
    return int(np.argmax(scores))


def _real_step_feedback(
    *,
    choice_idx: int,
    action: np.ndarray,
    prev_box_dist: float,
    next_box_dist: float,
    prev_clearance: float,
    next_clearance: float,
    agent_hazard_hit: bool,
    box_hazard_hit: bool,
    goal_success: bool,
    physics_loss: float | None,
) -> str:
    status = (
        "goal_success"
        if goal_success
        else ("agent_hazard" if agent_hazard_hit else ("box_hazard" if box_hazard_hit else "running"))
    )
    loss_text = "n/a" if physics_loss is None else f"{physics_loss:.5f}"
    choice = choice_idx + 1 if choice_idx >= 0 else 0
    return (
        f"Last chosen candidate={choice}, force={_format_vec(action)}. "
        f"Real one-step result: box_dist {prev_box_dist:.3f}->{next_box_dist:.3f}, "
        f"box_goal_progress={prev_box_dist - next_box_dist:+.3f}, "
        f"system_clearance {prev_clearance:.3f}->{next_clearance:.3f}, "
        f"physics_train_loss={loss_text}, status={status}."
    )


# ---------------------------------------------------------------------------
# Online learned physics
# ---------------------------------------------------------------------------

class TransitionReplay:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 2), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return int(self.size)

    def append(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> None:
        i = self.cursor
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.actions[i] = np.asarray(action, dtype=np.float32).reshape(2)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return self.obs[idx], self.actions[idx], self.next_obs[idx]


class PhysicsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PushPrediction:
    choice_id: int
    action: np.ndarray
    agent_xy: np.ndarray
    box_xy: np.ndarray
    min_agent_clearance: float
    min_box_clearance: float
    min_system_clearance: float
    safe: bool
    box_goal_progress: float
    final_box_goal_dist: float
    agent_push_pose_progress: float
    final_agent_push_pose_dist: float
    final_push_slot_score: float
    final_push_behind: float
    final_push_lateral_error: float
    score: float
    model_ready: bool


class OnlinePushPhysics:
    def __init__(
        self,
        cfg: PointPushHazardConfig,
        *,
        obs_dim: int,
        device: torch.device,
        hidden_dim: int,
        lr: float,
        batch_size: int,
        replay_capacity: int,
        grad_clip: float,
    ):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.device = device
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.replay = TransitionReplay(replay_capacity, obs_dim=self.obs_dim)
        self.model = PhysicsMLP(input_dim=self.obs_dim + 2, hidden_dim=hidden_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=1e-5)
        self.train_steps = 0
        self.last_loss: float | None = None
        self.loaded_from_ckpt = False

    @property
    def num_samples(self) -> int:
        return len(self.replay)

    def ready(self, warmup: int) -> bool:
        return self.loaded_from_ckpt or self.num_samples >= int(warmup)

    def _normalize_obs_action(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        arena = max(1e-6, float(self.cfg.arena_half))
        agent_speed = max(1e-6, float(self.cfg.max_agent_speed))
        box_speed = max(1e-6, float(self.cfg.max_box_speed))
        obs[0:2] /= arena
        obs[2:4] /= agent_speed
        obs[4:6] /= arena
        obs[6:8] /= box_speed
        obs[8:10] /= arena
        haz = obs[10:].reshape(-1, 3)
        haz[:, 0:2] /= arena
        haz[:, 2] /= arena
        return np.concatenate([obs, np.asarray(action, dtype=np.float32).reshape(2)], axis=0)

    def _normalize_target(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        arena = max(1e-6, float(self.cfg.arena_half))
        agent_speed = max(1e-6, float(self.cfg.max_agent_speed))
        box_speed = max(1e-6, float(self.cfg.max_box_speed))
        target = np.empty(8, dtype=np.float32)
        target[0:2] = (next_obs[0:2] - obs[0:2]) / arena
        target[2:4] = (next_obs[2:4] - obs[2:4]) / agent_speed
        target[4:6] = (next_obs[4:6] - obs[4:6]) / arena
        target[6:8] = (next_obs[6:8] - obs[6:8]) / box_speed
        return target

    def add_transition(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> None:
        self.replay.append(obs, action, next_obs)

    def train_updates(self, n_updates: int) -> float | None:
        if self.num_samples <= 0:
            return self.last_loss
        self.model.train()
        loss_value: float | None = None
        for _ in range(max(0, int(n_updates))):
            obs_b, action_b, next_b = self.replay.sample(self.batch_size)
            x = np.stack([
                self._normalize_obs_action(obs_b[i], action_b[i])
                for i in range(obs_b.shape[0])
            ], axis=0)
            y = np.stack([
                self._normalize_target(obs_b[i], next_b[i])
                for i in range(obs_b.shape[0])
            ], axis=0)
            x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(y.astype(np.float32)).to(self.device)
            pred = self.model(x_t)
            loss = F.mse_loss(pred, y_t)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            self.train_steps += 1
        self.model.eval()
        self.last_loss = loss_value
        return self.last_loss

    @torch.inference_mode()
    def predict_next_parts(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        box_pos: np.ndarray,
        box_vel: np.ndarray,
        goal: np.ndarray,
        hazards: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = _build_obs_from_parts(agent_pos, agent_vel, box_pos, box_vel, goal, hazards, self.cfg)
        x = self._normalize_obs_action(obs, action)
        pred = self.model(torch.from_numpy(x.astype(np.float32)).to(self.device).reshape(1, -1))
        delta = pred.detach().cpu().numpy().reshape(8)

        arena = float(self.cfg.arena_half)
        agent_speed = float(self.cfg.max_agent_speed)
        box_speed = float(self.cfg.max_box_speed)
        next_agent_pos = agent_pos + delta[0:2] * arena
        next_agent_vel = agent_vel + delta[2:4] * agent_speed
        next_box_pos = box_pos + delta[4:6] * arena
        next_box_vel = box_vel + delta[6:8] * box_speed

        agent_limit = float(self.cfg.arena_half) - float(self.cfg.agent_radius)
        box_limit = float(self.cfg.arena_half) - float(self.cfg.box_half_size)
        next_agent_pos = np.clip(next_agent_pos, -agent_limit, agent_limit)
        next_box_pos = np.clip(next_box_pos, -box_limit, box_limit)
        for vel, max_speed in ((next_agent_vel, agent_speed), (next_box_vel, box_speed)):
            speed = float(np.linalg.norm(vel))
            if speed > max_speed:
                vel *= max_speed / speed

        return (
            next_agent_pos.astype(np.float32),
            next_agent_vel.astype(np.float32),
            next_box_pos.astype(np.float32),
            next_box_vel.astype(np.float32),
        )

    def rollout_candidate(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        choice_id: int,
        horizon: int,
        safety_margin: float,
        clearance_weight: float,
        approach_weight: float,
        push_slot_weight: float,
        clearance_cap: float,
        model_ready: bool,
    ) -> PushPrediction:
        agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, self.cfg)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        start_dist = float(np.linalg.norm(box_pos - goal))
        start_push_pose_dist, _start_behind, _start_lateral, _start_slot = _push_pose_metrics(
            agent_pos,
            box_pos,
            goal,
            self.cfg,
        )
        min_agent_clear = _agent_clearance(agent_pos, hazards, self.cfg)
        min_box_clear = _box_clearance(box_pos, hazards, self.cfg)
        agent_traj = [agent_pos.copy()]
        box_traj = [box_pos.copy()]
        ap, av = agent_pos.copy(), agent_vel.copy()
        bp, bv = box_pos.copy(), box_vel.copy()

        if model_ready:
            for _ in range(max(1, int(horizon))):
                ap, av, bp, bv = self.predict_next_parts(ap, av, bp, bv, goal, hazards, action)
                agent_traj.append(ap.copy())
                box_traj.append(bp.copy())
                min_agent_clear = min(min_agent_clear, _agent_clearance(ap, hazards, self.cfg))
                min_box_clear = min(min_box_clear, _box_clearance(bp, hazards, self.cfg))

        final_dist = float(np.linalg.norm(bp - goal))
        progress = start_dist - final_dist
        final_push_pose_dist, final_behind, final_lateral, final_slot = _push_pose_metrics(
            ap,
            bp,
            goal,
            self.cfg,
        )
        push_pose_progress = start_push_pose_dist - final_push_pose_dist
        min_system = min(min_agent_clear, min_box_clear)
        safe = bool(min_agent_clear >= float(safety_margin) and min_box_clear >= float(safety_margin))
        if float(clearance_cap) > 0:
            clearance_reward = min(min_system, float(safety_margin) + float(clearance_cap))
        else:
            clearance_reward = min_system

        # Phase-aware scoring: blend weights based on how close agent is to
        # the push slot.  _start_slot ~ 0 means agent is far / misaligned,
        # _start_slot ~ 1 means agent is right behind the box ready to push.
        slot_blend = float(np.clip(_start_slot, 0.0, 1.0))
        # approach phase (slot_blend~0): prioritise getting behind the box
        # push phase    (slot_blend~1): prioritise box-goal progress
        eff_approach = float(approach_weight) * (1.0 - slot_blend)
        eff_push_slot = float(push_slot_weight) * (1.0 - 0.6 * slot_blend)
        eff_progress = 1.0 + 1.0 * slot_blend  # 1.0 → 2.0 as agent aligns

        score = (
            eff_progress * progress
            + eff_approach * push_pose_progress
            + eff_push_slot * final_slot
            + float(clearance_weight) * clearance_reward
        )
        if model_ready and not safe:
            score -= 10.0
        return PushPrediction(
            choice_id=int(choice_id),
            action=action.astype(np.float32),
            agent_xy=np.asarray(agent_traj, dtype=np.float32),
            box_xy=np.asarray(box_traj, dtype=np.float32),
            min_agent_clearance=float(min_agent_clear),
            min_box_clearance=float(min_box_clear),
            min_system_clearance=float(min_system),
            safe=bool(safe),
            box_goal_progress=float(progress),
            final_box_goal_dist=float(final_dist),
            agent_push_pose_progress=float(push_pose_progress),
            final_agent_push_pose_dist=float(final_push_pose_dist),
            final_push_slot_score=float(final_slot),
            final_push_behind=float(final_behind),
            final_push_lateral_error=float(final_lateral),
            score=float(score),
            model_ready=bool(model_ready),
        )

    def rollout_candidates(
        self,
        obs: np.ndarray,
        candidates: list[np.ndarray],
        *,
        horizon: int,
        safety_margin: float,
        clearance_weight: float,
        approach_weight: float,
        push_slot_weight: float,
        clearance_cap: float,
        warmup: int,
    ) -> list[PushPrediction]:
        ready = self.ready(warmup)
        return [
            self.rollout_candidate(
                obs,
                c,
                choice_id=i + 1,
                horizon=horizon,
                safety_margin=safety_margin,
                clearance_weight=clearance_weight,
                approach_weight=approach_weight,
                push_slot_weight=push_slot_weight,
                clearance_cap=clearance_cap,
                model_ready=ready,
            )
            for i, c in enumerate(candidates)
        ]

    @staticmethod
    def choose_teacher(predictions: list[PushPrediction]) -> int:
        if not predictions or not any(p.model_ready for p in predictions):
            return -1
        safe = [p for p in predictions if p.safe]
        pool = safe if safe else predictions
        best = max(pool, key=lambda p: (p.score, p.min_system_clearance, p.box_goal_progress))
        return int(best.choice_id - 1)

    def save(self, path: str) -> None:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "train_steps": self.train_steps,
            "last_loss": self.last_loss,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if int(ckpt.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError(f"Physics checkpoint obs_dim mismatch: {path}")
        self.model.load_state_dict(ckpt["model"])
        self.train_steps = int(ckpt.get("train_steps", 0))
        self.last_loss = ckpt.get("last_loss", None)
        self.loaded_from_ckpt = True
        self.model.eval()


# ---------------------------------------------------------------------------
# Rendering and prompting
# ---------------------------------------------------------------------------

_TRAJ_COLORS = [
    (35, 120, 220),
    (30, 155, 95),
    (190, 115, 35),
    (145, 70, 190),
    (20, 150, 165),
    (205, 70, 120),
    (110, 140, 35),
    (80, 95, 200),
    (125, 90, 55),
    (30, 130, 130),
    (170, 90, 150),
    (85, 150, 65),
    (160, 80, 45),
    (75, 115, 180),
    (130, 120, 30),
    (150, 75, 75),
]
_RISK_COLOR = (185, 45, 45)


def _font(size: int, bold: bool = False):
    paths = []
    if bold:
        paths.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ])
    else:
        paths.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ])
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, img_size: int, fill) -> None:
    font = _font(15, bold=True)
    r = 12
    x = int(np.clip(xy[0], r + 2, img_size - r - 2))
    y = int(np.clip(xy[1], r + 2, img_size - r - 2))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=(25, 25, 25), width=2)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    lum = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    text_fill = (255, 255, 255) if lum < 130 else (20, 20, 20)
    draw.text((x - tw // 2, y - th // 2 - 1), text, fill=text_fill, font=font)


def _render_candidate_sub_image(
    renderer: PointPushHazardRenderer,
    pred: PushPrediction,
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
    sub_size: int,
) -> Image.Image:
    """Render a small sub-image showing predicted agent+box after rollout."""
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)
    # Create a small renderer with same world coords but smaller pixel size
    small_renderer = PointPushHazardRenderer(
        arena_half=cfg.arena_half,
        agent_radius=cfg.agent_radius,
        box_half_size=cfg.box_half_size,
        goal_radius=cfg.goal_radius,
        img_size=sub_size,
    )
    # Predicted final positions from the rollout
    pred_agent_xy = pred.agent_xy[-1] if len(pred.agent_xy) > 0 else agent_pos
    pred_box_xy = pred.box_xy[-1] if len(pred.box_xy) > 0 else box_pos

    # Render the base scene with predicted positions
    frame = small_renderer.render(
        agent_xy=pred_agent_xy,
        box_xy=pred_box_xy,
        goal_xy=goal,
        hazards=hazards,
        agent_trail=list(pred.agent_xy) if len(pred.agent_xy) >= 2 else None,
        box_trail=list(pred.box_xy) if len(pred.box_xy) >= 2 else None,
    )
    sub_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(sub_img)

    # Draw only the candidate number. Keep internal safety and score signals out
    # of the VLM image so it judges the predicted scene itself.
    color = _TRAJ_COLORS[(pred.choice_id - 1) % len(_TRAJ_COLORS)]
    tag = f"{pred.choice_id}"
    font_label = _font(max(11, sub_size // 8), bold=True)

    # Label top-left
    draw.rectangle([0, 0, sub_size // 3, sub_size // 5], fill=color)
    lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    text_fill = (255, 255, 255) if lum < 130 else (20, 20, 20)
    draw.text((4, 2), tag, fill=text_fill, font=font_label)

    return sub_img


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill,
    width: int = 2,
    dash_len: int = 8,
    gap_len: int = 5,
) -> None:
    """Draw a dashed line from start to end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash_len, length)
        p0 = (int(start[0] + ux * pos), int(start[1] + uy * pos))
        p1 = (int(start[0] + ux * seg_end), int(start[1] + uy * seg_end))
        draw.line([p0, p1], fill=fill, width=width)
        pos = seg_end + gap_len


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    tip: tuple[int, int],
    tail: tuple[int, int],
    fill,
    size: int = 10,
) -> None:
    dx = tip[0] - tail[0]
    dy = tip[1] - tail[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base_x = tip[0] - ux * size
    base_y = tip[1] - uy * size
    points = [
        tip,
        (int(base_x + px * size * 0.55), int(base_y + py * size * 0.55)),
        (int(base_x - px * size * 0.55), int(base_y - py * size * 0.55)),
    ]
    draw.polygon(points, fill=fill)


def _draw_box_goal_hint(
    draw: ImageDraw.ImageDraw,
    renderer: PointPushHazardRenderer,
    box_pos: np.ndarray,
    goal: np.ndarray,
) -> None:
    """Draw a visual reference from the current box position to the goal."""
    box_px = renderer.world_to_pixel(float(box_pos[0]), float(box_pos[1]))
    goal_px = renderer.world_to_pixel(float(goal[0]), float(goal[1]))
    fill = (35, 150, 80)
    _draw_dashed_line(
        draw,
        box_px,
        goal_px,
        fill=fill,
        width=2,
        dash_len=10,
        gap_len=6,
    )
    _draw_arrowhead(draw, goal_px, box_px, fill=fill, size=10)


def annotate_push_predictions(
    base_image: Image.Image,
    candidates: list[np.ndarray],
    predictions: list[PushPrediction],
    renderer: PointPushHazardRenderer,
    *,
    agent_world_xy: np.ndarray,
    arrow_length_world: float,
    model_ready: bool,
    obs: np.ndarray | None = None,
    cfg: PointPushHazardConfig | None = None,
) -> Image.Image:
    if not model_ready or obs is None or cfg is None:
        annotated_base = base_image.copy()
        if obs is not None and cfg is not None:
            _agent_pos, _agent_vel, box_pos, _box_vel, goal, _hazards = _obs_parts(obs, cfg)
            _draw_box_goal_hint(ImageDraw.Draw(annotated_base), renderer, box_pos, goal)
        return annotate_candidates(
            annotated_base,
            candidates,
            renderer,
            agent_world_xy=agent_world_xy,
            arrow_length_world=arrow_length_world,
        )

    n = len(predictions)
    if n == 0:
        return base_image.copy()

    # --- Draw trajectories + push line on the MAIN image ---
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    img_size = img.size[0]

    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)

    # Draw a direct box-to-goal reference on every prompt image.
    _draw_box_goal_hint(draw, renderer, box_pos, goal)

    # P1: Draw push-line (ideal push direction: goal ← box ← push_pose)
    push_pose, push_dir, push_gap = _push_pose_for_box(box_pos, goal, cfg)
    # Draw dashed line from push_pose through box to goal
    line_start = box_pos - push_dir * (push_gap + 0.3)
    line_end = box_pos + push_dir * float(np.linalg.norm(goal - box_pos)) * 0.8
    ls_px = renderer.world_to_pixel(float(line_start[0]), float(line_start[1]))
    le_px = renderer.world_to_pixel(float(line_end[0]), float(line_end[1]))
    # Draw as dashed line (alternating segments)
    _draw_dashed_line(draw, ls_px, le_px, fill=(100, 200, 100), width=2, dash_len=8, gap_len=5)
    # Draw push pose marker (cross)
    pp_px = renderer.world_to_pixel(float(push_pose[0]), float(push_pose[1]))
    cross_r = 6
    draw.line([(pp_px[0] - cross_r, pp_px[1] - cross_r), (pp_px[0] + cross_r, pp_px[1] + cross_r)],
              fill=(100, 200, 100), width=3)
    draw.line([(pp_px[0] + cross_r, pp_px[1] - cross_r), (pp_px[0] - cross_r, pp_px[1] + cross_r)],
              fill=(100, 200, 100), width=3)

    # Draw predicted trajectories for each candidate on main image
    agent_px = renderer.world_to_pixel(float(agent_world_xy[0]), float(agent_world_xy[1]))
    for idx, pred in enumerate(predictions):
        color = _TRAJ_COLORS[idx % len(_TRAJ_COLORS)]

        # Number label (positioned along action direction, drawn first as background)
        action = np.asarray(pred.action, dtype=np.float32)
        mag = float(np.linalg.norm(action))
        if mag > 1e-6:
            unit = action / mag
            label_r = float(arrow_length_world) * (1.25 + 2.0 * mag)
            label_world = np.asarray(agent_world_xy, dtype=np.float32) + unit * label_r
            label_xy = renderer.world_to_pixel(float(label_world[0]), float(label_world[1]))
        else:
            label_xy = agent_px

        # Agent trajectory (solid colored line)
        agent_traj = np.asarray(pred.agent_xy, dtype=np.float32)
        if agent_traj.shape[0] >= 2:
            pts = [renderer.world_to_pixel(float(p[0]), float(p[1])) for p in agent_traj]
            # Draw connecting ray: agent → label (thin) to show direction
            draw.line([agent_px, label_xy], fill=color, width=1)
            # Draw trajectory curve (thick)
            for j in range(1, len(pts)):
                draw.line([pts[j - 1], pts[j]], fill=color, width=4)
            agent_end = pts[-1]
            draw.ellipse([agent_end[0] - 4, agent_end[1] - 4, agent_end[0] + 4, agent_end[1] + 4],
                         fill=color)
            # Connect label to trajectory endpoint
            draw.line([label_xy, agent_end], fill=color, width=1)
        else:
            agent_end = agent_px

        # Box trajectory (dashed line, same color) — only if box moved
        box_traj = np.asarray(pred.box_xy, dtype=np.float32)
        box_moved = box_traj.shape[0] >= 2 and float(np.linalg.norm(box_traj[-1] - box_traj[0])) > 0.005
        if box_moved:
            box_pts = [renderer.world_to_pixel(float(p[0]), float(p[1])) for p in box_traj]
            for j in range(1, len(box_pts)):
                _draw_dashed_line(draw, box_pts[j - 1], box_pts[j],
                                  fill=color, width=3, dash_len=6, gap_len=4)
            # Ghost box at predicted endpoint (small square outline)
            box_end = box_pts[-1]
            ghost_r = max(4, renderer.world_scale(float(cfg.box_half_size)) // 2)
            draw.rectangle([box_end[0] - ghost_r, box_end[1] - ghost_r,
                            box_end[0] + ghost_r, box_end[1] + ghost_r],
                           outline=color, width=2)

        _draw_label(draw, label_xy, str(pred.choice_id), img_size, fill=color)

    # --- Composite: main image + sub-image grid ---
    main_size = img.size[0]
    n_cols = min(n, 2)
    n_rows = math.ceil(n / n_cols)
    sub_size = main_size // max(n_cols, 2)

    grid_w = n_cols * sub_size
    grid_h = n_rows * sub_size
    total_w = max(main_size, grid_w)
    total_h = main_size + grid_h + 20

    composite = Image.new("RGB", (total_w, total_h), (240, 240, 238))
    x_offset = (total_w - main_size) // 2
    composite.paste(img, (x_offset, 0))

    draw_comp = ImageDraw.Draw(composite)
    label_font = _font(13, bold=True)
    draw_comp.text(
        (4, main_size + 2),
        "Predicted outcomes per candidate (solid=agent, dashed=box, green ✕=ideal push position):",
        fill=(40, 40, 40),
        font=label_font,
    )

    grid_x_offset = (total_w - grid_w) // 2
    for i, pred in enumerate(predictions):
        sub_img = _render_candidate_sub_image(renderer, pred, obs, cfg, sub_size)
        col = i % n_cols
        row = i // n_cols
        px = grid_x_offset + col * sub_size
        py = main_size + 20 + row * sub_size
        composite.paste(sub_img, (px, py))
        border_color = _TRAJ_COLORS[(pred.choice_id - 1) % len(_TRAJ_COLORS)]
        draw_comp.rectangle(
            [px, py, px + sub_size - 1, py + sub_size - 1],
            outline=border_color,
            width=2,
        )

    return composite


def _prediction_text(predictions: list[PushPrediction]) -> str:
    if not predictions or not any(p.model_ready for p in predictions):
        rows = []
        for p in predictions:
            rows.append(f"{p.choice_id}: force={_format_vec(p.action)}")
        return "\n".join(rows)
    rows = []
    for p in predictions:
        pred_agent = p.agent_xy[-1] if len(p.agent_xy) > 0 else p.action
        pred_box = p.box_xy[-1] if len(p.box_xy) > 0 else p.action
        rows.append(
            f"{p.choice_id}: force={_format_vec(p.action)}, "
            f"pred_agent={_format_vec(pred_agent)}, pred_box={_format_vec(pred_box)}, "
            f"agent_clear={p.min_agent_clearance:+.3f}, box_clear={p.min_box_clearance:+.3f}, "
            f"push_pose_progress={p.agent_push_pose_progress:+.3f}, "
            f"push_pose_dist={p.final_agent_push_pose_dist:.3f}, "
            f"push_slot={p.final_push_slot_score:.2f}, "
            f"box_progress={p.box_goal_progress:+.3f}, "
            f"box_goal_dist={p.final_box_goal_dist:.3f}"
        )
    return "\n".join(rows)


def build_push_prompt(
    *,
    predictions: list[PushPrediction],
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
    physics_samples: int,
    physics_loss: float | None,
    warmup: int,
    physics_ready: bool,
) -> str:
    agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)
    box_goal_vec = goal - box_pos
    agent_box_vec = box_pos - agent_pos
    dist_box_goal = float(np.linalg.norm(box_goal_vec))
    dist_agent_box = float(np.linalg.norm(agent_box_vec))
    agent_clear = _agent_clearance(agent_pos, hazards, cfg)
    box_clear = _box_clearance(box_pos, hazards, cfg)
    loss_text = "n/a" if physics_loss is None else f"{physics_loss:.5f}"
    n_candidates = len(predictions)

    push_pose, push_dir, push_gap = _push_pose_for_box(box_pos, goal, cfg)
    push_pose_dist, push_behind, push_lateral, push_slot = _push_pose_metrics(
        agent_pos, box_pos, goal, cfg,
    )
    # Determine current pushing phase
    if push_pose_dist > 0.35:
        phase = "APPROACH: Agent is far from the push position. Priority: move behind the box first."
    elif push_slot < 0.4:
        phase = "ALIGN: Agent is near the box but not well aligned behind it. Priority: get directly behind the box relative to the goal direction."
    else:
        phase = "PUSH: Agent is behind the box and aligned. Priority: push the box straight toward the goal."

    if physics_ready:
        visual = (
            f"The image has two parts:\n"
            f"- TOP: current scene (blue agent, orange box, green goal, red hazards)\n"
            f"  A green dashed arrow connects the orange box to the green goal as a reference.\n"
            f"- BOTTOM: a grid of {n_candidates} sub-images, each showing the PREDICTED future state if that candidate action is applied.\n"
            f"  Each sub-image shows predicted agent trail (blue) and box trail (orange).\n"
            f"  The number label identifies the candidate."
        )
        model_note = (
            "The sub-images come from an online learned physics model. "
            "Compare predicted agent and box positions across candidates to choose wisely."
        )
    else:
        visual = (
            f"- Green dashed arrow = direct reference from the current orange box to the green goal\n"
            f"- Orange numbered arrows (1-{n_candidates}) = candidate force choices while the learned model warms up"
        )
        model_note = "The learned physics model is warming up; use the force arrows and the box-to-goal reference."

    return f"""You control a blue point agent that must push an orange square box into the green goal.

{visual}

{model_note}

KEY STRATEGY — The agent can only push, not pull. To push the box toward the goal:
1. APPROACH: First move the agent to the OPPOSITE side of the box from the goal (get behind it).
2. ALIGN: Position yourself directly behind the box along the box-to-goal line.
3. PUSH: Once behind and aligned, push forward through the box toward the goal.
Never push from the side or front — this sends the box away from the goal!

HARD SAFETY CONSTRAINT:
- Never touch any red hazard circle. If the agent or box touches a hazard, it dies immediately and the episode fails.
- Reject candidates whose predicted agent path or box path enters, touches, or scrapes a hazard.
- When two candidates make similar progress, choose the one with larger hazard clearance.

Current phase: {phase}
- push_direction_to_goal={_format_vec(push_dir)}
- ideal_push_position={_format_vec(push_pose)} (behind box, opposite to goal)
- dist_agent_to_push_position={push_pose_dist:.3f}
- push_alignment_score={push_slot:.2f} (1.0=perfectly behind box)

Current measured state:
- agent_position={_format_vec(agent_pos)}, agent_velocity={_format_vec(agent_vel)}
- box_position={_format_vec(box_pos)}, box_velocity={_format_vec(box_vel)}
- vector_agent_to_box={_format_vec(agent_box_vec)}, dist_agent_to_box={dist_agent_box:.3f}
- vector_box_to_goal={_format_vec(box_goal_vec)}, dist_box_to_goal={dist_box_goal:.3f}
- agent_hazard_clearance={agent_clear:.3f}, box_hazard_clearance={box_clear:.3f}
- learned_physics_samples={physics_samples}, warmup={int(warmup)}, latest_physics_loss={loss_text}

Candidate summaries:
{_prediction_text(predictions)}

Choose the best candidate. In APPROACH/ALIGN phase, pick the candidate that moves the agent
behind the box (toward the ideal push position). In PUSH phase, pick the candidate that pushes
the box toward the goal while keeping both agent and box safely away from hazards.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


# ---------------------------------------------------------------------------
# Choice scoring and optional LoRA micro-training
# ---------------------------------------------------------------------------

@dataclass
class ChoiceScore:
    choice_idx: int
    best_choice_idx: int
    raw_score: float
    best_raw_score: float
    normalized_score: float
    best_normalized_score: float
    score_gap: float
    rank: int
    safe: bool
    teacher_match: bool
    candidate_raw_scores: list[float]
    candidate_normalized_scores: list[float]


def score_vlm_choice_for_push(
    choice_idx: int,
    predictions: list[PushPrediction],
    *,
    safety_margin: float,
    progress_weight: float = 1.0,
    clearance_weight: float = 0.35,
    approach_weight: float = 0.65,
    push_slot_weight: float = 0.45,
    final_dist_weight: float = 0.08,
    safe_bonus: float = 0.35,
    unsafe_penalty: float = 2.0,
    clearance_cap: float = 1.0,
) -> ChoiceScore:
    """Score a VLM/PIVOT choice with PointPush-specific learned-physics criteria.

    The score favors predicted box progress and low final box-goal distance,
    but explicitly prices both agent and box safety. This is intentionally
    separate from the env reward: it is a teacher/evaluator for visual choices,
    not the environment's scalar return.
    """
    if not predictions or not any(p.model_ready for p in predictions):
        return ChoiceScore(
            choice_idx=int(choice_idx),
            best_choice_idx=-1,
            raw_score=float("nan"),
            best_raw_score=float("nan"),
            normalized_score=float("nan"),
            best_normalized_score=float("nan"),
            score_gap=float("nan"),
            rank=-1,
            safe=False,
            teacher_match=False,
            candidate_raw_scores=[],
            candidate_normalized_scores=[],
        )

    # Use the first prediction's starting push slot score to determine phase.
    # All predictions share the same starting obs, so any pred works.
    start_slot = float(predictions[0].final_push_slot_score) if predictions else 0.0
    # Approximate start_slot from the trajectory start rather than end:
    # final_push_slot_score is after rollout; we want the *starting* slot.
    # Use a rough heuristic: if most candidates have high slot, agent is
    # already behind the box.  Take min of final slots as a conservative
    # proxy for starting alignment.
    start_slot_proxy = float(np.clip(
        min(p.final_push_slot_score for p in predictions), 0.0, 1.0
    ))

    raw_scores: list[float] = []
    for pred in predictions:
        clearance = float(pred.min_system_clearance)
        if float(clearance_cap) > 0:
            clearance_reward = min(clearance, float(safety_margin) + float(clearance_cap))
        else:
            clearance_reward = clearance
        clearance_shortfall = max(0.0, float(safety_margin) - clearance)

        # Phase-aware weights (mirrors rollout_candidate logic)
        eff_progress = float(progress_weight) * (1.0 + 1.0 * start_slot_proxy)
        eff_approach = float(approach_weight) * (1.0 - start_slot_proxy)
        eff_push_slot = float(push_slot_weight) * (1.0 - 0.6 * start_slot_proxy)

        raw = (
            eff_progress * float(pred.box_goal_progress)
            + eff_approach * float(pred.agent_push_pose_progress)
            + eff_push_slot * float(pred.final_push_slot_score)
            + float(clearance_weight) * clearance_reward
            - float(final_dist_weight) * float(pred.final_box_goal_dist)
            + (float(safe_bonus) if pred.safe else -float(unsafe_penalty))
            - 4.0 * clearance_shortfall
        )
        raw_scores.append(float(raw))

    raw_arr = np.asarray(raw_scores, dtype=np.float32)
    best_idx = int(np.argmax(raw_arr))
    lo = float(np.min(raw_arr))
    hi = float(np.max(raw_arr))
    if hi - lo > 1e-8:
        norm_arr = (raw_arr - lo) / (hi - lo)
    else:
        norm_arr = np.ones_like(raw_arr, dtype=np.float32)

    if 0 <= int(choice_idx) < len(predictions):
        chosen_raw = float(raw_arr[int(choice_idx)])
        chosen_norm = float(norm_arr[int(choice_idx)])
        rank = 1 + int(np.sum(raw_arr > chosen_raw))
        safe = bool(predictions[int(choice_idx)].safe)
    else:
        chosen_raw = float("nan")
        chosen_norm = float("nan")
        rank = -1
        safe = False

    return ChoiceScore(
        choice_idx=int(choice_idx),
        best_choice_idx=best_idx,
        raw_score=chosen_raw,
        best_raw_score=float(raw_arr[best_idx]),
        normalized_score=chosen_norm,
        best_normalized_score=float(norm_arr[best_idx]),
        score_gap=float(raw_arr[best_idx] - chosen_raw) if math.isfinite(chosen_raw) else float("nan"),
        rank=int(rank),
        safe=bool(safe),
        teacher_match=bool(int(choice_idx) == best_idx),
        candidate_raw_scores=[float(x) for x in raw_arr.tolist()],
        candidate_normalized_scores=[float(x) for x in norm_arr.tolist()],
    )


def _teacher_choice_from_push_scores(
    predictions: list[PushPrediction],
    *,
    safety_margin: float,
    clearance_weight: float = 0.35,
    approach_weight: float = 0.65,
    push_slot_weight: float = 0.45,
    clearance_cap: float = 1.0,
) -> int:
    score = score_vlm_choice_for_push(
        -1,
        predictions,
        safety_margin=safety_margin,
        clearance_weight=clearance_weight,
        approach_weight=approach_weight,
        push_slot_weight=push_slot_weight,
        clearance_cap=clearance_cap,
    )
    return int(score.best_choice_idx)


def _lora_reason_for_choice(choice_idx: int, predictions: list[PushPrediction], score: ChoiceScore) -> str:
    if not (0 <= choice_idx < len(predictions)):
        return "The learned push-physics scorer could not find a valid candidate."
    pred = predictions[choice_idx]
    return (
        f"Candidate {choice_idx + 1} has the best learned push score "
        f"({score.best_normalized_score:.2f} normalized): predicted progress toward push pose "
        f"{pred.agent_push_pose_progress:.2f}, push-slot readiness {pred.final_push_slot_score:.2f}, "
        f"box progress {pred.box_goal_progress:.2f}, final box-goal distance {pred.final_box_goal_dist:.2f}, "
        f"agent clearance {pred.min_agent_clearance:.2f}, and box clearance {pred.min_box_clearance:.2f}."
    )


@dataclass
class LoraSample:
    image: Image.Image
    prompt_text: str
    target_text: str
    choice_idx: int
    source_choice_idx: int
    source_score: float
    teacher_score: float


def _assistant_only_labels(processor, messages: list[dict[str, Any]], batch) -> torch.Tensor:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("LoRA training messages must end with an assistant response.")

    prompt_batch = processor.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"]
    prompt_ids = prompt_batch["input_ids"].to(input_ids.device)
    prompt_len = int(prompt_ids.shape[1])
    if (
        prompt_ids.shape[0] != input_ids.shape[0]
        or prompt_len >= int(input_ids.shape[1])
        or not torch.equal(input_ids[:, :prompt_len], prompt_ids)
    ):
        raise RuntimeError(
            "Assistant response boundary did not match the full chat template; "
            "refusing to train with incorrectly masked labels."
        )

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0, -100)
    return labels


class LoraMicroTrainer:
    def __init__(self, args: argparse.Namespace, model, processor):
        self.enabled = bool(args.enable_lora_update)
        self.args = args
        self.model = model
        self.processor = processor
        self.samples: list[LoraSample] = []
        self.optimizer = None
        self.last_loss: float | None = None

        if not self.enabled:
            return
        if model is None or processor is None:
            raise RuntimeError("--enable_lora_update requires --pilot_mode vlm.")
        try:
            import accelerate  # noqa: F401
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA updates require peft and accelerate. Install on the GPU machine, "
                "for example: pip install peft accelerate transformers"
            ) from exc

        lora_cfg = LoraConfig(
            r=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        )
        self.model = get_peft_model(model, lora_cfg)
        if args.lora_gradient_checkpointing:
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(args.lora_lr),
        )
        os.makedirs(args.lora_out_dir, exist_ok=True)

    def add_scored_sample(
        self,
        image: Image.Image,
        prompt_text: str,
        choice_score: ChoiceScore,
        predictions: list[PushPrediction],
        physics_loss: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        teacher_idx = int(choice_score.best_choice_idx)
        if teacher_idx < 0:
            return
        # Only add LoRA samples when the physics model is well-trained enough
        # to produce trustworthy teacher signals.
        max_loss = float(self.args.lora_physics_loss_gate)
        if max_loss > 0 and physics_loss is not None and float(physics_loss) > max_loss:
            return
        if bool(self.args.lora_only_mistakes) and choice_score.teacher_match:
            return
        if (
            math.isfinite(float(choice_score.score_gap))
            and float(choice_score.score_gap) < float(self.args.lora_min_score_gap)
        ):
            return

        choice = teacher_idx + 1
        reason = _lora_reason_for_choice(teacher_idx, predictions, choice_score)
        target_text = json.dumps({"choice": choice, "reason": reason})
        self.samples.append(LoraSample(
            image=image.copy(),
            prompt_text=prompt_text,
            target_text=target_text,
            choice_idx=teacher_idx,
            source_choice_idx=int(choice_score.choice_idx),
            source_score=float(choice_score.normalized_score),
            teacher_score=float(choice_score.best_normalized_score),
        ))
        max_samples = int(self.args.lora_buffer_size)
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]

    def maybe_update(self, global_step: int) -> float | None:
        if not self.enabled or not self.samples:
            return self.last_loss
        if global_step <= 0 or global_step % max(1, int(self.args.lora_update_every)) != 0:
            return self.last_loss

        assert self.optimizer is not None
        self.model.train()
        for _ in range(max(1, int(self.args.lora_steps_per_update))):
            sample = random.choice(self.samples)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample.image},
                        {"type": "text", "text": sample.prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample.target_text}],
                },
            ]
            batch = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            labels = _assistant_only_labels(self.processor, messages, batch)
            out = self.model(**batch, labels=labels)
            loss = out.loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(self.args.lora_grad_clip) > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    float(self.args.lora_grad_clip),
                )
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu().item())

        self.model.eval()
        if self.args.lora_save_every_update:
            self.model.save_pretrained(self.args.lora_out_dir)
            try:
                self.processor.save_pretrained(self.args.lora_out_dir)
            except Exception:
                pass
        return self.last_loss

    def save(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.args.lora_out_dir, exist_ok=True)
        self.model.save_pretrained(self.args.lora_out_dir)
        try:
            self.processor.save_pretrained(self.args.lora_out_dir)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

@dataclass
class PivotResult:
    action: np.ndarray
    choice_idx: int
    parsed_ok: bool
    confidence: float
    uncertainty: float
    mean_logprob: float
    reason: str | None
    raw_text: str
    fallback_used: bool
    fallback_reason: str
    physics_override_used: bool
    physics_override_reason: str
    annotated_image: Image.Image
    prompt_text: str
    predictions: list[PushPrediction]
    teacher_choice_idx: int
    choice_score: ChoiceScore
    n_candidates: int


def _apply_physics_gate(
    choice_idx: int,
    teacher_idx: int,
    predictions: list[PushPrediction],
    *,
    gate_enabled: bool,
    gate_margin: float,
) -> tuple[int, bool, str]:
    if not gate_enabled or choice_idx < 0 or teacher_idx < 0:
        return choice_idx, False, ""
    if choice_idx >= len(predictions) or teacher_idx >= len(predictions):
        return choice_idx, False, ""
    chosen = predictions[choice_idx]
    teacher = predictions[teacher_idx]
    if not chosen.model_ready or not teacher.model_ready:
        return choice_idx, False, ""
    if chosen.safe:
        return choice_idx, False, ""
    if teacher.safe and teacher.score >= chosen.score + float(gate_margin):
        return teacher_idx, True, f"physics_gate_overrode_{choice_idx + 1}_with_{teacher_idx + 1}"
    return choice_idx, False, ""


def choose_pivot_action(
    *,
    model,
    processor,
    base_image: Image.Image,
    renderer: PointPushHazardRenderer,
    obs: np.ndarray,
    cfg: PointPushHazardConfig,
    physics: OnlinePushPhysics,
    args: argparse.Namespace,
    held_action: np.ndarray,
    held_choice_idx: int,
) -> PivotResult:
    candidates = generate_candidates(args.pivot_n_dirs, args.pivot_n_mags)
    predictions = physics.rollout_candidates(
        obs,
        candidates,
        horizon=args.learned_rollout_horizon,
        safety_margin=args.safety_margin,
        clearance_weight=args.clearance_weight,
        approach_weight=args.approach_weight,
        push_slot_weight=args.push_slot_weight,
        clearance_cap=args.clearance_cap,
        warmup=args.physics_warmup,
    )
    ready = physics.ready(args.physics_warmup)
    agent_pos = np.asarray(obs[0:2], dtype=np.float32)
    annotated = annotate_push_predictions(
        base_image,
        candidates,
        predictions,
        renderer,
        agent_world_xy=agent_pos,
        arrow_length_world=args.pivot_arrow_len,
        model_ready=ready,
        obs=obs,
        cfg=cfg,
    )
    prompt_text = build_push_prompt(
        predictions=predictions,
        obs=obs,
        cfg=cfg,
        physics_samples=physics.num_samples,
        physics_loss=physics.last_loss,
        warmup=args.physics_warmup,
        physics_ready=ready,
    )
    teacher_idx = _teacher_choice_from_push_scores(
        predictions,
        safety_margin=args.safety_margin,
        clearance_weight=args.clearance_weight,
        approach_weight=args.approach_weight,
        push_slot_weight=args.push_slot_weight,
        clearance_cap=args.clearance_cap,
    )

    choice_idx = -1
    parsed_ok = False
    confidence = 0.5
    uncertainty = 0.5
    mean_logprob = float("nan")
    reason = None
    raw_text = ""
    fallback_used = False
    fallback_reason = ""

    if args.pilot_mode == "vlm":
        assert model is not None and processor is not None
        best_raw = None
        for attempt in range(max(0, int(args.vlm_retries)) + 1):
            p_text = prompt_text
            if attempt > 0:
                p_text += "\nREMINDER: Output ONLY JSON. Start with '{'.\n"
            raw = _call_vlm_select(
                model,
                processor,
                annotated,
                p_text,
                n_candidates=len(candidates),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                enable_thinking=args.enable_thinking,
            )
            if best_raw is None:
                best_raw = raw
            if raw.parsed_ok:
                best_raw = raw
                break
        assert best_raw is not None
        parsed_ok = bool(best_raw.parsed_ok)
        confidence = float(best_raw.confidence)
        uncertainty = float(best_raw.uncertainty)
        mean_logprob = float(best_raw.mean_logprob)
        raw_text = best_raw.raw_text
        reason = best_raw.reason
        if parsed_ok and best_raw.choice_idx is not None:
            choice_idx = int(best_raw.choice_idx)
        elif held_choice_idx >= 0:
            fallback_used = True
            fallback_reason = "vlm_parse_failed_hold_previous_choice"
            choice_idx = int(held_choice_idx)
        else:
            fallback_used = True
            fallback_reason = "vlm_parse_failed_local_push_heuristic"
            choice_idx = _local_push_heuristic(obs, cfg, candidates)

    elif args.pilot_mode == "local":
        heuristic_idx = _local_push_heuristic(obs, cfg, candidates)
        if args.local_policy == "learned" and teacher_idx >= 0:
            choice_idx = int(teacher_idx)
            reason = "local_teacher_from_learned_physics"
        elif args.local_policy == "hybrid" and teacher_idx >= 0:
            teacher = predictions[teacher_idx]
            if teacher.safe and teacher.box_goal_progress > 0.02:
                choice_idx = int(teacher_idx)
                reason = "local_hybrid_learned_safe_progress"
            else:
                choice_idx = heuristic_idx
                reason = "local_hybrid_heuristic_guard"
        else:
            choice_idx = heuristic_idx
            reason = "local_push_warmup_heuristic"
        parsed_ok = True
        confidence = 1.0
        uncertainty = 0.0
        raw_text = json.dumps({"choice": choice_idx + 1, "reason": reason})

    elif args.pilot_mode == "random":
        choice_idx = int(np.random.randint(0, len(candidates)))
        parsed_ok = True
        confidence = 0.5
        uncertainty = 0.5
        reason = "random_candidate"
        raw_text = json.dumps({"choice": choice_idx + 1, "reason": reason})
    else:
        raise ValueError(f"Unknown pilot_mode: {args.pilot_mode}")

    gated_idx, gate_used, gate_reason = _apply_physics_gate(
        choice_idx,
        teacher_idx,
        predictions,
        gate_enabled=bool(args.physics_gate),
        gate_margin=float(args.physics_gate_margin),
    )
    choice_idx = gated_idx
    choice_score = score_vlm_choice_for_push(
        choice_idx,
        predictions,
        safety_margin=args.safety_margin,
        clearance_weight=args.clearance_weight,
        approach_weight=args.approach_weight,
        push_slot_weight=args.push_slot_weight,
        clearance_cap=args.clearance_cap,
    )
    if 0 <= choice_idx < len(candidates):
        action = candidates[choice_idx].copy()
    elif held_choice_idx >= 0:
        action = held_action.copy()
    else:
        action = np.zeros(2, dtype=np.float32)

    return PivotResult(
        action=np.clip(action, -1.0, 1.0).astype(np.float32),
        choice_idx=int(choice_idx),
        parsed_ok=bool(parsed_ok),
        confidence=float(confidence),
        uncertainty=float(uncertainty),
        mean_logprob=float(mean_logprob),
        reason=reason,
        raw_text=raw_text,
        fallback_used=bool(fallback_used),
        fallback_reason=fallback_reason,
        physics_override_used=bool(gate_used),
        physics_override_reason=gate_reason,
        annotated_image=annotated,
        prompt_text=prompt_text,
        predictions=predictions,
        teacher_choice_idx=int(teacher_idx),
        choice_score=choice_score,
        n_candidates=len(candidates),
    )


def _prediction_log(predictions: list[PushPrediction]) -> dict[str, Any]:
    return {
        "candidate_actions": [p.action.tolist() for p in predictions],
        "candidate_safe": [bool(p.safe) for p in predictions],
        "candidate_agent_min_clearance": [float(p.min_agent_clearance) for p in predictions],
        "candidate_box_min_clearance": [float(p.min_box_clearance) for p in predictions],
        "candidate_agent_push_pose_progress": [float(p.agent_push_pose_progress) for p in predictions],
        "candidate_final_agent_push_pose_dist": [float(p.final_agent_push_pose_dist) for p in predictions],
        "candidate_final_push_slot_score": [float(p.final_push_slot_score) for p in predictions],
        "candidate_final_push_behind": [float(p.final_push_behind) for p in predictions],
        "candidate_final_push_lateral_error": [float(p.final_push_lateral_error) for p in predictions],
        "candidate_box_goal_progress": [float(p.box_goal_progress) for p in predictions],
        "candidate_final_box_goal_dist": [float(p.final_box_goal_dist) for p in predictions],
        "candidate_score": [float(p.score) for p in predictions],
    }


def _choice_score_log(score: ChoiceScore) -> dict[str, Any]:
    return {
        "vlm_choice_score_raw": float(score.raw_score),
        "vlm_choice_score_norm": float(score.normalized_score),
        "vlm_choice_rank": int(score.rank),
        "vlm_choice_safe": bool(score.safe),
        "scored_teacher_choice": int(score.best_choice_idx + 1) if score.best_choice_idx >= 0 else 0,
        "scored_teacher_score_raw": float(score.best_raw_score),
        "scored_teacher_score_norm": float(score.best_normalized_score),
        "vlm_choice_score_gap": float(score.score_gap),
        "vlm_choice_teacher_match": bool(score.teacher_match),
        "candidate_choice_raw_scores": score.candidate_raw_scores,
        "candidate_choice_norm_scores": score.candidate_normalized_scores,
    }


# ---------------------------------------------------------------------------
# Smoke and eval
# ---------------------------------------------------------------------------

def _make_physics(args: argparse.Namespace, cfg: PointPushHazardConfig, obs_dim: int) -> OnlinePushPhysics:
    device = torch.device("cuda" if args.physics_device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.physics_device == "auto" else args.physics_device
    ))
    physics = OnlinePushPhysics(
        cfg,
        obs_dim=obs_dim,
        device=device,
        hidden_dim=args.physics_hidden_dim,
        lr=args.physics_lr,
        batch_size=args.physics_batch_size,
        replay_capacity=args.physics_replay_capacity,
        grad_clip=args.physics_grad_clip,
    )
    if args.resume_physics and os.path.exists(args.physics_ckpt):
        physics.load(args.physics_ckpt)
        print(f"Loaded learned push physics checkpoint: {args.physics_ckpt}")
    return physics


def _warmup_physics(
    physics: OnlinePushPhysics,
    cfg: PointPushHazardConfig,
    args: argparse.Namespace,
) -> None:
    """Pre-fill the physics replay buffer with diverse transitions.

    Uses the same discrete PIVOT actions as evaluation, mixing random
    candidates with box-seeking candidates so the replay buffer contains
    plenty of agent-box contact data.
    """
    n_steps = int(args.physics_warmup)
    if n_steps <= 0:
        return
    if physics.loaded_from_ckpt:
        print(f"  physics loaded from checkpoint, skipping warmup collection")
        return

    warmup_env = make_env(cfg=cfg, seed=args.seed + 999_000, with_renderer=False)
    obs, _info = warmup_env.reset(seed=args.seed + 999_000)
    obs = np.asarray(obs, dtype=np.float32)
    warmup_candidates = generate_candidates(args.pivot_n_dirs, args.pivot_n_mags)
    if not warmup_candidates:
        raise ValueError("Physics warmup requires at least one PIVOT candidate action.")
    print(
        f"  warmup actions: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags "
        f"= {len(warmup_candidates)} candidates"
    )

    contact_transitions = 0
    ep_count = 0
    t0 = time.time()

    for step in range(n_steps):
        agent_pos = obs[0:2]
        box_pos = obs[4:6]
        to_box = box_pos - agent_pos
        dist_to_box = float(np.linalg.norm(to_box))

        # 50% random candidate, 50% candidate most aligned toward the box.
        if np.random.rand() < 0.5 and dist_to_box > 1e-6:
            direction = to_box / dist_to_box
            action = max(
                warmup_candidates,
                key=lambda candidate: float(np.dot(candidate, direction)),
            ).copy()
        else:
            action = warmup_candidates[np.random.randint(len(warmup_candidates))].copy()

        next_obs, _reward, terminated, truncated, _info = warmup_env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        physics.add_transition(obs, action, next_obs)

        # Track if box moved (indicates contact)
        box_delta = float(np.linalg.norm(next_obs[4:6] - obs[4:6]))
        if box_delta > 0.005:
            contact_transitions += 1

        # Train physics model online during warmup
        if physics.num_samples >= min(32, n_steps // 4):
            physics.train_updates(args.physics_updates_per_step)

        obs = next_obs
        if terminated or truncated:
            ep_count += 1
            obs, _info = warmup_env.reset(seed=args.seed + 999_000 + ep_count)
            obs = np.asarray(obs, dtype=np.float32)

    warmup_env.close()
    elapsed = time.time() - t0
    print(
        f"  warmup done: {n_steps} steps, {contact_transitions} contact transitions "
        f"({contact_transitions / max(1, n_steps):.0%}), "
        f"{ep_count} resets, loss={physics.last_loss}, {elapsed:.1f}s"
    )


def run_smoke_test(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = PointPushHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=True)
    renderer = env._renderer
    assert renderer is not None
    physics = _make_physics(args, cfg, env.obs_dim)

    # Use dedicated warmup with box-seeking exploration
    smoke_warmup_args = argparse.Namespace(**{
        **vars(args),
        "physics_warmup": max(int(args.smoke_train_steps), int(args.physics_warmup)),
    })
    _warmup_physics(physics, cfg, smoke_warmup_args)

    obs, _info = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)
    frame = env.render(info_text="PointPush learned-physics PIVOT smoke")
    image = Image.fromarray(frame)
    local_args = argparse.Namespace(**{**vars(args), "pilot_mode": "local"})
    result = choose_pivot_action(
        model=None,
        processor=None,
        base_image=image,
        renderer=renderer,
        obs=obs,
        cfg=cfg,
        physics=physics,
        args=local_args,
        held_action=np.zeros(2, dtype=np.float32),
        held_choice_idx=-1,
    )

    out_dir = os.path.dirname(args.smoke_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.annotated_image.save(args.smoke_out)
    env.close()
    print(f"Smoke image saved: {args.smoke_out}")
    print(f"  physics samples: {physics.num_samples}")
    print(f"  physics train steps: {physics.train_steps}")
    print(f"  physics ready: {physics.ready(args.physics_warmup)}")
    print(f"  latest physics loss: {physics.last_loss}")
    print(f"  selected choice: {result.choice_idx + 1 if result.choice_idx >= 0 else 0}")


def _save_prompt_image(result: PivotResult, run_dir: str, ep: int, t: int) -> str:
    img_dir = os.path.join(run_dir, "prompt_images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"ep{ep:03d}_t{t:03d}.png")
    result.annotated_image.save(path)
    return path


def evaluate(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = PointPushHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=0, with_renderer=True)
    renderer = env._renderer
    assert renderer is not None
    use_physics = not bool(args.no_physics)
    physics = _make_physics(args, cfg, env.obs_dim)

    print("PointPush " + ("learned-physics PIVOT" if use_physics else "plain PIVOT (no physics)"))
    print(f"  pilot={args.pilot_mode}, episodes={args.episodes}")
    if use_physics:
        print(f"  physics warmup={args.physics_warmup}, horizon={args.learned_rollout_horizon}")
        # Dedicated warmup: collect diverse transitions (including box contacts)
        # before any VLM/eval episodes start.
        _warmup_physics(physics, cfg, args)

    model = None
    processor = None
    if args.pilot_mode == "vlm":
        model, processor = _load_local_vlm(args)
    lora_trainer = LoraMicroTrainer(args, model, processor)
    if lora_trainer.enabled:
        model = lora_trainer.model

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ""
    if args.save_log:
        run_dir = os.path.join(args.log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
    if args.save_gif:
        os.makedirs(args.gif_dir, exist_ok=True)

    ep_returns: list[float] = []
    ep_successes: list[int] = []
    ep_hazard_hits: list[int] = []
    ep_final_dist: list[float] = []
    ep_min_agent_clear: list[float] = []
    ep_min_box_clear: list[float] = []
    ep_vlm_calls: list[int] = []
    ep_gate_overrides: list[int] = []
    ep_physics_losses: list[float] = []
    ep_lora_losses: list[float] = []
    ep_vlm_choice_scores: list[float] = []
    ep_vlm_teacher_matches: list[int] = []
    global_steps = 0

    for ep in range(int(args.episodes)):
        obs, _info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        held_action = np.zeros(2, dtype=np.float32)
        held_choice_idx = -1
        previous_feedback = ""
        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []
        log_rows: list[dict[str, Any]] = []
        min_agent_clear = float("inf")
        min_box_clear = float("inf")
        vlm_calls = 0
        parse_ok = 0
        parse_total = 0
        gate_overrides = 0

        for t in range(int(args.max_steps)):
            agent_pos, agent_vel, box_pos, box_vel, goal, hazards = _obs_parts(obs, cfg)
            current_agent_clear = _agent_clearance(agent_pos, hazards, cfg)
            current_box_clear = _box_clearance(box_pos, hazards, cfg)
            current_system_clear = min(current_agent_clear, current_box_clear)
            prev_box_dist = float(np.linalg.norm(box_pos - goal))
            min_agent_clear = min(min_agent_clear, current_agent_clear)
            min_box_clear = min(min_box_clear, current_box_clear)

            frame = env.render(info_text=f"ep={ep} t={t}")
            image = Image.fromarray(frame)
            if args.save_gif and t % max(1, int(args.gif_every)) == 0:
                frames.append(image.copy())

            vlm_called = args.pilot_mode == "vlm" and (t % max(1, int(args.vlm_every)) == 0)
            result: PivotResult | None = None
            prompt_image_path = ""
            if args.pilot_mode != "vlm" or vlm_called:
                result = choose_pivot_action(
                    model=model,
                    processor=processor,
                    base_image=image,
                    renderer=renderer,
                    obs=obs,
                    cfg=cfg,
                    physics=physics,
                    args=args,
                    held_action=held_action,
                    held_choice_idx=held_choice_idx,
                )
                held_action = result.action.copy()
                held_choice_idx = result.choice_idx
                vlm_calls += int(args.pilot_mode == "vlm")
                parse_total += int(args.pilot_mode == "vlm")
                parse_ok += int(args.pilot_mode == "vlm" and result.parsed_ok)
                gate_overrides += int(result.physics_override_used)
                if args.pilot_mode == "vlm":
                    if math.isfinite(float(result.choice_score.normalized_score)):
                        ep_vlm_choice_scores.append(float(result.choice_score.normalized_score))
                    ep_vlm_teacher_matches.append(int(result.choice_score.teacher_match))
                    lora_trainer.add_scored_sample(
                        result.annotated_image,
                        result.prompt_text,
                        result.choice_score,
                        result.predictions,
                        physics_loss=physics.last_loss,
                    )
                if args.save_prompt_images and args.save_log:
                    prompt_image_path = _save_prompt_image(result, run_dir, ep, t)

            exec_action = np.clip(held_action, -1.0, 1.0).astype(np.float32)
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            physics_loss: float | None = None
            if use_physics:
                physics.add_transition(prev_obs, exec_action, obs)
                physics_loss = physics.train_updates(args.physics_updates_per_step)
                if physics_loss is not None:
                    ep_physics_losses.append(float(physics_loss))
            lora_loss = lora_trainer.maybe_update(global_steps)
            if lora_loss is not None:
                ep_lora_losses.append(float(lora_loss))
            global_steps += 1

            next_agent_pos, _next_agent_vel, next_box_pos, _next_box_vel, _goal, next_hazards = _obs_parts(obs, cfg)
            next_agent_clear = _agent_clearance(next_agent_pos, next_hazards, cfg)
            next_box_clear = _box_clearance(next_box_pos, next_hazards, cfg)
            next_system_clear = min(next_agent_clear, next_box_clear)
            next_box_dist = float(np.linalg.norm(next_box_pos - goal))
            min_agent_clear = min(min_agent_clear, next_agent_clear)
            min_box_clear = min(min_box_clear, next_box_clear)
            ep_return += float(reward)
            last_info = dict(info or {})

            previous_feedback = _real_step_feedback(
                choice_idx=held_choice_idx,
                action=exec_action,
                prev_box_dist=prev_box_dist,
                next_box_dist=next_box_dist,
                prev_clearance=current_system_clear,
                next_clearance=next_system_clear,
                agent_hazard_hit=bool(last_info.get("agent_hazard_hit", False)),
                box_hazard_hit=bool(last_info.get("box_hazard_hit", False)),
                goal_success=bool(last_info.get("goal_success", False)),
                physics_loss=physics_loss,
            )

            if args.debug_print:
                choice = held_choice_idx + 1 if held_choice_idx >= 0 else 0
                print(
                    f"[ep {ep:03d} t={t:03d}] choice={choice} "
                    f"box_dist={next_box_dist:.2f} agent_clear={next_agent_clear:.2f} "
                    f"box_clear={next_box_clear:.2f} loss={physics_loss}"
                )

            if args.save_log:
                row: dict[str, Any] = {
                    "t": int(t),
                    "pilot_mode": args.pilot_mode,
                    "agent_pos": agent_pos.tolist(),
                    "agent_vel": agent_vel.tolist(),
                    "box_pos": box_pos.tolist(),
                    "box_vel": box_vel.tolist(),
                    "goal": goal.tolist(),
                    "exec": exec_action.tolist(),
                    "choice": int(held_choice_idx + 1) if held_choice_idx >= 0 else 0,
                    "reward": float(reward),
                    "dist_box_to_goal": float(last_info.get("dist_box_to_goal", next_box_dist)),
                    "agent_hazard_clearance": float(next_agent_clear),
                    "box_hazard_clearance": float(next_box_clear),
                    "agent_hazard_hit": bool(last_info.get("agent_hazard_hit", False)),
                    "box_hazard_hit": bool(last_info.get("box_hazard_hit", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "physics_samples": int(physics.num_samples),
                    "physics_train_steps": int(physics.train_steps),
                    "physics_ready": bool(physics.ready(args.physics_warmup)),
                    "physics_loss": None if physics_loss is None else float(physics_loss),
                    "lora_loss": None if lora_loss is None else float(lora_loss),
                    "lora_buffer_size": int(len(lora_trainer.samples)),
                    "real_step_feedback": previous_feedback,
                    "vlm_called": bool(vlm_called),
                    "prompt_image": prompt_image_path,
                }
                if result is not None:
                    row.update({
                        "vlm_parsed_ok": bool(result.parsed_ok),
                        "vlm_confidence": float(result.confidence),
                        "vlm_uncertainty": float(result.uncertainty),
                        "vlm_logprob": float(result.mean_logprob),
                        "vlm_reason": result.reason,
                        "vlm_raw": result.raw_text,
                        "fallback_used": bool(result.fallback_used),
                        "fallback_reason": result.fallback_reason,
                        "physics_override_used": bool(result.physics_override_used),
                        "physics_override_reason": result.physics_override_reason,
                        "teacher_choice": int(result.teacher_choice_idx + 1) if result.teacher_choice_idx >= 0 else 0,
                        "pivot_n_candidates": int(result.n_candidates),
                    })
                    row.update(_prediction_log(result.predictions))
                    row.update(_choice_score_log(result.choice_score))
                log_rows.append(row)

            if terminated or truncated:
                break

        success = bool(last_info.get("goal_success", False))
        hazard_hit = bool(last_info.get("hazard_hit", False))
        final_dist = float(last_info.get("dist_box_to_goal", float("nan")))
        term_reason = last_info.get("termination_reason", "timeout")
        ep_returns.append(ep_return)
        ep_successes.append(int(success))
        ep_hazard_hits.append(int(hazard_hit))
        ep_final_dist.append(final_dist)
        ep_min_agent_clear.append(min_agent_clear)
        ep_min_box_clear.append(min_box_clear)
        ep_vlm_calls.append(vlm_calls)
        ep_gate_overrides.append(gate_overrides)

        flag = "OK " if success else ("HIT" if hazard_hit else "---")
        print(
            f"[ep {ep:03d}] [{flag}] return={ep_return:7.1f} steps={env.t:3d} "
            f"box_dist={final_dist:.2f} min_agent_clear={min_agent_clear:.2f} "
            f"min_box_clear={min_box_clear:.2f} reason={term_reason}"
        )

        if args.save_log:
            log_path = os.path.join(run_dir, f"ep{ep:03d}.jsonl")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "type": "episode_meta",
                    "episode": ep,
                    "pilot_mode": args.pilot_mode,
                    "seed": args.seed + ep,
                    "learned_physics_pivot": use_physics,
                    "future_oracle_rollout": False,
                    "use_diffusion": False,
                    "env": "PointPushHazardEnv",
                    "args": vars(args),
                }) + "\n")
                for row in log_rows:
                    f.write(json.dumps(row) + "\n")
                f.write(json.dumps({
                    "type": "episode_end",
                    "return": ep_return,
                    "success": success,
                    "hazard_hit": hazard_hit,
                    "dist_box_to_goal": final_dist,
                    "steps": int(env.t),
                    "min_agent_hazard_clearance": min_agent_clear,
                    "min_box_hazard_clearance": min_box_clear,
                    "vlm_calls": vlm_calls,
                    "parse_ok": parse_ok,
                    "parse_total": parse_total,
                    "physics_gate_overrides": gate_overrides,
                    "mean_vlm_choice_score": float(np.nanmean(ep_vlm_choice_scores)) if ep_vlm_choice_scores else float("nan"),
                    "vlm_teacher_match_rate": float(np.mean(ep_vlm_teacher_matches)) if ep_vlm_teacher_matches else float("nan"),
                }) + "\n")

        if args.save_gif and frames:
            tag = "ok" if success else ("hit" if hazard_hit else "timeout")
            gif_path = os.path.join(args.gif_dir, f"ep{ep:03d}_{tag}.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / max(1e-6, float(args.gif_fps))),
                loop=0,
            )

    env.close()

    n = max(1, int(args.episodes))
    succ_rate = float(np.mean(ep_successes))
    hit_rate = float(np.mean(ep_hazard_hits))
    timeout_rate = 1.0 - succ_rate - hit_rate
    mean_loss = float(np.nanmean(ep_physics_losses)) if ep_physics_losses else float("nan")
    mean_lora_loss = float(np.nanmean(ep_lora_losses)) if ep_lora_losses else float("nan")
    mean_vlm_choice_score = float(np.nanmean(ep_vlm_choice_scores)) if ep_vlm_choice_scores else float("nan")
    vlm_teacher_match_rate = float(np.mean(ep_vlm_teacher_matches)) if ep_vlm_teacher_matches else float("nan")
    print("\n" + "=" * 60)
    print("PointPush learned-physics PIVOT  Diffusion: OFF  Oracle rollout: OFF")
    print(f"  Success:    {sum(ep_successes):3d}/{n} ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n} ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n} ({timeout_rate:.1%})")
    print(f"  Mean return: {float(np.mean(ep_returns)):.1f}")
    print(f"  Mean final box dist: {float(np.nanmean(ep_final_dist)):.2f}")
    print(f"  Min agent clearance: {float(np.nanmin(ep_min_agent_clear)):.2f}")
    print(f"  Min box clearance: {float(np.nanmin(ep_min_box_clear)):.2f}")
    print(f"  Physics samples: {physics.num_samples}, train_steps={physics.train_steps}, mean_loss={mean_loss:.6f}")
    print(f"  VLM calls/ep: {float(np.mean(ep_vlm_calls)):.1f}")
    print(f"  VLM choice score: {mean_vlm_choice_score:.3f}" if ep_vlm_choice_scores else "  VLM choice score: n/a")
    print(f"  VLM teacher match: {vlm_teacher_match_rate:.1%}" if ep_vlm_teacher_matches else "  VLM teacher match: n/a")
    print(f"  Physics gate overrides: {int(np.sum(ep_gate_overrides))}")
    print(f"  LoRA samples: {len(lora_trainer.samples)}, mean_lora_loss={mean_lora_loss:.6f}")
    print("=" * 60)

    if use_physics and args.save_physics_ckpt:
        physics.save(args.physics_ckpt)
        print(f"Learned push physics checkpoint saved: {args.physics_ckpt}")

    if args.enable_lora_update:
        lora_trainer.save()
        print(f"LoRA adapter saved: {args.lora_out_dir}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "env": "PointPushHazardEnv",
            "learned_physics_pivot": use_physics,
            "future_oracle_rollout": False,
            "use_diffusion": False,
            "episodes": int(args.episodes),
            "success_rate": succ_rate,
            "hazard_hit_rate": hit_rate,
            "timeout_rate": timeout_rate,
            "mean_return": float(np.mean(ep_returns)),
            "mean_final_box_goal_dist": float(np.nanmean(ep_final_dist)),
            "min_agent_hazard_clearance": float(np.nanmin(ep_min_agent_clear)),
            "min_box_hazard_clearance": float(np.nanmin(ep_min_box_clear)),
            "physics_samples": int(physics.num_samples),
            "physics_train_steps": int(physics.train_steps),
            "mean_physics_loss": mean_loss,
            "mean_vlm_calls": float(np.mean(ep_vlm_calls)),
            "mean_vlm_choice_score": mean_vlm_choice_score,
            "vlm_teacher_match_rate": vlm_teacher_match_rate,
            "physics_gate_overrides": int(np.sum(ep_gate_overrides)),
            "lora_enabled": bool(args.enable_lora_update),
            "lora_samples": int(len(lora_trainer.samples)),
            "mean_lora_loss": mean_lora_loss,
            "args": vars(args),
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {os.path.join(run_dir, 'summary.json')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned-physics PIVOT on PointPushHazardEnv without diffusion or oracle rollouts"
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot_mode", type=str, default="vlm", choices=["vlm", "local", "random"])
    parser.add_argument("--local_policy", type=str, default="heuristic",
                        choices=["heuristic", "hybrid", "learned"],
                        help="Selection rule for --pilot_mode local; learned predictions are still logged/rendered.")

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vlm_every", type=int, default=3)
    parser.add_argument("--vlm_retries", type=int, default=1)

    parser.add_argument("--pivot_n_dirs", type=int, default=8)
    parser.add_argument("--pivot_n_mags", type=int, default=1)
    parser.add_argument("--pivot_arrow_len", type=float, default=0.6)

    parser.add_argument("--learned_rollout_horizon", type=int, default=3)
    parser.add_argument("--safety_margin", type=float, default=0.12)
    parser.add_argument("--clearance_weight", type=float, default=0.20)
    parser.add_argument(
        "--clearance_cap",
        type=float,
        default=1.0,
        help="Cap positive clearance reward at safety_margin + this value; <=0 disables the cap.",
    )
    parser.add_argument(
        "--approach_weight",
        type=float,
        default=0.65,
        help="Reward predicted progress toward the behind-box push pose before contact.",
    )
    parser.add_argument(
        "--push_slot_weight",
        type=float,
        default=0.45,
        help="Reward ending near the aligned behind-box push slot.",
    )
    parser.add_argument("--no_physics", action="store_true",
                        help="Disable learned physics entirely: pure VLM+PIVOT with simple arrow candidates.")
    parser.add_argument("--physics_warmup", type=int, default=25600)
    parser.add_argument("--physics_batch_size", type=int, default=256)
    parser.add_argument("--physics_lr", type=float, default=1e-3)
    parser.add_argument("--physics_hidden_dim", type=int, default=256)
    parser.add_argument("--physics_replay_capacity", type=int, default=100000)
    parser.add_argument("--physics_updates_per_step", type=int, default=1)
    parser.add_argument("--physics_grad_clip", type=float, default=5.0)
    parser.add_argument("--physics_device", type=str, default="auto")
    parser.add_argument("--physics_ckpt", type=str, default="hazard/learned_push_physics_agent.pt")
    parser.add_argument("--resume_physics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_physics_ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physics_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--physics_gate_margin", type=float, default=0.25)

    parser.add_argument("--enable_lora_update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_update_every", type=int, default=50)
    parser.add_argument("--lora_steps_per_update", type=int, default=4)
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_out_dir", type=str, default="hazard/lora_pointpush_learned_physics_pivot")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lora_buffer_size", type=int, default=512)
    parser.add_argument("--lora_grad_clip", type=float, default=1.0)
    parser.add_argument("--lora_save_every_update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_only_mistakes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--lora_min_score_gap",
        type=float,
        default=0.0,
        help="Only add LoRA samples when best scored candidate beats VLM choice by at least this raw score gap.",
    )
    parser.add_argument(
        "--lora_physics_loss_gate",
        type=float,
        default=0.1,
        help="Only produce LoRA teacher samples when physics loss is below this threshold. <=0 disables.",
    )

    parser.add_argument("--save_log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_dir", type=str, default="hazard/logs_pointpush_learned_physics_pivot")
    parser.add_argument("--save_prompt_images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_pointpush_learned_physics_pivot")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_out", type=str, default="hazard/pointpush_learned_physics_pivot_smoke.png")
    parser.add_argument("--smoke_train_steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_lora_update and args.pilot_mode != "vlm":
        raise RuntimeError("--enable_lora_update requires --pilot_mode vlm.")
    if args.smoke_test:
        run_smoke_test(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
