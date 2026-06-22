"""
Learned-physics PIVOT controller for PointHazardEnv.

This entry point removes diffusion and oracle dynamics rollouts from the
control path. A small online torch model learns one-step point-mass physics
from real executed transitions:

    (obs_t, action_t) -> (pos_{t+1}, vel_{t+1}) - (pos_t, vel_t)

The learned model then recursively predicts short trajectories for each PIVOT
candidate action. Those predicted trajectories are drawn on the image and the
VLM chooses a numbered candidate. The predictions are learned from real
feedback only; they are not the environment's hand-coded dynamics.

Examples:
    python shared_autonomy_hazard_learned_physics_pivot.py --smoke_test

    python shared_autonomy_hazard_learned_physics_pivot.py \
        --pilot_mode local --episodes 2 --no-save_gif

    python shared_autonomy_hazard_learned_physics_pivot.py \
        --model_path Qwen/Qwen3-VL-32B-Instruct --episodes 10

Optional LoRA micro-updates are available with --enable_lora_update on a GPU
machine with transformers, peft, and accelerate installed.
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

from env_pointhazard import PointHazardConfig, make_env
from hazard_renderer import HazardRenderer
from pivot_vlm import _call_vlm_select, annotate_candidates, generate_candidates


# ---------------------------------------------------------------------------
# Local VLM loading
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
            "VLM mode requires transformers. Install it on the GPU machine, "
            "or run with --pilot_mode local for the learned-physics smoke path."
        ) from exc

    print(f"Loading local VLM: {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    dtype = _torch_dtype_from_arg(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype if dtype != "auto" else "auto",
    }
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model.eval()
    print("Local VLM loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _obs_parts(obs: np.ndarray, cfg: PointHazardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_haz = int(cfg.n_hazards)
    pos = obs[0:2].astype(np.float32)
    vel = obs[2:4].astype(np.float32)
    goal = obs[4:6].astype(np.float32)
    hazards = obs[6:6 + 3 * n_haz].reshape(n_haz, 3).astype(np.float32)
    return pos, vel, goal, hazards


def _sort_hazards_by_pos(pos: np.ndarray, hazards: np.ndarray) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return hazards
    order = np.argsort(np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1))
    return hazards[order].astype(np.float32)


def _build_obs_from_parts(
    pos: np.ndarray,
    vel: np.ndarray,
    goal: np.ndarray,
    hazards: np.ndarray,
    cfg: PointHazardConfig,
) -> np.ndarray:
    hazards_sorted = _sort_hazards_by_pos(pos, hazards)
    obs = np.empty(4 + 2 + 3 * int(cfg.n_hazards), dtype=np.float32)
    obs[0:2] = np.asarray(pos, dtype=np.float32)
    obs[2:4] = np.asarray(vel, dtype=np.float32)
    obs[4:6] = np.asarray(goal, dtype=np.float32)
    obs[6:] = hazards_sorted.reshape(-1)
    return obs


def _clearance_to_hazards(pos: np.ndarray, hazards: np.ndarray, cfg: PointHazardConfig) -> float:
    pos = np.asarray(pos, dtype=np.float32).reshape(2)
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return float("inf")
    center_d = np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)
    clearances = center_d - hazards[:, 2] - float(cfg.agent_radius)
    return float(np.min(clearances))


def _nearest_hazard_vector(pos: np.ndarray, hazards: np.ndarray) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=np.float32)
    if hazards.size == 0:
        return np.zeros(2, dtype=np.float32)
    idx = int(np.argmin(np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)))
    return (hazards[idx, :2] - pos).astype(np.float32)


def _format_vec(v: np.ndarray) -> str:
    return f"[{float(v[0]):+.2f}, {float(v[1]):+.2f}]"


def _candidate_text(candidates: list[np.ndarray]) -> str:
    rows = []
    for idx, action in enumerate(candidates, start=1):
        mag = float(np.linalg.norm(action))
        angle = math.degrees(math.atan2(float(action[1]), float(action[0])))
        rows.append(
            f"{idx}: force={_format_vec(action)}, magnitude={mag:.2f}, angle_deg={angle:+.0f}"
        )
    return "\n".join(rows)


def _real_step_feedback(
    *,
    choice_idx: int,
    action: np.ndarray,
    prev_speed: float,
    next_speed: float,
    goal_progress: float,
    clearance_delta: float,
    next_clearance: float,
    hazard_hit: bool,
    goal_success: bool,
) -> str:
    status = "goal_success" if goal_success else ("hazard_hit" if hazard_hit else "running")
    choice = choice_idx + 1 if choice_idx >= 0 else 0
    return (
        f"Last chosen candidate={choice}, force={_format_vec(action)}. "
        f"Result: speed {prev_speed:.3f}->{next_speed:.3f}, "
        f"goal_progress={goal_progress:+.3f}, "
        f"clearance_change={clearance_delta:+.3f}, "
        f"clearance={next_clearance:.3f}, status={status}."
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
        if self.size <= 0:
            raise ValueError("Replay buffer is empty")
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
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LearnedTrajectory:
    choice_id: int
    action: np.ndarray
    trajectory_xy: np.ndarray
    final_vel: np.ndarray
    min_clearance: float
    safe: bool
    goal_progress: float
    final_dist_to_goal: float
    score: float
    model_ready: bool


class OnlinePhysicsAgent:
    def __init__(
        self,
        cfg: PointHazardConfig,
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
        self.warmed_up = False

    @property
    def num_samples(self) -> int:
        return len(self.replay)

    def ready(self) -> bool:
        return self.loaded_from_ckpt or self.warmed_up

    def _normalize_obs_action(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        action = np.asarray(action, dtype=np.float32).reshape(2)
        arena = max(1e-6, float(self.cfg.arena_half))
        max_speed = max(1e-6, float(self.cfg.max_speed))

        obs[0:2] /= arena
        obs[2:4] /= max_speed
        obs[4:6] /= arena
        haz = obs[6:].reshape(-1, 3)
        haz[:, 0:2] /= arena
        haz[:, 2] /= arena
        return np.concatenate([obs, action], axis=0).astype(np.float32)

    def _normalize_target(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        arena = max(1e-6, float(self.cfg.arena_half))
        max_speed = max(1e-6, float(self.cfg.max_speed))
        target = np.empty(4, dtype=np.float32)
        target[0:2] = (next_obs[0:2] - obs[0:2]) / arena
        target[2:4] = (next_obs[2:4] - obs[2:4]) / max_speed
        return target

    def add_transition(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> None:
        self.replay.append(obs, action, next_obs)

    def train_updates(self, n_updates: int) -> float | None:
        if self.num_samples <= 0:
            return self.last_loss

        loss_value: float | None = None
        self.model.train()
        for _ in range(max(0, int(n_updates))):
            obs_b, action_b, next_obs_b = self.replay.sample(self.batch_size)
            x = np.stack([
                self._normalize_obs_action(obs_b[i], action_b[i])
                for i in range(obs_b.shape[0])
            ], axis=0)
            y = np.stack([
                self._normalize_target(obs_b[i], next_obs_b[i])
                for i in range(obs_b.shape[0])
            ], axis=0)
            x_t = torch.from_numpy(x).to(self.device)
            y_t = torch.from_numpy(y).to(self.device)

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
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        hazards: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = _build_obs_from_parts(pos, vel, goal, hazards, self.cfg)
        x = self._normalize_obs_action(obs, action)
        x_t = torch.from_numpy(x).to(self.device).reshape(1, -1)
        pred = self.model(x_t).detach().cpu().numpy().reshape(4)

        arena = float(self.cfg.arena_half)
        max_speed = float(self.cfg.max_speed)
        delta_pos = pred[0:2] * arena
        delta_vel = pred[2:4] * max_speed

        next_pos = np.asarray(pos, dtype=np.float32) + delta_pos.astype(np.float32)
        next_vel = np.asarray(vel, dtype=np.float32) + delta_vel.astype(np.float32)

        arena_limit = float(self.cfg.arena_half) - float(self.cfg.agent_radius)
        next_pos = np.clip(next_pos, -arena_limit, arena_limit).astype(np.float32)
        speed = float(np.linalg.norm(next_vel))
        if speed > max_speed:
            next_vel = next_vel * (max_speed / speed)
        return next_pos.astype(np.float32), next_vel.astype(np.float32)

    @torch.inference_mode()
    def rollout_candidate(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        choice_id: int,
        horizon: int,
        safety_margin: float,
        clearance_weight: float,
        model_ready: bool,
    ) -> LearnedTrajectory:
        pos, vel, goal, hazards = _obs_parts(obs, self.cfg)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

        start_dist = float(np.linalg.norm(goal - pos))
        min_clearance = _clearance_to_hazards(pos, hazards, self.cfg)
        trajectory = [pos.copy()]
        p = pos.copy()
        v = vel.copy()

        if model_ready:
            for _ in range(max(1, int(horizon))):
                p, v = self.predict_next_parts(p, v, goal, hazards, action)
                trajectory.append(p.copy())
                min_clearance = min(min_clearance, _clearance_to_hazards(p, hazards, self.cfg))
        else:
            # Warmup visualization is handled as arrows. Keep a one-point
            # trajectory for scoring logs without pretending to know physics.
            pass

        final_dist = float(np.linalg.norm(goal - p))
        goal_progress = start_dist - final_dist
        safe = bool(min_clearance >= float(safety_margin))
        score = goal_progress + float(clearance_weight) * min_clearance
        if model_ready and not safe:
            score -= 10.0

        return LearnedTrajectory(
            choice_id=int(choice_id),
            action=action.astype(np.float32),
            trajectory_xy=np.asarray(trajectory, dtype=np.float32),
            final_vel=v.astype(np.float32),
            min_clearance=float(min_clearance),
            safe=safe,
            goal_progress=float(goal_progress),
            final_dist_to_goal=float(final_dist),
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
    ) -> list[LearnedTrajectory]:
        model_ready = self.ready()
        return [
            self.rollout_candidate(
                obs,
                action,
                choice_id=i + 1,
                horizon=horizon,
                safety_margin=safety_margin,
                clearance_weight=clearance_weight,
                model_ready=model_ready,
            )
            for i, action in enumerate(candidates)
        ]

    @staticmethod
    def choose_teacher(predictions: list[LearnedTrajectory]) -> int:
        if not predictions:
            return -1
        ready = any(p.model_ready for p in predictions)
        if not ready:
            return -1
        safe = [p for p in predictions if p.safe]
        pool = safe if safe else predictions
        best = max(pool, key=lambda p: (p.score, p.min_clearance, p.goal_progress))
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
            "cfg": {
                "n_hazards": int(self.cfg.n_hazards),
                "arena_half": float(self.cfg.arena_half),
                "max_speed": float(self.cfg.max_speed),
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if int(ckpt.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError(f"Physics checkpoint obs_dim does not match: {path}")
        self.model.load_state_dict(ckpt["model"])
        self.train_steps = int(ckpt.get("train_steps", 0))
        self.last_loss = ckpt.get("last_loss", None)
        self.loaded_from_ckpt = True
        self.warmed_up = True
        self.model.eval()


# ---------------------------------------------------------------------------
# Learned trajectory rendering and prompting
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


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    img_size: int,
    *,
    fill: tuple[int, int, int] = (45, 45, 45),
    outline: tuple[int, int, int] = (25, 25, 25),
) -> None:
    font = _font(15, bold=True)
    r = 12
    x = int(np.clip(xy[0], r + 2, img_size - r - 2))
    y = int(np.clip(xy[1], r + 2, img_size - r - 2))
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    text_fill = (255, 255, 255) if luminance < 130 else (20, 20, 20)
    draw.text((x - tw // 2, y - th // 2 - 1), text, fill=text_fill, font=font)


def _render_hazard_candidate_card(
    pred: LearnedTrajectory,
    obs: np.ndarray,
    cfg: PointHazardConfig,
    card_size: int,
) -> Image.Image:
    """Render a sub-image card showing the predicted agent state for one candidate."""
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    small_renderer = HazardRenderer(
        arena_half=cfg.arena_half,
        agent_radius=cfg.agent_radius,
        goal_radius=cfg.goal_radius,
        img_size=card_size,
    )
    traj = np.asarray(pred.trajectory_xy, dtype=np.float32)
    pred_pos = traj[-1] if traj.shape[0] > 0 else pos
    trail = list(traj) if traj.shape[0] >= 2 else None

    frame = small_renderer.render(
        agent_xy=pred_pos,
        goal_xy=goal,
        hazards=hazards,
        trail=trail,
    )
    card = Image.fromarray(frame)
    draw = ImageDraw.Draw(card)

    color = _TRAJ_COLORS[(pred.choice_id - 1) % len(_TRAJ_COLORS)]
    tag = str(pred.choice_id)
    font_label = _font(13, bold=True)
    bbox = draw.textbbox((0, 0), tag, font=font_label)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 3
    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=color)
    lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    text_fill = (255, 255, 255) if lum < 130 else (20, 20, 20)
    draw.text((pad, pad - 1), tag, fill=text_fill, font=font_label)

    return card


def annotate_learned_predictions(
    base_image: Image.Image,
    candidates: list[np.ndarray],
    predictions: list[LearnedTrajectory],
    renderer: HazardRenderer,
    *,
    agent_world_xy: np.ndarray,
    arrow_length_world: float,
    model_ready: bool,
    obs: np.ndarray | None = None,
    cfg: PointHazardConfig | None = None,
) -> Image.Image:
    if not model_ready:
        return annotate_candidates(
            base_image,
            candidates,
            renderer,
            agent_world_xy=agent_world_xy,
            arrow_length_world=arrow_length_world,
        )

    n = len(predictions)
    if n == 0 or obs is None or cfg is None:
        return base_image.copy()

    # --- Main image: current state + numbered direction arrows ---
    main_img = base_image.copy()
    main_size = main_img.size[0]
    draw_main = ImageDraw.Draw(main_img)
    agent_px = renderer.world_to_pixel(float(agent_world_xy[0]), float(agent_world_xy[1]))

    for idx, pred in enumerate(predictions):
        color = _TRAJ_COLORS[idx % len(_TRAJ_COLORS)]
        action = np.asarray(pred.action, dtype=np.float32)
        mag = float(np.linalg.norm(action))
        if mag < 1e-6:
            continue
        unit = action / mag
        end_world = np.asarray(agent_world_xy, dtype=np.float32) + unit * arrow_length_world * mag
        end_px = renderer.world_to_pixel(float(end_world[0]), float(end_world[1]))
        draw_main.line([agent_px, end_px], fill=color, width=3)
        draw_main.ellipse([end_px[0] - 3, end_px[1] - 3, end_px[0] + 3, end_px[1] + 3], fill=color)
        label_world = np.asarray(agent_world_xy, dtype=np.float32) + unit * arrow_length_world * mag * 1.35
        label_px = renderer.world_to_pixel(float(label_world[0]), float(label_world[1]))
        _draw_label(draw_main, label_px, str(pred.choice_id), main_size,
                    fill=color, outline=(25, 25, 25))

    # --- Sub-image grid below ---
    n_cols = 2
    n_rows = math.ceil(n / n_cols)
    sub_size = main_size // n_cols

    gap = 4
    grid_w = n_cols * sub_size + (n_cols - 1) * gap
    grid_h = n_rows * sub_size + (n_rows - 1) * gap
    total_w = max(main_size, grid_w)
    total_h = main_size + gap + grid_h

    composite = Image.new("RGB", (total_w, total_h), (230, 230, 228))
    draw_comp = ImageDraw.Draw(composite)

    # Paste main image centered
    x_offset = (total_w - main_size) // 2
    composite.paste(main_img, (x_offset, 0))
    draw_comp.rectangle([x_offset - 1, -1, x_offset + main_size, main_size],
                        outline=(60, 60, 60), width=2)

    # Paste candidate cards
    grid_x_offset = (total_w - grid_w) // 2
    for i, pred in enumerate(predictions):
        col = i % n_cols
        row = i // n_cols
        px = grid_x_offset + col * (sub_size + gap)
        py = main_size + gap + row * (sub_size + gap)
        card = _render_hazard_candidate_card(pred, obs, cfg, sub_size)
        composite.paste(card, (px, py))
        border_color = _TRAJ_COLORS[(pred.choice_id - 1) % len(_TRAJ_COLORS)]
        draw_comp.rectangle([px - 1, py - 1, px + sub_size, py + sub_size],
                            outline=border_color, width=3)

    return composite


def _prediction_text(predictions: list[LearnedTrajectory]) -> str:
    rows = []
    for pred in predictions:
        mag = float(np.linalg.norm(pred.action))
        angle = math.degrees(math.atan2(float(pred.action[1]), float(pred.action[0])))
        rows.append(
            f"{pred.choice_id}: force={_format_vec(pred.action)}, "
            f"magnitude={mag:.2f}, angle_deg={angle:+.0f}"
        )
    return "\n".join(rows)


def build_learned_physics_prompt(
    *,
    predictions: list[LearnedTrajectory],
    obs: np.ndarray,
    cfg: PointHazardConfig,
) -> str:
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    speed = float(np.linalg.norm(vel))
    goal_vec = goal - pos
    dist_to_goal = float(np.linalg.norm(goal_vec))
    n_candidates = len(predictions)

    return f"""You control a point-mass agent in a 2D hazard arena.

The image has two parts:
- TOP: current scene (blue agent, green goal "G", red hazards, blue trail = recent path, blue arrow = velocity)
- BOTTOM: {n_candidates} sub-images, each showing the PREDICTED future state if that numbered candidate action is applied.
  Each sub-image shows the predicted agent position and trail (blue).

Current state:
- position={_format_vec(pos)}, velocity={_format_vec(vel)}, speed={speed:.3f}
- vector_to_goal={_format_vec(goal_vec)}, dist_to_goal={dist_to_goal:.3f}

Candidate forces:
{_prediction_text(predictions)}

HARD SAFETY CONSTRAINT:
- Never touch any red hazard circle. Touching a hazard kills the agent immediately and the episode fails.
- Reject candidates whose predicted path enters, touches, or scrapes a hazard.
- When two candidates make similar goal progress, choose the one with larger hazard clearance.

Choose the best numbered candidate to reach the goal while avoiding hazards.
Compare the predicted agent positions across sub-images to judge which path is safest and most direct.

OUTPUT JSON only:
{{"choice": <number>, "reason": "brief explanation"}}
"""


# ---------------------------------------------------------------------------
# Optional LoRA micro-training
# ---------------------------------------------------------------------------

@dataclass
class LoraSample:
    image: Image.Image
    prompt_text: str
    target_text: str
    choice_idx: int


class LoraMicroTrainer:
    def __init__(self, args: argparse.Namespace, model, processor):
        self.update_enabled = bool(args.enable_lora_update)
        self.enabled = bool(args.enable_lora_update or args.resume_lora)
        self.args = args
        self.model = model
        self.processor = processor
        self.samples: list[LoraSample] = []
        self.optimizer = None
        self.last_loss: float | None = None

        if not self.enabled:
            return
        if model is None or processor is None:
            raise RuntimeError("LoRA loading/updating requires --pilot_mode vlm.")
        try:
            import accelerate  # noqa: F401
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA loading/updating requires peft and accelerate. Install on the GPU "
                "machine, for example: pip install peft accelerate transformers"
            ) from exc

        adapter_config = os.path.join(args.lora_out_dir, "adapter_config.json")
        if args.resume_lora:
            if not os.path.exists(adapter_config):
                raise FileNotFoundError(
                    f"--resume_lora was set, but no adapter_config.json was found in {args.lora_out_dir}"
                )
            self.model = PeftModel.from_pretrained(
                model,
                args.lora_out_dir,
                is_trainable=self.update_enabled,
            )
            print(f"Loaded LoRA adapter: {args.lora_out_dir}")
        else:
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
        if self.update_enabled:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=float(args.lora_lr),
            )
            os.makedirs(args.lora_out_dir, exist_ok=True)

    def add_teacher_sample(
        self,
        image: Image.Image,
        prompt_text: str,
        choice_idx: int,
        predictions: list[LearnedTrajectory],
    ) -> None:
        if not self.update_enabled or choice_idx < 0:
            return
        choice = int(choice_idx) + 1
        reason = "Online learned physics predicts this candidate has the best safe goal progress."
        if 0 <= choice_idx < len(predictions):
            pred = predictions[choice_idx]
            reason = (
                f"Online learned physics predicts candidate {choice} with "
                f"clearance {pred.min_clearance:.2f} and progress {pred.goal_progress:.2f}."
            )
        target_text = json.dumps({"choice": choice, "reason": reason})
        self.samples.append(LoraSample(image.copy(), prompt_text, target_text, int(choice_idx)))
        max_samples = int(self.args.lora_buffer_size)
        if len(self.samples) > max_samples:
            self.samples = self.samples[-max_samples:]

    def maybe_update(self, global_step: int) -> float | None:
        if not self.update_enabled or not self.samples:
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

            labels = batch["input_ids"].clone()
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
        if not self.update_enabled:
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
class LearnedPhysicsPivotResult:
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
    predictions: list[LearnedTrajectory]
    teacher_choice_idx: int
    n_candidates: int


def _nearest_candidate_to_vector(candidates: list[np.ndarray], vec: np.ndarray) -> int:
    if not candidates:
        return -1
    vec = np.asarray(vec, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return 0
    unit = vec / norm
    scores = [float(np.dot(c / max(1e-6, float(np.linalg.norm(c))), unit)) for c in candidates]
    return int(np.argmax(scores))


def _local_warmup_choice(obs: np.ndarray, cfg: PointHazardConfig, candidates: list[np.ndarray]) -> int:
    pos, vel, goal, hazards = _obs_parts(obs, cfg)
    to_goal = goal - pos
    goal_norm = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / goal_norm if goal_norm > 1e-6 else np.zeros(2, dtype=np.float32)
    clearance = _clearance_to_hazards(pos, hazards, cfg)
    nearest = _nearest_hazard_vector(pos, hazards)
    away_norm = float(np.linalg.norm(nearest))
    away = -nearest / away_norm if away_norm > 1e-6 else np.zeros(2, dtype=np.float32)
    desired = goal_dir - 0.20 * vel + away * max(0.0, 0.8 - clearance)
    return _nearest_candidate_to_vector(candidates, desired)


def choose_learned_physics_pivot_action(
    *,
    model,
    processor,
    base_image: Image.Image,
    renderer: HazardRenderer,
    obs: np.ndarray,
    cfg: PointHazardConfig,
    physics: OnlinePhysicsAgent,
    args: argparse.Namespace,
    held_action: np.ndarray,
    held_choice_idx: int,
) -> LearnedPhysicsPivotResult:
    candidates = generate_candidates(args.pivot_n_dirs, args.pivot_n_mags)
    predictions = physics.rollout_candidates(
        obs,
        candidates,
        horizon=args.learned_rollout_horizon,
        safety_margin=args.safety_margin,
        clearance_weight=args.clearance_weight,
    )
    model_ready = physics.ready()
    annotated = annotate_learned_predictions(
        base_image,
        candidates,
        predictions,
        renderer,
        agent_world_xy=np.asarray(obs[:2], dtype=np.float32),
        arrow_length_world=args.pivot_arrow_len,
        model_ready=model_ready,
        obs=obs,
        cfg=cfg,
    )
    prompt_text = build_learned_physics_prompt(
        predictions=predictions,
        obs=obs,
        cfg=cfg,
    )
    teacher_idx = physics.choose_teacher(predictions)  # for LoRA labels only

    raw = None
    choice_idx = -1
    parsed_ok = False
    confidence = 0.5
    uncertainty = 0.5
    mean_logprob = float("nan")
    raw_text = ""
    reason = None
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
            fallback_reason = "vlm_parse_failed_zero_action"
            choice_idx = -1

    elif args.pilot_mode == "local":
        choice_idx = _local_warmup_choice(obs, cfg, candidates)
        reason = "local_goal_velocity_heuristic"
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

    if 0 <= choice_idx < len(candidates):
        action = candidates[choice_idx].copy()
    elif held_choice_idx >= 0:
        action = held_action.copy()
    else:
        action = np.zeros(2, dtype=np.float32)

    return LearnedPhysicsPivotResult(
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
        physics_override_used=False,
        physics_override_reason="",
        annotated_image=annotated,
        prompt_text=prompt_text,
        predictions=predictions,
        teacher_choice_idx=int(teacher_idx),
        n_candidates=len(candidates),
    )


def _save_prompt_image(result: LearnedPhysicsPivotResult, run_dir: str, ep: int, t: int) -> str:
    img_dir = os.path.join(run_dir, "prompt_images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"ep{ep:03d}_t{t:03d}.png")
    result.annotated_image.save(path)
    return path


def _prediction_log(predictions: list[LearnedTrajectory]) -> dict[str, Any]:
    return {
        "candidate_actions": [p.action.tolist() for p in predictions],
        "candidate_min_clearance": [float(p.min_clearance) for p in predictions],
        "candidate_safe": [bool(p.safe) for p in predictions],
        "candidate_goal_progress": [float(p.goal_progress) for p in predictions],
        "candidate_final_dist": [float(p.final_dist_to_goal) for p in predictions],
        "candidate_score": [float(p.score) for p in predictions],
    }


# ---------------------------------------------------------------------------
# Smoke test and evaluation
# ---------------------------------------------------------------------------

def _make_physics_agent(args: argparse.Namespace, cfg: PointHazardConfig, obs_dim: int) -> OnlinePhysicsAgent:
    if args.physics_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.physics_device)
    agent = OnlinePhysicsAgent(
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
        agent.load(args.physics_ckpt)
        print(f"Loaded learned physics checkpoint: {args.physics_ckpt}")
    return agent


def _run_physics_warmup(
    args: argparse.Namespace,
    env,
    physics: OnlinePhysicsAgent,
) -> None:
    """Run random episodes to train the physics model before VLM evaluation."""
    n_eps = int(args.physics_warmup_episodes)
    if n_eps <= 0:
        return
    if physics.loaded_from_ckpt:
        print(f"Physics model loaded from checkpoint, skipping warmup.")
        return
    print(f"Warming up physics model with {n_eps} random episodes ...")
    total_steps = 0
    for ep in range(n_eps):
        obs, _ = env.reset(seed=args.seed + 100000 + ep)
        obs = np.asarray(obs, dtype=np.float32)
        for _ in range(args.max_steps):
            action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            physics.add_transition(obs, action, next_obs)
            physics.train_updates(args.physics_updates_per_step)
            obs = next_obs
            total_steps += 1
            if terminated or truncated:
                break
    physics.warmed_up = True
    print(
        f"  Warmup done: {n_eps} episodes, {total_steps} steps, "
        f"{physics.num_samples} samples, loss={physics.last_loss}"
    )


def run_smoke_test(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)
    physics = _make_physics_agent(args, cfg, env.obs_dim)

    obs, _info = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32)
    for step in range(max(0, int(args.smoke_train_steps))):
        action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        next_obs, _reward, terminated, truncated, _info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        physics.add_transition(obs, action, next_obs)
        physics.train_updates(args.physics_updates_per_step)
        obs = next_obs
        if terminated or truncated:
            obs, _info = env.reset(seed=args.seed + step + 1)
            obs = np.asarray(obs, dtype=np.float32)
    physics.warmed_up = True

    frame = env.render()
    if frame is None:
        frame = np.zeros((cfg.render_size, cfg.render_size, 3), dtype=np.uint8)
    image = Image.fromarray(frame)
    result = choose_learned_physics_pivot_action(
        model=None,
        processor=None,
        base_image=image,
        renderer=renderer,
        obs=obs,
        cfg=cfg,
        physics=physics,
        args=argparse.Namespace(**{**vars(args), "pilot_mode": "local"}),
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
    print(f"  physics ready: {physics.ready()}")
    print(f"  latest physics loss: {physics.last_loss}")
    print(f"  teacher choice: {result.teacher_choice_idx + 1 if result.teacher_choice_idx >= 0 else 0}")


def evaluate(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = PointHazardConfig(max_episode_steps=args.max_steps)
    env = make_env(cfg=cfg, seed=0, with_renderer=False)
    renderer = HazardRenderer.from_env(env)
    env.attach_renderer(renderer)

    vlm_model = None
    vlm_processor = None
    if args.pilot_mode == "vlm":
        vlm_model, vlm_processor = _load_local_vlm(args)

    physics = _make_physics_agent(args, cfg, env.obs_dim)
    _run_physics_warmup(args, env, physics)

    lora_trainer = LoraMicroTrainer(args, vlm_model, vlm_processor)
    if lora_trainer.enabled:
        vlm_model = lora_trainer.model

    print("Learned-physics PIVOT, no diffusion, no oracle dynamics rollout")
    print(f"  candidates: {args.pivot_n_dirs} dirs x {args.pivot_n_mags} mags")
    print(f"  physics warmup_episodes={args.physics_warmup_episodes}, horizon={args.learned_rollout_horizon}")
    print(f"  physics device={physics.device}, samples={physics.num_samples}")
    if args.pilot_mode == "vlm":
        print(f"  VLM: {args.model_path}")
    if args.enable_lora_update:
        print(f"  LoRA updates: every={args.lora_update_every}, out={args.lora_out_dir}")

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
    ep_min_clearances: list[float] = []
    ep_physics_losses: list[float] = []
    ep_vlm_calls: list[int] = []
    ep_parse_ok: list[int] = []
    ep_parse_total: list[int] = []
    global_steps = 0

    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        _pos0, _vel0, goal, _haz0 = _obs_parts(obs, cfg)

        held_action = np.zeros(2, dtype=np.float32)
        held_choice_idx = -1
        previous_feedback = ""
        ep_return = 0.0
        last_info: dict[str, Any] = {}
        frames: list[Image.Image] = []
        log_rows: list[dict[str, Any]] = []
        ep_min_clearance = float("inf")
        vlm_calls = 0
        parse_ok_count = 0
        parse_total = 0

        for t in range(args.max_steps):
            frame = env.render()
            if frame is None:
                frame = np.zeros((cfg.render_size, cfg.render_size, 3), dtype=np.uint8)
            image = Image.fromarray(frame)
            if args.save_gif and t % args.gif_every == 0:
                frames.append(image.copy())

            pos, vel, _goal, hazards = _obs_parts(obs, cfg)
            current_clearance = _clearance_to_hazards(pos, hazards, cfg)
            prev_speed = float(np.linalg.norm(vel))
            prev_dist = float(np.linalg.norm(goal - pos))
            ep_min_clearance = min(ep_min_clearance, current_clearance)

            vlm_called = False
            if args.pilot_mode == "vlm":
                vlm_called = (t % max(1, int(args.vlm_every)) == 0)
            if args.pilot_mode != "vlm" or vlm_called:
                result = choose_learned_physics_pivot_action(
                    model=vlm_model,
                    processor=vlm_processor,
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
                parse_ok_count += int(args.pilot_mode == "vlm" and result.parsed_ok)

                if result.teacher_choice_idx >= 0:
                    lora_trainer.add_teacher_sample(
                        result.annotated_image,
                        result.prompt_text,
                        result.teacher_choice_idx,
                        result.predictions,
                    )
                prompt_image_path = ""
                if args.save_prompt_images and args.save_log:
                    prompt_image_path = _save_prompt_image(result, run_dir, ep, t)
            else:
                result = None
                prompt_image_path = ""

            exec_action = np.clip(held_action, -1.0, 1.0).astype(np.float32)
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(exec_action)
            obs = np.asarray(obs, dtype=np.float32)
            physics.add_transition(prev_obs, exec_action, obs)
            physics_loss = physics.train_updates(args.physics_updates_per_step)
            if physics_loss is not None:
                ep_physics_losses.append(float(physics_loss))

            lora_loss = lora_trainer.maybe_update(global_steps)

            next_pos, next_vel, _next_goal, next_hazards = _obs_parts(obs, cfg)
            next_clearance = _clearance_to_hazards(next_pos, next_hazards, cfg)
            next_speed = float(np.linalg.norm(next_vel))
            next_dist = float(np.linalg.norm(goal - next_pos))
            goal_progress = prev_dist - next_dist
            clearance_delta = next_clearance - current_clearance
            ep_min_clearance = min(ep_min_clearance, next_clearance)
            ep_return += float(reward)
            last_info = dict(info or {})
            global_steps += 1

            previous_feedback = _real_step_feedback(
                choice_idx=held_choice_idx,
                action=exec_action,
                prev_speed=prev_speed,
                next_speed=next_speed,
                goal_progress=goal_progress,
                clearance_delta=clearance_delta,
                next_clearance=next_clearance,
                hazard_hit=bool(last_info.get("hazard_hit", False)),
                goal_success=bool(last_info.get("goal_success", False)),
            )

            if args.debug_print:
                choice = held_choice_idx + 1 if held_choice_idx >= 0 else 0
                ready = physics.ready()
                print(
                    f"  [ep {ep:03d} t={t:03d}] choice={choice} "
                    f"action={_format_vec(exec_action)} ready={ready} "
                    f"loss={physics_loss} clear={next_clearance:.2f}"
                )

            if args.save_log:
                row = {
                    "t": int(t),
                    "pilot_mode": args.pilot_mode,
                    "agent_pos": pos.tolist(),
                    "agent_vel": vel.tolist(),
                    "goal": goal.tolist(),
                    "current_clearance": float(current_clearance),
                    "next_clearance": float(next_clearance),
                    "dist_to_goal": float(last_info.get("dist_to_goal", float("nan"))),
                    "prev_speed": float(prev_speed),
                    "next_speed": float(next_speed),
                    "goal_progress": float(goal_progress),
                    "clearance_delta": float(clearance_delta),
                    "choice": int(held_choice_idx + 1) if held_choice_idx >= 0 else 0,
                    "exec": exec_action.tolist(),
                    "reward": float(reward),
                    "goal_success": bool(last_info.get("goal_success", False)),
                    "hazard_hit": bool(last_info.get("hazard_hit", False)),
                    "real_step_feedback": previous_feedback,
                    "physics_samples": int(physics.num_samples),
                    "physics_train_steps": int(physics.train_steps),
                    "physics_ready": bool(physics.ready()),
                    "physics_loss": None if physics_loss is None else float(physics_loss),
                    "lora_loss": None if lora_loss is None else float(lora_loss),
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
                        "pivot_n_candidates": int(result.n_candidates),
                    })
                    row.update(_prediction_log(result.predictions))
                log_rows.append(row)

            if terminated or truncated:
                break

        success = bool(last_info.get("goal_success", False))
        hazard_hit = bool(last_info.get("hazard_hit", False))
        dist = float(last_info.get("dist_to_goal", float("nan")))
        term_reason = last_info.get("termination_reason", "timeout")

        ep_returns.append(ep_return)
        ep_successes.append(int(success))
        ep_hazard_hits.append(int(hazard_hit))
        ep_final_dist.append(dist)
        ep_min_clearances.append(ep_min_clearance)
        ep_vlm_calls.append(vlm_calls)
        ep_parse_ok.append(parse_ok_count)
        ep_parse_total.append(parse_total)

        flag = "OK " if success else ("HIT" if hazard_hit else "---")
        print(
            f"[ep {ep:03d}] [{flag}] return={ep_return:7.1f} steps={t + 1:3d} "
            f"dist={dist:.2f} min_clear={ep_min_clearance:.2f} "
            f"phys_samples={physics.num_samples} reason={term_reason}"
        )

        if args.save_log:
            log_path = os.path.join(run_dir, f"ep{ep:03d}.jsonl")
            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "type": "episode_meta",
                    "episode": ep,
                    "pilot_mode": args.pilot_mode,
                    "model": args.model_path,
                    "seed": args.seed + ep,
                    "goal": goal.tolist(),
                    "learned_physics_pivot": True,
                    "future_oracle_rollout": False,
                    "use_diffusion": False,
                    "safe_expert": False,
                    "args": vars(args),
                }) + "\n")
                for row in log_rows:
                    f.write(json.dumps(row) + "\n")
                f.write(json.dumps({
                    "type": "episode_end",
                    "return": ep_return,
                    "success": success,
                    "hazard_hit": hazard_hit,
                    "dist": dist,
                    "steps": t + 1,
                    "min_clearance": ep_min_clearance,
                    "vlm_calls": vlm_calls,
                    "parse_ok": parse_ok_count,
                    "parse_total": parse_total,
                }) + "\n")

        if args.save_gif and frames:
            dur = int(1000 / max(1e-6, args.gif_fps))
            tag = "ok" if success else ("hit" if hazard_hit else "timeout")
            gif_path = os.path.join(args.gif_dir, f"ep{ep:03d}_{tag}.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=dur,
                loop=0,
            )

    env.close()

    n = int(args.episodes)
    succ_rate = float(np.mean(ep_successes))
    hit_rate = float(np.mean(ep_hazard_hits))
    timeout_rate = 1.0 - succ_rate - hit_rate
    mean_ret = float(np.mean(ep_returns))
    mean_dist = float(np.nanmean(ep_final_dist))
    mean_min_clearance = float(np.nanmean(ep_min_clearances))
    mean_vlm_calls = float(np.mean(ep_vlm_calls))
    parse_total_sum = int(np.sum(ep_parse_total))
    parse_success_rate = float(np.sum(ep_parse_ok) / parse_total_sum) if parse_total_sum else float("nan")
    mean_loss = float(np.nanmean(ep_physics_losses)) if ep_physics_losses else float("nan")

    print(f"\n{'=' * 60}")
    print("Learned-physics PIVOT  Diffusion: OFF  Oracle rollout: OFF")
    print(f"  Pilot: {args.pilot_mode}  Episodes: {n}")
    print(f"  Success:    {sum(ep_successes):3d}/{n}  ({succ_rate:.1%})")
    print(f"  Hazard hit: {sum(ep_hazard_hits):3d}/{n}  ({hit_rate:.1%})")
    print(f"  Timeout:    {n - sum(ep_successes) - sum(ep_hazard_hits):3d}/{n}  ({timeout_rate:.1%})")
    print(f"  Mean return: {mean_ret:.1f}")
    print(f"  Mean final dist: {mean_dist:.2f}")
    print(f"  Mean min clearance: {mean_min_clearance:.2f}")
    print(f"  Physics samples: {physics.num_samples}, train_steps={physics.train_steps}, mean_loss={mean_loss:.6f}")
    print(f"  VLM calls/ep: {mean_vlm_calls:.1f}")
    print(f"  Parse success: {parse_success_rate:.1%}" if parse_total_sum else "  Parse success: n/a")
    print(f"{'=' * 60}")

    if args.save_physics_ckpt:
        physics.save(args.physics_ckpt)
        print(f"Learned physics checkpoint saved: {args.physics_ckpt}")

    if args.enable_lora_update:
        lora_trainer.save()
        print(f"LoRA adapter saved: {args.lora_out_dir}")

    if args.save_log:
        summary = {
            "pilot_mode": args.pilot_mode,
            "model": args.model_path,
            "learned_physics_pivot": True,
            "future_oracle_rollout": False,
            "use_diffusion": False,
            "episodes": n,
            "success_rate": succ_rate,
            "hazard_hit_rate": hit_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_ret,
            "mean_final_dist": mean_dist,
            "mean_min_clearance": mean_min_clearance,
            "physics_samples": int(physics.num_samples),
            "physics_train_steps": int(physics.train_steps),
            "mean_physics_loss": mean_loss,
            "mean_vlm_calls": mean_vlm_calls,
            "parse_success_rate": parse_success_rate,
            "lora_enabled": bool(args.enable_lora_update),
            "args": vars(args),
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned-physics PIVOT on PointHazardEnv without diffusion or oracle rollouts")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--pilot_mode", type=str, default="vlm",
                        choices=["vlm", "local", "random"])

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
    parser.add_argument("--vlm_every", type=int, default=1)
    parser.add_argument("--vlm_retries", type=int, default=1)

    parser.add_argument("--pivot_n_dirs", type=int, default=8)
    parser.add_argument("--pivot_n_mags", type=int, default=1)
    parser.add_argument("--pivot_arrow_len", type=float, default=0.6)

    parser.add_argument("--learned_rollout_horizon", type=int, default=3)
    parser.add_argument("--safety_margin", type=float, default=0.15)
    parser.add_argument("--clearance_weight", type=float, default=0.25)
    parser.add_argument("--physics_warmup_episodes", type=int, default=100)
    parser.add_argument("--physics_batch_size", type=int, default=256)
    parser.add_argument("--physics_lr", type=float, default=1e-3)
    parser.add_argument("--physics_hidden_dim", type=int, default=256)
    parser.add_argument("--physics_replay_capacity", type=int, default=100000)
    parser.add_argument("--physics_updates_per_step", type=int, default=1)
    parser.add_argument("--physics_grad_clip", type=float, default=5.0)
    parser.add_argument("--physics_device", type=str, default="auto")
    parser.add_argument("--physics_ckpt", type=str, default="hazard/learned_physics_agent.pt")
    parser.add_argument("--resume_physics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_physics_ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_lora_update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume_lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_update_every", type=int, default=50)
    parser.add_argument("--lora_steps_per_update", type=int, default=4)
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_out_dir", type=str, default="hazard/lora_learned_physics_pivot")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lora_buffer_size", type=int, default=512)
    parser.add_argument("--lora_grad_clip", type=float, default=1.0)
    parser.add_argument("--lora_save_every_update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--save_log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_dir", type=str, default="hazard/logs_hazard_learned_physics_pivot")
    parser.add_argument("--save_prompt_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif_dir", type=str, default="hazard/gifs_hazard_learned_physics_pivot")
    parser.add_argument("--gif_fps", type=float, default=10.0)
    parser.add_argument("--gif_every", type=int, default=2)
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_out", type=str, default="/tmp/learned_physics_pivot_smoke.png")
    parser.add_argument("--smoke_train_steps", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_lora_update and args.pilot_mode != "vlm":
        raise RuntimeError("--enable_lora_update requires --pilot_mode vlm.")
    if args.resume_lora and args.pilot_mode != "vlm":
        raise RuntimeError("--resume_lora requires --pilot_mode vlm.")
    if args.smoke_test:
        run_smoke_test(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
