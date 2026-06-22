"""
Qwen-backed direct VLA baseline for PointHazardEnv.

This is the direct-action VLA baseline: it does not use PIVOT candidates,
candidate arrows, or rollout visualizations. The policy is:

    Qwen-VL(image + task/state prompt) -> hidden state -> action head -> [fx, fy]

The action head is trained from target actions with MSE/flow/diffusion loss.
By default those targets are SafeExpert demonstrations. For a score-teacher
LoRA comparison, use --train_target physics_score to relabel each training
state with the candidate action maximizing
goal_progress + clearance_weight * min_clearance. By default the Qwen backbone
is frozen and only the regression head is trained, which is the cheapest
same-backbone VLA adaptation. Use --train_lora for a direct-VLA LoRA baseline,
or --train_vlm if you want to fine-tune the whole Qwen backbone.

Examples:
    python direct_vla_hazard.py --mode collect_expert \
        --episodes 1000 \
        --seed 42 \
        --out hazard/direct_vla_expert_demos.npz

    python direct_vla_hazard.py --mode train \
        --dataset hazard/direct_vla_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_head.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_hazard.py --mode train \
        --dataset hazard/direct_vla_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_diffusion_head.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --action_head_type diffusion \
        --train_lora \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_hazard.py --mode train \
        --dataset hazard/direct_vla_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_flow_head.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --action_head_type flow \
        --train_lora \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_hazard.py --mode train \
        --dataset hazard/direct_vla_expert_demos.npz \
        --ckpt hazard/direct_qwen_vla_lora_physics_score.pt \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --train_lora \
        --train_target physics_score \
        --epochs 3 \
        --batch_size 1 \
        --seed 42

    python direct_vla_hazard.py --mode eval \
        --ckpt hazard/direct_qwen_vla_head.pt \
        --episodes 100 \
        --seed 1000 \
        --summary hazard/direct_qwen_vla_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from env_pointhazard import PointHazardConfig, make_env
from hazard_renderer import HazardRenderer
from safe_expert import SafeExpert, SafeExpertConfig


TASK_TEXT = "Reach the green goal while avoiding red circular hazards."


# ---------------------------------------------------------------------------
# Reproducibility and device helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def get_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def torch_dtype_from_arg(name: str):
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


def module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


# ---------------------------------------------------------------------------
# Observation, rendering, and prompt helpers
# ---------------------------------------------------------------------------

def obs_parts(obs: np.ndarray, cfg: PointHazardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_hazards = int(cfg.n_hazards)
    pos = obs[0:2].astype(np.float32)
    vel = obs[2:4].astype(np.float32)
    goal = obs[4:6].astype(np.float32)
    hazards = obs[6:6 + 3 * n_hazards].reshape(n_hazards, 3).astype(np.float32)
    return pos, vel, goal, hazards


def clearance_to_hazards(pos: np.ndarray, hazards: np.ndarray, cfg: PointHazardConfig) -> float:
    hazards = np.asarray(hazards, dtype=np.float32).reshape(-1, 3)
    if hazards.size == 0:
        return float("inf")
    pos = np.asarray(pos, dtype=np.float32).reshape(2)
    center_d = np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)
    clearances = center_d - hazards[:, 2] - float(cfg.agent_radius)
    return float(np.min(clearances))


def obs_clearance(obs: np.ndarray, cfg: PointHazardConfig) -> float:
    pos, _, _, hazards = obs_parts(obs, cfg)
    return clearance_to_hazards(pos, hazards, cfg)


def simulate_point_hazard_step(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    cfg: PointHazardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One PointHazardEnv dynamics step, without mutating an env instance."""
    pos = np.asarray(pos, dtype=np.float32).reshape(2).copy()
    vel = np.asarray(vel, dtype=np.float32).reshape(2).copy()
    action = np.asarray(action, dtype=np.float32).reshape(2)
    action = np.clip(action, -1.0, 1.0)

    force = float(cfg.force_scale) * action
    new_vel = vel + (force - float(cfg.drag) * vel) * float(cfg.dt)
    speed = float(np.linalg.norm(new_vel))
    if speed > float(cfg.max_speed):
        new_vel = new_vel * (float(cfg.max_speed) / speed)

    new_pos = pos + new_vel * float(cfg.dt)
    arena_limit = float(cfg.arena_half) - float(cfg.agent_radius)
    for i in range(2):
        if new_pos[i] < -arena_limit:
            new_pos[i] = -arena_limit
            new_vel[i] = -new_vel[i] * 0.5
        elif new_pos[i] > arena_limit:
            new_pos[i] = arena_limit
            new_vel[i] = -new_vel[i] * 0.5

    return new_pos.astype(np.float32), new_vel.astype(np.float32)


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


def score_constant_action_rollout(
    obs: np.ndarray,
    cfg: PointHazardConfig,
    action: np.ndarray,
    *,
    horizon: int,
    safety_margin: float,
    clearance_weight: float,
    unsafe_penalty: float,
) -> tuple[float, float, float, bool]:
    pos, vel, goal, hazards = obs_parts(obs, cfg)
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

    start_dist = float(np.linalg.norm(goal - pos))
    min_clearance = clearance_to_hazards(pos, hazards, cfg)
    safe = min_clearance >= float(safety_margin)
    p = pos.copy()
    v = vel.copy()

    for _ in range(max(1, int(horizon))):
        p, v = simulate_point_hazard_step(p, v, action, cfg)
        clear = clearance_to_hazards(p, hazards, cfg)
        min_clearance = min(min_clearance, clear)
        if clear < float(safety_margin):
            safe = False

    final_dist = float(np.linalg.norm(goal - p))
    goal_progress = start_dist - final_dist
    score = goal_progress + float(clearance_weight) * min_clearance
    if not safe:
        score -= float(unsafe_penalty)
    return float(score), float(min_clearance), float(goal_progress), bool(safe)


def relabel_actions_with_physics_score(
    obs: np.ndarray,
    actions: np.ndarray,
    cfg: PointHazardConfig,
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
                ob,
                cfg,
                candidate,
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
        score, clearance, progress, safe = scored[best_idx]
        target_actions[i] = np.asarray(pool[best_idx], dtype=np.float32)
        best_scores.append(score)
        best_clearances.append(clearance)
        best_progresses.append(progress)
        best_safe.append(float(safe))
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


def make_renderer(cfg: PointHazardConfig, render_size: int) -> HazardRenderer:
    return HazardRenderer(
        arena_half=cfg.arena_half,
        agent_radius=cfg.agent_radius,
        goal_radius=cfg.goal_radius,
        img_size=int(render_size),
    )


def render_obs_pil(obs: np.ndarray, cfg: PointHazardConfig, renderer: HazardRenderer) -> Image.Image:
    pos, vel, goal, hazards = obs_parts(obs, cfg)
    arr = renderer.render(
        agent_xy=pos,
        goal_xy=goal,
        hazards=hazards,
        vel_xy=vel,
        trail=None,
        info_text=None,
    )
    return Image.fromarray(arr)


def format_vec(v: np.ndarray) -> str:
    return f"[{float(v[0]):+.3f}, {float(v[1]):+.3f}]"


def build_vla_prompt(obs: np.ndarray, cfg: PointHazardConfig, prompt_mode: str = "image_state") -> str:
    prompt_mode = str(prompt_mode)
    if prompt_mode == "image_only":
        return f"""You are a direct vision-language-action controller for a 2D point-mass robot.

Task: {TASK_TEXT}

Visual elements:
- Red filled circles are hazards. Touching one ends the episode.
- Blue dot is the controlled agent.
- Green circle marked G is the goal.
- Blue arrow shows current velocity.

Predict the next continuous action [fx, fy], where each component is in [-1, 1].
The action should move toward the goal while staying clear of hazards.
"""
    if prompt_mode != "image_state":
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    pos, vel, goal, hazards = obs_parts(obs, cfg)
    goal_vec = goal - pos
    nearest_idx = int(np.argmin(np.linalg.norm(hazards[:, :2] - pos[None, :], axis=1)))
    nearest = hazards[nearest_idx]
    nearest_vec = nearest[:2] - pos
    clearance = clearance_to_hazards(pos, hazards, cfg)
    speed = float(np.linalg.norm(vel))
    dist_to_goal = float(np.linalg.norm(goal_vec))

    return f"""You are a direct vision-language-action controller for a 2D point-mass robot.

Task: {TASK_TEXT}

Visual elements:
- Red filled circles are hazards. Touching one ends the episode.
- Blue dot is the controlled agent.
- Green circle marked G is the goal.
- Blue arrow shows current velocity.

Current measured state:
- position={format_vec(pos)}
- velocity={format_vec(vel)}, speed={speed:.3f}
- goal_position={format_vec(goal)}
- vector_to_goal={format_vec(goal_vec)}, dist_to_goal={dist_to_goal:.3f}
- nearest_hazard_center={format_vec(nearest[:2])}, radius={float(nearest[2]):.3f}
- vector_to_nearest_hazard_center={format_vec(nearest_vec)}
- current_body_to_hazard_clearance={clearance:.3f}

Predict the next continuous action [fx, fy], where each component is in [-1, 1].
The action should move toward the goal while staying clear of hazards.
"""


def expert_act_obs(obs: np.ndarray, cfg: PointHazardConfig) -> np.ndarray:
    """Return an obs copy in the relative-hazard format SafeExpert.act expects."""
    obs = np.asarray(obs, dtype=np.float32).copy()
    pos = obs[0:2].astype(np.float32)
    n_hazards = int(cfg.n_hazards)
    hazards = obs[6:6 + 3 * n_hazards].reshape(n_hazards, 3).copy()
    hazards[:, 0:2] -= pos[None, :]
    obs[6:6 + 3 * n_hazards] = hazards.reshape(-1)
    return obs


def finite_mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.nanmean(np.asarray(xs, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Dataset and Qwen batch construction
# ---------------------------------------------------------------------------

class QwenVLADataset(Dataset):
    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        *,
        cfg: PointHazardConfig,
        render_size: int,
        prompt_mode: str,
    ):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.cfg = cfg
        self.render_size = int(render_size)
        self.prompt_mode = str(prompt_mode)
        self._renderer: HazardRenderer | None = None

        if self.obs.ndim != 2:
            raise ValueError(f"Expected obs shape (N, D), got {self.obs.shape}")
        if self.actions.ndim != 2 or self.actions.shape[1] != 2:
            raise ValueError(f"Expected actions shape (N, 2), got {self.actions.shape}")
        if self.obs.shape[0] != self.actions.shape[0]:
            raise ValueError("obs and actions must have the same number of rows")

    @property
    def renderer(self) -> HazardRenderer:
        if self._renderer is None:
            self._renderer = make_renderer(self.cfg, self.render_size)
        return self._renderer

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        obs = self.obs[int(idx)]
        return {
            "image": render_obs_pil(obs, self.cfg, self.renderer),
            "prompt": build_vla_prompt(obs, self.cfg, self.prompt_mode),
            "action": self.actions[int(idx)].astype(np.float32),
        }


def qwen_vla_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in items],
        "prompts": [item["prompt"] for item in items],
        "actions": torch.from_numpy(np.stack([item["action"] for item in items], axis=0).astype(np.float32)),
    }


def load_npz_dataset(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    meta: dict[str, Any] = {}
    if "metadata" in data:
        meta["metadata"] = data["metadata"].tolist()
    for key in ("env_config_json", "expert_config_json", "task_text"):
        if key in data:
            value = data[key]
            meta[key] = str(value.item()) if np.asarray(value).shape == () else str(value)
    return obs, actions, meta


def cfg_from_dataset_meta(meta: dict[str, Any]) -> PointHazardConfig:
    raw = meta.get("env_config_json")
    if not raw:
        return PointHazardConfig()
    try:
        data = json.loads(str(raw))
        known = {f.name for f in PointHazardConfig.__dataclass_fields__.values()}
        return PointHazardConfig(**{k: v for k, v in data.items() if k in known})
    except (TypeError, json.JSONDecodeError):
        return PointHazardConfig()


def split_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    val_frac: float,
    seed: int,
    max_train_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(obs.shape[0])
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    if max_train_samples is not None and max_train_samples > 0:
        order = order[: min(len(order), int(max_train_samples))]
        n = int(order.shape[0])
    n_val = int(round(float(val_frac) * n))
    n_val = max(0, min(n_val, n - 1)) if n > 1 else 0
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return obs[train_idx], actions[train_idx], obs[val_idx], actions[val_idx]


def build_processor_inputs(processor, images: list[Image.Image], prompts: list[str], device: torch.device) -> dict[str, Any]:
    texts = []
    for image, prompt in zip(images, prompts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            texts.append(text)
        except TypeError:
            batch = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return move_batch_to_device(batch, device)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return move_batch_to_device(batch, device)


# ---------------------------------------------------------------------------
# Qwen VLA model
# ---------------------------------------------------------------------------

def load_qwen_backbone(args: argparse.Namespace):
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Qwen VLA requires transformers installed on the GPU machine.") from exc

    print(f"Loading Qwen VLM backbone: {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    dtype = torch_dtype_from_arg(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs)
    if not args.device_map:
        model.to(get_device(args.device))
    model.eval()
    print("Qwen backbone loaded.")
    return model, processor


def infer_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise RuntimeError("Could not infer Qwen hidden size from model config.")


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int, head_hidden_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_hidden_dim = int(head_hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_hidden.float())


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1, 1)
        half = self.dim // 2
        if half <= 0:
            return t.new_zeros((t.shape[0], 0))
        freqs = torch.exp(
            torch.linspace(
                0.0,
                math.log(10000.0),
                half,
                device=t.device,
                dtype=t.dtype,
            )
            * -1.0
        )
        args = t * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def make_beta_schedule_tensor(
    schedule: str,
    num_steps: int,
    beta_min: float,
    beta_max: float,
) -> torch.Tensor:
    if int(num_steps) <= 1:
        raise ValueError("num_steps must be > 1 for a diffusion action head.")
    schedule = str(schedule)
    if schedule == "linear":
        betas = torch.linspace(float(beta_min), float(beta_max), int(num_steps), dtype=torch.float32)
    elif schedule == "sigmoid":
        grid = torch.linspace(-6.0, 6.0, int(num_steps), dtype=torch.float32)
        betas = torch.sigmoid(grid) * (float(beta_max) - float(beta_min)) + float(beta_min)
    else:
        raise ValueError(f"Unknown diffusion beta schedule: {schedule}")
    return betas.clamp(1e-8, 0.999)


class GenerativeActionMixin:
    def sample_actions(self, pooled_hidden: torch.Tensor, *, n_samples: int = 1) -> torch.Tensor:
        raise NotImplementedError


class DiffusionActionHead(nn.Module, GenerativeActionMixin):
    """Qwen-conditioned DDPM action expert for continuous [fx, fy]."""

    def __init__(
        self,
        hidden_size: int,
        *,
        cond_dim: int = 256,
        head_hidden_dim: int = 512,
        time_dim: int = 64,
        dropout: float = 0.05,
        num_steps: int = 50,
        beta_schedule: str = "sigmoid",
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.cond_dim = int(cond_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.time_dim = int(time_dim)
        self.dropout = float(dropout)
        self.num_steps = int(num_steps)
        self.beta_schedule = str(beta_schedule)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        self.cond_net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.cond_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.time_embed = SinusoidalTimeEmbedding(self.time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + self.cond_dim + self.time_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, 2),
        )

        betas = make_beta_schedule_tensor(self.beta_schedule, self.num_steps, self.beta_min, self.beta_max)
        alphas = (1.0 - betas).clamp(1e-8, 1.0)
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars), persistent=False)

    def _predict_eps(self, noisy_action: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_norm = t.float() / max(1, self.num_steps - 1)
        h = torch.cat([noisy_action.float(), cond.float(), self.time_embed(t_norm)], dim=-1)
        return self.net(h)

    def loss(self, pooled_hidden: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        target_action = target_action.float().clamp(-1.0, 1.0)
        cond = self.cond_net(pooled_hidden.float())
        b = int(target_action.shape[0])
        t = torch.randint(0, self.num_steps, (b,), device=target_action.device)
        eps = torch.randn_like(target_action)
        a = self.sqrt_alpha_bars[t].view(-1, 1).to(target_action.device)
        am1 = self.sqrt_one_minus_alpha_bars[t].view(-1, 1).to(target_action.device)
        noisy_action = a * target_action + am1 * eps
        pred_eps = self._predict_eps(noisy_action, t, cond)
        return F.mse_loss(pred_eps, eps)

    @torch.no_grad()
    def sample_actions(self, pooled_hidden: torch.Tensor, *, n_samples: int = 1) -> torch.Tensor:
        self.eval()
        pooled_hidden = pooled_hidden.float()
        b = int(pooled_hidden.shape[0])
        n_samples = max(1, int(n_samples))
        cond = self.cond_net(pooled_hidden)
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, dim=0)
        action = torch.randn((b * n_samples, 2), device=pooled_hidden.device, dtype=torch.float32)

        for t_int in reversed(range(self.num_steps)):
            t = torch.full((action.shape[0],), t_int, device=action.device, dtype=torch.long)
            eps_theta = self._predict_eps(action, t, cond)
            beta_t = self.betas[t].view(-1, 1).to(action.device)
            alpha_t = self.alphas[t].view(-1, 1).to(action.device)
            abar_t = self.alpha_bars[t].view(-1, 1).to(action.device)
            coef = (1.0 - alpha_t) / torch.sqrt((1.0 - abar_t).clamp_min(1e-8))
            mean = (action - coef * eps_theta) / torch.sqrt(alpha_t.clamp_min(1e-8))
            if t_int > 0:
                action = mean + torch.sqrt(beta_t.clamp_min(1e-8)) * torch.randn_like(action)
            else:
                action = mean
            action = action.clamp(-2.0, 2.0)

        action = action.tanh()
        if n_samples > 1:
            action = action.view(b, n_samples, 2).mean(dim=1)
        return action.clamp(-1.0, 1.0)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.sample_actions(pooled_hidden, n_samples=1)


class FlowActionHead(nn.Module, GenerativeActionMixin):
    """Qwen-conditioned rectified-flow action expert."""

    def __init__(
        self,
        hidden_size: int,
        *,
        cond_dim: int = 256,
        head_hidden_dim: int = 512,
        time_dim: int = 64,
        dropout: float = 0.05,
        sample_steps: int = 16,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.cond_dim = int(cond_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.time_dim = int(time_dim)
        self.dropout = float(dropout)
        self.sample_steps = int(sample_steps)

        self.cond_net = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.cond_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.time_embed = SinusoidalTimeEmbedding(self.time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + self.cond_dim + self.time_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, 2),
        )

    def _predict_velocity(self, action_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = torch.cat([action_t.float(), cond.float(), self.time_embed(t.float())], dim=-1)
        return self.net(h)

    def loss(self, pooled_hidden: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        target_action = target_action.float().clamp(-1.0, 1.0)
        cond = self.cond_net(pooled_hidden.float())
        noise = torch.randn_like(target_action)
        t = torch.rand((target_action.shape[0], 1), device=target_action.device)
        action_t = (1.0 - t) * noise + t * target_action
        target_velocity = target_action - noise
        pred_velocity = self._predict_velocity(action_t, t.squeeze(-1), cond)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_actions(self, pooled_hidden: torch.Tensor, *, n_samples: int = 1) -> torch.Tensor:
        self.eval()
        pooled_hidden = pooled_hidden.float()
        b = int(pooled_hidden.shape[0])
        n_samples = max(1, int(n_samples))
        cond = self.cond_net(pooled_hidden)
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, dim=0)
        action = torch.randn((b * n_samples, 2), device=pooled_hidden.device, dtype=torch.float32)
        steps = max(1, int(self.sample_steps))
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((action.shape[0],), (i + 0.5) / steps, device=action.device)
            velocity = self._predict_velocity(action, t, cond)
            action = (action + dt * velocity).clamp(-2.0, 2.0)
        action = action.tanh()
        if n_samples > 1:
            action = action.view(b, n_samples, 2).mean(dim=1)
        return action.clamp(-1.0, 1.0)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.sample_actions(pooled_hidden, n_samples=1)


def action_head_type(action_head: nn.Module) -> str:
    if isinstance(action_head, DiffusionActionHead):
        return "diffusion"
    if isinstance(action_head, FlowActionHead):
        return "flow"
    return "regression"


def make_action_head(args: argparse.Namespace, hidden_size: int) -> nn.Module:
    head_type = str(getattr(args, "action_head_type", "regression"))
    if head_type == "regression":
        return ActionHead(
            hidden_size=hidden_size,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.head_dropout,
        )
    if head_type == "diffusion":
        return DiffusionActionHead(
            hidden_size=hidden_size,
            cond_dim=args.generative_cond_dim,
            head_hidden_dim=args.head_hidden_dim,
            time_dim=args.generative_time_dim,
            dropout=args.head_dropout,
            num_steps=args.diffusion_steps,
            beta_schedule=args.diffusion_beta_schedule,
            beta_min=args.diffusion_beta_min,
            beta_max=args.diffusion_beta_max,
        )
    if head_type == "flow":
        return FlowActionHead(
            hidden_size=hidden_size,
            cond_dim=args.generative_cond_dim,
            head_hidden_dim=args.head_hidden_dim,
            time_dim=args.generative_time_dim,
            dropout=args.head_dropout,
            sample_steps=args.flow_sample_steps,
        )
    raise ValueError(f"Unknown action_head_type: {head_type}")


def make_action_head_from_ckpt(ckpt: dict[str, Any]) -> nn.Module:
    head_type = str(ckpt.get("action_head_type", "regression"))
    cfg = ckpt["action_head_config"]
    if head_type == "regression":
        return ActionHead(
            hidden_size=int(cfg["hidden_size"]),
            head_hidden_dim=int(cfg.get("head_hidden_dim", 512)),
            dropout=float(cfg.get("dropout", 0.05)),
        )
    if head_type == "diffusion":
        return DiffusionActionHead(
            hidden_size=int(cfg["hidden_size"]),
            cond_dim=int(cfg.get("cond_dim", 256)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", 512)),
            time_dim=int(cfg.get("time_dim", 64)),
            dropout=float(cfg.get("dropout", 0.05)),
            num_steps=int(cfg.get("num_steps", 50)),
            beta_schedule=str(cfg.get("beta_schedule", "sigmoid")),
            beta_min=float(cfg.get("beta_min", 1e-4)),
            beta_max=float(cfg.get("beta_max", 0.02)),
        )
    if head_type == "flow":
        return FlowActionHead(
            hidden_size=int(cfg["hidden_size"]),
            cond_dim=int(cfg.get("cond_dim", 256)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", 512)),
            time_dim=int(cfg.get("time_dim", 64)),
            dropout=float(cfg.get("dropout", 0.05)),
            sample_steps=int(cfg.get("sample_steps", 16)),
        )
    raise ValueError(f"Unknown checkpoint action_head_type: {head_type}")


def action_head_config(action_head: nn.Module, args: argparse.Namespace, hidden_size: int) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "hidden_size": int(hidden_size),
        "head_hidden_dim": int(args.head_hidden_dim),
        "dropout": float(args.head_dropout),
    }
    if isinstance(action_head, DiffusionActionHead):
        cfg.update(
            {
                "cond_dim": int(action_head.cond_dim),
                "time_dim": int(action_head.time_dim),
                "num_steps": int(action_head.num_steps),
                "beta_schedule": str(action_head.beta_schedule),
                "beta_min": float(action_head.beta_min),
                "beta_max": float(action_head.beta_max),
            }
        )
    elif isinstance(action_head, FlowActionHead):
        cfg.update(
            {
                "cond_dim": int(action_head.cond_dim),
                "time_dim": int(action_head.time_dim),
                "sample_steps": int(action_head.sample_steps),
            }
        )
    return cfg


def pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, -1, :]
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    gather_idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    return hidden.gather(dim=1, index=gather_idx).squeeze(1)


def qwen_forward_pooled(
    *,
    qwen,
    processor,
    images: list[Image.Image],
    prompts: list[str],
    train_vlm: bool,
    train_lora: bool = False,
) -> torch.Tensor:
    device = module_device(qwen)
    batch = build_processor_inputs(processor, images, prompts, device)

    train_backbone = bool(train_vlm or train_lora)
    ctx = torch.enable_grad() if train_backbone else torch.no_grad()
    with ctx:
        out = qwen(
            **batch,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        pooled = pool_last_token(hidden, batch.get("attention_mask"))
        if not train_backbone:
            pooled = pooled.detach()

    return pooled


def qwen_forward_action(
    *,
    qwen,
    action_head: nn.Module,
    processor,
    images: list[Image.Image],
    prompts: list[str],
    train_vlm: bool,
    train_lora: bool = False,
    action_samples: int = 1,
) -> torch.Tensor:
    pooled = qwen_forward_pooled(
        qwen=qwen,
        processor=processor,
        images=images,
        prompts=prompts,
        train_vlm=train_vlm,
        train_lora=train_lora,
    )
    if module_device(action_head) != pooled.device:
        action_head.to(pooled.device)
    if isinstance(action_head, GenerativeActionMixin):
        return action_head.sample_actions(pooled, n_samples=action_samples)
    return action_head(pooled)


def is_lora_parameter(name: str) -> bool:
    return "lora_" in str(name)


def set_qwen_trainability(qwen, *, train_vlm: bool, train_lora: bool = False) -> None:
    train_backbone = bool(train_vlm or train_lora)
    if train_vlm:
        for p in qwen.parameters():
            p.requires_grad = True
    elif train_lora:
        for name, p in qwen.named_parameters():
            p.requires_grad = is_lora_parameter(name)
    else:
        for p in qwen.parameters():
            p.requires_grad = False
    qwen.train(train_backbone)


def lora_target_modules_from_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    modules = [part.strip() for part in str(value).split(",") if part.strip()]
    return modules or None


def lora_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "r": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": lora_target_modules_from_arg(args.lora_target_modules),
        "gradient_checkpointing": bool(args.lora_gradient_checkpointing),
    }


def apply_lora_adapter(
    qwen,
    args: argparse.Namespace,
    *,
    training: bool,
    lora_config: dict[str, Any] | None = None,
):
    try:
        import accelerate  # noqa: F401
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "Direct VLA LoRA training/eval requires peft and accelerate on the "
            "GPU machine, for example: pip install peft accelerate transformers"
        ) from exc

    cfg = dict(lora_config or lora_config_from_args(args))
    peft_cfg = LoraConfig(
        r=int(cfg.get("r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias=str(cfg.get("bias", "none")),
        task_type=str(cfg.get("task_type", "CAUSAL_LM")),
        target_modules=cfg.get("target_modules", None),
    )
    model = get_peft_model(qwen, peft_cfg)
    if bool(cfg.get("gradient_checkpointing", False)) and training:
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if training and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def extract_lora_state_dict(qwen) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in qwen.state_dict().items()
        if is_lora_parameter(name)
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def collect_expert(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    cfg = PointHazardConfig()
    cfg.max_episode_steps = int(args.max_steps)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)
    expert_cfg = SafeExpertConfig(
        grid_res=args.expert_grid_res,
        safety_margin=args.expert_safety_margin,
        desired_speed=args.expert_desired_speed,
        speed_noise_std=args.expert_speed_noise_std,
        lookahead_dist=args.expert_lookahead_dist,
        waypoint_advance_dist=args.expert_waypoint_advance_dist,
        kp=args.expert_kp,
        use_drag_feedforward=not args.no_expert_drag_feedforward,
        hazard_slowdown_dist=args.expert_hazard_slowdown_dist,
        hazard_min_speed_factor=args.expert_hazard_min_speed_factor,
    )
    expert = SafeExpert(env, cfg=expert_cfg)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    ep_returns: list[float] = []
    ep_steps: list[int] = []
    ep_final_dists: list[float] = []
    ep_min_clearances: list[float] = []

    n_success = 0
    n_hazard = 0
    n_timeout = 0
    n_plan_fail = 0
    n_discarded_transitions = 0
    t0 = time.time()

    for ep in range(int(args.episodes)):
        obs, _ = env.reset(seed=int(args.seed) + ep)
        obs = np.asarray(obs, dtype=np.float32)
        plan_ok = expert.plan(env.pos.copy(), env.goal.copy(), env.hazards.copy())
        if not plan_ok:
            n_plan_fail += 1

        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_return = 0.0
        ep_min_clearance = obs_clearance(obs, cfg)
        last_info: dict[str, Any] = {"dist_to_goal": float(np.linalg.norm(env.pos - env.goal))}
        hit_hazard = False
        success = False

        if plan_ok:
            for _step in range(int(args.max_steps)):
                action = expert.act(expert_act_obs(obs, cfg))
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
                    if info.get("termination_reason") == "goal":
                        success = True
                    elif info.get("termination_reason") == "hazard":
                        hit_hazard = True
                    break
                if truncated:
                    break

        if hit_hazard:
            n_hazard += 1
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
        ep_final_dists.append(float(last_info.get("dist_to_goal", np.nan)))
        ep_min_clearances.append(float(ep_min_clearance))

        if (ep + 1) % int(args.log_every) == 0 or ep == int(args.episodes) - 1:
            elapsed = time.time() - t0
            print(
                f"[collect_expert] ep {ep + 1}/{args.episodes} "
                f"success={n_success} hazard={n_hazard} timeout={n_timeout} "
                f"plan_fail={n_plan_fail} kept_transitions={len(all_obs)} "
                f"elapsed={elapsed:.0f}s"
            )

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    action_arr = np.asarray(all_actions, dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = np.array(
        [
            int(args.episodes),
            n_success,
            n_hazard,
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
        "hazard_hit_rate": n_hazard / max(1, int(args.episodes)),
        "timeout_rate": n_timeout / max(1, int(args.episodes)),
        "plan_fail_rate": n_plan_fail / max(1, int(args.episodes)),
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

    print(f"\nExpert dataset saved: {out_path}")
    print(f"Summary saved: {summary_path}")
    print(f"  obs={obs_arr.shape} actions={action_arr.shape}")
    print(
        f"  success={summary['success_rate']:.1%} "
        f"hazard={summary['hazard_hit_rate']:.1%} "
        f"timeout={summary['timeout_rate']:.1%} "
        f"transitions={len(all_obs)}"
    )


def run_epoch(
    *,
    qwen,
    action_head: nn.Module,
    processor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    train_vlm: bool,
    train_lora: bool,
    grad_clip: float,
) -> float:
    training = optimizer is not None
    set_qwen_trainability(
        qwen,
        train_vlm=train_vlm and training,
        train_lora=train_lora and training,
    )
    action_head.train(training)
    losses: list[float] = []

    for batch in loader:
        pooled = qwen_forward_pooled(
            qwen=qwen,
            processor=processor,
            images=batch["images"],
            prompts=batch["prompts"],
            train_vlm=train_vlm and training,
            train_lora=train_lora and training,
        )
        if module_device(action_head) != pooled.device:
            action_head.to(pooled.device)
        target = batch["actions"].to(pooled.device)
        if hasattr(action_head, "loss"):
            loss = action_head.loss(pooled, target)
        else:
            pred = action_head(pooled)
            loss = F.mse_loss(pred, target)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for p in action_head.parameters() if p.requires_grad]
            if train_vlm or train_lora:
                params.extend([p for p in qwen.parameters() if p.requires_grad])
            if grad_clip > 0 and params:
                nn.utils.clip_grad_norm_(params, float(grad_clip))
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return finite_mean(losses)


def train_policy(args: argparse.Namespace) -> None:
    if bool(args.train_vlm) and bool(args.train_lora):
        raise ValueError("--train_vlm and --train_lora are mutually exclusive.")

    set_seed(args.seed)
    obs, actions, dataset_meta = load_npz_dataset(args.dataset)
    cfg = cfg_from_dataset_meta(dataset_meta)

    expected_obs_dim = 4 + 2 + 3 * int(cfg.n_hazards)
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
        train_obs,
        train_actions,
        cfg=cfg,
        render_size=args.render_size,
        prompt_mode=args.prompt_mode,
    )
    val_ds = QwenVLADataset(
        val_obs,
        val_actions,
        cfg=cfg,
        render_size=args.render_size,
        prompt_mode=args.prompt_mode,
    ) if len(val_obs) > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=qwen_vla_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=qwen_vla_collate,
        drop_last=False,
    ) if val_ds is not None else None

    qwen, processor = load_qwen_backbone(args)
    hidden_size = infer_hidden_size(qwen)
    if args.train_lora:
        qwen = apply_lora_adapter(qwen, args, training=True)
    action_head = make_action_head(args, hidden_size).to(module_device(qwen))
    set_qwen_trainability(
        qwen,
        train_vlm=bool(args.train_vlm),
        train_lora=bool(args.train_lora),
    )

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

    print("Training Qwen direct VLA")
    print(f"  model_path={args.model_path}")
    print(f"  action_head_type={action_head_type(action_head)}")
    if isinstance(action_head, DiffusionActionHead):
        print(
            f"  diffusion_steps={action_head.num_steps} "
            f"beta={action_head.beta_schedule}[{action_head.beta_min:g},{action_head.beta_max:g}] "
            f"cond_dim={action_head.cond_dim}"
        )
    elif isinstance(action_head, FlowActionHead):
        print(
            f"  flow_sample_steps={action_head.sample_steps} "
            f"cond_dim={action_head.cond_dim}"
        )
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
            qwen=qwen,
            action_head=action_head,
            processor=processor,
            loader=train_loader,
            optimizer=optimizer,
            train_vlm=bool(args.train_vlm),
            train_lora=bool(args.train_lora),
            grad_clip=args.grad_clip,
        )
        val_loss = float("nan")
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(
                    qwen=qwen,
                    action_head=action_head,
                    processor=processor,
                    loader=val_loader,
                    optimizer=None,
                    train_vlm=False,
                    train_lora=False,
                    grad_clip=0.0,
                )
        monitor = val_loss if math.isfinite(val_loss) else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if monitor < best_loss:
            best_loss = monitor
            best_epoch = epoch
            save_checkpoint(
                path=ckpt_path,
                qwen=qwen,
                action_head=action_head,
                cfg=cfg,
                args=args,
                hidden_size=hidden_size,
                dataset_meta=dataset_meta,
                history=history,
                best_epoch=best_epoch,
                best_loss=best_loss,
                physics_target_summary=physics_target_summary,
            )

        elapsed = time.time() - t0
        print(
            f"[train] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_epoch={best_epoch} elapsed={elapsed:.1f}s"
        )

    print(f"Best Qwen VLA action-head checkpoint saved: {ckpt_path}")


def save_checkpoint(
    *,
    path: Path,
    qwen,
    action_head: nn.Module,
    cfg: PointHazardConfig,
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
            qwen,
            args,
            training=False,
            lora_config=ckpt.get("lora_config") or None,
        )
        lora_state = ckpt.get("qwen_lora_state", None)
        if not lora_state:
            raise RuntimeError(
                "Checkpoint is marked train_lora=True but has no qwen_lora_state."
            )
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
    cfg: PointHazardConfig,
    renderer: HazardRenderer,
    prompt_mode: str,
    action_samples: int = 1,
) -> np.ndarray:
    image = render_obs_pil(obs, cfg, renderer)
    prompt = build_vla_prompt(obs, cfg, prompt_mode)
    pred = qwen_forward_action(
        qwen=qwen,
        action_head=action_head,
        processor=processor,
        images=[image],
        prompts=[prompt],
        train_vlm=False,
        action_samples=action_samples,
    )
    action = pred.squeeze(0).detach().cpu().numpy()
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def eval_policy(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    qwen, processor, action_head, ckpt = load_qwen_vla_from_ckpt(args)

    cfg_data = ckpt.get("env_config", asdict(PointHazardConfig()))
    cfg = PointHazardConfig(**cfg_data)
    cfg.max_episode_steps = int(args.max_steps)
    render_size = int(args.render_size if args.render_size > 0 else ckpt.get("render_size", 128))
    renderer = make_renderer(cfg, render_size)
    env = make_env(cfg=cfg, seed=args.seed, with_renderer=False)

    n_success = 0
    n_hazard = 0
    n_timeout = 0
    ep_returns: list[float] = []
    ep_steps: list[float] = []
    ep_final_dists: list[float] = []
    ep_min_clearances: list[float] = []
    ep_action_norms: list[float] = []
    episodes_log: list[dict[str, Any]] = []

    t0 = time.time()
    for ep in range(int(args.episodes)):
        obs, _ = env.reset(seed=int(args.seed) + ep)
        obs = np.asarray(obs, dtype=np.float32)
        ep_return = 0.0
        ep_min_clearance = obs_clearance(obs, cfg)
        action_norms: list[float] = []
        last_info: dict[str, Any] = {"dist_to_goal": float(np.linalg.norm(env.pos - env.goal))}
        outcome = "timeout"
        steps = 0

        for t in range(int(args.max_steps)):
            action = predict_action(
                qwen=qwen,
                processor=processor,
                action_head=action_head,
                obs=obs,
                cfg=cfg,
                renderer=renderer,
                prompt_mode=args.prompt_mode,
                action_samples=args.action_samples,
            )
            action_norms.append(float(np.linalg.norm(action)))
            obs, reward, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            last_info = dict(info)
            ep_return += float(reward)
            ep_min_clearance = min(ep_min_clearance, obs_clearance(obs, cfg))
            steps = t + 1

            if terminated:
                if info.get("termination_reason") == "goal":
                    outcome = "success"
                    n_success += 1
                elif info.get("termination_reason") == "hazard":
                    outcome = "hazard"
                    n_hazard += 1
                break
            if truncated:
                outcome = "timeout"
                n_timeout += 1
                break
        else:
            outcome = "timeout"
            n_timeout += 1

        final_dist = float(last_info.get("dist_to_goal", np.nan))
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

    summary = {
        "method": "direct_qwen_vla",
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
        "hazard_hit_rate": n_hazard / max(1, int(args.episodes)),
        "timeout_rate": n_timeout / max(1, int(args.episodes)),
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

    print("\nDirect Qwen VLA eval complete")
    print(f"Summary saved: {summary_path}")
    print(f"  success={summary['success_rate']:.1%}")
    print(f"  hazard={summary['hazard_hit_rate']:.1%}")
    print(f"  timeout={summary['timeout_rate']:.1%}")
    print(f"  return={summary['mean_return']:.2f}")
    print(f"  final_dist={summary['mean_final_dist']:.2f}")
    print(f"  min_clearance={summary['mean_min_clearance']:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen-backed direct VLA baseline for PointHazardEnv")
    parser.add_argument("--mode", type=str, required=True, choices=["collect_expert", "train", "eval"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--render_size", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=20)

    # Qwen backbone.
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="image_state",
        choices=["image_only", "image_state", "from_ckpt"],
        help=(
            "Prompt variant. Use image_only for image+task text only; "
            "image_state adds numeric state text. Eval can use from_ckpt."
        ),
    )

    # collect_expert.
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out", type=str, default="hazard/direct_vla_expert_demos.npz")
    parser.add_argument("--collect_summary", type=str, default="")
    parser.add_argument("--keep_hazard_episodes", action="store_true")
    parser.add_argument("--expert_action_noise", type=float, default=0.0)
    parser.add_argument("--expert_grid_res", type=int, default=100)
    parser.add_argument("--expert_safety_margin", type=float, default=0.15)
    parser.add_argument("--expert_desired_speed", type=float, default=2.5)
    parser.add_argument("--expert_speed_noise_std", type=float, default=0.4)
    parser.add_argument("--expert_lookahead_dist", type=float, default=0.6)
    parser.add_argument("--expert_waypoint_advance_dist", type=float, default=0.3)
    parser.add_argument("--expert_kp", type=float, default=3.0)
    parser.add_argument("--no_expert_drag_feedforward", action="store_true")
    parser.add_argument("--expert_hazard_slowdown_dist", type=float, default=1.2)
    parser.add_argument("--expert_hazard_min_speed_factor", type=float, default=0.4)

    # train.
    parser.add_argument("--dataset", type=str, default="hazard/direct_vla_expert_demos.npz")
    parser.add_argument("--ckpt", type=str, default="hazard/direct_qwen_vla_head.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--train_target",
        type=str,
        default="expert",
        choices=["expert", "physics_score"],
        help=(
            "Training action labels. expert keeps the dataset actions; "
            "physics_score relabels each obs with the candidate action maximizing "
            "goal_progress + physics_clearance_weight * min_clearance."
        ),
    )
    parser.add_argument("--physics_rollout_horizon", type=int, default=12)
    parser.add_argument("--physics_clearance_weight", type=float, default=0.25)
    parser.add_argument("--physics_safety_margin", type=float, default=0.15)
    parser.add_argument("--physics_unsafe_penalty", type=float, default=10.0)
    parser.add_argument("--physics_n_dirs", type=int, default=8)
    parser.add_argument("--physics_n_mags", type=int, default=2)
    parser.add_argument(
        "--physics_action_magnitudes",
        type=str,
        default="",
        help="Comma-separated candidate action magnitudes. Empty uses defaults from physics_n_mags.",
    )
    parser.add_argument("--physics_include_brake", action="store_true")
    parser.add_argument(
        "--physics_include_dataset_action",
        action="store_true",
        help="Also score the original dataset action as a candidate before relabeling.",
    )
    parser.add_argument("--physics_target_log_every", type=int, default=5000)
    parser.add_argument(
        "--action_head_type",
        type=str,
        default="regression",
        choices=["regression", "diffusion", "flow"],
        help="Action decoder: original MSE regression head, DDPM head, or rectified-flow head.",
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
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)

    # eval.
    parser.add_argument("--action_samples", type=int, default=1)
    parser.add_argument("--summary", type=str, default="hazard/direct_qwen_vla_eval_summary.json")
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
