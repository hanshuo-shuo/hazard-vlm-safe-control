#!/usr/bin/env python3
"""Offline smoke runner for the two supported safety-accounting backends.

This command never imports an OpenRouter/VLM client.  It uses a deterministic
direct-goal/zero-action router and writes one replay-ready JSON artifact per
episode.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_pointhazard import PointHazardConfig
from envs import PointHazardAdapter, SafetyGymGoalAdapter, SemanticSafetyPointGoalAdapter
from evaluation.schemas import build_episode_artifact
from evaluation.semantic_evaluator import evaluate_scene_manifest


def build_backend(args: argparse.Namespace) -> Any:
    if args.environment_backend == "point_hazard":
        return PointHazardAdapter(
            PointHazardConfig(
                n_hazards=args.n_hazards,
                max_episode_steps=args.steps,
            ),
            with_renderer=True,
        )
    if args.semantic_terrain:
        return SemanticSafetyPointGoalAdapter(
            env_id=args.env_id,
            render_mode="rgb_array",
            capability=args.capability,
        )
    return SafetyGymGoalAdapter(env_id=args.env_id, render_mode="rgb_array")


def direct_goal_action(observation: Any, action_space: Any, backend: str) -> np.ndarray:
    """Deterministic router that receives no environment or evaluator object."""
    if backend == "point_hazard":
        array = np.asarray(observation, dtype=np.float32).reshape(-1)
        direction = array[4:6] - array[0:2]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            direction = direction / norm
        action = direction.astype(np.float32)
    elif isinstance(observation, dict) and "desired_goal" in observation and "achieved_goal" in observation:
        direction = np.asarray(observation["desired_goal"], dtype=np.float32) - np.asarray(observation["achieved_goal"], dtype=np.float32)
        action = np.zeros(action_space.shape, dtype=np.float32)
        if action.size >= 2 and np.linalg.norm(direction) > 1e-8:
            action[:2] = direction[:2] / np.linalg.norm(direction[:2])
    else:
        # Safety-Gymnasium's native observation layout is intentionally not
        # reverse-engineered into a privileged heuristic.  Zero action is a
        # deterministic, public-observation-safe fallback for this slice.
        action = np.zeros(action_space.shape, dtype=np.float32)
    return np.asarray(np.clip(action, action_space.low, action_space.high), dtype=np.float32)


def run_episode(adapter: Any, seed: int, steps: int, capability: str) -> tuple[Any, Any]:
    initial_obs, _ = adapter.reset(seed=seed)
    # Exercise the real policy-facing RGB boundary once per episode.  This is
    # deliberately the unannotated frame returned by the backend; no scene
    # manifest or evaluator context is used to construct the policy input.
    adapter.render_public_rgb()
    for _ in range(steps):
        action = direct_goal_action(
            adapter.public_observation(), adapter.action_space, adapter.backend
        )
        _obs, _reward, _cost, terminated, truncated, _info = adapter.step(action)
        if terminated or truncated:
            break
    context = adapter.evaluator_context()
    if not context.terminated and not context.truncated:
        context = replace(context, truncated=True, termination_reason="smoke_step_limit")
    if isinstance(adapter, SemanticSafetyPointGoalAdapter):
        semantic = evaluate_scene_manifest(
            context.trajectory,
            context.scene_manifest,
            capability,
        )
        context = replace(
            context,
            semantic_violations=semantic.per_step_violation,
        )
    artifact = build_episode_artifact(
        context,
        seed=seed,
        initial_observation=initial_obs,
        router="direct_goal",
        zone_source="none",
        enforcement="none",
    )
    return artifact, context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-backend",
        "--environment_backend",
        choices=("point_hazard", "safety_gym_goal"),
        default="safety_gym_goal",
    )
    parser.add_argument("--env-id", "--env_id", default="SafetyPointGoal1-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--artifact-dir", "--artifact_dir", type=Path, default=Path("results/safety_gym_smoke"))
    parser.add_argument("--n-hazards", "--n_hazards", type=int, default=8)
    parser.add_argument("--semantic-terrain", action="store_true")
    parser.add_argument("--capability", choices=("wheeled_non_waterproof", "amphibious"), default="wheeled_non_waterproof")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise SystemExit("--episodes and --steps must be positive")
    try:
        adapter = build_backend(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        for offset in range(args.episodes):
            seed = args.seed + offset
            artifact, _context = run_episode(adapter, seed, args.steps, args.capability)
            path = args.artifact_dir / f"episode_{seed}.json"
            artifact.write(path)
            print(
                f"backend={artifact.environment_backend} "
                f"env_id={artifact.environment_id} seed={artifact.seed} "
                f"return={sum(artifact.rewards):.6f} "
                f"native_cost={artifact.native_cost_total:.6f} "
                f"semantic_violation={artifact.semantic_violation} "
                f"success={artifact.success} "
                f"episode_length={len(artifact.actions)} "
                f"termination={artifact.termination_reason} "
                f"artifact={path}"
            )
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
