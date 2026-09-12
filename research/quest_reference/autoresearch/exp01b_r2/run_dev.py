#!/usr/bin/env python3
"""Run one fixed-tier EXP-01B-R2 development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.exp01b_r2 import SCHEMA_VERSION  # noqa: E402
from autoresearch.exp01b_r2 import train as candidate  # noqa: E402
from autoresearch.exp01b_r2.evaluator import (  # noqa: E402
    aggregate_seed_results,
    evaluate_candidate,
    load_fixed_models,
    parameter_count,
)
from autoresearch.exp01b_r2.integrity import file_sha256, verify_protected_files  # noqa: E402
from autoresearch.exp01b_r2.prepare_dev import (  # noqa: E402
    build_dev_datasets,
    dataset_audit,
)


CANDIDATE_PATH = Path(candidate.__file__).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = os.environ.get("AR_SOURCE_COMMIT")
        branch = os.environ.get("AR_SOURCE_BRANCH")
        dirty = os.environ.get("AR_SOURCE_DIRTY")
        return {
            "commit": commit,
            "branch": branch,
            "dirty": None if dirty is None else dirty != "0",
        }


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    result = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    else:
        result["gpu_name"] = None
        result["gpu_capability"] = None
    return result


def _resolve_device(value: str, require_cuda: bool) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("this run requires an allocated CUDA GPU")
    return device


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _report(result: Mapping[str, Any]) -> str:
    objective = result["aggregate"]["objective"]
    shortcuts = result["aggregate"]["shortcuts"]
    lines = [
        f"# EXP-01B-R2 development run — {result['run_id']}",
        "",
        f"- Status: **{result['status']}**",
        f"- Tier: `{result['tier']}`",
        f"- Scientific authority: `{str(result['scientific_authority']).lower()}`",
        f"- Candidate SHA-256: `{result['candidate_sha256']}`",
        f"- Device: `{result['runtime']['device']}` / `{result['runtime']['gpu_name']}`",
        "",
        "| Development metric | Value |",
        "|---|---:|",
        f"| appearance-OOD regret ↓ | {objective['appearance_ood_regret']:.6f} |",
        f"| worst dev regret ↓ | {objective['worst_dev_regret']:.6f} |",
        f"| worst dev false-safe ↓ | {objective['worst_dev_false_safe']:.6f} |",
        f"| worst dev pair ranking ↑ | {objective['worst_dev_pair_ranking']:.6f} |",
        f"| worst dev field mIoU ↑ | {objective['worst_dev_field_miou']:.6f} |",
        "",
        "| Shortcut guard | Worst value |",
        "|---|---:|",
        f"| image-shuffle pair delta ↑ | {shortcuts['min_image_shuffle_pair_delta']:.6f} |",
        f"| terrain-erasure positive rate ↑ | {shortcuts['min_terrain_erasure_positive_rate']:.6f} |",
        f"| irrelevant-swap accuracy ↑ | {shortcuts['min_irrelevant_swap_accuracy']:.6f} |",
        f"| same-image accuracy ↑ | {shortcuts['min_same_image_accuracy']:.6f} |",
        "",
        "This artifact is development-only and cannot alter the frozen project decision.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    _refuse_nonempty(output)
    started = time.monotonic()
    integrity = verify_protected_files()
    datasets, contract = build_dev_datasets(args.tier)
    tier = contract["tier"]
    device = _resolve_device(args.device, bool(args.require_cuda))

    torch.manual_seed(int(tier["model_seeds"][0]))
    np.random.seed(int(tier["model_seeds"][0]) % (2**32))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    fixed_models = load_fixed_models(contract["config"])
    seed_results: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    saved_arrays: dict[str, np.ndarray] = {}
    checkpoints: dict[str, str] = {}
    candidate_parameters: int | None = None

    for model_seed in tier["model_seeds"]:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, history = candidate.train_candidate(
            datasets["train"],
            datasets["validation"],
            seed=int(model_seed),
            device=device,
            budget=tier,
        )
        if int(history.get("epochs", 0)) > int(tier["max_epochs"]):
            raise PermissionError("candidate exceeded the fixed epoch budget")
        if float(history.get("elapsed_seconds", float("inf"))) > float(tier["max_train_seconds_per_seed"]) + 5:
            raise TimeoutError("candidate exceeded the fixed training-time budget")
        count = parameter_count(model)
        if candidate_parameters is None:
            candidate_parameters = count
        elif candidate_parameters != count:
            raise ValueError("candidate parameter count changed across seeds")
        split_result, arrays = evaluate_candidate(
            model,
            datasets,
            fixed_models=fixed_models,
            source_config=contract["source"],
            tier=tier,
            model_seed=int(model_seed),
            device=device,
        )
        seed_key = str(model_seed)
        seed_results[seed_key] = split_result
        histories[seed_key] = history
        saved_arrays.update(arrays)
        if args.save_checkpoint:
            checkpoint_path = output / f"candidate_seed_{model_seed}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            checkpoints[seed_key] = file_sha256(checkpoint_path)
        del model

    arrays_path = output / "PER_SCENE_METRICS.npz"
    np.savez_compressed(arrays_path, **saved_arrays)
    elapsed = time.monotonic() - started
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "EXP-01B-R2-DEV",
        "run_id": args.run_id,
        "tier": args.tier,
        "status": "COMPLETE",
        "artifact_status": "DEVELOPMENT_ONLY",
        "scientific_authority": False,
        "formal_holdout_evaluated": False,
        "candidate_path": str(CANDIDATE_PATH.relative_to(ROOT)),
        "candidate_sha256": file_sha256(CANDIDATE_PATH),
        "candidate_parameter_count": int(candidate_parameters or 0),
        "git": _git_metadata(),
        "runtime": {
            **_runtime_metadata(device),
            "elapsed_seconds": elapsed,
            "peak_vram_mb": peak_vram_mb,
        },
        "contract": {
            "config_sha256": contract["config_sha256"],
            "source_split_sha256": contract["source_split_sha256"],
            "tier": tier,
            "protected_integrity": integrity,
        },
        "dataset_audit": dataset_audit(datasets),
        "training": histories,
        "per_seed": seed_results,
        "aggregate": aggregate_seed_results(seed_results),
        "per_scene_metrics_sha256": file_sha256(arrays_path),
        "checkpoint_sha256": checkpoints,
    }
    _write_json(output / "RESULTS.json", result)
    (output / "REPORT.md").write_text(_report(result), encoding="utf-8")
    print(f"AR_RUN_ID: {args.run_id}")
    print(f"AR_TIER: {args.tier}")
    print(f"AR_APPEARANCE_REGRET: {result['aggregate']['objective']['appearance_ood_regret']:.9f}")
    print(f"AR_WORST_FALSE_SAFE: {result['aggregate']['objective']['worst_dev_false_safe']:.9f}")
    print(f"AR_RESULT: {output / 'RESULTS.json'}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tier", choices=("smoke", "quick", "confirm"), default="quick")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in args.run_id):
        parser.error("run-id may contain only letters, digits, dot, underscore, and hyphen")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception as error:
        output = Path(parsed.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "FAILURE.json", {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": "EXP-01B-R2-DEV",
            "run_id": parsed.run_id,
            "tier": parsed.tier,
            "status": "CRASH",
            "scientific_authority": False,
            "candidate_sha256": file_sha256(CANDIDATE_PATH),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise
