"""Fixed development data preparation for EXP-01B-R2.

The loop deliberately reuses already-consumed EXP-01B distributions as dev
diagnostics.  It does not create, expose, or evaluate a new formal holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")
SOURCE_SPLIT_PATH = ROOT / "configs" / "exp01b_splits.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_config(config: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    from autoresearch.exp01b_r2 import SCHEMA_VERSION
    from scripts.run_exp01b_simulator_field import audit_splits

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported autoresearch config schema")
    if config.get("artifact_status") != "DEVELOPMENT_ONLY":
        raise PermissionError("autoresearch artifact cannot claim formal status")
    if config.get("scientific_authority") is not False:
        raise PermissionError("development loop cannot carry scientific authority")
    if config.get("formal_holdout_available_to_loop") is not False:
        raise PermissionError("formal holdout must remain unavailable")
    if config.get("source_split_file") != "configs/exp01b_splits.json":
        raise ValueError("source split file changed")
    audit_splits(source)

    required_tiers = {"smoke", "quick", "confirm"}
    if set(config.get("tiers", {})) != required_tiers:
        raise ValueError("smoke, quick, and confirm tiers are required")
    source_counts = {
        "train": int(source["splits"]["train"]["scene_count"]),
        "validation": int(source["splits"]["validation"]["scene_count"]),
        "iid": int(source["splits"]["iid"]["scene_count"]),
        "appearance_ood": int(source["splits"]["appearance_ood"]["scene_count"]),
        "joint_ood": int(source["splits"]["joint_ood"]["scene_count"]),
    }
    key_map = {
        "train_scenes": "train",
        "validation_scenes": "validation",
        "iid_scenes": "iid",
        "appearance_ood_scenes": "appearance_ood",
        "joint_ood_scenes": "joint_ood",
    }
    for tier_name, tier in config["tiers"].items():
        if not tier.get("model_seeds"):
            raise ValueError(f"{tier_name} needs at least one model seed")
        for tier_key, split in key_map.items():
            count = int(tier[tier_key])
            if count < 2 or count > source_counts[split]:
                raise ValueError(f"invalid {tier_name}.{tier_key}: {count}")
        if int(tier["max_epochs"]) < 1 or float(tier["max_train_seconds_per_seed"]) <= 0:
            raise ValueError(f"invalid budget for tier {tier_name}")

    objective = config.get("objective", {})
    if objective.get("primary_split") != "appearance_ood":
        raise ValueError("appearance OOD must remain the development target")
    if objective.get("primary_arm") != "factorized_no_cf" or objective.get("primary_metric") != "regret":
        raise ValueError("primary decision estimand changed")


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    config, source = read_json(CONFIG_PATH), read_json(SOURCE_SPLIT_PATH)
    audit_config(config, source)
    return config, source


def build_dev_datasets(tier_name: str) -> tuple[dict[str, tuple[Any, ...]], dict[str, Any]]:
    from evaluation.simulator_spatial_field import build_rgb_scenes

    config, source = load_contract()
    if tier_name not in config["tiers"]:
        raise ValueError(f"unknown tier: {tier_name}")
    tier = config["tiers"][tier_name]
    count_keys = {
        "train": "train_scenes",
        "validation": "validation_scenes",
        "iid": "iid_scenes",
        "appearance_ood": "appearance_ood_scenes",
        "joint_ood": "joint_ood_scenes",
    }
    datasets = {}
    for split, count_key in count_keys.items():
        spec = dict(source["splits"][split])
        spec["scene_count"] = int(tier[count_key])
        datasets[split] = build_rgb_scenes(
            split,
            spec,
            image_size=int(source["image_size"]),
            field_size=int(source["field_size"]),
        )
    return datasets, {
        "config": config,
        "source": source,
        "tier": dict(tier),
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_split_sha256": file_sha256(SOURCE_SPLIT_PATH),
    }


def dataset_audit(datasets: Mapping[str, tuple[Any, ...]]) -> dict[str, Any]:
    result = {}
    for split, scenes in datasets.items():
        rgb_hash = hashlib.sha256()
        field_hash = hashlib.sha256()
        for scene in scenes:
            rgb_hash.update(scene.rgb.tobytes())
            field_hash.update(scene.field_target.tobytes())
        result[split] = {
            "scene_count": len(scenes),
            "scene_ids": [scene.scene.scene_id for scene in scenes],
            "appearance_families": sorted({scene.family for scene in scenes}),
            "rgb_sha256": rgb_hash.hexdigest(),
            "field_sha256": field_hash.hexdigest(),
        }
    return result
