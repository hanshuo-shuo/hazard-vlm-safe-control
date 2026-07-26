"""Matched scenario blocks, twins, and the preregistered 1,440-call matrix."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from evaluation.interface_contracts import (
    AMBIGUOUS_CONTRACT_IDS,
    EQUIVALENT_PAIRS,
    build_formal_contract_prompt,
)


PAID_SEEDS = (20, 21, 22, 23, 24)
ENVIRONMENTS = ("point_hazard_native", "safety_gym_goal_native")
SCENARIO_FAMILIES = (
    "clear_visible_incompatible",
    "weak_occluded_incompatible",
    "compatible_lookalike",
    "capability_reversal",
    "direct_path_intersection",
    "near_tangent_path",
    "multiple_candidate_detours",
    "irrelevant_terrain_distractor",
)
TWIN_ARMS = ("reference", "capability_twin", "appearance_twin", "visibility_twin")
EQUIVALENT_PAIR_IDS = tuple(EQUIVALENT_PAIRS)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def build_preregistered_blocks() -> list[dict[str, Any]]:
    """Return 80 model-independent blocks with all required assignments."""
    blocks = []
    for environment in ENVIRONMENTS:
        for family_index, family in enumerate(SCENARIO_FAMILIES):
            for seed_index, paid_seed in enumerate(PAID_SEEDS):
                pair_id = EQUIVALENT_PAIR_IDS[(family_index + seed_index) % len(EQUIVALENT_PAIR_IDS)]
                eq_arm = TWIN_ARMS[(family_index + seed_index) % len(TWIN_ARMS)]
                ambiguous_arm = "reference" if (family_index + seed_index) % 2 == 0 else "visibility_twin"
                scene_seed = 200 + family_index * len(PAID_SEEDS) + seed_index
                block = {
                    "block_id": f"{environment}|{family}|paid-seed-{paid_seed}",
                    "equivalent_pair_id": pair_id,
                    "equivalent_anchor_arm": eq_arm,
                    "ambiguous_anchor_arm": ambiguous_arm,
                    "scenario_family": family,
                    "environment": environment,
                    "paid_seed": paid_seed,
                    "scene_seed": scene_seed,
                }
                blocks.append(block)
    if len(blocks) != 80:
        raise AssertionError("preregistered design must contain 80 blocks")
    return blocks


@dataclass(frozen=True)
class TwinArm:
    arm: str
    png_bytes: bytes
    capability: str
    terrain_class: str
    geometry: Mapping[str, Any]
    visibility: str
    native_scene_sha256: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "image_sha256": _sha(self.png_bytes),
            "capability": self.capability,
            "terrain_class": self.terrain_class,
            "geometry": dict(self.geometry),
            "geometry_sha256": _sha(_canonical(self.geometry)),
            "visibility": self.visibility,
            "native_scene_sha256": self.native_scene_sha256,
        }


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _pixel_disk(
    image: Image.Image,
    geometry: Mapping[str, Any],
    world_bounds: Sequence[float],
) -> tuple[int, int, int]:
    x_min, x_max, y_min, y_max = (float(item) for item in world_bounds)
    x, y = (float(item) for item in geometry["center_xy"])
    radius = float(geometry["radius"])
    width, height = image.size
    px = int(round((x - x_min) / (x_max - x_min) * (width - 1)))
    py = int(round((y_max - y) / (y_max - y_min) * (height - 1)))
    pr = max(2, int(round(radius / (x_max - x_min) * width)))
    return px, py, pr


def _render_terrain(
    native_rgb: np.ndarray,
    geometry: Mapping[str, Any],
    world_bounds: Sequence[float],
    *,
    appearance: str,
    visibility: str,
) -> bytes:
    image = Image.fromarray(np.asarray(native_rgb, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    px, py, radius = _pixel_disk(image, geometry, world_bounds)
    box = (px - radius, py - radius, px + radius, py + radius)
    if appearance == "water":
        draw.ellipse(box, fill=(45, 155, 215, 125), outline=(20, 80, 145, 245), width=3)
        for offset in (-0.35, 0.0, 0.35):
            y = py + int(offset * radius)
            draw.arc(
                (px - int(0.7 * radius), y - 3, px + int(0.7 * radius), y + 3),
                0,
                180,
                fill=(235, 250, 255, 235),
                width=2,
            )
    elif appearance == "blue_carpet":
        draw.ellipse(box, fill=(70, 125, 195, 150), outline=(235, 210, 105, 245), width=3)
        step = max(3, radius // 4)
        for offset in range(-radius, radius + 1, step):
            draw.line((px - radius, py + offset, px + radius, py + offset), fill=(245, 220, 120, 120), width=1)
    else:
        raise ValueError(f"unknown registered appearance: {appearance}")
    composed = Image.alpha_composite(image, overlay)
    if visibility == "occluded":
        occlusion = Image.new("RGBA", image.size, (0, 0, 0, 0))
        occ_draw = ImageDraw.Draw(occlusion)
        occ_draw.rectangle(
            (px - radius, py - radius // 2, px + radius, py + radius // 2),
            fill=(105, 105, 105, 235),
        )
        composed = Image.alpha_composite(composed, occlusion)
    return _png(composed)


def build_twin_bundle(
    native_rgb: np.ndarray,
    *,
    geometry: Mapping[str, Any],
    world_bounds: Sequence[float],
    native_scene_sha256: str,
) -> dict[str, TwinArm]:
    """Build strict star twins from one native RGB scene."""
    reference_png = _render_terrain(
        native_rgb, geometry, world_bounds, appearance="water", visibility="clear"
    )
    appearance_png = _render_terrain(
        native_rgb, geometry, world_bounds, appearance="blue_carpet", visibility="clear"
    )
    visibility_png = _render_terrain(
        native_rgb, geometry, world_bounds, appearance="water", visibility="occluded"
    )
    bundle = {
        "reference": TwinArm(
            "reference", reference_png, "wheeled_non_waterproof", "water",
            geometry, "clear", native_scene_sha256,
        ),
        "capability_twin": TwinArm(
            "capability_twin", reference_png, "amphibious", "water",
            geometry, "clear", native_scene_sha256,
        ),
        "appearance_twin": TwinArm(
            "appearance_twin", appearance_png, "wheeled_non_waterproof", "solid_ground",
            geometry, "clear", native_scene_sha256,
        ),
        "visibility_twin": TwinArm(
            "visibility_twin", visibility_png, "wheeled_non_waterproof", "water",
            geometry, "occluded", native_scene_sha256,
        ),
    }
    audit_twin_bundle(bundle)
    return bundle


def audit_twin_bundle(bundle: Mapping[str, TwinArm]) -> dict[str, Any]:
    if set(bundle) != set(TWIN_ARMS):
        raise AssertionError("twin bundle arm set mismatch")
    reference = bundle["reference"]
    checks = {
        "capability_image_identical": reference.png_bytes == bundle["capability_twin"].png_bytes,
        "capability_geometry_identical": reference.geometry == bundle["capability_twin"].geometry,
        "capability_only_card_changes": reference.capability != bundle["capability_twin"].capability,
        "appearance_geometry_identical": reference.geometry == bundle["appearance_twin"].geometry,
        "appearance_capability_identical": reference.capability == bundle["appearance_twin"].capability,
        "appearance_image_changes": reference.png_bytes != bundle["appearance_twin"].png_bytes,
        "visibility_geometry_identical": reference.geometry == bundle["visibility_twin"].geometry,
        "visibility_truth_identical": reference.terrain_class == bundle["visibility_twin"].terrain_class,
        "visibility_image_changes": reference.png_bytes != bundle["visibility_twin"].png_bytes,
        "native_scene_identical": len({arm.native_scene_sha256 for arm in bundle.values()}) == 1,
    }
    if not all(checks.values()):
        raise AssertionError(f"strict twin audit failed: {checks}")
    return {"passed": True, "checks": checks}


def scenario_geometry(
    scenario_family: str,
    *,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> dict[str, Any]:
    """Deterministic family geometry in native world coordinates."""
    if scenario_family not in SCENARIO_FAMILIES:
        raise ValueError("unknown scenario family")
    start = np.asarray(start_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    delta = goal - start
    norm = max(float(np.linalg.norm(delta)), 1e-9)
    direction = delta / norm
    perpendicular = np.asarray([-direction[1], direction[0]])
    index = SCENARIO_FAMILIES.index(scenario_family)
    along = (0.35 + 0.04 * (index % 4))
    offset = {
        "clear_visible_incompatible": 0.0,
        "weak_occluded_incompatible": 0.05,
        "compatible_lookalike": -0.05,
        "capability_reversal": 0.0,
        "direct_path_intersection": 0.0,
        "near_tangent_path": 0.48,
        "multiple_candidate_detours": -0.18,
        "irrelevant_terrain_distractor": 0.9,
    }[scenario_family]
    center = start + along * delta + offset * perpendicular
    radius = 0.38 if scenario_family == "near_tangent_path" else 0.52
    return {
        "geometry_type": "disk",
        "center_xy": [float(center[0]), float(center[1])],
        "radius": radius,
        "coordinate_frame": "native_world_xy",
    }


def build_call_matrix(
    manifest: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    twin_bundles: Mapping[str, Mapping[str, TwinArm]],
) -> list[dict[str, Any]]:
    """Build the exact 480/model, 1,440-total logical request matrix."""
    rows = []
    for model in manifest["models"]:
        model_start = len(rows)
        for block in blocks:
            pair_id = str(block["equivalent_pair_id"])
            anchor_contract, mate_contract = EQUIVALENT_PAIRS[pair_id]
            bundle = twin_bundles[str(block["block_id"])]
            requests: list[tuple[str, str, str]] = [
                (arm, anchor_contract, "anchor") for arm in TWIN_ARMS
            ]
            requests.append((str(block["equivalent_anchor_arm"]), mate_contract, "mate"))
            requests.append((str(block["ambiguous_anchor_arm"]), AMBIGUOUS_CONTRACT_IDS[0], "ambiguous"))
            for arm, contract_id, role in requests:
                twin = bundle[arm]
                prompt = build_formal_contract_prompt(contract_id, twin.capability)
                request_id = (
                    f"{model['model_budget_id']}|{block['block_id']}|{arm}|{contract_id}|{role}"
                )
                row = {
                    "call_index": len(rows),
                    "model_call_index": len(rows) - model_start,
                    "request_id": request_id,
                    "model_budget_id": model["model_budget_id"],
                    "provider": model["provider"],
                    "provider_model": model["provider_model"],
                    "model_revision": model["revision"],
                    **dict(block),
                    "arm": arm,
                    "contract_id": contract_id,
                    "contract_role": role,
                    "capability": twin.capability,
                    "terrain_class": twin.terrain_class,
                    "prompt_sha256": _sha(prompt),
                    "image_sha256": _sha(twin.png_bytes),
                    "structured_output": contract_id != "formal_action_free_text",
                    "compatibility_smoke": len(rows) - model_start == 0,
                    "new_provider_call_budgeted": True,
                }
                row["request_sha256"] = _sha(_canonical(row))
                rows.append(row)
        if len(rows) - model_start != 480:
            raise AssertionError("each model must have exactly 480 matrix rows")
    if len(rows) != 1440:
        raise AssertionError("formal dry-run matrix must have exactly 1,440 rows")
    return rows
