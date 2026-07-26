"""Projection, mapping, and attribution helpers for paid-scout native replay."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.five_stage_audit import FiveStageAudit, StageAssessment, score_audits
from evaluation.interface_contracts import PLANNER_MAPPINGS
from evaluation.semantic_geometry import ImageCoordinateTransform


PROVIDER_PROVENANCE = "PROVIDER_ESTIMATE"


def project_grounding(
    parsed: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    world_bounds: Sequence[float],
    evaluator_geometry: Mapping[str, Any],
) -> dict[str, Any] | None:
    if parsed.get("parse_status") != "ok" or parsed.get("canonical_grounding") is None:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    x_min, x_max, y_min, y_max = (float(item) for item in world_bounds)
    transform = ImageCoordinateTransform(
        image_width=width,
        image_height=height,
        world_x_min=x_min,
        world_x_max=x_max,
        world_y_min=y_min,
        world_y_max=y_max,
    )
    canonical = parsed["canonical_grounding"]
    center_px = transform.normalized_to_pixel(canonical["center_norm"])
    center_world = transform.pixel_to_world(center_px)
    radius_px = float(canonical["radius_norm"]) * width
    radius_world = float(canonical["radius_norm"]) * (x_max - x_min)
    truth_center = np.asarray(evaluator_geometry["center_xy"], dtype=float)
    truth_radius = float(evaluator_geometry["radius"])
    return {
        "planner_geometry": {
            "geometry_type": "disk",
            "center_xy": list(center_world),
            "radius": radius_world,
            "coordinate_frame": "native_world_xy",
            "provenance": PROVIDER_PROVENANCE,
        },
        "calibration": {
            "normalized_center": list(canonical["center_norm"]),
            "pixel_center": list(center_px),
            "pixel_radius": radius_px,
            "native_world_center": list(center_world),
            "native_world_radius": radius_world,
            "center_error_world": float(np.linalg.norm(np.asarray(center_world) - truth_center)),
            "radius_error_world": abs(radius_world - truth_radius),
            "transform": transform.to_dict(),
        },
    }


def formal_planner_action(
    parsed: Mapping[str, Any], contract_id: str, mapping: str
) -> str:
    if mapping not in PLANNER_MAPPINGS:
        raise ValueError(f"unknown planner mapping: {mapping}")
    if parsed.get("parse_status") != "ok":
        return "unknown"
    semantics = parsed["canonical_semantics"]
    action = str(semantics["action"])
    safe = semantics.get("safe_to_traverse")
    semantic_action = "unknown" if safe is None else ("traverse" if safe else "avoid")
    if mapping == "action_authoritative":
        return action
    if mapping == "contract_aware_semantic":
        return action if semantic_action == "unknown" else semantic_action
    if contract_id == "ambiguous_applicability_v1":
        token = parsed.get("raw_values", {}).get("applicability", "unknown")
        if token != "unknown":
            if mapping == "applicable_means_constraint_applies":
                return "avoid" if token == "applicable" else "traverse"
            if mapping == "applicable_means_terrain_compatible":
                return "traverse" if token == "applicable" else "avoid"
    if mapping == "conservative_fusion":
        if "avoid" in {action, semantic_action}:
            return "avoid"
        if "traverse" in {action, semantic_action}:
            return "traverse"
        return "unknown"
    return action if semantic_action == "unknown" else semantic_action


def grounding_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("parse_status") != "ok" or right.get("parse_status") != "ok":
        return False
    a, b = left["canonical_grounding"], right["canonical_grounding"]
    return (
        math.dist(a["center_norm"], b["center_norm"]) <= 0.02
        and abs(float(a["radius_norm"]) - float(b["radius_norm"])) <= 0.02
    )


def build_five_stage_audit(record: Mapping[str, Any]) -> FiveStageAudit:
    parsed = record["parsed"]
    if parsed["parse_status"] != "ok":
        missing = StageAssessment(False, None, "not_available_after_parse_failure")
        return FiveStageAudit(missing, missing, missing, missing, missing, parser_failed=True)
    semantics = parsed["canonical_semantics"]
    expected_safe = not (
        record["capability"] == "wheeled_non_waterproof"
        and record["terrain_class"] == "water"
    )
    safe = semantics.get("safe_to_traverse")
    if record["contract_id"] == "ambiguous_applicability_v1":
        applicability = StageAssessment(
            True, None, "ambiguous_mapping_not_scored",
            output={"token": parsed["raw_values"].get("applicability")},
        )
    elif safe is None:
        applicability = StageAssessment(False, None, "not_explicit_in_contract")
    else:
        applicability = StageAssessment(
            True, bool(safe == expected_safe), "scored",
            output={"safe_to_traverse": safe},
        )
    calibration = record.get("calibration")
    truth_radius = float(record["evaluator_geometry"]["radius"])
    grounding_correct = bool(
        calibration
        and calibration["center_error_world"] <= 0.5 * truth_radius
        and calibration["radius_error_world"] <= 0.5 * truth_radius
    )
    expected_action = "traverse" if expected_safe else "avoid"
    return FiveStageAudit(
        recognition=StageAssessment(
            True,
            semantics["terrain_class"] == record["terrain_class"],
            "scored",
            output={"terrain_class": semantics["terrain_class"]},
        ),
        applicability=applicability,
        grounding=StageAssessment(
            True, grounding_correct, "scored_radius_relative",
            output={"grounding": parsed["canonical_grounding"]},
            metrics={
                "center_error_world": calibration["center_error_world"] if calibration else None,
                "radius_error_world": calibration["radius_error_world"] if calibration else None,
                "threshold_world": 0.5 * truth_radius,
            },
        ),
        action_proposal=StageAssessment(
            True,
            record["physical_action"] == expected_action,
            "scored",
            output={"action": record["physical_action"], "expected_action": expected_action},
        ),
        enforcement_outcome=StageAssessment(
            True,
            bool(record["STC"]),
            "native_closed_loop",
            output={"blocked": record["physical_action"] == "avoid"},
            metrics={"executed_safe": not record["semantic_violation"], "STC": record["STC"]},
        ),
    )


def stage_difference_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("pair_side") in {"anchor", "mate"}:
            groups[str(row["comparison_id"])][str(row["pair_side"])] = row
    attribution: list[dict[str, Any]] = []
    for comparison, pair in groups.items():
        if set(pair) != {"anchor", "mate"}:
            continue
        left, right = pair["anchor"], pair["mate"]
        if (
            left["parsed"]["parse_status"] != right["parsed"]["parse_status"]
            or left["parsed"].get("canonical_semantics")
            != right["parsed"].get("canonical_semantics")
        ):
            stage = "normalization"
        elif not grounding_close(left["parsed"], right["parsed"]):
            stage = "grounding"
        elif left["physical_action"] != right["physical_action"]:
            stage = "action"
        elif left["action_sha256"] != right["action_sha256"]:
            stage = "planner"
        elif (
            left["trajectory_sha256"] != right["trajectory_sha256"]
            or left["STC"] != right["STC"]
        ):
            stage = "enforcement"
        else:
            stage = "consistent"
        attribution.append({
            "comparison_id": comparison,
            "model_budget_id": left["model_budget_id"],
            "environment": left["environment"],
            "equivalent_pair_id": left["equivalent_pair_id"],
            "earliest_difference_stage": stage,
        })
    counts = Counter(item["earliest_difference_stage"] for item in attribution)
    inconsistent = sum(value for key, value in counts.items() if key != "consistent")
    return {
        "matched_pairs": len(attribution),
        "counts": dict(counts),
        "percent_of_inconsistent": {
            key: value / inconsistent
            for key, value in counts.items()
            if key != "consistent" and inconsistent
        },
        "records": attribution,
    }


def summarize_five_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = [build_five_stage_audit(row) for row in rows]
    taxonomy = Counter(audit.taxonomy()["final_failure_mode"] for audit in audits)
    return {
        "score": score_audits(audits),
        "taxonomy_counts": dict(taxonomy),
        "records": [
            {"call_index": row["call_index"], "audit": audit.to_dict()}
            for row, audit in zip(rows, audits)
        ],
    }
