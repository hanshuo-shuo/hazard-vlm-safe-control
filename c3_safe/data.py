"""Split-first transition/twin contracts and action-free teacher requests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from c3_safe.artifacts import ROOT, digest, file_sha
from c3_safe.costs import Capability, Rules, relation_cost, semantic_contact

TEACHER_PROMPT = (
    "Describe visible ground regions by environmental properties only. "
    "Return JSON with a regions list; each region has channel (water_ingress_demand, "
    "low_traction_demand, or fragile_surface_property), polygon_norm_xy (at least "
    "three [x,y] vertices in [0,1], origin top-left), and confidence in [0,1]. "
    "Use an empty list for no visible supported region. Include unknown_regions "
    "for areas whose properties cannot be established from the image. "
    "Do not infer fragility merely from green color or grass."
)


def load_spec(path=None):
    spec = json.loads(Path(path or ROOT / "configs/c3_safe_foundation.json").read_text())
    for axis in ("seed_splits", "appearance_splits"):
        groups = list(spec[axis].values())
        flattened = [x for group in groups for x in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError(f"overlapping {axis}")
    for axis in ("capability_splits", "rule_splits"):
        test = set(map(tuple, spec[axis]["test"]))
        if test & (set(map(tuple, spec[axis]["train"])) | set(map(tuple, spec[axis]["validation"]))):
            raise ValueError(f"held-out combinations leaked in {axis}")
    return spec


def validate_row_split(row, spec):
    split = row["split"]
    if split not in spec["seed_splits"]:
        raise ValueError("unregistered split")
    for key, axis in (("geometry_seed", "seed_splits"), ("appearance_id", "appearance_splits"),
                      ("capability_vector", "capability_splits"), ("rule_vector", "rule_splits")):
        value = list(row[key]) if key.endswith("vector") else row[key]
        if value not in spec[axis][split]:
            raise ValueError(f"{key} is outside the frozen {split} split")


def audit_splits(rows, spec):
    owners = {key: {} for key in ("geometry_hash", "image_sha256", "twin_group_id")}
    counts = {split: 0 for split in spec["seed_splits"]}
    for row in rows:
        validate_row_split(row, spec)
        counts[row["split"]] += 1
        for key, seen in owners.items():
            value = row[key]
            if value in seen and seen[value] != row["split"]:
                raise ValueError(f"cross-split leakage in {key}")
            seen[value] = row["split"]
    return {"rows_per_split": counts, "cross_split_overlap": {key: 0 for key in owners},
            "all_declared_splits_present": all(counts.values())}


def counterfactual_twin(row, capability, rules, spec):
    """Relabel the identical observed transition within its allowed split.

    Neither the action, next state, RGB, field nor swept footprint is regenerated.
    A held-out combination is rejected even when it is used only as a target.
    """
    validate_row_split(row, spec)
    twin = deepcopy(row)
    twin["capability_vector"] = list(capability)
    twin["rule_vector"] = list(rules)
    validate_row_split(twin, spec)
    cap, rule = Capability(*capability), Rules(*rules)
    twin["oracle_contact_violation"] = semantic_contact(
        twin["motion_samples"], twin["robot_radius"], twin["oracle_regions"], cap, rule,
    )
    exposure = twin["exposure"]
    twin["oracle_soft_cost"] = relation_cost(exposure["values"], cap, rule) if exposure["validity"] == "valid" else {
        "validity": "unknown_projection", "capability_cost": None, "rule_cost": None, "semantic_cost": None}
    twin["intervention"] = "same_transition_capability_rule"
    return twin


def teacher_request(image_path, *, image_id):
    """Only the RGB and constant environmental-property prompt leave this API."""
    return {"schema_version": "c3-action-free-spatial-request-v1", "image_id": image_id,
            "image_path": str(image_path), "image_sha256": file_sha(image_path),
            "prompt": TEACHER_PROMPT, "prompt_sha256": hashlib.sha256(TEACHER_PROMPT.encode()).hexdigest(),
            "status": "NOT_SUBMITTED"}


def validate_teacher_record(record, request):
    required = ("teacher_model", "teacher_revision", "prompt_sha256", "image_sha256",
                "raw_response", "response_sha256", "field_path", "field_sha256")
    if any(not record.get(key) for key in required):
        raise ValueError("teacher record lacks real model, response or field provenance")
    for key in ("prompt_sha256", "image_sha256"):
        if record[key] != request[key]:
            raise ValueError(f"teacher {key} mismatch")
    if record["response_sha256"] != hashlib.sha256(record["raw_response"].encode()).hexdigest():
        raise ValueError("teacher response hash mismatch")
    if file_sha(record["field_path"]) != record["field_sha256"]:
        raise ValueError("teacher field hash mismatch")
    if any(key in record for key in ("action", "candidate", "waypoint", "trajectory", "evaluator_truth")):
        raise ValueError("action-free teacher record contains forbidden fields")
    return True
