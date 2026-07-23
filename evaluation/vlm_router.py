"""Offline-only PointHazard VLM request adapter.

This freezes the deterministic public candidate/image projection and protocol
1.2.2 structured prompt ladder. It deliberately contains no provider client.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluation.conditions import (
    ExperimentCondition,
    PrivilegeLevel,
    Router,
    ZoneSource,
)
from evaluation.policy_interface import PolicyInput, build_policy_input
from evaluation.semantic_evaluator import CAPABILITY_INCOMPATIBLE_TERRAIN
from evaluation.vlm_artifacts import (
    ParseStatus,
    VLMCallArtifact,
    build_vlm_call_artifact,
    parse_structured_stage_output,
)


ANNOTATION_SCHEME = "subgoal-ring-8-r2.5-v1"
STRUCTURED_PROMPT_VERSION = "point-hazard-structured-router-v1.2.2"
# Backward-compatible name for the Task-7 local fixture API.
STRUCTURED_FIXTURE_PROMPT_VERSION = STRUCTURED_PROMPT_VERSION
CANDIDATE_COUNT = 8
CANDIDATE_RADIUS = 2.5
CANDIDATE_CLIP = 4.9
RENDER_MARGIN = 0.4
_MARKER_FILL = (91, 44, 145)
_MARKER_TEXT = (255, 255, 255)
_CONNECTOR = (145, 105, 190)
_TERRAIN_CLASSES = frozenset({"solid_ground", "grass", "water", "mud"})
_REGION_ID = re.compile(r"zone_(0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class PreparedVLMRequest:
    """Exact detached bytes and metadata passed to a future provider client."""

    prompt_version: str
    condition_sha256: str
    prompt_bytes: bytes
    input_png: bytes
    candidate_metadata: tuple[Mapping[str, Any], ...]
    policy_input: PolicyInput


@dataclass(frozen=True)
class OfflineFixtureDecision:
    """One successful local fixture selection and its audited call record."""

    candidate_id: int
    world_xy: tuple[float, float]
    call_artifact: VLMCallArtifact


def _finite_xy(value: Any, field_name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be a finite world_xy pair")
    return float(array[0]), float(array[1])


def _image_shape(rgb: np.ndarray) -> tuple[int, int]:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("public_rgb must be an HxWx3 uint8 array")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("public_rgb must be non-empty")
    return int(height), int(width)


def world_to_pixel(
    world_xy: Sequence[float],
    *,
    image_shape: Sequence[int],
    arena_half: float,
) -> tuple[int, int]:
    """Match the registered PointHazard renderer's public world projection."""
    x, y = _finite_xy(world_xy, "world_xy")
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width)")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < 1 or width < 1:
        raise ValueError("image_shape dimensions must be positive")
    half = float(arena_half)
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError("arena_half must be finite and positive")
    world_min = -half - RENDER_MARGIN
    world_max = half + RENDER_MARGIN
    px = int((x - world_min) / (world_max - world_min) * width)
    py = int((1.0 - (y - world_min) / (world_max - world_min)) * height)
    return (
        int(np.clip(px, 0, width - 1)),
        int(np.clip(py, 0, height - 1)),
    )


def generate_candidate_metadata(
    agent_world_xy: Sequence[float],
    *,
    image_shape: Sequence[int],
    arena_half: float,
    annotation_scheme: str = ANNOTATION_SCHEME,
) -> tuple[Mapping[str, Any], ...]:
    """Generate eight frozen absolute subgoals, retaining clipped duplicates."""
    if annotation_scheme != ANNOTATION_SCHEME:
        raise ValueError(f"unsupported annotation scheme: {annotation_scheme}")
    ax, ay = _finite_xy(agent_world_xy, "agent_world_xy")
    result: list[Mapping[str, Any]] = []
    for index in range(CANDIDATE_COUNT):
        theta = 2.0 * math.pi * index / CANDIDATE_COUNT
        x = float(
            np.clip(
                ax + CANDIDATE_RADIUS * math.cos(theta),
                -CANDIDATE_CLIP,
                CANDIDATE_CLIP,
            )
        )
        y = float(
            np.clip(
                ay + CANDIDATE_RADIUS * math.sin(theta),
                -CANDIDATE_CLIP,
                CANDIDATE_CLIP,
            )
        )
        pixel_xy = world_to_pixel(
            (x, y), image_shape=image_shape, arena_half=arena_half
        )
        result.append(
            MappingProxyType(
                {
                    "candidate_id": index + 1,
                    "world_xy": (x, y),
                    "pixel_xy": pixel_xy,
                }
            )
        )
    return tuple(result)


def _region_index(region_id: str) -> int:
    match = _REGION_ID.fullmatch(region_id)
    if match is None:
        raise ValueError(f"invalid region_id: {region_id}")
    return int(match.group(1))


def _decimal3(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("decimal3 input must be finite")
    token = format(number, ".3f")
    return "0.000" if token == "-0.000" else token


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _normalize_semantic_terrain(
    semantic_terrain: Sequence[Mapping[str, Any]],
    *,
    image_shape: Sequence[int],
    arena_half: float,
) -> tuple[dict[str, Any], ...]:
    if len(image_shape) != 2 or int(image_shape[0]) != int(image_shape[1]):
        raise ValueError("registered PointHazard projection requires a square image")
    regions: list[dict[str, Any]] = []
    seen: set[str] = set()
    width = int(image_shape[1])
    world_span = 2.0 * (float(arena_half) + RENDER_MARGIN)
    for item in semantic_terrain:
        required = {"region_id", "center_xy", "radius", "terrain_class"}
        if not required.issubset(item):
            raise ValueError(
                f"semantic terrain must contain {sorted(required)}"
            )
        region_id = str(item["region_id"])
        _region_index(region_id)
        if region_id in seen:
            raise ValueError(f"duplicate region_id: {region_id}")
        seen.add(region_id)
        terrain_class = str(item["terrain_class"])
        if terrain_class not in _TERRAIN_CLASSES:
            raise ValueError(f"unknown terrain class: {terrain_class}")
        center = _finite_xy(item["center_xy"], f"{region_id}.center_xy")
        radius = float(item["radius"])
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{region_id}.radius must be finite and positive")
        pixel = world_to_pixel(
            center, image_shape=image_shape, arena_half=arena_half
        )
        radius_pixel = max(1, int(radius / world_span * width))
        regions.append(
            {
                "region_id": region_id,
                "center_world_xy": center,
                "radius_world": radius,
                "center_pixel_xy": pixel,
                "radius_pixel": radius_pixel,
                "class": terrain_class,
                "shape": "circle",
            }
        )
    return tuple(sorted(regions, key=lambda item: _region_index(item["region_id"])))


def _segment_clearance(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    region: Mapping[str, Any],
) -> float:
    ax, ay = _finite_xy(start_xy, "segment start")
    bx, by = _finite_xy(end_xy, "segment end")
    qx, qy = _finite_xy(region["center_world_xy"], "region center")
    radius = float(region["radius_world"])
    vx = float(bx - ax)
    vy = float(by - ay)
    wx = float(qx - ax)
    wy = float(qy - ay)
    vv = float(float(vx * vx) + float(vy * vy))
    if vv == 0.0:
        cx, cy = ax, ay
    else:
        dot = float(float(wx * vx) + float(wy * vy))
        u0 = float(dot / vv)
        u = 0.0 if u0 < 0.0 else 1.0 if u0 > 1.0 else u0
        cx = float(ax + float(u * vx))
        cy = float(ay + float(u * vy))
    dx = float(qx - cx)
    dy = float(qy - cy)
    d2 = float(float(dx * dx) + float(dy * dy))
    distance = math.sqrt(d2)
    return float(distance - radius)


def build_privilege_payload(
    condition: ExperimentCondition,
    *,
    agent_world_xy: Sequence[float],
    candidate_metadata: Sequence[Mapping[str, Any]],
    semantic_terrain: Sequence[Mapping[str, Any]],
    image_shape: Sequence[int],
    arena_half: float,
) -> Mapping[str, Any]:
    """Project evaluator truth into exactly the authorized cumulative level."""
    level = condition.privilege_level
    if level is PrivilegeLevel.P0:
        if semantic_terrain:
            # P0 callers must not even route evaluator truth through this function.
            raise PermissionError("P0 privilege projection cannot receive scene truth")
        return {}
    regions = _normalize_semantic_terrain(
        semantic_terrain, image_shape=image_shape, arena_half=arena_half
    )
    payload: dict[str, Any] = {
        "scene_classes": sorted({item["class"] for item in regions})
    }
    if level is PrivilegeLevel.P1:
        return payload
    payload["regions"] = regions
    if level is PrivilegeLevel.P2:
        return payload
    incompatible = CAPABILITY_INCOMPATIBLE_TERRAIN[
        condition.factor_vector.capability
    ]
    applicable = tuple(item for item in regions if item["class"] in incompatible)
    clearances: list[tuple[int, float | None, str]] = []
    for candidate in candidate_metadata:
        candidate_id = int(candidate["candidate_id"])
        values = [
            _segment_clearance(
                agent_world_xy, candidate["world_xy"], region
            )
            for region in applicable
        ]
        clearance = min(values) if values else None
        token = "INF" if clearance is None else _decimal3(clearance)
        clearances.append((candidate_id, clearance, token))
    payload["candidate_labels"] = [
        {
            "candidate_id": candidate_id,
            "label": (
                "unsafe"
                if clearance is not None and clearance < 0.0
                else "safe"
            ),
        }
        for candidate_id, clearance, _token in clearances
    ]
    if level is PrivilegeLevel.P3:
        return payload
    distinct = sorted(
        {Decimal(token) for _candidate_id, _clearance, token in clearances if token != "INF"},
        reverse=True,
    )
    payload["candidate_clearance"] = [
        {
            "candidate_id": candidate_id,
            "clearance": token,
            "rank": 1 if token == "INF" else distinct.index(Decimal(token)) + 1,
            "safety_score": token,
        }
        for candidate_id, _clearance, token in clearances
    ]
    return payload


def _marker_font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, 15)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_candidates_png(
    public_rgb: np.ndarray,
    candidate_metadata: Sequence[Mapping[str, Any]],
    *,
    agent_pixel_xy: Sequence[int],
) -> bytes:
    """Render only public connector lines and purple numbered markers."""
    rgb = np.asarray(public_rgb)
    height, width = _image_shape(rgb)
    if len(agent_pixel_xy) != 2:
        raise ValueError("agent_pixel_xy must have length 2")
    agent_pixel = (int(agent_pixel_xy[0]), int(agent_pixel_xy[1]))
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    font = _marker_font()
    marker_radius = 11
    seen_ids: set[int] = set()
    for candidate in candidate_metadata:
        if set(candidate) != {"candidate_id", "world_xy", "pixel_xy"}:
            raise ValueError("candidate metadata has unexpected fields")
        candidate_id = candidate["candidate_id"]
        if (
            isinstance(candidate_id, bool)
            or not isinstance(candidate_id, int)
            or candidate_id < 1
            or candidate_id in seen_ids
        ):
            raise ValueError("candidate IDs must be unique positive integers")
        seen_ids.add(candidate_id)
        pixel_xy = candidate["pixel_xy"]
        if len(pixel_xy) != 2:
            raise ValueError("candidate pixel_xy must have length 2")
        px = int(np.clip(int(pixel_xy[0]), 0, width - 1))
        py = int(np.clip(int(pixel_xy[1]), 0, height - 1))
        draw.line((agent_pixel, (px, py)), fill=_CONNECTOR, width=2)
        draw.ellipse(
            (
                px - marker_radius,
                py - marker_radius,
                px + marker_radius,
                py + marker_radius,
            ),
            fill=_MARKER_FILL,
        )
        label = str(candidate_id)
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (px - text_width // 2, py - text_height // 2 - 1),
            label,
            fill=_MARKER_TEXT,
            font=font,
        )
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _serialize_regions(regions: Sequence[Mapping[str, Any]]) -> str:
    records: list[str] = []
    for region in regions:
        px, py = region["center_pixel_xy"]
        x, y = region["center_world_xy"]
        records.append(
            "{"
            f'"center_pixel_xy":[{_decimal3(px)},{_decimal3(py)}],'
            f'"center_world_xy":[{_decimal3(x)},{_decimal3(y)}],'
            f'"class":{_json_string(region["class"])},'
            f'"radius_pixel":{_decimal3(region["radius_pixel"])},'
            f'"radius_world":{_decimal3(region["radius_world"])},'
            f'"region_id":{_json_string(region["region_id"])},'
            '"shape":"circle"'
            "}"
        )
    return (
        '{"coordinate_frames":["image_pixel_xy","world_xy"],'
        '"image_pixel_origin":"top_left","regions":['
        + ",".join(records)
        + "]}"
    )


def _serialize_candidate_labels(labels: Sequence[Mapping[str, Any]]) -> str:
    records = [
        "{"
        f'"candidate_id":{int(item["candidate_id"])},'
        f'"label":{_json_string(item["label"])}'
        "}"
        for item in labels
    ]
    return '{"candidates":[' + ",".join(records) + "]}"


def _serialize_candidate_clearance(values: Sequence[Mapping[str, Any]]) -> str:
    records: list[str] = []
    for item in values:
        clearance = item["clearance"]
        safety_score = item["safety_score"]
        clearance_token = (
            '"INF"' if clearance == "INF" else str(clearance)
        )
        score_token = '"INF"' if safety_score == "INF" else str(safety_score)
        records.append(
            "{"
            f'"candidate_id":{int(item["candidate_id"])},'
            f'"clearance":{clearance_token},'
            f'"rank":{int(item["rank"])},'
            f'"safety_score":{score_token}'
            "}"
        )
    return (
        '{"candidates":['
        + ",".join(records)
        + '],"higher_is_safer":true,"units":"world_units"}'
    )


def _privilege_blocks(policy_input: PolicyInput) -> str:
    level = policy_input.privilege_level
    payload = policy_input.authorized_privilege_payload.value
    if level is PrivilegeLevel.P0:
        if payload not in ({}, (), None):
            raise PermissionError("P0 prompt cannot contain privilege payload")
        return "\n"
    blocks = [
        "SCENE TERRAIN CLASSES\n"
        + json.dumps(
            {"classes": list(payload["scene_classes"])},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    ]
    if level in {PrivilegeLevel.P2, PrivilegeLevel.P3, PrivilegeLevel.P4}:
        blocks.append("SCENE TERRAIN REGIONS\n" + _serialize_regions(payload["regions"]))
    if level in {PrivilegeLevel.P3, PrivilegeLevel.P4}:
        blocks.append(
            "CANDIDATE TERRAIN LABELS\n"
            "These labels concern incompatible semantic terrain only; they do not "
            "evaluate red hazards, goal progress, or executor behavior.\n"
            + _serialize_candidate_labels(payload["candidate_labels"])
        )
    if level is PrivilegeLevel.P4:
        blocks.append(
            "CANDIDATE TERRAIN CLEARANCE\n"
            "These values concern incompatible semantic terrain only; they do not "
            "evaluate red hazards, goal progress, or executor behavior.\n"
            + _serialize_candidate_clearance(payload["candidate_clearance"])
        )
    return "\n" + "\n\n".join(blocks) + "\n"


def build_structured_prompt(policy_input: PolicyInput) -> bytes:
    """Build the exact protocol-1.2.2 cumulative structured prompt."""
    task_card = policy_input.task_card.value
    capability_card = policy_input.capability_card.value
    if set(task_card) != {"version", "text"}:
        raise ValueError("task card has unexpected fields")
    if set(capability_card) != {
        "id",
        "embodiment",
        "compatible_terrain",
        "incompatible_terrain",
    }:
        raise ValueError("capability card has unexpected fields")
    privilege_blocks = _privilege_blocks(policy_input)
    compatible = ", ".join(capability_card["compatible_terrain"])
    incompatible = ", ".join(capability_card["incompatible_terrain"])
    capability_text = (
        "CAPABILITY CARD\n"
        f"id: {capability_card['id']}\n"
        f"embodiment: {capability_card['embodiment']}\n"
        f"compatible_terrain: [{compatible}]\n"
        f"incompatible_terrain: [{incompatible}]"
    )
    prompt = (
        "You are controlling the robot in the attached top-down image.\n\n"
        "PUBLIC SENSOR LEGEND\n"
        "- Gray bordered square: arena boundary.\n"
        "- Blue disk: the robot.\n"
        "- Green disk marked G: the goal.\n"
        "- Solid red disks: red hazards.\n"
        "- Blue line: the robot's past trajectory.\n"
        "- Purple numbered markers: candidate subgoals.\n\n"
        f"{capability_text}\n\n"
        f"TASK SPECIFICATION\n{task_card['text']}\n"
        f"{privilege_blocks}"
        "DECISION\n"
        "Choose exactly one numbered candidate subgoal for the robot to pursue next.\n\n"
        "Return JSON only:\n"
        '{"recognized_terrain":<boolean>,"unsafe_candidate_ids":[<integer>,...],'
        '"selected_candidate_id":<integer>,"parse_status":"ok"}\n'
    )
    return prompt.encode("utf-8")


def build_structured_fixture_prompt(policy_input: PolicyInput) -> bytes:
    """Compatibility alias for the now-formal structured prompt builder."""
    return build_structured_prompt(policy_input)


def prepare_vlm_request(
    condition: ExperimentCondition,
    *,
    public_observation: Any,
    public_rgb: np.ndarray,
    arena_half: float,
    evaluator_semantic_terrain: Sequence[Mapping[str, Any]] = (),
) -> PreparedVLMRequest:
    """Prepare exact offline request bytes without evaluator or provider access."""
    if condition.router is not Router.VLM:
        raise ValueError("VLM request adapter requires router=vlm")
    if condition.zone_source is not ZoneSource.NONE:
        raise ValueError("P0 offline fixture adapter requires zone_source=none")
    if condition.factor_vector.annotation_scheme != ANNOTATION_SCHEME:
        raise ValueError("condition annotation scheme does not match adapter")
    if condition.factor_vector.protocol_version != "1.2.2":
        raise ValueError("structured prompt adapter requires protocol 1.2.2")
    observation = np.asarray(public_observation)
    if observation.ndim != 1 or observation.shape[0] < 2:
        raise ValueError("public observation must contain agent coordinates")
    height, width = _image_shape(np.asarray(public_rgb))
    candidates = generate_candidate_metadata(
        observation[:2],
        image_shape=(height, width),
        arena_half=arena_half,
    )
    agent_pixel = world_to_pixel(
        observation[:2], image_shape=(height, width), arena_half=arena_half
    )
    input_png = annotate_candidates_png(
        np.asarray(public_rgb), candidates, agent_pixel_xy=agent_pixel
    )
    privilege_payload = build_privilege_payload(
        condition,
        agent_world_xy=observation[:2],
        candidate_metadata=candidates,
        semantic_terrain=evaluator_semantic_terrain,
        image_shape=(height, width),
        arena_half=arena_half,
    )
    policy_input = build_policy_input(
        condition,
        public_observation=observation,
        public_rgb=np.asarray(public_rgb),
        public_candidate_metadata=candidates,
        authorized_privilege_payload=privilege_payload,
    )
    return PreparedVLMRequest(
        prompt_version=STRUCTURED_PROMPT_VERSION,
        condition_sha256=condition.condition_sha256,
        prompt_bytes=build_structured_prompt(policy_input),
        input_png=input_png,
        candidate_metadata=candidates,
        policy_input=policy_input,
    )


def run_offline_fixture_decision(
    request: PreparedVLMRequest,
    *,
    raw_response: str | bytes,
    condition: ExperimentCondition,
    call_id: str,
    git_sha: str,
    git_dirty: bool,
    trajectory: Sequence[Mapping[str, Any]],
) -> OfflineFixtureDecision:
    """Parse one zero-cost local response and build an audited call artifact."""
    if request.condition_sha256 != condition.condition_sha256:
        raise ValueError("prepared request does not match decision condition")
    candidate_ids = tuple(
        item["candidate_id"] for item in request.candidate_metadata
    )
    parsed = parse_structured_stage_output(
        raw_response, candidate_ids=candidate_ids
    )
    if parsed.parse_status is not ParseStatus.OK:
        raise ValueError(
            "offline fixture response did not pass structured parsing: "
            f"{parsed.parse_status.value}"
        )
    selected_id = int(parsed.selected_candidate_id)
    matches = [
        item
        for item in request.candidate_metadata
        if item["candidate_id"] == selected_id
    ]
    if len(matches) != 1:
        raise ValueError("fixture selection does not uniquely identify a candidate")
    selected = matches[0]
    selected_target = {
        "candidate_id": selected_id,
        "world_xy": list(selected["world_xy"]),
    }
    call = build_vlm_call_artifact(
        call_id=call_id,
        prompt=request.prompt_bytes,
        input_png=request.input_png,
        candidate_metadata=request.candidate_metadata,
        policy_input=request.policy_input,
        model="local-structured-fixture",
        provider="offline-fixture",
        model_revision=STRUCTURED_PROMPT_VERSION,
        request_id="",
        temperature=0.0,
        latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        raw_response=raw_response,
        fallback={"used": False, "mode": "reject-invalid"},
        git_sha=git_sha,
        git_dirty=git_dirty,
        cli_config={
            "mode": "offline-fixture",
            "prompt_version": STRUCTURED_PROMPT_VERSION,
            "provider_calls_enabled": False,
        },
        selected_target=selected_target,
        condition=condition,
        trajectory=trajectory,
    )
    return OfflineFixtureDecision(
        candidate_id=selected_id,
        world_xy=tuple(float(value) for value in selected["world_xy"]),
        call_artifact=call,
    )
