"""Strict structured VLM output and byte-reconstructable per-call artifacts.

This module deliberately has no provider client.  It defines the gate that a
future client must pass before a call can become valid experiment evidence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from evaluation.conditions import ExperimentCondition, PrivilegeLevel
from evaluation.policy_interface import PolicyInput, ProvenanceTag


VLM_CALL_SCHEMA_VERSION = "point-hazard-vlm-call-v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SECRET_FIELD_FRAGMENTS = ("api_key", "apikey", "authorization", "bearer", "secret")


class ParseStatus(str, Enum):
    OK = "ok"
    JSON_ERROR = "json_error"
    SCHEMA_ERROR = "schema_error"


def _json_copy(value: Any, field_name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain finite JSON values") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _reject_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if any(fragment in lowered for fragment in _SECRET_FIELD_FRAGMENTS):
                raise PermissionError(f"secret field must not enter artifact: {path}.{key}")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")


def _decode_json_object(raw_response: bytes) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member: {key}")
            result[key] = value
        return result

    text = raw_response.decode("utf-8")
    value = json.loads(text, object_pairs_hook=object_pairs)
    if not isinstance(value, Mapping):
        raise ValueError("structured response must be a JSON object")
    return value


def _valid_candidate_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        or isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


@dataclass(frozen=True)
class StructuredStageOutput:
    """Machine-judgeable recognition and action result for one model call."""

    recognized_terrain: bool | None
    unsafe_candidate_ids: tuple[str | int, ...]
    selected_candidate_id: str | int | None
    parse_status: ParseStatus
    free_text_reason: str | None = None
    parse_error: str | None = None

    def __post_init__(self) -> None:
        status = ParseStatus(self.parse_status)
        object.__setattr__(self, "parse_status", status)
        object.__setattr__(self, "unsafe_candidate_ids", tuple(self.unsafe_candidate_ids))
        if status is ParseStatus.OK:
            if not isinstance(self.recognized_terrain, bool):
                raise ValueError("recognized_terrain must be boolean for an ok parse")
            if not _valid_candidate_id(self.selected_candidate_id):
                raise ValueError("selected_candidate_id must be a candidate ID")
            if self.parse_error is not None:
                raise ValueError("ok parse cannot contain parse_error")
        elif (
            self.recognized_terrain is not None
            or self.unsafe_candidate_ids
            or self.selected_candidate_id is not None
            or not self.parse_error
        ):
            raise ValueError("failed parse must expose no partial machine labels")
        if any(not _valid_candidate_id(item) for item in self.unsafe_candidate_ids):
            raise ValueError("unsafe_candidate_ids contains an invalid ID")
        if len(set(self.unsafe_candidate_ids)) != len(self.unsafe_candidate_ids):
            raise ValueError("unsafe_candidate_ids must not contain duplicates")
        if self.free_text_reason is not None and not isinstance(self.free_text_reason, str):
            raise ValueError("free_text_reason must be a string or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognized_terrain": self.recognized_terrain,
            "unsafe_candidate_ids": list(self.unsafe_candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "parse_status": self.parse_status.value,
            "free_text_reason": self.free_text_reason,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructuredStageOutput":
        expected = {
            "recognized_terrain",
            "unsafe_candidate_ids",
            "selected_candidate_id",
            "parse_status",
            "free_text_reason",
            "parse_error",
        }
        if set(value) != expected:
            raise ValueError(f"structured output must contain exactly {sorted(expected)}")
        return cls(**dict(value))


def parse_structured_stage_output(
    raw_response: str | bytes,
    *,
    candidate_ids: Sequence[str | int],
) -> StructuredStageOutput:
    """Strictly parse a response without recovering partial quantitative labels."""
    raw_bytes = raw_response.encode("utf-8") if isinstance(raw_response, str) else bytes(raw_response)
    try:
        value = _decode_json_object(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return StructuredStageOutput(
            None, (), None, ParseStatus.JSON_ERROR, parse_error=str(exc)
        )

    required = {
        "recognized_terrain",
        "unsafe_candidate_ids",
        "selected_candidate_id",
        "parse_status",
    }
    allowed = required | {"free_text_reason"}
    try:
        if set(value) != required and set(value) != allowed:
            raise ValueError(f"response keys must be exactly {sorted(required)} plus optional free_text_reason")
        if value["parse_status"] != ParseStatus.OK.value:
            raise ValueError("model-emitted parse_status must equal 'ok'")
        if not isinstance(value["recognized_terrain"], bool):
            raise ValueError("recognized_terrain must be boolean")
        unsafe = value["unsafe_candidate_ids"]
        if not isinstance(unsafe, list):
            raise ValueError("unsafe_candidate_ids must be a JSON array")
        known_ids = tuple(candidate_ids)
        if any(not _valid_candidate_id(item) for item in known_ids):
            raise ValueError("candidate metadata contains an invalid candidate ID")
        if len(set(known_ids)) != len(known_ids):
            raise ValueError("candidate metadata contains duplicate candidate IDs")
        if any(item not in known_ids for item in unsafe):
            raise ValueError("unsafe_candidate_ids contains an unknown candidate")
        selected = value["selected_candidate_id"]
        if selected not in known_ids:
            raise ValueError("selected_candidate_id is not in candidate metadata")
        reason = value.get("free_text_reason")
        return StructuredStageOutput(
            recognized_terrain=value["recognized_terrain"],
            unsafe_candidate_ids=tuple(unsafe),
            selected_candidate_id=selected,
            parse_status=ParseStatus.OK,
            free_text_reason=reason,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return StructuredStageOutput(
            None, (), None, ParseStatus.SCHEMA_ERROR, parse_error=str(exc)
        )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str, field_name: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not canonical base64") from exc


def _validate_xy(value: Any, field_name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")
    for coordinate in value:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            raise ValueError(f"{field_name} coordinates must be finite numbers")


def _validate_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    for index, candidate in enumerate(candidates):
        required = {"candidate_id", "world_xy", "pixel_xy"}
        if not required.issubset(candidate):
            raise ValueError(
                f"candidate_metadata[{index}] must contain {sorted(required)}"
            )
        if not _valid_candidate_id(candidate["candidate_id"]):
            raise ValueError(f"candidate_metadata[{index}] has an invalid candidate_id")
        _validate_xy(candidate["world_xy"], f"candidate_metadata[{index}].world_xy")
        _validate_xy(candidate["pixel_xy"], f"candidate_metadata[{index}].pixel_xy")


@dataclass(frozen=True)
class VLMCallArtifact:
    """One complete request/response record with exact input reconstruction."""

    call_id: str
    prompt_utf8_base64: str
    prompt_sha256: str
    input_png_base64: str
    image_sha256: str
    candidate_metadata: tuple[Mapping[str, Any], ...]
    authorized_information_tags: Mapping[str, str]
    policy_input_base64: str
    policy_input_sha256: str
    model: str
    provider: str
    model_revision: str
    request_id: str
    temperature: float
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_response_utf8_base64: str
    structured_parse: StructuredStageOutput
    fallback: Mapping[str, Any]
    git_sha: str
    git_dirty: bool
    cli_config: Mapping[str, Any]
    selected_target: Mapping[str, Any]
    condition: Mapping[str, Any]
    condition_sha256: str
    trajectory: tuple[Mapping[str, Any], ...]
    schema_version: str = VLM_CALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("call_id", "model", "provider", "model_revision", "git_sha"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.request_id, str):
            raise ValueError("request_id must be a string")
        if self.schema_version != VLM_CALL_SCHEMA_VERSION:
            raise ValueError(f"unsupported VLM call schema: {self.schema_version}")
        for name in ("temperature", "latency_seconds", "cost_usd"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.authorized_information_tags, Mapping):
            raise ValueError("authorized_information_tags must be a field-to-tag mapping")
        tags = {
            str(field): ProvenanceTag(tag).value
            for field, tag in self.authorized_information_tags.items()
        }
        expected_tag_fields = {
            "public_observation",
            "public_rgb",
            "task_card",
            "capability_card",
            "authorized_privilege_payload",
            "public_candidate_metadata",
        }
        if set(tags) != expected_tag_fields:
            raise ValueError(
                "authorized_information_tags must identify every policy field"
            )
        if ProvenanceTag.EVAL_ONLY.value in tags.values():
            raise PermissionError("EVAL_ONLY tag must not enter an authorized call payload")
        object.__setattr__(
            self, "authorized_information_tags", MappingProxyType(tags)
        )
        for name in ("candidate_metadata", "trajectory"):
            object.__setattr__(
                self,
                name,
                tuple(_freeze_json(_json_copy(dict(item), name)) for item in getattr(self, name)),
            )
        _validate_candidates(self.candidate_metadata)
        for name in ("fallback", "cli_config", "selected_target", "condition"):
            copied = _json_copy(dict(getattr(self, name)), name)
            _reject_secrets(copied, name)
            object.__setattr__(self, name, _freeze_json(copied))
        if not isinstance(self.structured_parse, StructuredStageOutput):
            object.__setattr__(
                self,
                "structured_parse",
                StructuredStageOutput.from_dict(self.structured_parse),
            )
        self.audit()

    @property
    def prompt_bytes(self) -> bytes:
        value = _unb64(self.prompt_utf8_base64, "prompt_utf8_base64")
        value.decode("utf-8")
        return value

    @property
    def image_png_bytes(self) -> bytes:
        return _unb64(self.input_png_base64, "input_png_base64")

    @property
    def policy_input_bytes(self) -> bytes:
        return _unb64(self.policy_input_base64, "policy_input_base64")

    @property
    def raw_response_bytes(self) -> bytes:
        value = _unb64(self.raw_response_utf8_base64, "raw_response_utf8_base64")
        value.decode("utf-8")
        return value

    @property
    def candidate_ids(self) -> tuple[str | int, ...]:
        try:
            return tuple(item["candidate_id"] for item in self.candidate_metadata)
        except KeyError as exc:
            raise ValueError("each candidate must contain candidate_id") from exc

    def audit(self) -> dict[str, Any]:
        if not self.image_png_bytes.startswith(PNG_SIGNATURE):
            raise ValueError("input image is not a PNG byte stream")
        try:
            policy_payload = _decode_json_object(self.policy_input_bytes)
            policy_fields = (
                "public_observation",
                "public_rgb",
                "task_card",
                "capability_card",
                "authorized_privilege_payload",
                "public_candidate_metadata",
            )
            policy_tags = {
                name: policy_payload[name]["provenance"] for name in policy_fields
            }
            privilege_level = PrivilegeLevel(policy_payload["privilege_level"])
            p0_has_no_privilege = not (
                privilege_level is PrivilegeLevel.P0
                and ProvenanceTag.AUTHORIZED_PRIVILEGE.value in policy_tags.values()
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
            policy_tags = {}
            p0_has_no_privilege = False
        selected_matches = True
        if self.structured_parse.parse_status is ParseStatus.OK:
            selected_id = self.structured_parse.selected_candidate_id
            matches = [
                item for item in self.candidate_metadata
                if item["candidate_id"] == selected_id
            ]
            selected_matches = (
                len(matches) == 1
                and self.selected_target.get("candidate_id") == selected_id
                and self.selected_target.get("world_xy") == matches[0]["world_xy"]
            )
        checks = {
            "prompt_hash_matches": _sha256(self.prompt_bytes) == self.prompt_sha256,
            "image_hash_matches": _sha256(self.image_png_bytes) == self.image_sha256,
            "policy_input_hash_matches": (
                _sha256(self.policy_input_bytes) == self.policy_input_sha256
            ),
            "condition_hash_matches": (
                ExperimentCondition.from_dict(self.condition).condition_sha256
                == self.condition_sha256
            ),
            "structured_reparse_matches": (
                parse_structured_stage_output(
                    self.raw_response_bytes, candidate_ids=self.candidate_ids
                ).to_dict()
                == self.structured_parse.to_dict()
            ),
            "candidate_ids_unique": (
                len(set(self.candidate_ids)) == len(self.candidate_ids)
            ),
            "authorized_tags_match_policy": (
                policy_tags == self.authorized_information_tags
            ),
            "p0_has_no_authorized_privilege": p0_has_no_privilege,
            "selected_target_matches_candidate": selected_matches,
        }
        if not all(checks.values()):
            failures = sorted(name for name, passed in checks.items() if not passed)
            raise ValueError(f"VLM call artifact audit failed: {failures}")
        return checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "prompt_utf8_base64": self.prompt_utf8_base64,
            "prompt_sha256": self.prompt_sha256,
            "input_png_base64": self.input_png_base64,
            "image_sha256": self.image_sha256,
            "candidate_metadata": [_thaw_json(item) for item in self.candidate_metadata],
            "authorized_information_tags": dict(self.authorized_information_tags),
            "policy_input_base64": self.policy_input_base64,
            "policy_input_sha256": self.policy_input_sha256,
            "model": self.model,
            "provider": self.provider,
            "model_revision": self.model_revision,
            "request_id": self.request_id,
            "temperature": self.temperature,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "raw_response_utf8_base64": self.raw_response_utf8_base64,
            "structured_parse": self.structured_parse.to_dict(),
            "fallback": _thaw_json(self.fallback),
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "cli_config": _thaw_json(self.cli_config),
            "selected_target": _thaw_json(self.selected_target),
            "condition": _thaw_json(self.condition),
            "condition_sha256": self.condition_sha256,
            "trajectory": [_thaw_json(item) for item in self.trajectory],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VLMCallArtifact":
        expected = {
            "schema_version", "call_id", "prompt_utf8_base64", "prompt_sha256",
            "input_png_base64", "image_sha256", "candidate_metadata",
            "authorized_information_tags", "policy_input_base64",
            "policy_input_sha256", "model", "provider", "model_revision",
            "request_id", "temperature", "latency_seconds", "input_tokens",
            "output_tokens", "cost_usd", "raw_response_utf8_base64",
            "structured_parse", "fallback", "git_sha", "git_dirty",
            "cli_config", "selected_target", "condition", "condition_sha256",
            "trajectory",
        }
        if set(value) != expected:
            raise ValueError(f"VLM call artifact must contain exactly {sorted(expected)}")
        data = dict(value)
        data["structured_parse"] = StructuredStageOutput.from_dict(
            data["structured_parse"]
        )
        return cls(**data)


def build_vlm_call_artifact(
    *,
    call_id: str,
    prompt: str | bytes,
    input_png: bytes,
    candidate_metadata: Sequence[Mapping[str, Any]],
    policy_input: PolicyInput,
    model: str,
    provider: str,
    model_revision: str,
    request_id: str,
    temperature: float,
    latency_seconds: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    raw_response: str | bytes,
    fallback: Mapping[str, Any],
    git_sha: str,
    git_dirty: bool,
    cli_config: Mapping[str, Any],
    selected_target: Mapping[str, Any],
    condition: ExperimentCondition,
    trajectory: Sequence[Mapping[str, Any]],
) -> VLMCallArtifact:
    """Build and immediately audit a complete call artifact."""
    prompt_bytes = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
    prompt_bytes.decode("utf-8")
    image_bytes = bytes(input_png)
    raw_bytes = raw_response.encode("utf-8") if isinstance(raw_response, str) else bytes(raw_response)
    raw_bytes.decode("utf-8")
    candidates = tuple(_json_copy(dict(item), "candidate_metadata") for item in candidate_metadata)
    _validate_candidates(candidates)
    try:
        candidate_ids = tuple(item["candidate_id"] for item in candidates)
    except KeyError as exc:
        raise ValueError("each candidate must contain candidate_id") from exc
    structured = parse_structured_stage_output(raw_bytes, candidate_ids=candidate_ids)
    tags = {
        "public_observation": policy_input.public_observation.provenance.value,
        "public_rgb": policy_input.public_rgb.provenance.value,
        "task_card": policy_input.task_card.provenance.value,
        "capability_card": policy_input.capability_card.provenance.value,
        "authorized_privilege_payload": (
            policy_input.authorized_privilege_payload.provenance.value
        ),
        "public_candidate_metadata": (
            policy_input.public_candidate_metadata.provenance.value
        ),
    }
    policy_bytes = policy_input.canonical_bytes()
    return VLMCallArtifact(
        call_id=call_id,
        prompt_utf8_base64=_b64(prompt_bytes),
        prompt_sha256=_sha256(prompt_bytes),
        input_png_base64=_b64(image_bytes),
        image_sha256=_sha256(image_bytes),
        candidate_metadata=candidates,
        authorized_information_tags=tags,
        policy_input_base64=_b64(policy_bytes),
        policy_input_sha256=_sha256(policy_bytes),
        model=model,
        provider=provider,
        model_revision=model_revision,
        request_id=request_id,
        temperature=temperature,
        latency_seconds=latency_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        raw_response_utf8_base64=_b64(raw_bytes),
        structured_parse=structured,
        fallback=fallback,
        git_sha=git_sha,
        git_dirty=git_dirty,
        cli_config=cli_config,
        selected_target=selected_target,
        condition=condition.to_dict(),
        condition_sha256=condition.condition_sha256,
        trajectory=tuple(trajectory),
    )
