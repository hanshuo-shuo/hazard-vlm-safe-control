"""Action-free teacher bundles and strict spatial responses; no model runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from c3_safe.geometry import CHANNELS


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def inside(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError("bundle paths must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle path escapes its root")
    return path


def validate_request(request, root):
    keys = {"schema_version", "image_id", "image_path", "image_sha256", "prompt", "prompt_sha256", "status"}
    if set(request) != keys:
        raise ValueError("teacher request must contain only image identity and prompt")
    if request["schema_version"] != "c3-action-free-spatial-request-v1" or request["status"] != "NOT_SUBMITTED":
        raise ValueError("unexpected request version or state")
    if hashlib.sha256(request["prompt"].encode()).hexdigest() != request["prompt_sha256"]:
        raise ValueError("prompt hash mismatch")
    if file_hash(inside(root, request["image_path"])) != request["image_sha256"]:
        raise ValueError("image hash mismatch")
    if request["image_id"] != request["image_sha256"]:
        raise ValueError("image identity must be its content hash")


def load_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text())
    expected = set(manifest["files"]) | {"BUNDLE_MANIFEST.json"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError("bundle contains unexpected or missing files")
    for name, sha in manifest["files"].items():
        if file_hash(inside(root, name)) != sha:
            raise ValueError(f"bundle integrity failure: {name}")
    requests = [json.loads(line) for line in (root / "REQUESTS.jsonl").read_text().splitlines()]
    allowed = {"REQUESTS.jsonl", "BUNDLE_MANIFEST.json"}
    seen = set()
    for request in requests:
        validate_request(request, root)
        if request["image_path"] != f"images/{request['image_sha256']}.png":
            raise ValueError("transport image filenames must contain only the content hash")
        allowed.add(request["image_path"])
        if request["image_sha256"] in seen or request["prompt_sha256"] != manifest["prompt_sha256"]:
            raise ValueError("duplicate image or changing teacher prompt")
        seen.add(request["image_sha256"])
    if actual != allowed:
        raise ValueError("teacher bundle may contain only RGB files and requests")
    if len(requests) != manifest["request_count"] or not requests:
        raise ValueError("bundle request count mismatch")
    return manifest, requests


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_response(raw):
    text = raw.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        lines = text.splitlines()
        if lines[0] not in (fence, fence + "json") or lines[-1] != fence:
            raise ValueError("invalid JSON fence")
        text = "\n".join(lines[1:-1])
    data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(data, dict) or set(data) != {"regions", "unknown_regions"}:
        raise ValueError("response requires regions and unknown_regions only")
    for key in ("regions", "unknown_regions"):
        if not isinstance(data[key], list) or len(data[key]) > 64:
            raise ValueError("invalid region list")
    for item in data["regions"]:
        if not isinstance(item, dict) or set(item) != {"channel", "polygon_norm_xy", "confidence"}:
            raise ValueError("invalid property region schema")
        if item["channel"] not in CHANNELS:
            raise ValueError("unregistered spatial channel")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite in [0,1]")
        polygon(item["polygon_norm_xy"])
    for item in data["unknown_regions"]:
        if not isinstance(item, dict) or "polygon_norm_xy" not in item or set(item) - {"polygon_norm_xy", "reason", "confidence"}:
            raise ValueError("unknown areas require explicit normalized polygons")
        if "confidence" in item:
            confidence = item["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("unknown-region confidence must be finite in [0,1]")
        polygon(item["polygon_norm_xy"])
    return data


def polygon(vertices):
    points = np.asarray(vertices)
    if points.dtype.kind not in "iuf":
        raise ValueError("polygon coordinates must be numbers")
    points = points.astype(float)
    if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 64:
        raise ValueError("polygon requires 3 to 64 xy vertices")
    if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError("polygon coordinates must be finite in [0,1]")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(np.unique(points, axis=0)) != len(points) or len(points) < 3:
        raise ValueError("duplicate or insufficient polygon vertices")
    area2 = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    if abs(area2) < 1e-10:
        raise ValueError("degenerate polygon")
    # Reject crossing nonadjacent edges instead of silently inventing a fill.
    def cross(a, b, c):
        u, v = b - a, c - a
        return u[0] * v[1] - u[1] * v[0]
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 2, len(points)):
            if i == 0 and j == len(points) - 1:
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
                raise ValueError("self-intersecting polygon")
    return points


def polygon_mask(vertices, width, height):
    points = polygon(vertices)
    y, x = np.mgrid[:height, :width]
    x, y = (x + .5) / width, (y + .5) / height
    mask = np.zeros((height, width), dtype=bool)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        if a[1] == b[1]:
            continue
        mask ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
    return mask


def rasterize_response(data, width, height):
    field = np.zeros((height, width, len(CHANNELS)), dtype=np.float32)
    known = np.ones((height, width), dtype=bool)
    for region in data["regions"]:
        mask = polygon_mask(region["polygon_norm_xy"], width, height)
        # Confidence is retained in the response, not confused with severity.
        field[..., CHANNELS.index(region["channel"])][mask] = 1.
    for region in data["unknown_regions"]:
        known[polygon_mask(region["polygon_norm_xy"], width, height)] = False
    return field, known
