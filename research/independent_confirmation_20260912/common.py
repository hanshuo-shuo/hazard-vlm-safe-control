"""Isolated imports and provenance for the prospectively frozen component study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "composition_audit_20260912"))
import run_audit as reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def verify_freeze(root):
    record = json.loads((Path(root) / "FREEZE.json").read_text())
    for name, expected in record["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    reference.verify_reference(protocol())
    return record


def manifest(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json"}


def verify_manifest(directory):
    record = json.loads((Path(directory) / "MANIFEST.json").read_text())
    for name, expected in record.items():
        if sha(Path(directory) / name) != expected:
            raise ValueError(f"Artifact changed: {directory}/{name}")
    return len(record)
