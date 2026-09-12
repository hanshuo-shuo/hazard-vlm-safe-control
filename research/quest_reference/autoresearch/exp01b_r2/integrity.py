"""Verify that the development evaluator and frozen inputs are unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("protected_hashes.json")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protected_files() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing integrity manifest: {MANIFEST_PATH}")
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "exp01b-r2-protected-files-v1":
        raise ValueError("unsupported protected-file manifest")
    checked = {}
    failures = []
    for relative, expected in value.get("sha256", {}).items():
        path = ROOT / relative
        actual = None if not path.is_file() else file_sha256(path)
        checked[relative] = actual
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        lines = [f"{item['path']}: expected {item['expected']}, got {item['actual']}" for item in failures]
        raise PermissionError("protected autoresearch inputs changed:\n" + "\n".join(lines))
    return {"manifest": str(MANIFEST_PATH.relative_to(ROOT)), "files": checked}


if __name__ == "__main__":
    result = verify_protected_files()
    print(json.dumps({"status": "PASS", "checked_files": len(result["files"])}, indent=2))
