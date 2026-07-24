"""Keep the rendered review overview in sync with the machine-readable registry."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "review_status_registry.json"
REVIEW_PATH = ROOT / "docs" / "RESEARCH_REVIEW_COMMENTS.md"

EXPECTED_IDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "M06",
    "M07",
    "M08",
)
ALLOWED_STATUSES = {"DONE", "PARTIAL", "OPEN", "BLOCKED", "DEFERRED"}


def _load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_review_status_registry_is_complete_and_unambiguous() -> None:
    registry = _load_registry()
    assert registry["status_owner"] == "docs/review_status_registry.json"
    items = registry["items"]
    assert tuple(item["id"] for item in items) == EXPECTED_IDS
    assert len({item["id"] for item in items}) == len(items)

    for item in items:
        assert item["severity"] in {"BLOCKER", "MAJOR"}
        assert item["status"] in ALLOWED_STATUSES
        assert isinstance(item["blocks_paid_run"], bool)
        assert item["implementation_level"]
        assert "remaining_gap" in item


def test_review_overview_matches_registry() -> None:
    expected = {item["id"]: item["status"] for item in _load_registry()["items"]}
    rendered: dict[str, str] = {}
    in_table = False
    for line in REVIEW_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("<!-- BEGIN REVIEW STATUS TABLE"):
            in_table = True
            continue
        if line.startswith("<!-- END REVIEW STATUS TABLE"):
            in_table = False
            continue
        if not in_table or not line.startswith("|") or line.startswith("|---"):
            continue
        fields = [field.strip() for field in line.strip("|").split("|")]
        if fields and fields[0] in expected:
            rendered[fields[0]] = fields[3]

    assert rendered == expected
