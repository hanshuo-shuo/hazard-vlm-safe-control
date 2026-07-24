#!/usr/bin/env python3
"""Render the human-readable review overview from the JSON status registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "review_status_registry.json"
REVIEW_PATH = ROOT / "docs" / "RESEARCH_REVIEW_COMMENTS.md"
BEGIN = "<!-- BEGIN REVIEW STATUS TABLE: generated from review_status_registry.json; do not edit status cells by hand. -->"
END = "<!-- END REVIEW STATUS TABLE -->"


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def render_table(registry: dict) -> str:
    lines = [
        BEGIN,
        "",
        "| ID | 严重度 | 主题 | 当前状态 | 是否阻止 paid run |",
        "|---|---|---|---|---|",
    ]
    for item in registry["items"]:
        blocker = "是" if item["blocks_paid_run"] else "否"
        lines.append(
            f"| {item['id']} | {item['severity']} | {item['title']} | "
            f"{item['status']} | {blocker} |"
        )
    lines.extend(["", END])
    return "\n".join(lines)


def render_document(source: str, registry: dict) -> str:
    start = source.find(BEGIN)
    end = source.find(END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("review document is missing the status-table markers")
    end += len(END)
    return source[:start] + render_table(registry) + source[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the generated status table in the review document",
    )
    args = parser.parse_args()

    registry = load_registry()
    current = REVIEW_PATH.read_text(encoding="utf-8")
    rendered = render_document(current, registry)
    if args.write:
        REVIEW_PATH.write_text(rendered, encoding="utf-8")
        return 0
    if rendered != current:
        raise SystemExit(
            "review status table is stale; run "
            "python scripts/render_review_status.py --write"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
