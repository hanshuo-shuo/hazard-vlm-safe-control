#!/usr/bin/env python3
"""Reparse immutable teacher responses with the explicit unknown confidence schema.

No generation is repeated. Parent records and their original parse outcomes are
preserved, so a parser correction cannot masquerade as improved model output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c3_safe.artifacts import file_sha, finish_run, new_run, run_cli, write_json
from c3_safe.teacher_io import inside, parse_response, rasterize_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    parent = json.loads((args.labels / "MANIFEST.json").read_text())
    for name, sha in parent["artifacts"].items():
        if file_sha(inside(args.labels, name)) != sha:
            raise ValueError(f"parent artifact changed: {name}")
    spec = json.loads((args.labels / "RUN_SPEC.json").read_text())
    paths = sorted((args.labels / "records").glob("*.json"))
    if len(paths) != spec["request_count"]:
        raise ValueError("parent generation run is incomplete")
    output = new_run(args.output)
    (output / "records").mkdir()
    (output / "fields").mkdir()
    spec["materialization"] = {"schema_version": "c3-spatial-materialization-v2", "new_model_generations": 0,
                               "parent_teacher_manifest_sha256": file_sha(args.labels / "MANIFEST.json"),
                               "change": "allow optional bounded confidence on explicitly unknown polygons"}
    write_json(output / "RUN_SPEC.json", spec)
    counts = {"accepted": 0, "invalid_responses": 0, "with_property_regions": 0, "unknown_only": 0}
    for path in paths:
        original = json.loads(path.read_text())
        raw = original["raw_response"]
        if hashlib.sha256(raw.encode()).hexdigest() != original["response_sha256"]:
            raise ValueError("parent raw response changed")
        record = dict(original)
        record["status_at_generation"] = original["status"]
        record["parent_record_sha256"] = file_sha(path)
        for key in ("field_path", "field_sha256", "parsed_response", "known_pixel_fraction", "parse_error"):
            record.pop(key, None)
        try:
            if original.get("truncated"):
                raise ValueError("parent generation was truncated")
            data = parse_response(raw)
            field, known = rasterize_response(data, record["width"], record["height"])
            relative = f"fields/{record['image_sha256']}.npz"
            np.savez_compressed(output / relative, requirement_field=field, known_mask=known)
            record.update(status="ACCEPTED", field_path=relative, field_sha256=file_sha(output / relative),
                          parsed_response=data, known_pixel_fraction=float(known.mean()))
            counts["accepted"] += 1
            counts["with_property_regions"] += bool(data["regions"])
            counts["unknown_only"] += not data["regions"] and bool(data["unknown_regions"])
        except (ValueError, TypeError, KeyError) as exc:
            record.update(status="INVALID_RESPONSE", parse_error=str(exc))
            counts["invalid_responses"] += 1
        write_json(output / "records" / path.name, record)
    summary = {"status": "POSTHOC_PARSER_CORRECTION_NOT_NEW_MODEL_EVIDENCE", **counts,
               "new_model_generations": 0, "parent_generation_count": len(paths),
               "parent_teacher_manifest_sha256": file_sha(args.labels / "MANIFEST.json"),
               "human_review": "NOT_PERFORMED", "scientific_gates_passed": []}
    finish_run(output, summary, source_paths=["scripts/materialize_c3_teacher_fields.py"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_cli(main)
