#!/usr/bin/env python3
"""Export only frozen image/prompt requests; simulator truth is never copied."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c3_safe.teacher_io import file_hash, load_bundle, save_json, validate_request


def prepare(dataset, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh teacher bundle directory")
    requests = [json.loads(line) for line in (dataset / "TEACHER_REQUESTS_NOT_SUBMITTED.jsonl").read_text().splitlines()]
    for request in requests:
        validate_request(request, dataset)
    if not requests or len({r["prompt_sha256"] for r in requests}) != 1:
        raise ValueError("expected one frozen prompt")
    (output / "images").mkdir(parents=True, exist_ok=True)
    exported = []
    for request in requests:
        target = f"images/{request['image_sha256']}.png"
        shutil.copyfile(dataset / request["image_path"], output / target)
        exported.append({**request, "image_path": target})
    (output / "REQUESTS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in exported))
    manifest = {"schema_version": "c3-action-free-bundle-v1", "request_count": len(exported),
                "prompt_sha256": requests[0]["prompt_sha256"],
                "source_dataset_manifest_sha256": file_hash(dataset / "MANIFEST.json"),
                "source_requests_sha256": file_hash(dataset / "TEACHER_REQUESTS_NOT_SUBMITTED.jsonl"),
                "content": "RGB images and constant environmental-property prompt only",
                "files": {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    save_json(output / "BUNDLE_MANIFEST.json", manifest)
    load_bundle(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.dataset, args.output)
    print(json.dumps({"requests": result["request_count"], "bundle": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
