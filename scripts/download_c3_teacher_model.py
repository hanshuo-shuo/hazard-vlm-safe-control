#!/usr/bin/env python3
"""Download a pinned, public Qwen checkpoint and verify every official file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            git.update(block)
    return sha.hexdigest(), git.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec["model_id"] != "Qwen/Qwen3-VL-8B-Instruct" or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("expected the pinned public Qwen3-VL-8B-Instruct model")
    for item in spec["files"]:
        if Path(item["path"]).name != item["path"] or not item["path"].endswith((".json", ".txt", ".safetensors")):
            raise ValueError("unexpected checkpoint file")
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "MODEL_VERIFIED.json"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=spec["model_id"], revision=spec["revision"],
                          local_dir=args.output, allow_patterns=[f["path"] for f in spec["files"]],
                          token=False, max_workers=4)
        verified = {}
        for item in spec["files"]:
            path = args.output / item["path"]
            if path.stat().st_size != item["size"]:
                raise ValueError(f"size mismatch: {item['path']}")
            sha, git = hashes(path)
            if item.get("sha256"):
                if sha != item["sha256"]:
                    raise ValueError(f"LFS SHA256 mismatch: {item['path']}")
            elif git != item["blob_id"]:
                raise ValueError(f"Git blob identity mismatch: {item['path']}")
            verified[item["path"]] = {"size": path.stat().st_size, "sha256": sha}
            print(json.dumps({"verified": item["path"], "bytes": path.stat().st_size}), flush=True)
        config = json.loads((args.output / "config.json").read_text())
        if config.get("model_type") != "qwen3_vl":
            raise ValueError("unexpected model architecture")
        result = {"status": "VERIFIED", "model_id": spec["model_id"], "revision": spec["revision"],
                  "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "download_spec_sha256": hashes(args.spec)[0], "files": verified,
                  "download_bytes": sum(f["size"] for f in verified.values()),
                  "code_execution_from_model_repository": False}
        report.write_text(json.dumps(result, indent=2) + "\n")
    except BaseException as exc:
        report.write_text(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "detail": str(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
