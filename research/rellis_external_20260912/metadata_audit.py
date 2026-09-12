"""Inspect split/pose metadata and lock pilot identities without image labels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def repository_head(repo):
    head = (repo / ".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        path = repo / ".git" / ref
        if path.exists():
            return path.read_text().strip()
        for line in (repo / ".git/packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
        raise ValueError("Cannot resolve pinned repository HEAD")
    return head


def record(line):
    a = line.split()
    m = re.search(r"frame(\d+)-(\d+)_(\d+)", a[0])
    return dict(sequence=a[0].split("/")[0], frame=int(m[1]),
                timestamp=float(m[2] + "." + m[3]), image=a[0], label=a[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    src = root / "research/rellis_external_20260912"
    protocol = json.loads((src / "qualification_protocol.json").read_text())
    records = {split: [record(s) for s in (root / "metadata/image_splits" / (split + ".lst")).read_text().splitlines() if s]
               for split in ["train", "val", "test"]}
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "official_commit": repository_head(root / "official_rellis"),
           "protocol_sha256": digest(src / "qualification_protocol.json"),
           "list_sha256": {s: digest(root / "metadata/image_splits" / (s + ".lst")) for s in records},
           "counts": {s: dict(collections.Counter(r["sequence"] for r in rr)) for s, rr in records.items()},
           "nearest_train_seconds": {}, "pose_metadata": {}}
    assert out["official_commit"] == protocol["official_repository_commit"]
    for split in ["val", "test"]:
        out["nearest_train_seconds"][split] = {}
        for seq in [f"{i:05}" for i in range(5)]:
            a = np.array([r["timestamp"] for r in records["train"] if r["sequence"] == seq])
            b = np.array([r["timestamp"] for r in records[split] if r["sequence"] == seq])
            if len(a) and len(b):
                d = np.abs(b[:, None] - a[None, :]).min(1)
                out["nearest_train_seconds"][split][seq] = {
                    "n": len(b), "min": float(d.min()), "median": float(np.median(d)),
                    "fraction_within_1s": float((d <= 1).mean()),
                    "fraction_within_5s": float((d <= 5).mean())}
    for seq in [f"{i:05}" for i in range(5)]:
        path = root / "metadata/poses/Rellis-3D" / seq / "poses.txt"
        x = np.loadtxt(path).reshape(-1, 3, 4)
        t = x[:, :, 3]
        out["pose_metadata"][seq] = {
            "rows": len(x), "first_translation": t[0].tolist(),
            "translation_min": t.min(0).tolist(), "translation_max": t.max(0).tolist(),
            "path_length_in_supplied_coordinates": float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum()),
            "has_timestamps": False,
            "image_frame_max": max(r["frame"] for rr in records.values() for r in rr if r["sequence"] == seq)}
    eligible = sorted([r for r in records["train"] if r["sequence"] == protocol["pilot"]["sequence"]], key=lambda r: r["timestamp"])
    selected = [eligible[int(round(i))] for i in np.linspace(0, len(eligible) - 1, protocol["pilot"]["count"])]
    assert len({r["image"] for r in selected}) == protocol["pilot"]["count"]
    (root / "artifacts/PILOT_SELECTION.json").write_text(json.dumps({"selection_uses_labels": False, "records": selected}, indent=2) + "\n")
    out["pilot_selection_sha256"] = digest(root / "artifacts/PILOT_SELECTION.json")
    out["decision"] = "Released split is not automatically a spatial split; pose rows must not be paired to image frame numbers without timestamp evidence. Released checkpoint remains unqualified for external confirmation."
    (root / "artifacts/METADATA_AUDIT.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
