"""Acquire exactly the preselected pilot RGB/labels and timestamp-matched LiDAR.

Only archive directories (filenames), not non-pilot image or label contents, are
inspected. The point cloud's filename timestamp supplies the match, not its ID.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile
from remote_zip import RemoteZipReader


def stamp(name):
    m = re.search(r"frame(\d+)-(\d+)_(\d+)", name)
    return float(m[2] + "." + m[3]) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    selected = json.loads((root / "artifacts/PILOT_SELECTION.json").read_text())["records"]
    sources = {"rgb": "1F3Leu0H_m6aPVpZITragfreO_SGtL2yV",
               "labels": "16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5",
               "lidar": ""}
    # Official timestamped PLY archive identity comes from the cloned README.
    readme = (root / "official_rellis/README.md").read_text()
    m = re.search(r"Ouster LiDAR with Color Annotation PLY Format[^\n]*file/d/([^/]+)/", readme)
    if not m:
        raise ValueError("Cannot resolve official timestamped PLY archive")
    sources["lidar"] = m[1]
    out = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": {}, "records": selected}
    target = root / "pilot"
    target.mkdir(exist_ok=True)
    for kind, file_id in sources.items():
        try:
            handle = RemoteZipReader(file_id)
            z = zipfile.ZipFile(handle)
            names = z.namelist()
            meta = {"status": "READING", "official_id": file_id, "archive_bytes": handle.size, "members": len(names)}
            out["sources"][kind] = meta
            if kind == "lidar":
                candidates = [(stamp(n), n) for n in names if "/00000/" in "/" + n and n.endswith(".ply")]
                candidates = [(t, n) for t, n in candidates if t is not None]
                if not candidates:
                    raise ValueError("No timestamped sequence-00000 point clouds")
            for r in selected:
                if kind == "lidar":
                    t, member = min(candidates, key=lambda x: abs(x[0] - r["timestamp"]))
                    r["sync_delta_seconds"] = abs(t - r["timestamp"])
                    if r["sync_delta_seconds"] > 0.05:
                        r[kind] = {"status": "NO_MATCH"}
                        continue
                else:
                    suffix = r["image"] if kind == "rgb" else r["label"]
                    matching = [n for n in names if n.endswith(suffix)]
                    if len(matching) != 1:
                        raise ValueError(f"Expected one {kind} member for {suffix}; got {len(matching)}")
                    member = matching[0]
                info = z.getinfo(member)
                if info.file_size > 30_000_000:
                    raise ValueError("Unexpected individual pilot file size")
                payload = z.read(member)  # ZIP verifies CRC.
                path = target / kind / Path(member).name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(payload)
                r[kind] = {"status": "DOWNLOADED", "member": member, "path": str(path.relative_to(root)),
                           "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            meta.update(status="COMPLETE", transferred_bytes=handle.transferred)
        except Exception as exc:
            out["sources"].setdefault(kind, {"official_id": file_id}).update(status="ERROR", error=repr(exc))
        (root / "artifacts/PILOT_ACQUISITION.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"kind": kind, "result": out["sources"][kind]}), flush=True)
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["complete_triples"] = sum(all(r.get(k, {}).get("status") == "DOWNLOADED" for k in sources) for r in selected)
    (root / "artifacts/PILOT_ACQUISITION.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
