"""Bounded official-source acquisition; runs on Quest, never opens test labels.

Google's public download confirmation is supported; login and quota errors are
recorded, not bypassed. Raw error bodies stay on Quest for access diagnosis.
"""
import argparse
import concurrent.futures
import hashlib
import html.parser
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

FILES = {
    "intrinsics": ("1NAigZTJYocRSOTfgFBddZYnDsI_CSpwK", 2_000_000),
    "extrinsics": ("19EOqWS9fDUFp4nsBrMCa69xs9LgIlS2e", 2_000_000),
    "poses": ("1V3PT_NJhA41N7TBLp5AbW31d0ztQDQOX", 250_000_000),
    "image_splits": ("1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ", 2_000_000),
    "image_examples": ("1wIig-LCie571DnK72p2zNAYYWeclEz1D", 20_000_000),
    "lidar_examples": ("1QikPnpmxneyCuwefr6m50fBOSB2ny4LC", 50_000_000),
}


class Confirmation(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.values = None, {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
        if tag == "input" and attrs.get("name"):
            self.values[attrs["name"]] = attrs.get("value", "")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, spec, destination):
    file_id, limit = spec
    archive = destination / (name + ".zip")
    result = {"name": name, "official_id": file_id, "attempts": []}
    if not archive.exists():
        urls = [f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                f"https://drive.google.com/uc?export=download&id={file_id}"]
        for index, url in enumerate(urls):
            entry = {"url": url}
            result["attempts"].append(entry)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RELLIS-research-acquisition/1.0"})
                with urllib.request.urlopen(req, timeout=45) as response:
                    entry.update(status=response.status,
                                 content_type=response.headers.get("Content-Type", ""))
                    prefix = response.read(4096)
                    if not prefix.startswith(b"PK"):
                        body = prefix + response.read(1_000_000)
                        (destination / f"{name}_attempt{index}.html").write_bytes(body)
                        parser = Confirmation()
                        parser.feed(body.decode("utf-8", "replace"))
                        if parser.action and index < 2:
                            target = urllib.parse.urlparse(parser.action)
                            if target.scheme == "https" and target.hostname == "drive.usercontent.google.com":
                                urls.append(parser.action + "?" + urllib.parse.urlencode(parser.values))
                        raise ValueError("Expected public ZIP; received non-archive response")
                    length = response.headers.get("Content-Length")
                    if length and int(length) > limit:
                        raise ValueError("Archive exceeds registered acquisition limit")
                    part = archive.with_suffix(".part")
                    total = len(prefix)
                    with part.open("wb") as f:
                        f.write(prefix)
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > limit:
                                raise ValueError("Archive exceeds registered acquisition limit")
                            f.write(chunk)
                    if not zipfile.is_zipfile(part):
                        raise ValueError("Incomplete or invalid ZIP")
                    part.rename(archive)
                break
            except (urllib.error.URLError, ValueError, OSError) as exc:
                entry["error"] = str(exc)
    if archive.exists() and zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            members = []
            for info in z.infolist():
                p = pathlib.PurePosixPath(info.filename)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Unsafe archive member")
                members.append({"name": info.filename, "bytes": info.file_size})
            # Metadata is small and permitted before the task freeze. Examples
            # are not extracted/viewed here: their scene identity must be checked.
            if name in {"intrinsics", "extrinsics", "poses", "image_splits"}:
                if sum(m["bytes"] for m in members) > 500_000_000:
                    raise ValueError("Unexpected metadata expansion")
                z.extractall(destination.parent / "metadata" / name)
        result.update(status="DOWNLOADED", bytes=archive.stat().st_size,
                      sha256=sha(archive), members=members)
    else:
        result["status"] = "UNAVAILABLE"
    print(json.dumps({"name": name, "status": result["status"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.root / "downloads"
    out.mkdir(parents=True, exist_ok=True)
    report = {"source": "https://github.com/unmannedlab/RELLIS-3D",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "source_sha256": sha(pathlib.Path(__file__)),
              "scope": "Official metadata and unopened examples; no training or final-test labels."}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, k, v, out): k for k, v in FILES.items()}
        files = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                files[name] = future.result()
            except Exception as exc:
                files[name] = {"status": "ERROR", "error": repr(exc)}
    report["files"] = files
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.root / "artifacts" / "ACQUISITION.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
