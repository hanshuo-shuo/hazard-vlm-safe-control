"""Independent development runs with exact source snapshots and artifact hashes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

from c3_safe import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_RUN = None


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_run(path):
    global _ACTIVE_RUN
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"choose a new run directory; refusing to overwrite {path}")
    path.mkdir(parents=True, exist_ok=True)
    _ACTIVE_RUN = path
    write_json(path / "RUN_STATUS.json", {"status": "RUNNING"})
    sources = set(ROOT.glob("c3_safe/*.py")) | {ROOT / p for p in (
        "configs/c3_safe_foundation.json", "env_pointhazard.py", "hazard_renderer.py", "mpc_expert.py", "safe_expert.py",
        "envs/protocol_env.py", "envs/point_hazard_adapter.py", "envs/safety_gym_goal_adapter.py",
        "evaluation/outcomes.py", "evaluation/semantic_evaluator.py", "evaluation/schemas.py")}
    entry = Path(sys.argv[0]).resolve()
    if entry.is_file() and entry.is_relative_to(ROOT):
        sources.add(entry)
    # Include local imported helpers (for example the saved-student comparison
    # imports the trainer's evaluator). Snapshot at run start, not after work.
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            module_path = Path(filename).resolve()
            if module_path.suffix == ".py" and module_path.is_relative_to(ROOT) and module_path.is_file():
                sources.add(module_path)
    hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        target = path / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[str(relative)] = file_sha(target)
    write_json(path / "SOURCE_SNAPSHOT.json", hashes)
    return path


def finish_run(path, summary, *, source_paths=()):
    path = Path(path)
    write_json(path / "SUMMARY.json", summary)
    source_hashes = json.loads((path / "SOURCE_SNAPSHOT.json").read_text())
    missing = set(source_paths) - set(source_hashes)
    if missing:
        raise RuntimeError(f"run entry point was not snapshotted before execution: {sorted(missing)}")
    drift = [name for name,sha in source_hashes.items() if file_sha(ROOT / name) != sha]
    if drift:
        raise RuntimeError(f"source changed during execution: {drift}")
    write_json(path / "RUN_STATUS.json", {"status": "COMPLETE", "scientific_validation": False})
    from evaluation.schemas import dependency_versions
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "DEVELOPMENT_INFRASTRUCTURE_NOT_VALIDATED",
        "provider_calls": 0,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "dependencies": dependency_versions(),
        "source_files": source_hashes,
        "source_snapshot_sha256": digest(source_hashes),
        "artifacts": {str(p.relative_to(path)): file_sha(p) for p in sorted(path.rglob("*")) if p.is_file()},
    }
    write_json(path / "MANIFEST.json", manifest)
    return manifest


def run_cli(main):
    """Retain partial artifacts and a visible failure record on CLI errors."""
    try:
        main()
    except BaseException as exc:
        if _ACTIVE_RUN is not None:
            status_path = _ACTIVE_RUN / "RUN_STATUS.json"
            status = json.loads(status_path.read_text())
            if status.get("status") != "COMPLETE":
                write_json(status_path, {"status":"FAILED", "error_type":type(exc).__name__, "detail":str(exc)})
        raise
