"""Acceptance tests for the pilot release and reproducibility lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.release_manifest import (
    PILOT_RELEASE_SCHEMA_VERSION,
    PilotReleaseManifest,
    sha256_file,
    validate_dependency_lock,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "pilot_release_manifest.json"
LOCK_PATH = ROOT / "requirements.lock"


def _write_manifest(tmp_path: Path, value: dict) -> Path:
    config = tmp_path / "configs"
    config.mkdir()
    path = config / "pilot_release_manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_checked_in_manifest_and_dependency_lock_audit_fail_closed() -> None:
    manifest = PilotReleaseManifest.load(MANIFEST_PATH, repository_root=ROOT)
    audit = manifest.audit()
    assert audit["schema_version"] == PILOT_RELEASE_SCHEMA_VERSION
    assert audit["status"] == "BLOCKED"
    assert audit["provider_calls_enabled"] is False
    assert audit["dependency_count"] == 10
    assert audit["authorization_blockers"] == [
        "release.release_date",
        "release.authorized_operator",
        "release.estimated_maximum_spend_usd",
        "pilot.scene_seeds",
        "models",
    ]
    with pytest.raises(PermissionError, match="paid provider calls are not authorized"):
        manifest.assert_provider_calls_authorized()


def test_lock_has_only_portable_exact_pins_and_matches_manifest() -> None:
    requirements = validate_dependency_lock(LOCK_PATH)
    assert "numpy==1.23.5" in requirements
    assert "Pillow==9.4.0" in requirements
    assert all("file://" not in requirement for requirement in requirements)
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert (
        value["reproducibility"]["dependency_lock"]["sha256"]
        == sha256_file(LOCK_PATH)
    )


@pytest.mark.parametrize(
    "bad_line",
    [
        "numpy>=1.23\n",
        "numpy @ file:///tmp/numpy.whl\n",
        "-e ../local-package\n",
    ],
)
def test_dependency_lock_rejects_ranges_and_local_sources(
    tmp_path: Path, bad_line: str
) -> None:
    path = tmp_path / "requirements.lock"
    path.write_text(bad_line, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_dependency_lock(path)


def test_manifest_rejects_tampered_referenced_file(tmp_path: Path) -> None:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROTOCOL.md").write_text("tampered", encoding="utf-8")
    (tmp_path / "requirements.lock").write_bytes(LOCK_PATH.read_bytes())
    path = _write_manifest(tmp_path, value)
    with pytest.raises(ValueError, match="protocol.document.sha256"):
        PilotReleaseManifest.load(path, repository_root=tmp_path)


def test_partial_authorization_cannot_enable_provider_calls(tmp_path: Path) -> None:
    value = copy.deepcopy(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    value["provider_calls_enabled"] = True
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROTOCOL.md").write_bytes(
        (ROOT / "docs" / "PROTOCOL.md").read_bytes()
    )
    (tmp_path / "requirements.lock").write_bytes(LOCK_PATH.read_bytes())
    path = _write_manifest(tmp_path, value)
    with pytest.raises(
        ValueError, match="provider_calls_enabled must equal the complete authorization state"
    ):
        PilotReleaseManifest.load(path, repository_root=tmp_path)


def test_fully_populated_authorization_requires_models_and_exact_pilot_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = copy.deepcopy(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    value["status"] = "AUTHORIZED"
    value["provider_calls_enabled"] = True
    value["release"].update(
        {
            "release_date": "2026-07-24",
            "authorized_operator": "release-operator",
            "estimated_maximum_spend_usd": 25.0,
        }
    )
    value["pilot"]["scene_seeds"] = list(range(200, 230))
    value["models"] = [
        {
            "provider": "local",
            "model": "open-model",
            "revision": "commit-a",
            "revision_date": "2026-07-20",
            "open_weights": True,
        },
        {
            "provider": "provider-a",
            "model": "closed-model-a",
            "revision": "revision-a",
            "revision_date": "2026-07-20",
            "open_weights": False,
        },
    ]
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROTOCOL.md").write_bytes(
        (ROOT / "docs" / "PROTOCOL.md").read_bytes()
    )
    (tmp_path / "requirements.lock").write_bytes(LOCK_PATH.read_bytes())
    path = _write_manifest(tmp_path, value)
    manifest = PilotReleaseManifest.load(path, repository_root=tmp_path)
    assert manifest.provider_calls_enabled is True
    assert manifest.authorization_blockers() == ()
    monkeypatch.setattr(
        "evaluation.release_manifest.subprocess.run",
        lambda args, **_kwargs: SimpleNamespace(
            stdout=("b" * 40 + "\n") if args[-1] == "HEAD" else ""
        ),
    )
    with pytest.raises(PermissionError, match="release git SHA does not match HEAD"):
        manifest.assert_provider_calls_authorized()
