import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_semantic_geometry_audit import assert_no_secrets


ROOT = Path(__file__).resolve().parents[1]


def run_audit(output, *extra):
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_semantic_geometry_audit.py",
            "--output",
            str(output),
            "--seeds",
            "0",
            "--dry-run",
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_runner_manifest_hash_and_resume(tmp_path):
    first = run_audit(tmp_path, "--geometry-source", "fixture")
    assert first.returncode == 0, first.stderr
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["row_count"] == 1
    assert manifest["provider_calls"] == 0
    second = run_audit(tmp_path, "--geometry-source", "fixture", "--resume")
    assert second.returncode == 0, second.stderr
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["resumed_rows"] == 1


def test_runner_partial_failure_is_manifested(tmp_path):
    result = run_audit(tmp_path, "--geometry-source", "detector")
    assert result.returncode == 1
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["failed_rows"][0]["error_type"] == "ValueError"
    assert manifest["missing_artifact_audit"] == [0]


def test_cached_detector_decomposition_row(tmp_path):
    cached = ROOT / "results/next_five_experiments/03_modern_perception_substitution/results.json"
    result = run_audit(
        tmp_path,
        "--geometry-source",
        "detector",
        "--cached-detector-results",
        str(cached),
        "--geometry-arm",
        "detector_center_oracle_radius",
    )
    assert result.returncode == 0, result.stderr
    row_path = next((tmp_path / "rows").glob("*.json"))
    row = json.loads(row_path.read_text())
    assert row["geometry_decomposition"]["geometry"]["radius"] > 0
    assert row["geometry_decomposition"]["contains_oracle_component"] is True
    assert row["geometry_decomposition"]["planner_execution_status"] == "not_run"


def test_fixed_planner_row_records_closed_loop_metrics(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_semantic_geometry_audit.py",
            "--output",
            str(tmp_path),
            "--seeds",
            "0",
            "--geometry-source",
            "fixture",
            "--max-steps",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads(next((tmp_path / "rows").glob("*.json")).read_text())
    assert row["result_status"] == "FIXED_PLANNER_INFRASTRUCTURE_RUN"
    assert row["closed_loop_metrics"]["path_length"] >= 0
    assert row["replan_count"] == 1


def test_secret_and_authorization_leakage_rejected():
    with pytest.raises(PermissionError):
        assert_no_secrets({"nested": {"Authorization": "Bearer secret"}})
    with pytest.raises(PermissionError):
        assert_no_secrets({"text": "Bearer secret"})


def test_terminated_marker_runner_is_fail_closed():
    result = subprocess.run(
        [sys.executable, "scripts/run_next_five_experiments.py", "--experiments", "5"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "TERMINATED" in result.stderr


def test_terminated_marker_override_cannot_enable_provider():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_next_five_experiments.py",
            "--experiments",
            "5",
            "--allow-terminated-marker-pilot",
            "--allow-provider-requests",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "cannot enable paid/provider" in result.stderr
