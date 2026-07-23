"""Fail-closed pilot release and reproducibility-manifest validation.

This module validates release metadata only.  It deliberately contains no
provider client and cannot itself authorize, schedule, or execute a paid run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from evaluation.offline_matrix import OFFLINE_MATRIX_SCHEMA_VERSION
from evaluation.vlm_router import STRUCTURED_PROMPT_VERSION


PILOT_RELEASE_SCHEMA_VERSION = "point-hazard-pilot-release-v1"
RELEASE_STATUSES = frozenset({"BLOCKED", "AUTHORIZED"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_EXACT_REQUIREMENT = re.compile(
    r"[A-Za-z0-9_.-]+==[^=<>!~;@\s]+(?:;\s*.+)?\Z"
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of exact file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dependency_lock(path: str | Path) -> tuple[str, ...]:
    """Reject local paths, editable installs, and non-exact active pins."""
    lock_path = Path(path)
    if not lock_path.is_file():
        raise ValueError(f"dependency lock does not exist: {lock_path}")
    requirements: list[str] = []
    for line_number, raw_line in enumerate(
        lock_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if (
            "file://" in lowered
            or lowered.startswith(("-e ", "--editable "))
            or " @ " in line
        ):
            raise ValueError(
                f"dependency lock line {line_number} uses a local or editable source"
            )
        if _EXACT_REQUIREMENT.fullmatch(line) is None:
            raise ValueError(
                f"dependency lock line {line_number} is not an exact version pin"
            )
        requirements.append(line)
    if not requirements:
        raise ValueError("dependency lock must contain at least one requirement")
    names = [
        re.split(r"==", item, maxsplit=1)[0].lower().replace("_", "-")
        for item in requirements
    ]
    if len(names) != len(set(names)):
        raise ValueError("dependency lock contains duplicate project names")
    return tuple(requirements)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _valid_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class PilotReleaseManifest:
    """Validated machine-readable release state.

    ``provider_calls_enabled`` is derived from the complete authorization state;
    a partially populated manifest always fails closed.
    """

    document: Mapping[str, Any]
    source_path: Path
    repository_root: Path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path | None = None,
    ) -> "PilotReleaseManifest":
        source_path = Path(path).resolve()
        root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else source_path.parent.parent
        )
        try:
            value = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("pilot release manifest must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("pilot release manifest must be a JSON object")
        manifest = cls(document=value, source_path=source_path, repository_root=root)
        manifest.audit()
        return manifest

    @property
    def status(self) -> str:
        return str(self.document["status"])

    @property
    def provider_calls_enabled(self) -> bool:
        return bool(self.document["provider_calls_enabled"])

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def authorization_blockers(self) -> tuple[str, ...]:
        release = self.document["release"]
        blockers: list[str] = []
        if release["release_date"] is None:
            blockers.append("release.release_date")
        if release["authorized_operator"] is None:
            blockers.append("release.authorized_operator")
        budget = release["estimated_maximum_spend_usd"]
        if budget is None:
            blockers.append("release.estimated_maximum_spend_usd")
        seeds = self.document["pilot"]["scene_seeds"]
        if not seeds:
            blockers.append("pilot.scene_seeds")
        models = self.document["models"]
        if not models:
            blockers.append("models")
        return tuple(blockers)

    def _audit_file_reference(self, value: Mapping[str, Any], name: str) -> Path:
        _exact_keys(value, {"path", "sha256"}, name)
        relative = value["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"{name}.path must be a non-empty repository-relative path")
        resolved = (self.repository_root / relative).resolve()
        try:
            resolved.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError(f"{name}.path escapes repository root") from exc
        _valid_sha256(value["sha256"], f"{name}.sha256")
        if not resolved.is_file():
            raise ValueError(f"{name}.path does not exist")
        if sha256_file(resolved) != value["sha256"]:
            raise ValueError(f"{name}.sha256 does not match file bytes")
        return resolved

    def audit(self) -> dict[str, Any]:
        expected = {
            "schema_version",
            "status",
            "provider_calls_enabled",
            "release",
            "protocol",
            "offline_gate",
            "reproducibility",
            "pilot",
            "analysis",
            "models",
        }
        _exact_keys(self.document, expected, "pilot release manifest")
        if self.document["schema_version"] != PILOT_RELEASE_SCHEMA_VERSION:
            raise ValueError("unsupported pilot release manifest schema")
        if self.status not in RELEASE_STATUSES:
            raise ValueError("status must be BLOCKED or AUTHORIZED")
        if not isinstance(self.document["provider_calls_enabled"], bool):
            raise ValueError("provider_calls_enabled must be boolean")

        release = self.document["release"]
        _exact_keys(
            release,
            {
                "release_date",
                "implementation_git_sha",
                "authorized_operator",
                "estimated_maximum_spend_usd",
            },
            "release",
        )
        if _GIT_SHA.fullmatch(str(release["implementation_git_sha"])) is None:
            raise ValueError("release.implementation_git_sha must be a lowercase git SHA")
        if release["release_date"] is not None:
            try:
                date.fromisoformat(release["release_date"])
            except (TypeError, ValueError) as exc:
                raise ValueError("release.release_date must use YYYY-MM-DD") from exc
        if release["authorized_operator"] is not None and (
            not isinstance(release["authorized_operator"], str)
            or not release["authorized_operator"].strip()
        ):
            raise ValueError("release.authorized_operator must be null or non-empty")
        budget = release["estimated_maximum_spend_usd"]
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0
        ):
            raise ValueError("estimated maximum spend must be null or positive")

        protocol = self.document["protocol"]
        _exact_keys(
            protocol,
            {"version", "document", "structured_prompt_version"},
            "protocol",
        )
        if protocol["version"] != "1.2.2":
            raise ValueError("pilot protocol version must be 1.2.2")
        if protocol["structured_prompt_version"] != STRUCTURED_PROMPT_VERSION:
            raise ValueError("structured prompt version does not match implementation")
        self._audit_file_reference(protocol["document"], "protocol.document")

        offline = self.document["offline_gate"]
        _exact_keys(
            offline,
            {
                "schema_version",
                "entry_count",
                "matrix_sha256",
                "provider_calls_enabled",
            },
            "offline_gate",
        )
        if offline != {
            "schema_version": OFFLINE_MATRIX_SCHEMA_VERSION,
            "entry_count": 14,
            "matrix_sha256": "3bf69114eb74e5bba0248b26e526c6dd1a254eeb8452c5ad429087950bc15725",
            "provider_calls_enabled": False,
        }:
            raise ValueError("offline gate identity does not match the frozen matrix")

        reproducibility = self.document["reproducibility"]
        _exact_keys(
            reproducibility,
            {"python_version", "platform", "dependency_lock"},
            "reproducibility",
        )
        if reproducibility["python_version"] != "3.10.18":
            raise ValueError("pilot runtime must pin Python 3.10.18")
        if reproducibility["platform"] != "macos-arm64":
            raise ValueError("pilot platform must be macos-arm64")
        lock_path = self._audit_file_reference(
            reproducibility["dependency_lock"], "reproducibility.dependency_lock"
        )
        locked_requirements = validate_dependency_lock(lock_path)

        pilot = self.document["pilot"]
        _exact_keys(
            pilot,
            {"split", "allocated_range", "scene_seeds", "family_count_bounds"},
            "pilot",
        )
        if pilot["split"] != "pilot" or pilot["allocated_range"] != [200, 299]:
            raise ValueError("pilot split must use the frozen 200-299 allocation")
        if pilot["family_count_bounds"] != [30, 50]:
            raise ValueError("pilot family count bounds must be 30-50")
        seeds = pilot["scene_seeds"]
        if (
            not isinstance(seeds, list)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(seeds) != len(set(seeds))
            or any(seed < 200 or seed > 299 for seed in seeds)
        ):
            raise ValueError("pilot scene_seeds must be unique integers in 200-299")
        if seeds and not 30 <= len(seeds) <= 50:
            raise ValueError("a frozen pilot must contain 30-50 scene families")

        analysis = self.document["analysis"]
        _exact_keys(
            analysis,
            {
                "primary_metric",
                "statistical_unit",
                "pilot_use",
                "formal_pooling_allowed",
            },
            "analysis",
        )
        if analysis != {
            "primary_metric": "safe_task_completion",
            "statistical_unit": "scenario_family",
            "pilot_use": ["go_no_go", "power_estimation", "prespecified_tuning"],
            "formal_pooling_allowed": False,
        }:
            raise ValueError("analysis block does not match the frozen pilot policy")

        models = self.document["models"]
        if not isinstance(models, list):
            raise ValueError("models must be a list")
        model_ids: set[tuple[str, str, str]] = set()
        for index, model in enumerate(models):
            if not isinstance(model, dict):
                raise ValueError(f"models[{index}] must be an object")
            _exact_keys(
                model,
                {"provider", "model", "revision", "revision_date", "open_weights"},
                f"models[{index}]",
            )
            for field in ("provider", "model", "revision"):
                if not isinstance(model[field], str) or not model[field]:
                    raise ValueError(f"models[{index}].{field} must be non-empty")
            try:
                date.fromisoformat(model["revision_date"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"models[{index}].revision_date must use YYYY-MM-DD"
                ) from exc
            if not isinstance(model["open_weights"], bool):
                raise ValueError(f"models[{index}].open_weights must be boolean")
            identity = (model["provider"], model["model"], model["revision"])
            if identity in model_ids:
                raise ValueError("model provider/model/revision identities must be unique")
            model_ids.add(identity)
        if models and (
            len(models) < 2 or not any(model["open_weights"] for model in models)
        ):
            raise ValueError(
                "pilot model list needs at least two models and one open-weight model"
            )

        blockers = self.authorization_blockers()
        authorized = not blockers
        if self.provider_calls_enabled != authorized:
            raise ValueError(
                "provider_calls_enabled must equal the complete authorization state"
            )
        expected_status = "AUTHORIZED" if authorized else "BLOCKED"
        if self.status != expected_status:
            raise ValueError(f"status must be {expected_status} for the populated fields")
        return {
            "schema_version": PILOT_RELEASE_SCHEMA_VERSION,
            "status": self.status,
            "provider_calls_enabled": self.provider_calls_enabled,
            "authorization_blockers": list(blockers),
            "dependency_count": len(locked_requirements),
            "manifest_sha256": self.sha256,
        }

    def assert_provider_calls_authorized(self) -> None:
        """Fail before a provider client can be connected."""
        blockers = self.authorization_blockers()
        if not self.provider_calls_enabled or blockers:
            joined = ", ".join(blockers) if blockers else "provider gate disabled"
            raise PermissionError(f"paid provider calls are not authorized: {joined}")
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PermissionError("unable to verify release git state") from exc
        expected = self.document["release"]["implementation_git_sha"]
        if head != expected:
            raise PermissionError(
                "paid provider calls are not authorized: release git SHA does not match HEAD"
            )
        if dirty:
            raise PermissionError(
                "paid provider calls are not authorized: working tree is dirty"
            )
