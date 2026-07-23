"""Frozen zero-network condition matrix for the PointHazard offline gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from evaluation.conditions import (
    EnforcementConfig,
    ExperimentCondition,
    FactorVector,
    PrivilegeLevel,
    Router,
    SeedSplit,
    ZoneSource,
)


OFFLINE_MATRIX_SCHEMA_VERSION = "point-hazard-offline-gate-v1"
PRIMARY_CAPABILITY = "wheeled_non_waterproof"
TWIN_CAPABILITY = "amphibious"


@dataclass(frozen=True)
class OfflineMatrixEntry:
    """One named condition in the provider-free integration matrix."""

    entry_id: str
    family: str
    condition: ExperimentCondition
    provider_calls_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.entry_id or not self.family:
            raise ValueError("offline matrix entry_id and family must be non-empty")
        if self.provider_calls_enabled:
            raise ValueError("offline gate entries cannot enable provider calls")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "family": self.family,
            "provider_calls_enabled": self.provider_calls_enabled,
            "condition_sha256": self.condition.condition_sha256,
            "condition": self.condition.to_dict(),
        }


@dataclass(frozen=True)
class OfflineGateMatrix:
    """Canonical matrix manifest used before any paid-run release."""

    entries: tuple[OfflineMatrixEntry, ...]
    schema_version: str = OFFLINE_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OFFLINE_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported offline matrix schema")
        ids = tuple(entry.entry_id for entry in self.entries)
        hashes = tuple(entry.condition.condition_sha256 for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("offline matrix entry IDs must be unique")
        if len(hashes) != len(set(hashes)):
            raise ValueError("offline matrix conditions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_calls_enabled": False,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _condition(
    enforcement: EnforcementConfig,
    *,
    router: Router,
    zone_source: ZoneSource,
    privilege: PrivilegeLevel,
    capability: str,
    seed: int,
    split: SeedSplit,
) -> ExperimentCondition:
    return ExperimentCondition(
        router=router,
        zone_source=zone_source,
        enforcement=enforcement,
        privilege_level=privilege,
        factor_vector=FactorVector(
            appearance="water-render-v1",
            task_spec_version="task-spec-v1",
            capability=capability,
            privilege_level=privilege,
            annotation_scheme="subgoal-ring-8-r2.5-v1",
            evaluator_version="point-center-discrete-v1.2.1",
            protocol_version="1.2.2",
        ),
        seed=seed,
        split=split,
    )


def build_offline_gate_matrix(
    enforcement: EnforcementConfig,
    *,
    seed: int = 0,
    split: SeedSplit = SeedSplit.DEV,
) -> OfflineGateMatrix:
    """Build the exact 14-condition provider-free development matrix."""
    entries: list[OfflineMatrixEntry] = []
    for router in (Router.DIRECT, Router.REPLAY):
        for zone_source in (ZoneSource.NONE, ZoneSource.ORACLE):
            entries.append(
                OfflineMatrixEntry(
                    entry_id=f"{router.value}-{zone_source.value}-P0-primary",
                    family="shared-router-zone-source",
                    condition=_condition(
                        enforcement,
                        router=router,
                        zone_source=zone_source,
                        privilege=PrivilegeLevel.P0,
                        capability=PRIMARY_CAPABILITY,
                        seed=seed,
                        split=split,
                    ),
                )
            )
    for capability, role in (
        (PRIMARY_CAPABILITY, "primary"),
        (TWIN_CAPABILITY, "capability-twin"),
    ):
        for privilege in PrivilegeLevel:
            entries.append(
                OfflineMatrixEntry(
                    entry_id=f"vlm-none-{privilege.value}-{role}",
                    family=f"vlm-privilege-{role}",
                    condition=_condition(
                        enforcement,
                        router=Router.VLM,
                        zone_source=ZoneSource.NONE,
                        privilege=privilege,
                        capability=capability,
                        seed=seed,
                        split=split,
                    ),
                )
            )
    return OfflineGateMatrix(entries=tuple(entries))
