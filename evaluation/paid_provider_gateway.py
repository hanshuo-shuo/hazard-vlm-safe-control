"""Single fail-closed gateway for every new paid provider request.

The ledger counts logical requests separately from transport attempts.  Both
reservations are durable and happen before transport.  Cache replay and fixtures
never mutate the ledger.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping


LEDGER_SCHEMA_VERSION = "paid-provider-ledger-v1"
MANIFEST_SCHEMA_VERSION = "interface-contract-pilot-manifest-v1"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PaidBudget:
    allowed_paid_seeds: tuple[int, ...]
    maximum_distinct_paid_seeds_per_model: int
    maximum_new_provider_calls_per_model: int
    maximum_attempts_per_request: int
    maximum_retries_per_request: int
    maximum_total_provider_attempts_per_model: int
    maximum_spend_usd: float | None = None

    def __post_init__(self) -> None:
        if self.allowed_paid_seeds != (20, 21, 22, 23, 24):
            raise ValueError("paid seeds must be frozen to exactly 20-24")
        if self.maximum_distinct_paid_seeds_per_model != 5:
            raise ValueError("maximum distinct paid seeds must equal five")
        if self.maximum_new_provider_calls_per_model != 480:
            raise ValueError("maximum new provider calls per model must equal 480")
        if self.maximum_attempts_per_request != 3:
            raise ValueError("maximum attempts per request must equal three")
        if self.maximum_retries_per_request != 2:
            raise ValueError("maximum retries per request must equal two")
        if self.maximum_attempts_per_request != self.maximum_retries_per_request + 1:
            raise ValueError("attempt and retry limits are inconsistent")
        if self.maximum_total_provider_attempts_per_model != 1440:
            raise ValueError("maximum total attempts per model must equal 1440")
        if self.maximum_spend_usd is not None and self.maximum_spend_usd <= 0:
            raise ValueError("maximum spend must be null or positive")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "PaidBudget":
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported interface-contract manifest")
        values = dict(manifest["paid_budget"])
        values["allowed_paid_seeds"] = tuple(
            int(seed) for seed in values["allowed_paid_seeds"]
        )
        return cls(**values)


@dataclass(frozen=True)
class ModelBudgetIdentity:
    model_budget_id: str
    provider: str | None
    provider_model: str | None
    revision: str | None
    historical_paid_seeds: tuple[int, ...]
    eligibility: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelBudgetIdentity":
        return cls(
            model_budget_id=str(value["model_budget_id"]),
            provider=value.get("provider"),
            provider_model=value.get("provider_model"),
            revision=value.get("revision"),
            historical_paid_seeds=tuple(int(seed) for seed in value["historical_paid_seeds"]),
            eligibility=str(value["eligibility"]),
        )

    @property
    def resolved(self) -> bool:
        return bool(
            self.provider
            and self.provider_model
            and self.revision
            and self.revision != "UNRESOLVED"
            and self.eligibility not in {"UNRESOLVED", "PAID_INELIGIBLE"}
        )


class PaidCallLedger:
    """Durable per-model seed/call/attempt ledger with an OS file lock."""

    def __init__(self, path: str | Path, manifest: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.manifest = dict(manifest)
        self.budget = PaidBudget.from_manifest(manifest)
        self.identities = {
            identity.model_budget_id: identity
            for identity in (
                ModelBudgetIdentity.from_mapping(value) for value in manifest["models"]
            )
        }
        if len(self.identities) != len(manifest["models"]):
            raise ValueError("duplicate canonical model budget ID")
        for identity in self.identities.values():
            if len(set(identity.historical_paid_seeds)) > self.budget.maximum_distinct_paid_seeds_per_model:
                if identity.eligibility != "PAID_INELIGIBLE":
                    raise ValueError(
                        f"historical paid seeds exceed five for {identity.model_budget_id}"
                    )

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "protocol_id": self.manifest["protocol_id"],
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "models": {
                model_id: {
                    "historical_paid_seeds": list(identity.historical_paid_seeds),
                    "reserved_paid_seeds": [],
                    "new_provider_calls": 0,
                    "provider_attempts": 0,
                    "compatibility_smoke_request_sha256": None,
                    "compatibility_smoke_status": "NOT_RUN",
                    "requests": {},
                }
                for model_id, identity in self.identities.items()
            },
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise RuntimeError("paid ledger schema mismatch")
        if value.get("protocol_id") != self.manifest["protocol_id"]:
            raise RuntimeError("paid ledger protocol mismatch")
        if set(value.get("models", {})) != set(self.identities):
            raise RuntimeError("paid ledger model identities differ from manifest")
        for model in value["models"].values():
            model.setdefault("compatibility_smoke_request_sha256", None)
            model.setdefault("compatibility_smoke_status", "NOT_RUN")
        return value

    def _write(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(value)
        data["updated_at"] = _utc_now()
        raw = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            value = self._read()
            try:
                yield value
            except Exception:
                raise
            else:
                self._write(value)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def reserve_request(
        self,
        model_budget_id: str,
        paid_seed: int,
        request_sha256: str,
        *,
        compatibility_smoke: bool | None = None,
    ) -> bool:
        """Reserve one logical call; return False when it was already reserved."""
        if model_budget_id not in self.identities:
            raise PermissionError("unregistered canonical model budget ID")
        identity = self.identities[model_budget_id]
        if not identity.resolved:
            raise PermissionError(f"model budget identity is not paid-eligible: {model_budget_id}")
        if paid_seed not in self.budget.allowed_paid_seeds:
            raise PermissionError("paid seed is outside the preregistered 20-24 set")
        with self._locked() as ledger:
            model = ledger["models"][model_budget_id]
            requests = model["requests"]
            if request_sha256 in requests:
                previous = requests[request_sha256]
                if int(previous["paid_seed"]) != int(paid_seed):
                    raise RuntimeError("request hash was previously registered under another seed")
                if compatibility_smoke is not None and bool(
                    previous.get("compatibility_smoke", False)
                ) != compatibility_smoke:
                    raise RuntimeError("request smoke designation changed after reservation")
                return False
            if compatibility_smoke is not None:
                smoke_sha = model["compatibility_smoke_request_sha256"]
                smoke_status = model["compatibility_smoke_status"]
                if compatibility_smoke:
                    if smoke_sha not in {None, request_sha256}:
                        raise PermissionError("a different compatibility smoke was already frozen")
                    if smoke_sha is None and int(model["new_provider_calls"]) != 0:
                        raise PermissionError("compatibility smoke must be the first formal request")
                    model["compatibility_smoke_request_sha256"] = request_sha256
                    model["compatibility_smoke_status"] = "RESERVED"
                elif smoke_status != "SUCCEEDED":
                    raise PermissionError(
                        "model is not eligible: compatibility smoke has not succeeded"
                    )
            paid_union = set(model["historical_paid_seeds"]) | set(model["reserved_paid_seeds"]) | {paid_seed}
            if len(paid_union) > self.budget.maximum_distinct_paid_seeds_per_model:
                raise PermissionError("five-distinct-paid-seed ceiling exceeded")
            if int(model["new_provider_calls"]) >= self.budget.maximum_new_provider_calls_per_model:
                raise PermissionError("480-new-provider-call ceiling exceeded")
            model["reserved_paid_seeds"] = sorted(
                set(model["reserved_paid_seeds"]) | {int(paid_seed)}
            )
            model["new_provider_calls"] = int(model["new_provider_calls"]) + 1
            requests[request_sha256] = {
                "paid_seed": int(paid_seed),
                "attempts": 0,
                "status": "RESERVED",
                "compatibility_smoke": bool(compatibility_smoke),
                "reserved_at": _utc_now(),
                "events": [],
            }
        return True

    def reserve_attempt(self, model_budget_id: str, request_sha256: str) -> int:
        """Durably count an attempt before transport and return its 1-based number."""
        with self._locked() as ledger:
            model = ledger["models"][model_budget_id]
            request = model["requests"].get(request_sha256)
            if request is None:
                raise RuntimeError("attempt cannot precede logical request reservation")
            if int(request["attempts"]) >= self.budget.maximum_attempts_per_request:
                raise PermissionError("per-request provider-attempt ceiling exceeded")
            if int(model["provider_attempts"]) >= self.budget.maximum_total_provider_attempts_per_model:
                raise PermissionError("per-model provider-attempt ceiling exceeded")
            request["attempts"] = int(request["attempts"]) + 1
            model["provider_attempts"] = int(model["provider_attempts"]) + 1
            attempt = int(request["attempts"])
            request["events"].append(
                {"event": "ATTEMPT_RESERVED", "attempt": attempt, "at": _utc_now()}
            )
            request["status"] = "ATTEMPTED"
            return attempt

    def record_outcome(
        self,
        model_budget_id: str,
        request_sha256: str,
        *,
        status: str,
        detail: str | None = None,
    ) -> None:
        allowed = {
            "SUCCEEDED",
            "FAILED",
            "TIMEOUT",
            "INVALID_RESPONSE",
            "MODEL_MISMATCH",
            "SMOKE_FAILED_ELIMINATED",
        }
        if status not in allowed:
            raise ValueError("unsupported provider outcome status")
        with self._locked() as ledger:
            model = ledger["models"][model_budget_id]
            request = model["requests"].get(request_sha256)
            if request is None:
                raise RuntimeError("outcome cannot precede request reservation")
            request["status"] = status
            request["events"].append(
                {"event": status, "detail": detail, "at": _utc_now()}
            )
            if request.get("compatibility_smoke", False):
                model["compatibility_smoke_status"] = status

    def snapshot(self) -> dict[str, Any]:
        # Audits and cache-only inspection must not rewrite the durable ledger.
        # Using ``_locked`` here used to refresh ``updated_at`` on every read,
        # which made provider-free dry-run generation dirty its own clean git
        # checkout before provenance was captured.
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                value = self._read()
                return json.loads(json.dumps(value))
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def audit(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        model_audits: dict[str, Any] = {}
        passed = True
        for model_id, value in snapshot["models"].items():
            paid_union = sorted(
                set(value["historical_paid_seeds"]) | set(value["reserved_paid_seeds"])
            )
            request_attempt_max = max(
                (int(item["attempts"]) for item in value["requests"].values()), default=0
            )
            checks = {
                "paid_seeds_subset_20_24": set(paid_union).issubset(self.budget.allowed_paid_seeds),
                "distinct_paid_seed_count_at_most_5": len(paid_union) <= 5,
                "new_provider_calls_at_most_480": int(value["new_provider_calls"]) <= 480,
                "attempts_per_request_at_most_3": request_attempt_max <= 3,
                "provider_attempts_at_most_1440": int(value["provider_attempts"]) <= 1440,
                "logical_call_count_matches_requests": int(value["new_provider_calls"])
                == len(value["requests"]),
                "compatibility_smoke_state_valid": value["compatibility_smoke_status"]
                in {
                    "NOT_RUN",
                    "RESERVED",
                    "SUCCEEDED",
                    "FAILED",
                    "TIMEOUT",
                    "INVALID_RESPONSE",
                    "MODEL_MISMATCH",
                    "SMOKE_FAILED_ELIMINATED",
                },
            }
            passed = passed and all(checks.values())
            model_audits[model_id] = {
                "paid_seed_union": paid_union,
                "new_provider_calls": int(value["new_provider_calls"]),
                "provider_attempts": int(value["provider_attempts"]),
                "maximum_request_attempts_observed": request_attempt_max,
                "compatibility_smoke_request_sha256": value[
                    "compatibility_smoke_request_sha256"
                ],
                "compatibility_smoke_status": value["compatibility_smoke_status"],
                "checks": checks,
            }
        return {
            "schema_version": "paid-provider-budget-audit-v1",
            "passed": passed,
            "budget": self.budget.__dict__,
            "models": model_audits,
        }


@dataclass(frozen=True)
class ProviderRequest:
    request_id: str
    model_budget_id: str
    provider: str
    provider_model: str
    model_revision: str
    paid_seed: int
    prompt_bytes: bytes
    image_bytes: bytes
    parameters: Mapping[str, Any]
    compatibility_smoke: bool = False

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_budget_id": self.model_budget_id,
            "provider": self.provider,
            "provider_model": self.provider_model,
            "model_revision": self.model_revision,
            "paid_seed": self.paid_seed,
            "prompt_sha256": _sha256(self.prompt_bytes),
            "image_sha256": _sha256(self.image_bytes),
            "parameters": dict(self.parameters),
            "compatibility_smoke": self.compatibility_smoke,
        }

    @property
    def sha256(self) -> str:
        return _sha256(_canonical(self.canonical_payload()))


Transport = Callable[[ProviderRequest, int], Mapping[str, Any]]
ResponseValidator = Callable[[ProviderRequest, Mapping[str, Any]], None]


class PaidProviderGateway:
    """Cache-first gateway. Network-capable transport must be explicitly injected."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        ledger: PaidCallLedger,
        cache_dir: str | Path,
        allow_provider_requests: bool = False,
        transport: Transport | None = None,
        response_validator: ResponseValidator | None = None,
    ) -> None:
        self.manifest = dict(manifest)
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.allow_provider_requests = allow_provider_requests
        self.transport = transport
        self.response_validator = response_validator

    def _identity_audit(self, request: ProviderRequest) -> ModelBudgetIdentity:
        identity = self.ledger.identities.get(request.model_budget_id)
        if identity is None:
            raise PermissionError("request uses an unregistered model budget identity")
        if not identity.resolved:
            raise PermissionError("request model identity/revision is unresolved")
        if (identity.provider, identity.provider_model, identity.revision) != (
            request.provider,
            request.provider_model,
            request.model_revision,
        ):
            raise PermissionError("provider/model/revision does not match canonical budget identity")
        return identity

    def call(self, request: ProviderRequest) -> dict[str, Any]:
        self._identity_audit(request)
        cache_path = self.cache_dir / f"{request.sha256}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("request_sha256") != request.sha256:
                raise RuntimeError("cache request hash mismatch")
            return {
                **cached,
                "call_origin": "cache_replay",
                "new_provider_call": False,
            }
        if not self.allow_provider_requests:
            raise PermissionError("cache miss: provider calls remain disabled")
        if not self.manifest.get("provider_calls_enabled", False):
            raise PermissionError("manifest does not authorize provider calls")
        if self.transport is None:
            raise RuntimeError("no provider transport was injected")

        self.ledger.reserve_request(
            request.model_budget_id,
            request.paid_seed,
            request.sha256,
            compatibility_smoke=request.compatibility_smoke,
        )
        last_error = "provider did not return a valid response"
        for _ in range(self.ledger.budget.maximum_attempts_per_request):
            attempt = self.ledger.reserve_attempt(request.model_budget_id, request.sha256)
            try:
                response = dict(self.transport(request, attempt))
            except TimeoutError as exc:
                last_error = str(exc)
                self.ledger.record_outcome(
                    request.model_budget_id,
                    request.sha256,
                    status="TIMEOUT",
                    detail=last_error,
                )
                continue
            except Exception as exc:
                last_error = str(exc)
                self.ledger.record_outcome(
                    request.model_budget_id,
                    request.sha256,
                    status="FAILED",
                    detail=last_error,
                )
                continue
            if response.get("provider_model") != request.provider_model:
                last_error = "returned model identity mismatch"
                self.ledger.record_outcome(
                    request.model_budget_id,
                    request.sha256,
                    status="MODEL_MISMATCH",
                    detail=last_error,
                )
                continue
            if not isinstance(response.get("raw_response"), str) or not response["raw_response"].strip():
                last_error = "empty or invalid provider response"
                self.ledger.record_outcome(
                    request.model_budget_id,
                    request.sha256,
                    status="INVALID_RESPONSE",
                    detail=last_error,
                )
                continue
            if self.response_validator is not None:
                try:
                    self.response_validator(request, response)
                except Exception as exc:
                    last_error = f"strict response validation failed: {exc}"
                    self.ledger.record_outcome(
                        request.model_budget_id,
                        request.sha256,
                        status="INVALID_RESPONSE",
                        detail=last_error,
                    )
                    continue
            record = {
                **response,
                "request_sha256": request.sha256,
                "request": request.canonical_payload(),
                "call_origin": "provider",
                "new_provider_call": True,
                "successful_attempt": attempt,
                "recorded_at": _utc_now(),
            }
            self.ledger.record_outcome(
                request.model_budget_id, request.sha256, status="SUCCEEDED"
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return record
        if request.compatibility_smoke:
            self.ledger.record_outcome(
                request.model_budget_id,
                request.sha256,
                status="SMOKE_FAILED_ELIMINATED",
                detail=last_error,
            )
        raise RuntimeError(f"provider request exhausted frozen attempts: {last_error}")


def load_pilot_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    PaidBudget.from_manifest(value)
    if value["calls_per_model"] != 480 or value["total_call_matrix_rows"] != 1440:
        raise ValueError("manifest call totals are not frozen to 480/1440")
    if value["provider_calls_enabled"] is not False or value["status"] != "BLOCKED":
        raise PermissionError("checked-in pilot manifest must remain blocked")
    return value
