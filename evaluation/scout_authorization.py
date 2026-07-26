"""Provider-free construction and validation of the 120-call scout subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.paid_provider_gateway import ProviderRequest


SCHEMA_VERSION = "interface-contract-scout-authorization-v1"
SCOUT_FAMILY = "direct_path_intersection"
SCOUT_ENVIRONMENTS = ("point_hazard_native", "safety_gym_goal_native")
SCOUT_SEEDS = (20, 21, 22, 23, 24)


def canonical_bytes(value: Any) -> bytes:
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_scout_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported scout authorization schema")
    design = value["design"]
    checks = {
        "blocked": value.get("status") == "BLOCKED",
        "provider_calls_disabled": value.get("provider_calls_enabled") is False,
        "family": design.get("scenario_family") == SCOUT_FAMILY,
        "environments": tuple(design.get("environments", ())) == SCOUT_ENVIRONMENTS,
        "paid_seeds": tuple(design.get("paid_seeds", ())) == SCOUT_SEEDS,
        "two_models": len(value.get("models", ())) == design.get("model_count") == 2,
        "calls_per_model": design.get("calls_per_model") == 60,
        "total_calls": design.get("total_new_provider_calls") == 120,
        "token_cap": value.get("request_parameters", {}).get("max_tokens") == 220,
        "deterministic": value.get("request_parameters", {}).get("temperature") == 0,
        "attempts": value.get("attempt_budget", {}).get(
            "maximum_provider_attempts_per_model"
        ) == 180,
        "positive_spend_cap": float(value.get("maximum_spend_usd", 0)) > 0,
        "blocked_reasons": bool(value.get("authorization_blockers")),
    }
    model_ids = [model["model_budget_id"] for model in value.get("models", ())]
    checks["unique_models"] = len(model_ids) == len(set(model_ids))
    checks["resolved_identities"] = all(
        all(model.get(field) for field in (
            "provider", "provider_model", "revision", "provider_endpoint_slug"
        ))
        for model in value.get("models", ())
    )
    checks["historical_seed_ceiling"] = all(
        set(model.get("historical_paid_seeds", ())).issubset(SCOUT_SEEDS)
        and len(set(model.get("historical_paid_seeds", ()))) <= 5
        for model in value.get("models", ())
    )
    if not all(checks.values()):
        raise ValueError(f"invalid scout authorization manifest: {checks}")
    return value


def _request_parameters(
    manifest: Mapping[str, Any], model: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        **dict(manifest["request_parameters"]),
        "seed": int(row["paid_seed"]),
        "provider": {
            "order": [str(model["provider_endpoint_slug"])],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {
                "prompt": float(model["input_price_usd_per_million"]),
                "completion": float(model["output_price_usd_per_million"]),
            },
        },
    }
    if row["structured_output"]:
        parameters["response_format"] = {"type": "json_object"}
    return parameters


def build_scout_matrix(
    manifest: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_dir: Path,
    image_dir: Path,
) -> list[dict[str, Any]]:
    templates = [
        dict(row)
        for row in source_rows
        if row["model_call_index"] < 480
        and row["scenario_family"] == SCOUT_FAMILY
        and row["model_budget_id"] == source_rows[0]["model_budget_id"]
    ]
    templates.sort(key=lambda row: int(row["model_call_index"]))
    if len(templates) != 60:
        raise ValueError("source matrix must provide exactly 60 scout templates")

    rows: list[dict[str, Any]] = []
    for model in manifest["models"]:
        for model_index, template in enumerate(templates):
            row = dict(template)
            row.update({
                "call_index": len(rows),
                "model_call_index": model_index,
                "model_budget_id": model["model_budget_id"],
                "provider": model["provider"],
                "provider_model": model["provider_model"],
                "model_revision": model["revision"],
                "provider_endpoint_slug": model["provider_endpoint_slug"],
                "compatibility_smoke": model_index == 0,
                "source_request_sha256": template["request_sha256"],
            })
            row["request_id"] = (
                f"{model['model_budget_id']}|{row['block_id']}|{row['arm']}|"
                f"{row['contract_id']}|{row['contract_role']}"
            )
            row["request_parameters"] = _request_parameters(manifest, model, row)
            prompt = (prompt_dir / f"{row['prompt_sha256']}.txt").read_bytes()
            image = (image_dir / f"{row['image_sha256']}.png").read_bytes()
            request = ProviderRequest(
                request_id=row["request_id"],
                model_budget_id=row["model_budget_id"],
                provider=row["provider"],
                provider_model=row["provider_model"],
                model_revision=row["model_revision"],
                paid_seed=int(row["paid_seed"]),
                prompt_bytes=prompt,
                image_bytes=image,
                parameters=row["request_parameters"],
                compatibility_smoke=bool(row["compatibility_smoke"]),
            )
            row["provider_request_sha256"] = request.sha256
            row.pop("request_sha256", None)
            row["request_sha256"] = sha256_bytes(canonical_bytes(row))
            rows.append(row)
    audit_scout_matrix(manifest, rows)
    return rows


def audit_scout_matrix(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    model_ids = [model["model_budget_id"] for model in manifest["models"]]
    checks = {
        "rows_are_120": len(rows) == 120,
        "calls_per_model_are_60": all(
            sum(row["model_budget_id"] == model_id for row in rows) == 60
            for model_id in model_ids
        ),
        "two_environments": set(row["environment"] for row in rows)
        == set(SCOUT_ENVIRONMENTS),
        "direct_path_only": {row["scenario_family"] for row in rows}
        == {SCOUT_FAMILY},
        "paid_seeds_20_24": all(
            {row["paid_seed"] for row in rows if row["model_budget_id"] == model_id}
            == set(SCOUT_SEEDS)
            for model_id in model_ids
        ),
        "one_first_smoke_per_model": all(
            sum(
                row["model_budget_id"] == model_id and row["compatibility_smoke"]
                for row in rows
            ) == 1
            and next(row for row in rows if row["model_budget_id"] == model_id)[
                "compatibility_smoke"
            ]
            for model_id in model_ids
        ),
        "five_pairs_per_environment_and_model": all(
            len({
                row["equivalent_pair_id"]
                for row in rows
                if row["model_budget_id"] == model_id
                and row["environment"] == environment
            }) == 5
            for model_id in model_ids
            for environment in SCOUT_ENVIRONMENTS
        ),
        "unique_logical_rows": len({row["request_sha256"] for row in rows}) == len(rows),
        "unique_provider_requests": len({
            row["provider_request_sha256"] for row in rows
        }) == len(rows),
        "frozen_token_cap": all(
            row["request_parameters"]["max_tokens"] == 220 for row in rows
        ),
        "fallbacks_disabled": all(
            row["request_parameters"]["provider"]["allow_fallbacks"] is False
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"scout matrix audit failed: {checks}")
    return {"passed": True, "checks": checks}
