# Paid-run release record

Status: `BLOCKED / NOT AUTHORIZED`

This record documents release readiness only. It does not authorize a provider
call, expenditure, pilot run, or promotion of fixture artifacts to scientific
results.

The machine-readable authority is
[`configs/pilot_release_manifest.json`](../configs/pilot_release_manifest.json).
`evaluation.release_manifest.PilotReleaseManifest` validates the referenced
protocol and dependency-lock bytes and fails closed unless every authorization
field is populated, the release SHA equals `HEAD`, and the working tree is
clean.

## Required release fields

| Field | Value |
|---|---|
| Release date | `NOT SET` |
| Technical baseline git SHA | `e353b92b2cc8a297202d2560df30b5505b6848f2` |
| Protocol version | `1.2.2` |
| Model list | `NOT FROZEN` |
| Pilot seed range | allocated `200–299`; exact 30–50-family subset `NOT FROZEN` |
| Estimated maximum spend | `NOT SET` |
| Authorized operator | `NOT SET` |

The manifest additionally keeps `provider_calls_enabled=false`, an empty exact
pilot seed list and an empty model list. The pilot allocation remains `200–299`
with a required frozen subset of 30–50 scenario families.

## Reproducibility lock

| Field | Value |
|---|---|
| Runtime | `CPython 3.10.18` |
| Platform | `macos-arm64` |
| Core lock | `requirements.lock` |
| Core lock SHA-256 | `2097feb031890c281a9c8e5e9b76ce1aae9e202f700c3fd0a25d4a07012e1418` |
| Protocol document SHA-256 | `d9ea8a7032764c1b609a70b57d47d2e1bac612677c5025e47bbf2cf753e277ce` |

The core lock contains only portable exact version pins; local `file://`,
editable and version-range dependencies are rejected by the release validator.
The optional Safety-Gymnasium stack remains isolated in
`requirements-safety-gym.txt` and is outside this PointHazard pilot lock.

## Passed offline gates

- B01 four-configuration 10,000-seed layout validation;
- unified condition serialization and hashes;
- direct/replay × none/oracle shared-enforcement harness;
- replay reconstruction and parity;
- static and dynamic minimal-permission boundary;
- forbidden-field and evaluator-only provenance rejection;
- shared STC truth table and episode reduction;
- strict structured response parsing;
- exact P0–P4 prompt and privilege projections;
- byte-reconstructable per-call artifact audit;
- registered semantic terrain and appearance/capability twins;
- 14-condition provider-free offline matrix.

Offline matrix:

```text
schema  = point-hazard-offline-gate-v1
entries = 14
sha256  = 3bf69114eb74e5bba0248b26e526c6dd1a254eeb8452c5ad429087950bc15725
provider_calls_enabled = false
```

Acceptance commands for the implementation SHA:

```text
python -m pytest -q \
  tests/test_offline_matrix.py \
  tests/test_unified_harness.py \
  tests/test_condition_contract.py
34 passed in 3.19s

LAYOUT_TEST_SEEDS=25 LAYOUT_TEST_WORKERS=1 \
  python -m pytest -q tests \
  --ignore=tests/test_safety_gym_goal_integration.py
88 passed in 87.48s

python -m pytest -q tests/test_safety_gym_goal_integration.py \
  -k 'not real_safety_gym_adapter_smoke_if_installed'
5 passed, 1 deselected in 0.33s

python -m compileall -q env_pointhazard.py envs evaluation tests
git diff --check
```

Task 11 acceptance:

```text
python -m pytest -q \
  tests/test_release_manifest.py \
  tests/test_offline_matrix.py \
  tests/test_condition_contract.py
28 passed
```

## Blocking fields

Paid execution remains blocked until the model list (with provider, immutable
revision and revision date), exact 30–50-family pilot seed subset, maximum spend,
release date, and authorized operator are explicitly filled in and approved in
a later clean commit. Phase 4 technical completion and a passing reproducibility
audit do not grant that authority.
