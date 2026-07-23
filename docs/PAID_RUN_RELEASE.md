# Paid-run release record

Status: `BLOCKED / NOT AUTHORIZED`

This record documents release readiness only. It does not authorize a provider
call, expenditure, pilot run, or promotion of fixture artifacts to scientific
results.

## Required release fields

| Field | Value |
|---|---|
| Release date | `NOT SET` |
| Implementation git SHA | `dd3d8c228494190ae3e32140d9bc67d85b20efba` |
| Protocol version | `1.2.2` |
| Model list | `NOT FROZEN` |
| Pilot seed range | allocated `200–299`; exact 30–50-family subset `NOT FROZEN` |
| Estimated maximum spend | `NOT SET` |
| Authorized operator | `NOT SET` |

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

## Blocking fields

Paid execution remains blocked until the model list, exact pilot seed subset,
maximum spend, release date, and authorized operator are explicitly filled in
and approved in a later commit. Phase 4 technical completion does not grant
that authority.
