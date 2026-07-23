# hazard-vlm-safe-control

Research prototype for auditing where closed-loop safety comes from in modular
VLM-guided robot control. The current validated development slice is
PointHazard with a unified condition contract, registered semantic terrain and
a shared direct/replay/offline-VLM harness. Historical VLM experiments remain
in the repository for provenance, but their reported semantic results are not
paper-valid.

## Current scope

The active offline matrix is:

```text
router      = direct | replay
zone_source = none   | oracle

offline development gate:
router          = vlm
zone_source     = none
privilege_level = P0 | P1 | P2 | P3 | P4
provider        = local structured fixture only
```

All four conditions use the same PointHazard environment, executor and
enforcement boundary:

```text
plan_to(public_observation, absolute_target, cost_map_payload)
```

Replay uses absolute world-coordinate targets, arrival-based switching at
`0.600`, and records target identity, planner events, STC components and
trajectory-divergence diagnostics. Oracle geometry enters only through an
explicit `PRIVILEGED` cost-map payload built on the evaluator side.

The VLM router can now execute the frozen 14-condition zero-network fixture
matrix and save per-call provenance; no real provider client is connected.
Detector and VLM zone sources are intentionally not connected yet. Paid
VLM/OpenRouter runs remain paused. The machine-readable pilot manifest at
[`configs/pilot_release_manifest.json`](configs/pilot_release_manifest.json)
validates the protocol and portable dependency lock but remains fail-closed
until the authorization fields in
[docs/PAID_RUN_RELEASE.md](docs/PAID_RUN_RELEASE.md) are explicitly populated.

## Repository map

| Path | Role |
|---|---|
| `env_pointhazard.py` | Frozen PointHazard dynamics and checked layout sampler |
| `envs/` | Permission-bounded environment adapters |
| `evaluation/conditions.py` | Canonical condition, enforcement and factor-vector contract |
| `evaluation/policy_interface.py` | Immutable minimal-permission routing-policy boundary |
| `evaluation/outcomes.py` | Shared Safe Task Completion reduction |
| `evaluation/vlm_artifacts.py` | Strict structured output and byte-reconstructable call provenance |
| `evaluation/vlm_router.py` | Offline P0–P4 candidate, prompt and structured fixture adapter |
| `evaluation/harness.py` | Unified direct/replay/offline-VLM runner and replay audit |
| `evaluation/schemas.py` | Episode artifact and provenance schema |
| `evaluation/release_manifest.py` | Fail-closed pilot release and lock validator |
| `evaluation/semantic_evaluator.py` | Evaluator-only semantic metrics |
| `configs/pilot_release_manifest.json` | Machine-readable blocked pilot release state |
| `mpc_expert.py` | Empirical CEM-MPC low-level controller |
| `tests/` | Condition, harness, layout and adapter gates |
| `docs/` | Current protocol, plans, review blockers and result registry |
| `legacy/` | Archived pre-accounting implementations |
| `outputs/` | Historical artifacts retained for forensic provenance |

The detailed status map is [STRUCTURE.md](STRUCTURE.md). Documentation starts at
[docs/README.md](docs/README.md).

## Setup and verification

Core PointHazard tests require NumPy and pytest. Optional Safety-Gym integration
uses the dependencies in `requirements-safety-gym.txt`.

```bash
python -m pytest -q \
  tests/test_release_manifest.py \
  tests/test_condition_contract.py \
  tests/test_policy_permissions.py \
  tests/test_stc_outcomes.py \
  tests/test_vlm_artifacts.py \
  tests/test_vlm_router.py \
  tests/test_unified_harness.py

LAYOUT_TEST_SEEDS=25 LAYOUT_TEST_WORKERS=1 \
  python -m pytest -q tests \
  --ignore=tests/test_safety_gym_goal_integration.py
```

The layout suite defaults to the formal 10,000-seed sweep. Use the shortened
environment variables only for local smoke testing.

## Claim boundary

This repository supports empirical rollout accounting: success, native cost,
semantic violation, clearance and safe task completion. It does not establish
formal safety, recursive feasibility, human-intent alignment, or superiority
over classical planning.

No semantic result currently has `VALIDATED` status. Check
[docs/RESULTS_REGISTRY.md](docs/RESULTS_REGISTRY.md) before citing any artifact.
