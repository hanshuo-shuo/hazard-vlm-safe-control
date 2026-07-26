# hazard-vlm-safe-control

Research prototype for auditing where closed-loop safety comes from in modular
vision-language robot control. The marker-based PointHazard VLM-waypoint
mainline is **TERMINATED** as of 2026-07-24 after its predeclared interface
stability kill condition fired. Those five-seed artifacts are
`PILOT_ONLY / NOT PAPER RESULT`; they remain for provenance and regression.

The active direction is interface-conditioned embodied safety built on
marker-free semantic geometry:

```text
public observation
  → recognition / capability-conditioned applicability / geometry
  → fixed planner
  → enforcement
  → STC and five-stage attribution
```

A frozen five-seed, two-model, two-native-environment scout now tests whether
semantically equivalent contracts change grounding, physical action,
trajectory and STC. It is `PILOT_ONLY`, but the cross-environment action/STC
continuation gate passed: native action and trajectory IEC are both 0.55,
CISR-EQ is 0.15, and CISR-MAP is 0.30. Start with the
[current handoff](docs/INTERFACE_CONTRACT_HANDOFF.md) and the
[illustrated report](docs/INTERFACE_CONTRACT_SCOUT_REPORT.md).

## Current scope

The maintained historical offline matrix is:

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

The VLM router can execute the frozen 14-condition zero-network fixture matrix
and save per-call provenance. It is now a regression surface, not the paper
mainline. New marker scale-up is refused by default.

`evaluation/semantic_geometry.py` supplies one marker-free schema for
none/oracle/fixture/detector and future cached VLM grounding. The provider-free
runner `scripts/run_semantic_geometry_audit.py` records exact model-visible
bytes, hashes, transforms, split sentinels, resume state and manifests. The
formal paid VLM/OpenRouter release remains paused; the separately scoped
five-seed interface-contract micro-pilot is complete and does not unlock that
release. The machine-readable pilot manifest at
[`configs/pilot_release_manifest.json`](configs/pilot_release_manifest.json)
validates the protocol and portable dependency lock but remains fail-closed
until the authorization fields in
[docs/PAID_RUN_RELEASE.md](docs/PAID_RUN_RELEASE.md) are explicitly populated.

## Development boundary

The current work is discovery on one auditable PointHazard slice. The maintained
surface is the condition contract, minimal-permission policy interface, exact
prompt/image reconstruction, replay identity, capability/appearance twins, STC
accounting, layout invariants, the unified executor, and registry/manifest
discipline.

The following are frozen archive assets: protocol lexical/numeric/serialization
expansion, P4 precision tuning, PointPush as an independent line, direct VLA
scaffolds, old PIVOT/B+ performance, renderer-palette detection, unpaired paid
VLM sweeps, and adding terrain names for scale. They remain available for
forensic reproduction, but do not receive new features, results, or baseline
status. The boundary and file-level ownership are recorded in
[`STRUCTURE.md`](STRUCTURE.md) and [`legacy/README.md`](legacy/README.md).

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
| `evaluation/semantic_geometry.py` | Marker-free regions, coordinate transforms and adapters |
| `evaluation/five_stage_audit.py` | Recognition→enforcement scoring and row taxonomy |
| `evaluation/geometry_calibration.py` | Detector decomposition metrics and dev/test discipline |
| `evaluation/capability_twins.py` | Capability/appearance twin invariants |
| `evaluation/interface_execution.py` | Grounding-to-native-controller executable bridge |
| `evaluation/scout_replay.py` | Provider-free native replay, CISR and five-stage analysis |
| `evaluation/scout_authorization.py` | Exact paid-scout scope and budget enforcement |
| `scripts/run_semantic_geometry_audit.py` | Unified zero-provider runner and manifests |
| `scripts/run_interface_contract_scout.py` | Frozen 120-call scout runner |
| `scripts/replay_interface_contract_scout.py` | Zero-call native replay and analysis |
| `configs/pilot_release_manifest.json` | Machine-readable blocked pilot release state |
| `mpc_expert.py` | Empirical CEM-MPC low-level controller |
| `tests/` | Condition, harness, layout and adapter gates |
| `docs/` | Current protocol, review status registry, plans, blockers and result registry |
| `legacy/` | Frozen archive boundary for superseded implementations and experiments |
| `outputs/` | Historical artifacts retained for forensic provenance |

The review-item status map is [docs/review_status_registry.json](docs/review_status_registry.json);
the repository map is [STRUCTURE.md](STRUCTURE.md). Documentation starts at
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
The current repository is ready for cached offline calibration, not for a new
provider pilot or a paid formal experiment. A future real API key, if separately
authorized, is hard-capped at five distinct scene seeds per key.
