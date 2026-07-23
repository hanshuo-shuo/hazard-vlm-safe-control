# hazard-vlm-safe-control

Research prototype for auditing where closed-loop safety comes from in modular
VLM-guided robot control. The current validated development slice is
PointHazard with a unified condition contract and a shared direct/replay
harness. Historical VLM experiments remain in the repository for provenance,
but their reported semantic results are not paper-valid.

## Current scope

The active offline matrix is:

```text
router      = direct | replay
zone_source = none   | oracle
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

Detector and VLM zone sources are intentionally not connected yet. Paid
VLM/OpenRouter runs remain paused until the offline gates in
[docs/CLAUDE_PLAN.md](docs/CLAUDE_PLAN.md) are complete.

## Repository map

| Path | Role |
|---|---|
| `env_pointhazard.py` | Frozen PointHazard dynamics and checked layout sampler |
| `envs/` | Permission-bounded environment adapters |
| `evaluation/conditions.py` | Canonical condition, enforcement and factor-vector contract |
| `evaluation/harness.py` | Unified direct/replay runner and replay audit |
| `evaluation/schemas.py` | Episode artifact and provenance schema |
| `evaluation/semantic_evaluator.py` | Evaluator-only semantic metrics |
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
  tests/test_condition_contract.py \
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
