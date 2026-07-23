# Project structure and status

Last reviewed: 2026-07-23.

The repository keeps top-level experiment modules in place because historical
scripts import them directly. New protocol-facing work belongs in `envs/`,
`evaluation/`, `tests/` and `docs/`; legacy experiments should not gain new
dependencies.

## Active core

| Path | Status | Purpose |
|---|---|---|
| `env_pointhazard.py` | `ACTIVE` | PointHazard dynamics and B01-validated semantic-zone sampler |
| `envs/protocol_env.py` | `ACTIVE` | Public environment boundary and evaluator-only context |
| `envs/point_hazard_adapter.py` | `ACTIVE` | PointHazard protocol adapter |
| `evaluation/conditions.py` | `ACTIVE` | Frozen condition and enforcement identity |
| `evaluation/policy_interface.py` | `ACTIVE` | Immutable task/capability/policy payload and permission audit |
| `evaluation/outcomes.py` | `ACTIVE` | Headline STC component reduction and truth table |
| `evaluation/vlm_artifacts.py` | `ACTIVE` | Structured recognition/action parse and audited per-call bytes |
| `evaluation/vlm_router.py` | `ACTIVE` | Offline P0–P4 candidates, annotated PNG, versioned prompt and decision audit |
| `evaluation/harness.py` | `ACTIVE` | Direct/replay × none/oracle vertical slice |
| `evaluation/schemas.py` | `ACTIVE` | Reproducible episode artifacts |
| `evaluation/release_manifest.py` | `ACTIVE` | Fail-closed pilot release and dependency-lock audit |
| `configs/pilot_release_manifest.json` | `ACTIVE/BLOCKED` | Machine-readable release state; provider calls disabled |
| `evaluation/semantic_evaluator.py` | `ACTIVE` | Evaluator-only semantic accounting |
| `mpc_expert.py` | `ACTIVE` | Shared empirical CEM-MPC executor |
| `tests/` | `ACTIVE` | Offline acceptance and regression gates |

## Optional infrastructure

| Path | Status | Boundary |
|---|---|---|
| `envs/safety_gym_goal_adapter.py` | `INFRA/BLOCKED` | Native adapter is reusable; semantic expansion is not a current result |
| `env_pointpushhazard.py`, `pointpush_*` | `INFRA` | Contact-task scaffold, outside the current PointHazard gate |
| `safe_expert.py` | `BASELINE` | Historical A* + PD controller |
| `zone_detector.py` | `BASELINE` | Renderer-specific color sanity baseline |
| `pivot_vlm.py` | `INFRA` | Candidate annotation and VLM parsing utilities |
| `direct_vla_*.py` | `BASELINE` | End-to-end VLA scaffolds |

## Historical paths

`subgoal_pivot_hazard.py`, the PointPush learned-physics script, `scripts/make_*`
and most committed `outputs/` reproduce earlier experiments. They are retained
for forensic use and are not the current harness. `legacy/` contains older
implementations that should remain isolated.

Invalidated artifacts live under `outputs/invalidated/`; do not delete or move
them without updating `docs/RESULTS_REGISTRY.md`.

## Documentation truth sources

| Document | Role |
|---|---|
| `docs/PROTOCOL.md` | Frozen experiment semantics |
| `docs/ICLR_PLAN.md` | Research strategy and go/no-go gates |
| `docs/CLAUDE_PLAN.md` | Current implementation sequence and acceptance evidence |
| `docs/RESEARCH_REVIEW_COMMENTS.md` | Open review blockers |
| `docs/RESULTS_REGISTRY.md` | Artifact validity status |

Superseded roadmaps, old result narratives and implementation prompts were
removed during the 2026-07-23 cleanup. Their raw artifacts and invalidation
records remain available through `outputs/` and the registry.

## Current gate

Tasks 1–11 are complete: B01 layout validation, unified condition contract,
direct/replay vertical slice, minimal-permission policy boundary, STC reduction,
the offline structured-output/per-call artifact gate, and the P0–P4 local
fixture request adapter. Registered semantic terrain, the P0–P4 offline VLM
episode harness and the frozen 14-condition offline matrix are also complete.
Detector integration and paid runs remain gated by `docs/CLAUDE_PLAN.md`; the
blocked authorization record is `docs/PAID_RUN_RELEASE.md`; its machine-readable
counterpart is `configs/pilot_release_manifest.json`.
