# Project structure and status

Last reviewed: 2026-07-26.

Research mainline status: marker-based PointHazard VLM waypoint is
`TERMINATED / PILOT_ONLY`. The active implementation surface is marker-free
semantic geometry plus Safety-Gym capability twins.

The repository keeps a small number of historical top-level modules in place
because old scripts and factor snapshots import them directly. New
protocol-facing work belongs in `envs/`, `evaluation/`, `tests/` and `docs/`.
Anything marked `FROZEN` or `ARCHIVED` is retained for provenance only and must
not gain new dependencies, claims, or experiment results.

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
| `evaluation/semantic_geometry.py` | `ACTIVE` | Shared marker-free geometry schema, transforms and adapters |
| `evaluation/five_stage_audit.py` | `ACTIVE` | Five-stage scoring and multi-label failure taxonomy |
| `evaluation/geometry_calibration.py` | `ACTIVE` | Detector decomposition metrics and calibration gates |
| `evaluation/capability_twins.py` | `ACTIVE` | Capability/appearance twin contracts |
| `evaluation/interface_contracts.py` | `ACTIVE/EXPLORATORY` | Equivalent-contract normalization, planner mappings and ranking-envelope analysis |
| `scripts/run_semantic_geometry_audit.py` | `ACTIVE` | Unified no-provider audit runner, resume and manifests |
| `scripts/run_interface_contract_audit.py` | `PILOT_ONLY` | Frozen five-seed, two-task, two-environment interface audit |
| `scripts/run_interface_contract_replacement.py` | `PILOT_ONLY` | Hash-matched replacement for the invalid GPT request arm |
| `scripts/analyze_interface_contract_results.py` | `ACTIVE/ANALYSIS` | Zero-provider combined analysis and paper gate |
| `scripts/run_interface_execution_bridge.py` | `PILOT_ONLY/EXECUTION` | Cached decisions replayed through fixed MPC and headless dynamics |
| `scripts/run_next_five_experiments.py` | `TERMINATED/REGRESSION` | Cached historical replay only; explicit override required |
| `mpc_expert.py` | `ACTIVE` | Shared empirical CEM-MPC executor |
| `tests/` | `ACTIVE` | Offline acceptance and regression gates |

## Optional infrastructure

| Path | Status | Boundary |
|---|---|---|
| `envs/safety_gym_goal_adapter.py` | `ACTIVE/INFRA` | Headless capability-twin harness; no Safety-Gym science result yet |
| `env_pointpushhazard.py`, `pointpush_*` | `FROZEN` | Contact-task scaffold; no independent PointPush science line |
| `safe_expert.py` | `FROZEN` | Historical A* + PD compatibility controller; not a current claim |
| `zone_detector.py` | `ARCHIVED` | Renderer-palette sanity plumbing only; not a perception baseline |
| `pivot_vlm.py`, `subgoal_pivot_hazard.py` | `FROZEN` | Historical PIVOT/B+ compatibility path and factor snapshot support |
| `direct_vla_*.py` | `ARCHIVED` | End-to-end VLA scaffolds; no training or performance work in this phase |

## Frozen and historical paths

`subgoal_pivot_hazard.py`, the PointPush learned-physics script, direct-VLA
scripts, `scripts/make_*`, the renderer-palette detector and most committed
`outputs/` reproduce earlier experiments. They are retained for forensic use and
are not the current harness. `legacy/` contains older implementations that
should remain isolated; its [README](legacy/README.md) is the archive boundary.

The following work is explicitly out of scope for the current phase:

- extending protocol lexical, numeric, P4 precision, or serialization details;
- optimizing old PIVOT/B+ performance or adding unpaired paid VLM sweeps;
- turning `zone_detector.py` into a perception baseline;
- expanding PointPush or direct VLA beyond frozen scaffolds;
- adding terrain names without a matched scientific control.

The active discovery surface is deliberately smaller: exact prompt/image
reconstruction, replay identity, capability and appearance twins, STC
components, layout invariant tests, the unified executor, and registry/run
manifest discipline.

Invalidated artifacts live under `outputs/invalidated/`; do not delete or move
them without updating `docs/RESULTS_REGISTRY.md`.

## Documentation truth sources

| Document | Role |
|---|---|
| `docs/PROTOCOL.md` | Frozen experiment semantics |
| `docs/ICLR_PLAN.md` | Research strategy and go/no-go gates |
| `docs/CLAUDE_PLAN.md` | Current implementation sequence and acceptance evidence |
| `docs/review_status_registry.json` | Machine-readable review-item status authority |
| `docs/RESEARCH_REVIEW_COMMENTS.md` | Review blocker narrative and rendered status snapshot |
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
Cached detector geometry is now accepted by the shared contract, while
segmentation masks and calibrated fixed-planner operating curves remain
unrun/TBD. Paid runs remain gated by the review registry and
`docs/CLAUDE_PLAN.md`; the
blocked authorization record is `docs/PAID_RUN_RELEASE.md`; its machine-readable
counterpart is `configs/pilot_release_manifest.json`.
