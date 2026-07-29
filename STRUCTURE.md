# Project Structure and Archive Policy

Updated: 2026-07-28.

The active research line is **interface-conditioned embodied safety**. The old
marker-waypoint line is terminated and kept only for provenance and regression.

## Start here

```text
RESEARCH_STORY.md        plain-English advisor brief
README.md                project entry point
docs/README.md           documentation map
docs/RESULTS_REGISTRY.md result-validity authority
```

## Active research code

| Area | Main paths | Role |
|---|---|---|
| Native environments | `env_pointhazard.py`, `envs/` | Public observation boundary and native adapters |
| Interface experiment | `evaluation/interface_contracts.py`, `evaluation/interface_scenarios.py` | Equivalent contracts and paired scenarios |
| Execution | `evaluation/interface_execution.py`, `mpc_expert.py` | Grounding conversion and fixed low-level control |
| Evaluation | `evaluation/interface_metrics.py`, `evaluation/scout_replay.py`, `evaluation/five_stage_audit.py` | IEC, CISR, native replay, and failure attribution |
| Release safety | `evaluation/scout_authorization.py`, `evaluation/paid_provider_gateway.py`, `configs/` | Fail-closed provider scope and budget checks |
| Reproducibility | `scripts/`, `tests/`, `results/` | Builders, tests, and checked-in evidence |

New scientific work should normally go into `envs/`, `evaluation/`, `scripts/`,
`tests/`, and `docs/`.

## Documentation layers

### 1. Current reading

| Document | Purpose |
|---|---|
| `RESEARCH_STORY.md` | Simple research story, figures, claim boundary, and next steps |
| `docs/INTERFACE_CONTRACT_PROTOCOL.md` | Frozen current experiment semantics |
| `docs/RESULTS_REGISTRY.md` | Whether an artifact may be cited |
| `docs/PAID_RUN_RELEASE.md` | Blocked general paid-run record |

### 2. Provenance-locked documents

The following files look like old planning material, but checked-in manifests
or tests hash or parse their exact paths. Moving or rewriting them would break
the historical evidence chain:

- `docs/ICLR_PLAN.md`;
- `docs/INTERFACE_CONTRACT_MAINLINE.md`;
- `docs/PROTOCOL.md`;
- `docs/RESEARCH_REVIEW_COMMENTS.md`;
- `docs/review_status_registry.json`.

They remain under `docs/` for reproducibility, not because they are the best
entry point for a reader.

### 3. Historical narratives

Superseded stories, completed handoffs, detailed internal reports, and old
implementation plans live under `legacy/docs/`. The July 2026 advisor rewrite
archive is in `legacy/docs/advisor_rewrite_2026-07/`.

## Frozen compatibility code

Some old top-level modules remain in place because historical scripts and tests
import them directly:

- `pivot_vlm.py`, `subgoal_pivot_hazard.py`, `zone_detector.py`;
- `safe_expert.py` and renderer compatibility modules;
- PointPush and direct-VLA scaffolds.

These files do not receive new research claims or features. Moving them without
updating all historical import paths would reduce reproducibility, so they stay
at their registered paths.

## Artifact policy

- `results/` contains current structured experiment evidence.
- `outputs/invalidated/` contains explicitly invalidated runs.
- older files directly under `outputs/` are historical and remain in place
  because `docs/RESULTS_REGISTRY.md` records those paths.
- `legacy/` contains superseded source, narratives, debug material, and assets.

Do not delete or rename an evidence file without updating the result registry
and checking every manifest that records its path or hash.

## Current gate

The 120-call scout is complete and cache-replayable. Its native action and
trajectory IEC are both 0.55, and its equivalent-contract CISR is 0.15. These
numbers justify a carefully scoped second scenario family, but they do not
authorize new paid calls or promote the pilot to a paper result.
