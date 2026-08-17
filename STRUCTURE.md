# Project Structure and Archive Policy

Updated: 2026-08-13.

仓库的 active research line 是 **C³-Safe**；旧 marker、interface-contract、PIVOT、
PointPush 和 direct-VLA 路线停止科学扩展，但保留原路径与 provenance。

## Start here

```text
README.md                              project entry point
RESEARCH_STORY.md                      current C³-Safe storyline
docs/C3_SAFE_MAINLINE.md               frozen mainline protocol
configs/c3_safe_mainline_manifest.json machine-readable planned scope
docs/RESULTS_REGISTRY.md               result-validity authority
docs/REPOSITORY_STATUS_2026-08-13.md   keep/freeze/reuse index
```

## Active C³-Safe foundation

| Area | Main paths | Role |
|---|---|---|
| PointHazard | `env_pointhazard.py`, `hazard_renderer.py` | 2-D dynamics, rendering and reliable layout source |
| Interface boundary | `envs/protocol_env.py`, `envs/point_hazard_adapter.py` | Keep policy inputs separate from evaluator truth |
| Safety-Gym | `envs/safety_gym_goal_adapter.py` | Second task; semantic variant must pass geometry/step gates |
| Capability twins | `evaluation/capability_twins.py` | Same-scene/different-capability counterfactual contract |
| Semantic truth | `evaluation/semantic_evaluator.py`, `evaluation/semantic_geometry.py` | Evaluator-only spatial fields, swept-footprint exposure and relation-cost truth |
| Calibration | `evaluation/geometry_calibration.py`, `evaluation/native_environment_gate.py` | Coordinate projection, footprint rasterization and native vertical-slice checks |
| Evaluation | `evaluation/outcomes.py`, `evaluation/schemas.py` | STC, native cost, semantic violation and provenance |
| Data coverage | `safe_expert.py`, `mpc_expert.py` | Safe/random/perturbed transition collection and reference baseline |
| Planned mainline | `docs/C3_SAFE_MAINLINE.md`, `configs/c3_safe_mainline_manifest.json` | Spatial labels, motion exposure, separated relation costs/critics and SAC-Lagrange scope |

No C³-Safe training implementation is registered yet. New code should follow the protocol and
land only after Gate 0 has a reproducible manifest.

## Frozen storyline support

The following code and artifacts remain available for motivation, failure analysis and regression:

- `evaluation/interface_contracts.py`, `interface_scenarios.py`, `interface_execution.py`,
  `interface_metrics.py`, `five_stage_audit.py`, `scout_replay.py`;
- `scripts/*interface_contract*`, corresponding `tests/`, `configs/interface_contract_*` and
  `results/interface_contract_*`;
- `pivot_vlm.py`, `subgoal_pivot_hazard.py`, old marker/waypoint helpers;
- `direct_vla_*.py`, PointPush files, learned-physics and diffusion scaffolds;
- historical images in `docs/assets/`, `outputs/` and archived narratives under `legacy/`.

They are not current baselines, do not receive new claims, and must not be silently folded into
C³-Safe results.

## Documentation layers

### Current reading

| Document | Purpose |
|---|---|
| `README.md` | Entry point and current boundary |
| `RESEARCH_STORY.md` | Advisor-facing story and retained pilot evidence |
| `docs/C3_SAFE_MAINLINE.md` | Mainline algorithm/protocol/gates |
| `docs/RESULTS_REGISTRY.md` | Artifact validity and allowed claims |
| `docs/REPOSITORY_STATUS_2026-08-13.md` | Physical keep/freeze/reuse map |

### Provenance-locked historical documents

`docs/ICLR_PLAN.md`, `docs/INTERFACE_CONTRACT_MAINLINE.md`, `docs/PROTOCOL.md`,
`docs/RESEARCH_REVIEW_COMMENTS.md` and `docs/review_status_registry.json` remain at their
original paths because manifests/tests may hash or parse them. They describe superseded plans;
the current decision is recorded in `docs/C3_SAFE_MAINLINE.md`.

### Historical narratives

Detailed interface-contract reports and prior research-story evolution remain under
`legacy/docs/` and the registered `results/` paths. The short current story should not be
reconstructed by reading those files first.

## Artifact policy

- `docs/RESULTS_REGISTRY.md` is the only result-state authority.
- `results/` retains machine-readable experiment artifacts at their registered paths.
- `outputs/invalidated/` retains explicitly invalidated runs.
- old files directly under `outputs/` remain because registry/provenance references them.
- no evidence file is deleted or renamed without updating every manifest and hash reference.

## Current gate

C³-Safe is planned but unrun. The next gate is infrastructure-only: disjoint splits, semantic
geometry calibration, native adapters, expert coverage, spatial teacher-label schema,
swept-footprint exposure truth and complete provenance. Passing a bridge or a manifest `COMPLETE`
status does not create a VALIDATED result.
