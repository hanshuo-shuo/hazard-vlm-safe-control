# Project Structure and Archive Policy

Updated: 2026-09-12.

仓库的 active research line 是 **C³-Safe**；旧 marker、interface-contract、PIVOT、
PointPush 和 direct-VLA 路线停止科学扩展，但保留原路径与 provenance。

## Start here

```text
README.md                              project entry point
RESEARCH_STORY.md                      current C³-Safe storyline
docs/C3_SAFE_MAINLINE.md               frozen mainline protocol
docs/C3_FOUNDATION_PROGRESS.md         implemented work, development evidence and next steps
docs/C3_FOUNDATION_RESULTS_2026-09-09.json  compact numerical evidence and run hashes
docs/C3_QUEST_PROGRESS_2026-09-12.md    Quest resources, real teacher checks and other-branch discovery
configs/c3_safe_mainline_manifest.json machine-readable planned scope
docs/RESULTS_REGISTRY.md               result-validity authority
docs/REPOSITORY_STATUS_2026-08-13.md   keep/freeze/reuse index
```

## Active C³-Safe foundation

| Area | Main paths | Role |
|---|---|---|
| Spatial/motion truth | `c3_safe/geometry.py`, `costs.py` | Calibrated fields, swept disks, q95 severity, binary contact, separate capability/rule costs |
| C3 Point adapter | `c3_safe/point_adapter.py` | Explicit typed regions, native dynamics and independent swept semantic truth |
| Native substeps | `c3_safe/native_motion.py` | Record selected MuJoCo body motion without changing native step results |
| Data and provenance | `c3_safe/data.py`, `artifacts.py` | Frozen split membership, same-transition twins, action-free requests and run-start source snapshots |
| Development expert | `c3_safe/oracle_control.py` | Shared A* + MPC router for blind and privileged oracle arms |
| Spatial student | `c3_safe/spatial_student.py`, `scripts/train_c3_spatial_baseline.py` | Optional PyTorch, RGB-only CNN, simulator-mask baseline and exposure-loss diagnostic |
| Real teacher I/O | `c3_safe/teacher_io.py`, `scripts/*c3_teacher*.py`, `setup/c3_teacher_*.sbatch` | RGB-only bundles, pinned official Qwen download, real GPU calls, immutable reparsing and coverage-aware evaluation |
| PointHazard | `env_pointhazard.py`, `hazard_renderer.py` | 2-D dynamics, rendering and reliable layout source |
| Interface boundary | `envs/protocol_env.py`, `envs/point_hazard_adapter.py` | Keep policy inputs separate from evaluator truth |
| Safety-Gym | `envs/safety_gym_goal_adapter.py` | Second task; semantic variant must pass geometry/step gates |
| Capability twins | `evaluation/capability_twins.py` | Same-scene/different-capability counterfactual contract |
| Semantic evaluation | `evaluation/semantic_evaluator.py`, `evaluation/semantic_geometry.py` | Versioned swept semantic evaluation and retained historical center/geometry protocols |
| Calibration | `evaluation/geometry_calibration.py`, `evaluation/native_environment_gate.py` | Coordinate projection, footprint rasterization and native vertical-slice checks |
| Evaluation | `evaluation/outcomes.py`, `evaluation/schemas.py` | STC, native cost, semantic violation and provenance |
| Data coverage | `safe_expert.py`, `mpc_expert.py` | Safe/random/perturbed transition collection and reference baseline |
| Planned mainline | `docs/C3_SAFE_MAINLINE.md`, `configs/c3_safe_mainline_manifest.json` | Spatial labels, motion exposure, separated relation costs/critics and SAC-Lagrange scope |

The spatial baseline trainer has run on a small development dataset. Teacher distillation,
learned counterfactual relations, independent cost critics and a SAC policy remain pending in
this branch. The discovered Quest branch has additional relation models and visual development
results; their separate scope is recorded in `docs/C3_QUEST_PROGRESS_2026-09-12.md`.
The complete formal Gate 0–2 requirements are unchanged.

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
| `docs/C3_FOUNDATION_PROGRESS.md` | Current implementation, exact development results, commands and remaining resources |
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
- new `results/c3_*/` raw datasets, checkpoints and source copies are local and ignored;
  checked-in numerical summaries and figures are in `docs/C3_FOUNDATION_RESULTS_2026-09-09.json`
  and `docs/assets/c3_foundation/`.

## Current gate

Geometry, native-step, split and oracle-coverage checks have development evidence. Full
Gate 0 still needs complete multi-terrain data and real teacher provenance; Gate 1 needs
teacher/student quality and a matched non-VLM comparison; Gate 2 needs an actual learned
policy and multi-seed joint-OOD evaluation. A run marked `COMPLETE` is not a VALIDATED result.
