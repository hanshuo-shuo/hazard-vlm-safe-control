# Next execution memo

> **ARCHIVED PRE-SCOUT SNAPSHOT.** This memo records the state before Experiment 0
> and the frozen 120-call scout were completed. Its commands and blockers are no
> longer current. Use
> [`../../docs/INTERFACE_CONTRACT_HANDOFF.md`](../../docs/INTERFACE_CONTRACT_HANDOFF.md)
> for execution state and
> [`../../docs/INTERFACE_CONTRACT_SCOUT_REPORT.md`](../../docs/INTERFACE_CONTRACT_SCOUT_REPORT.md)
> for results.

Date: 2026-07-26

## 2026-07-26 interface-contract update

- Completed a separately scoped five-seed interface-contract micro-pilot with
  two tasks and two renderer/environment contracts.
- Valid science arms are Gemini Flash-Lite, Qwen3-VL and the hash-matched
  Mistral replacement. GPT-5-mini is retained as an invalid request-interface
  arm and excluded from ranking.
- Field order, output mode, marker IDs, candidate order, planner mapping and one
  model-pair ranking are sensitive; executed ambiguous-mapping STC spans
  0.50–0.80.
- The ICLR mainline remains unrecovered because semantic-contract sensitivity
  failed the two-environment gate and Safety-Gym remains headless.
- Formal paid release remains blocked. See
  [INTERFACE_CONTRACT_MAINLINE.md](../../docs/INTERFACE_CONTRACT_MAINLINE.md) for the next
  native-environment gate.

## Completed

- Frozen marker-based PointHazard VLM waypoint as
  `TERMINATED / PILOT_ONLY / NOT PAPER RESULT` across docs, registry and CLI.
- Made the historical runner explicit-override and cache-only. It cannot combine
  the terminated-marker override with provider requests.
- Added the marker-free semantic geometry schema, deterministic hashes,
  provider/normalized/pixel/world transforms, letterbox handling and
  none/oracle/fixture/detector adapters.
- Added five-stage scoring and deterministic multi-label failure taxonomy.
- Added detector disk metrics, all requested decomposition arm IDs,
  single-parameter operating-curve validation and calibration/test budget gates.
- Added capability/appearance twin contracts and water, mud/rough and
  clearance/footprint protocol families.
- Added a unified provider-free runner with dry-run, resume, partial recovery,
  exact model-visible bytes/hash, row artifacts, manifest, secret audit and
  one-time test sentinel.
- Added systematic offline tests. No external provider request was made.

## Not completed

- No new scientific calibration was run; all improvement claims remain `TBD`.
- Cached detector artifacts currently contain disk-like center/radius output,
  not real segmentation masks.
- The unified runner records zero-action infrastructure smoke trajectories; a
  fixed-planner closed-loop sweep still needs to be connected to every geometry
  representation.
- Mud/rough and clearance/footprint are protocol fixtures. Clearance does not
  yet modify physical dynamics.
- No Safety-Gym provider pilot, multi-model experiment or formal test split was
  run.

## Current blockers

Science blockers:

- fresh dev scene-family split not yet frozen beyond the seed allocation;
- no calibrated detector geometry operating curve;
- no marker-free cross-model/cross-environment evidence;
- no causal estimate of perception error × enforcement.

Engineering blockers:

- detector box/polygon/mask cost-map projection must be connected to the fixed
  planner;
- real-mask artifacts are absent;
- Safety-Gym shortest-path/scripted baselines need backend-specific planning;
- a release artifact must freeze calibration trial budget and selected config.

Paid-run blockers:

- paid release remains `BLOCKED / NOT AUTHORIZED`;
- model/provider immutable revisions, operator, date and budget are unset;
- a marker-free provider protocol and exact seed/key partition are absent;
- each future real API key must be limited to at most five distinct seeds.

## Science versus infrastructure

Science evidence remains only the registered five-seed pilot and its bounded
negative/diagnostic findings. The new schema, runner, transforms, audits,
manifests, twins and tests are infrastructure. Passing them does not establish
better safety, applicability dominance, Safety-Gym replication or paper
readiness.

## Next provider-free commands

```bash
python scripts/run_semantic_geometry_audit.py \
  --environment point_hazard \
  --geometry-source fixture \
  --seeds 0:5 \
  --split dev \
  --dry-run \
  --output results/semantic_geometry_fixture_smoke

python scripts/run_semantic_geometry_audit.py \
  --environment point_hazard \
  --geometry-source detector \
  --cached-detector-results results/next_five_experiments/03_modern_perception_substitution/results.json \
  --seeds 0:5 \
  --split dev \
  --dry-run \
  --output results/semantic_geometry_detector_replay

python scripts/run_semantic_geometry_audit.py \
  --environment safety_gym \
  --geometry-source fixture \
  --capability wheeled_non_waterproof \
  --seeds 0:5 \
  --split dev \
  --max-steps 20 \
  --output results/safety_gym_geometry_smoke

LAYOUT_TEST_SEEDS=25 LAYOUT_TEST_WORKERS=1 \
  python -m pytest -q tests \
  --ignore=tests/test_safety_gym_goal_integration.py
```

Historical marker replay, if needed for regression only:

```bash
python scripts/run_next_five_experiments.py \
  --allow-terminated-marker-pilot \
  --experiments 1 2 3 4 5
```

This command is cache-only and remains `PILOT_ONLY`.

## Future paid-run prerequisites

1. connect and validate the fixed planner for all non-fabricated geometry arms;
2. complete bounded dev calibration and freeze its chosen config/hash;
3. pass capability-twin and no-peeking tests in Safety-Gym;
4. freeze the marker-free schema, prompts, models, exact pilot seeds and
   analysis;
5. partition seeds so each real key sees no more than five;
6. obtain explicit paid-release authorization in a clean commit.

## Stop criteria

Stop or redirect if marker-free effects disappear on fresh seeds, capability
twins cannot preserve invariants, detector calibration only shifts
conservatism without useful STC, the finding remains PointHazard-only, or
artifact audit cannot prove evaluator truth stayed outside the policy.

## Verification completed

```text
python -m compileall -q env_pointhazard.py envs evaluation scripts tests
PASS

LAYOUT_TEST_SEEDS=25 LAYOUT_TEST_WORKERS=1 \
  python -m pytest -q tests \
  --ignore=tests/test_safety_gym_goal_integration.py
132 passed in 91.00s

python -m pytest -q tests/test_safety_gym_goal_integration.py \
  -k 'not real_safety_gym_adapter_smoke_if_installed'
5 passed, 1 deselected in 0.31s
```

The full Safety-Gym file was also attempted: five tests passed before the
macOS `rgb_array` GUI smoke blocked; that process was terminated and is not
counted as a pass. Three five-seed, zero-provider integration manifests were
generated under `results/semantic_geometry_fixture_smoke/`,
`results/semantic_geometry_detector_replay/` and
  `results/safety_gym_geometry_smoke/`; each reports 5 rows, 0 failed rows,
0 missing artifacts and 0 provider calls. A fourth
`results/semantic_geometry_detector_fixed_planner_smoke/` manifest exercises
the cached detector disk through the fixed MPC for a deliberately partial
20-step infrastructure horizon; it is not a calibrated performance result.
