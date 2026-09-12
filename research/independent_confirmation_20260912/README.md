# Independent composition confirmation, 2026-09-12

This prospectively frozen follow-up implements the first two next steps in
`docs/C3_PAPER_PROGRESS_2026-09-12.md`: remove generator role/location shortcuts,
then compare fixed recipes with five new model seeds and explicit calibration.
It is a component study, not approval of a formal C3-Safe gate.

Completed results: [paper addendum](reports/PAPER_ADDENDUM.md),
[summary](reports/SUMMARY.json), [independent verification](reports/VERIFICATION.json),
and [scientific interpretation](INTERPRETATION.md). All 15 GPU tasks and the
cached verification/report job completed successfully. The primary joint
comparison passes; calibrated acceptance at the fixed 0.02 threshold is zero.

All dataset generation, tests, training, prediction, statistical analysis,
checkpoint replay and figure rendering execute on Quest in Slurm allocations.
Local files are source, retrieved summaries, and presentation artifacts.

## Frozen design

- Original, left/right-balanced, and independent-geometry training recipes.
- Five paired model seeds; 800 training and 100 validation images per recipe.
- The captured ResNet-18 recipe, optimizer, augmentation and 24-epoch ceiling
  remain unchanged. Early stopping may produce different actual step counts.
- Random geometry and anonymous paths use separate random streams. Paths are
  not accepted/rejected based on risk, preferred roles or target optima.
- Two property-swap variants per scene, preserving footprints and background.
- Three random tests of 600 pairs each; 200 legacy-regression pairs.
- A separate 300-pair calibration set. The exported checkpoint and calibration
  are persisted before the runner opens any test array.
- Main metrics exclude the identically zero B card. No numerical pooling with
  the earlier all-four-card pilot or three-channel/q95 mainline.
- One predeclared joint primary claim: balanced vs original on new appearance,
  requiring positive regret reduction and a false-safe noninferiority guard.
  The independent-geometry extension and other contrasts are secondary.

`protocol.json` defines exact seeds, families, scores, thresholds, failure
conditions and uncertainty calculations. `FREEZE.json`, retrieved after the run,
records the source identities and Quest timestamp before data generation.

## Runtime artifacts

Quest root:

```text
/projects/p33100/siosio/hazard_independent_confirmation_20260912/
    FREEZE.json
    JOB_IDS.txt
    REPORT_DEPENDENCIES.json
    research/                   frozen source copy
    datasets/                   RGB, masks, footprints, truth, geometry, hashes
    runs/{legacy,balanced,independent}/seed_{0,1,2,3,4}/
    reports/                    summary, independent verification, addendum, plots
    logs/                       all attempts, including pre-freeze failure
    report_deps/                isolated plotting packages
```

Training reuses `/home/shv7753/envs/hazard-exp01b-r2/bin/python`. Plot dependencies
are installed under this study's `report_deps/`; the shared training and Qwen
environments are not modified. The initial preflight exposed a negative-stride
NumPy/PyTorch compatibility issue; it was fixed before the prospective freeze.
All 13 final preflight tests pass on Quest, including tie/eligibility accounting,
calibration order statistics, factor streams and bootstrap pairing.

Recorded jobs: preflight `6176243` (failed before any study data), `6176335`
(11 tests passed), `6176471` (13 final tests passed); plotting setup `6176517`;
dataset `6176529`; GPU array `6176550`; aggregate/verification/report `6176551`.
Completion and outcome must be read from the retrieved result/accounting records.

Post-freeze verification amendment: the original verifier repeatedly decompressed
the dataset NPZ within scalar coverage loops. Job 6176551 completed aggregation
and the first checkpoint replay, then was deliberately stopped to avoid a long
CPU run. `verify_cached.py` eagerly reads the same arrays before executing the
same metric formulas and tolerances. The frozen `verify.py`, all model outputs,
and the already computed SUMMARY.json remain unchanged. The amendment timestamp,
old/new verifier hashes and summary hash are retained in
`VERIFIER_IO_AMENDMENT.txt`; the new verifier also records both source hashes in
VERIFICATION.json. `resume_report.sbatch` runs this verification and the original
frozen report builder without repeating aggregation or training.
The resumed job is `6177655`, completed with exit 0:0. All 15 checkpoints pass
the four-image CPU replay; maximum probability difference is 0.00259561 and
every replayed binary field agrees with the stored GPU result.

## Reproduction

The supplied sbatch files identify an already used root. Do not resubmit them
unchanged. For a new run, create a fresh remote root and copy the source tree,
then use the Python entry points with `--root` inside suitable Slurm allocations:

```bash
python research/independent_confirmation_20260912/test_study.py
python research/independent_confirmation_20260912/freeze.py --root NEW_ROOT
python research/independent_confirmation_20260912/prepare.py --root NEW_ROOT
python research/independent_confirmation_20260912/run.py --root NEW_ROOT --task-index 0
# Complete task indices 0-14 before analysis.
python research/independent_confirmation_20260912/aggregate.py --root NEW_ROOT
python research/independent_confirmation_20260912/verify_cached.py --root NEW_ROOT
PYTHONPATH=NEW_ROOT/report_deps python research/independent_confirmation_20260912/report.py --root NEW_ROOT
```

Task indices 0-4 are original, 5-9 are balanced, and 10-14 are independent.
The freeze assumes code is deployed as `NEW_ROOT/research/...`, preserving the
reference import paths. Existing data/run/report directories are not overwritten.

The checkpoint compatibility check uses four fixed images per model on a Quest
CPU. Its predeclared tolerance is maximum probability difference <=0.01 and
binary agreement >=0.995. It does not replace the archived GPU predictions.
All decision/cost/coverage metrics are independently reconstructed from arrays.

## Scientific boundary

The simulator provides two explicit synthetic properties and a public union-cost
formula. The property permutation is not a new counterfactual learning loss.
Shape/appearance shift here does not establish capability-combination holdout,
perspective or occlusion robustness, VLM supervision, or closed-loop safety.

Calibration uses a pair-block maximum residual and a fixed split-conformal order
statistic. Only the IID random split shares the calibration distribution. Zero
acceptance is recorded as zero coverage and undefined conditional risk. The
general calibration framework is described by [Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6).
The block-score construction and crossed seed/scene bootstrap are explicit study
choices, not claims of a novel conformal method or an exact five-seed guarantee.
