# Composition audit and adaptive repair, 2026-09-12

This directory is a new experiment around a byte-preserved deployed Quest snapshot.
It does not edit or extend the protected historical autoresearch loop in place.
Evidence class: `PILOT_ONLY / DEVELOPMENT_DIAGNOSTIC`.

- `protocol.json`: frozen first-stage audit, three new model seeds.
- `balanced_repair/protocol.json`: repair declared after the first audit seed's failure.
- `run_audit.py`: isolated reference imports, paired renderer intervention, fixed controls and checkpoint export.
- `run_balanced.py`: exactly half of training/validation images receive a property swap; same recipe and image counts.
- `aggregate.py`: hash validation, paired scene bootstrap conditional on three fixed model seeds.
- `verify_predictions.py`: independently reconstructs costs/metrics from stored fields and footprints; CPU-replays six models.
- `compare_history.py`: verifies saved historical array hashes and runs the unmodified historical comparator.
- `SUMMARY.json`, `PREDICTION_VERIFICATION.json`, `HISTORICAL_COMPARISONS.json`: checked-in findings.
- `SLURM_ACCOUNTING.txt`: six completed jobs, arrays 6161267 and 6161459.

The small frozen relation checkpoints are tracked explicitly (about 36 KB combined).
The six new visual checkpoints and prediction arrays are retained outside Git:

```
Local: /Users/hanshuo/Desktop/hazard/results/c3_composition_audit_20260912/
Quest: /projects/p33100/siosio/hazard_composition_audit_20260912/
```

Each `runs/seed_{0,1,2}` and `repair_runs/seed_{0,1,2}` contains a visual model, raw predictions,
target fields, candidate footprints, permutations, scene metrics, training history, dataset fingerprints
and hash manifest. This is 1,200 unique RGB images / 600 scene pairs, evaluated by six models;
7,200 model-image evaluations are not 7,200 independent scenes. Each pair uses identical footprints.
The learned/analytic regret comparison includes A/B/C/D. The changed-optimum paired metric uses
the eligible cards A and D; card B has zero target cost, and card C's union cost is invariant to the swap.

## Replay

The original execution environment is `/home/shv7753/envs/hazard-exp01b-r2/bin/python` on Quest:
Python 3.10.21, Torch 2.5.1+cu124, torchvision 0.20.1, NumPy 1.23.5, Pillow 9.4.0.
GPU type varies between A100 40 GB and 80 GB; no runtime comparison is claimed.

From the repository root, with NumPy, Torch and torchvision available:

```bash
python3 research/composition_audit_20260912/run_audit.py --self-test
python3 research/composition_audit_20260912/aggregate.py \
  --root results/c3_composition_audit_20260912 \
  --output research/composition_audit_20260912/SUMMARY.json
python3 research/composition_audit_20260912/verify_predictions.py
python3 research/composition_audit_20260912/compare_history.py
```

`compare_history.py` needs the recovered old result directories at
`results/c3_quest_teacher_20260911/old_quest_runs/`. Their remote source is
`/home/shv7753/hazard-autoresearch-exp01b-r2/autoresearch/exp01b_r2/runs/`.
Every one of the 17 recovered `PER_SCENE_METRICS.npz` hashes is checked against its own RESULTS.json.
Their findings are not pooled with the fresh audit.

For a new execution, use a **new output path**; the runner refuses to overwrite existing output:

```bash
python3 research/composition_audit_20260912/run_audit.py \
  --seed-index 0 --output /absolute/new/run/seed_0
python3 research/composition_audit_20260912/run_balanced.py \
  --seed-index 0 --output /absolute/new/repair/seed_0
```

Do not re-submit the supplied historical sbatch files without changing their output root.
Already-consumed test probes are diagnostics, not a new formal holdout. The balanced repair changes
both train and validation distributions; it is not an isolated loss ablation.

## Interpretation details

The reference generator's high-resolution audit has fixed optimal candidate roles. Our targets use
16×16 area-downsampled fields and footprints, so the role optimum need not remain exact at that
resolution. We score actual target costs and never infer the correct decision from a role name.
The role-metadata arm is explicitly privileged and does not establish leakage into learned inputs.
The spatial-mean control uses the original training field average and the legitimate candidate footprint.
Fixed no-RGB controls are unchanged in the repair stage for comparison.

CPU replays are a numerical compatibility check, not a replacement for the archived GPU arrays.
One maximum-pixel difference exceeded an initial 0.001 threshold (0.0012233); all twelve replayed
binary fields match exactly. The verification records actual errors and checks field agreement.
Zero-cost card B can have different predicted tie sets in a learned head without any target regret.
Nontrivial A/C/D choice-set differences are therefore recorded separately.

## Paper

Editable source: `../../paper/compositional_visual_risk/manuscript.md`.
Template and build code sit beside it. PDF: `../../output/pdf/compositional_visual_risk_working_paper.pdf`.
The PDF has been rasterized and visually inspected; figure PDFs are exportable vector artifacts.
See the paper README for runtime and regeneration details.
