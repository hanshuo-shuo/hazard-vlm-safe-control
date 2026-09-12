# Pro-directed three-experiment decision round

This is the bounded next round requested in [PRO_REVIEW.md](PRO_REVIEW.md).
All three experiments, certification, verification and plotting ran on Quest.
The result supports a measurement/postprocessing/decision-unit study; it does
not establish a new safety algorithm, VLM teacher, learned public relation or RL policy.

Read first:

- [Chinese decisions and progress](../../docs/PRO_DECISION_ROUND_PROGRESS_2026-09-12.md).
- [Current English paper addendum](artifacts/publication/PAPER_ADDENDUM.md).
- [Registered protocol](protocol.json).
- [Independent verification](artifacts/final_report/VERIFICATION.json).
- [Anonymous continuous-geometry split audit](artifacts/CANONICAL_SPLIT_AUDIT.json).
- [Fixed-baseline certificate check](artifacts/publication/FIXED_BASELINE_CHECK.json).
- [Initial report and machine-readable decision](artifacts/final_report/DECISION.json).

## Scientific result

1. **E1, no training:** five old checkpoints, inherited -2 logits versus restored
   bias; exact oracle feasibility, direct 256/512 mean-exposure references,
   nested residual scopes, and a privileged boundary-only oracle diagnostic.
   Simple postprocessing already meets the diagnostic safety/usefulness screen.
   Old low16 and common-reference danger labels differ in about 2.6%-3.3% of entries.
2. **E2, matched 2x2x5:** balanced-anchor/independent geometry by 48/64 source
   supervision, identical RGB per scene across source rasters, continuous
   geometry rejection only, exactly 960 updates and final checkpoints.
   Geometry effects pass at both rasters. Raster equivalence is supported for
   the independent scheme; the anchor scheme's interval is inconclusive.
3. **E3, one final confirmation:** one deployment decision per independent scene,
   five fixed models, 1,600 calibration and 2,000 test scenes per domain,
   12 registered thresholds, exact-binomial certification with a 120-rule
   family correction. IID screening passes with eta 98.82%, and the already
   registered fixed restored-bias baseline also passes with eta 96.93%.
   Threshold search is not necessary to pass this research screen.

Tau=0.02 is a **synthetic cost** danger cutoff. Epsilon=0.05 is conditional
unsafe frequency; delta=0.05 controls certification family error; alpha=0.05
is the separate residual-bound failure rate. These parameters are not interchangeable.
The guarantee concerns the registered IID mixture of image variants and A/C/D
cards, not each subgroup or a robot episode. Source-to-target results are stress
tests; labeled target certification is explicit adaptation.

## Provenance and artifacts

Quest root:

```text
/projects/p33100/siosio/hazard_pro_decision_round_20260912/
    research/                         source snapshots and preserved reference
    FREEZE_{E1,E2,E3,REPORT}.json       staged source/protocol identities
    e1_data/, e1_runs/, e1_report/     consumed-data diagnostics
    e2_data/, e2_runs/, e2_report/     matched factorial and 20 checkpoints
    e3_data/, e3_runs/, e3_report/     independent certificates and final tests
    final_report/                     independently verified initial report
    publication/                      figure QA refinements and fixed-baseline interpretation
    logs/, JOB_IDS.txt, SLURM_ACCOUNTING.txt
```

Local `artifacts/` contains compact copied reports, figures, freeze records,
dataset audits and selected logs. Full images, masks, predictions and weights
remain on Quest. Old models are read from
`/projects/p33100/siosio/hazard_independent_confirmation_20260912/`; they are not
modified. The training runtime is `/home/shv7753/envs/hazard-exp01b-r2/bin/python`.
Plotting reuses the prior study's isolated `report_deps` without changing environments.

Twelve numerical/protocol tests plus four certification tests passed. Repeating
the first twelve at staged freezes does not create additional distinct tests.
Independent verification checks 257 data/run/report manifest entries, scalar
ties and counts, binomial p-values, selected-rule identity and 80 CPU replay
images from all 20 new checkpoints. Binary agreement is 100%; maximum replay
probability difference is 0.00635160, within the fixed 0.01 tolerance.

A final metadata-only audit (Quest job 6183553) finds 10,200 distinct field
geometries and complete scene geometries across the E2/E3 banks, with no exact
duplicates. Identities exclude scene IDs, seeds, appearance/property labels,
source raster and candidate ordering. This checks exact stored continuous
geometry, not equivalence under arbitrary geometric symmetries.

The initial report's figures were refined in a separate publication artifact
to show the 5% reference line on comparable axes and leave space above eta=1.
The publication adds the already-certified fixed-threshold control explicitly.
This changes presentation/interpretation only: no new predictions, threshold
choice, test evaluation or modification of the frozen initial report.

## Reproduction

The sbatch files point to a completed immutable run. Do not resubmit them
unchanged. Deploy the source tree under a **new** Quest root and keep the
relative `research/quest_reference` and `research/composition_audit_20260912`
imports. Use the staged sbatch files as templates, replacing only the new root
and log paths before freezing. Each stage refuses existing output directories.

- Run `test_round.py`, freeze E1, then `prepare_e1.py`, five `run_e1.py` tasks and
  `aggregate_e1.py`. The E1 reference/feasibility gate controls continuation.
- Freeze E2, run `prepare_e2.py`, task indices 0-19 of `run_e2.py`, and `aggregate_e2.py`.
  Indices 0-4/5-9/10-14/15-19 are anchor48/anchor64/independent48/independent64.
- Run `test_certification.py`, freeze E3, create the disjoint E3 banks, run five
  `run_e3.py` tasks and `aggregate_e3.py`. All selector/rule certificates must be
  saved before tests are opened.
- Freeze the report sources, run `verify_round.py`, then `report_round.py` and
  optionally the presentation-only `publication.py` using the plot runtime.
- Run `audit_split_identity.py --root NEW_ROOT` for the final anonymous
  continuous-geometry duplicate check.

The exact job sequence, staged source lists and runtime commands are recorded
in the checked-in sbatch files. Every numerical stage uses a Slurm allocation.

## Stop decision

This round ends after the registered comparisons. Do not increase model sizes,
search more rasters/checkpoints or reuse these tests to repair the story.
Retain the simple fixed baseline and the public formula. A broader methods
paper needs independent-task or independently observed physical-outcome evidence;
VLM labeling and RL are not prerequisites for making the current diagnosis useful.
