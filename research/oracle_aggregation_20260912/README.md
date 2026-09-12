# Same-reference oracle aggregation study

This directory implements Pro's next task after `00880e6`: one discovery analysis
on consumed assets, followed only because its signal was substantive by one
prospective normal-IID confirmation. There is no training or new risk certification.

Read [PRO_REVIEW.md](PRO_REVIEW.md), [discovery protocol](protocol.json),
[confirmation protocol](confirmation_protocol.json), and the explicitly timed
[direction addendum](confirmation_direction_addendum.json).

The final scientific status is `SUPPORTED_WITH_EXACT_SCENE_DIRECTION_CHECK` for
the specified normal synthetic task. This is not a general impossibility result,
a new mathematical identity or calibration method, or evidence of cross-task
transfer. The RELLIS expansion is closed.

## Assets and reproduction

Quest root: `/projects/p33100/siosio/hazard_oracle_aggregation_20260912`.
Read-only prior root: `/projects/p33100/siosio/hazard_pro_decision_round_20260912`.
The sibling `research/pro_decision_round_20260912` on Quest is a symlink to the
prior frozen source. That source installs the preserved `quest_reference`
namespace through its own `composition_audit_20260912/run_audit.py`.
Python: `/home/shv7753/envs/hazard-exp01b-r2/bin/python`.

Run in a new empty output root with the corresponding batch paths updated;
scripts refuse to overwrite existing stage directories:

1. `decompose.py --root ROOT`: common-footprint reconstruction of consumed E3
   IID/appearance banks, each2,000 scenes; old inputs and predictions are checked.
2. `report.py --root ROOT`: signed exposure/cost terms, fixed-selection and
   reselection summaries, predeclared boundary/margin conditions.
3. `verify.py --root ROOT`: six contract tests and independent scalar replay.
4. After a substantive discovery decision, `confirm.py --root ROOT`: freeze new
   protocol/checkpoints/code, generate one2,000-scene IID bank, infer existing
   weights on CPU, record hypotheses and full descriptive report.
5. `verify_confirmation.py --root ROOT`: independent counts, blockwise costs,
   beta inversion, and stricter exact scene-level direction check. SciPy is
   loaded from the prior RELLIS qualification dependency directory only as a
   numerical library; no RELLIS data or experiments are reopened.
6. `publication_v3.py` and `final_publication.py`: task illustration, signed
   discovery plot, confirmation plot and tables. Matplotlib comes from the
   existing independent-confirmation `report_deps`.

The original direction bootstrap degenerates to lower1 when every observed
change is conservative. The addendum was frozen after job submission but before
fresh results were inspected or written; it preserves the original output and
requires an additional exact, scene-level check. That check gives lower.941566
for68/68 changed scenes, conditional on the fixed model bank. H3 retains an
approximate crossed bootstrap with only five seeds. This limitation is explicit.

## Evidence locations

- `discovery/` on Quest: arrays, old input hashes and full stage manifest.
- `report/`: consumed-data statistics; no recertification.
- `confirmation/`: freeze, records, arrays, logits, hypotheses and summary.
- `verification/`, `confirmation_verification/`: independent computational checks.
- `publication/`, `publication_v2/`, `publication_v3/`: versioned presentation
  changes. Earlier color/legend deficiencies are preserved; onlyv3 is current.
- `final_publication/`: discovery versus confirmation figure and full tables.
- `artifacts/`: selected reports, records, logs and figures archived in Git.
  Full stage manifests also name arrays/logits retained only on Quest; they are
  not manifests claiming every raw file is present in this compact Git archive.

See [Chinese progress](../../docs/ORACLE_AGGREGATION_PROGRESS_2026-09-12.md),
[manuscript](../../paper/visual_risk_diagnosis/manuscript.md), and
[appendix](../../paper/visual_risk_diagnosis/appendix.md).

No new confirmation will be automatically generated after this completed bank.
The next work is claim review, nine-page manuscript editing and anonymous
submission preparation, with author decisions handled by the human authors.
