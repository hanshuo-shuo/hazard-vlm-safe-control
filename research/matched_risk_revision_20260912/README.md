# Complete empirical-risk comparison and manuscript revision

Parent: `41182b0`. Read [strict review](PRO_REVIEW.md), [protocol](protocol.json),
[full tables](artifacts/verification_publication/TABLES.md) and
[independent verification](artifacts/verification_publication/VERIFICATION.json).

This round is closed with **STOP_NO_NEW_SCENES**. Both complete score-threshold
families use identical empirical risk ceilings and threshold selection. The
primary analysis fixes the uniform-selected action; reselected policies,
individual model curves and development-selected threshold transfer are retained.
It consumes only the two previously used 2,000-scene banks. There are no new
models, operators, sample banks, subgroup searches or risk certificates.

The exact-target gains are clearer at low risk ceilings. Learned point estimates
favor PLIC at all eight listed ceilings on both banks, but the pointwise paired
intervals cross zero at .5--4%. At 1% risk ceiling, appended learned eta gains
.45 pp with interval [-.13,1.49]; at 2%, .30 pp [-.19,.66]. This does not prove
equivalence or absence of spatial information. It does not meet the frozen
operational screen for spending a further independent sample bank either.

## Definitions and provenance

- Danger remains reference cost > .02. Only the acceptance-score threshold varies.
- Every distinct score enters the full family with complete equal-score groups.
  Zero acceptance has undefined risk. Conditional risk need not be monotonic.
- At each ceiling, maximize safe accepts, then minimize unsafe accepts, then
  minimize threshold. Five-model pooling uses one common threshold, on shared
  scenes. Both operators have identical selection and evaluation permissions.
- Empirical envelopes are selected on consumed outcome labels. They are not
  deployable guarantees. Pointwise bootstrap intervals preserve paired scenes
  and resample models, but do not remove selection optimism or imply simultaneous
  coverage. Model inference is limited by five training seeds.
- Operational continuation gate: see protocol. It was fixed before reading
  these full curves, not before the earlier uses of these data. It is a work-
  stopping convention, not a newly confirmatory hypothesis or a venue standard.
- [ALL_CURVES.csv.gz](artifacts/analysis/ALL_CURVES.csv.gz) contains every raw
  threshold point (166,414 non-reject-all rows). Full extracted decision arrays
  remain on Quest; manifests name those Quest-only arrays as well as local files.
- Error-cancellation counts remain outcome-conditioned retrospective patterns.
  No causal fraction, independent-module intervention or failure predictor is claimed.

## Reproduction on Quest

Root: `/projects/p33100/siosio/hazard_matched_risk_revision_20260912`.
Inputs: previous same-information round's immutable `development/data.npz` and
`appended/data.npz`, addressed in the protocol. Old operators are never changed.

1. Run `test_curves.py`, then `analyze.py --root ROOT` in an empty output root.
   It records source/protocol hashes before reading inputs and outputs complete
   curves, descriptive intervals, transfer results and the continuation decision.
2. Run `verify_plot.py --root ROOT`: original-array scalar action/score replay;
   independent binary-search counts in separately sorted safe/unsafe lists at
   every threshold; independent full-family optimum checks; figures and tables.
3. `build_paper.sbatch` snapshots the manuscript and compiles/renders it using
   TeX Live 2026 and the prior round's pinned PyMuPDF 1.26.7 renderer.

Four tests pass; independent checks replay 96,000 scores and verify 166,414
threshold rows. Scientific data analysis, plotting and document compilation all
run on Quest. Local operations are editing, copying, viewing and Git.

The manuscript now has two contributions, with reconstruction an explanatory
intervention. It is a controlled diagnostic case study. Neither computation
completion nor this revision asserts ICLR main-conference maturity. Human author
review, venue choice and the complete anonymous experimental supplement remain
pending; the deliverable ZIP is manuscript source only.

Final build `6207364` has nine main pages and fourteen total pages. All pages
have visual-review coverage recorded in [DELIVERY.json](artifacts/delivery/DELIVERY.json):
all predecessor pages inspected, changed pages 8 and 12 re-inspected, remaining
pages proven pixel-identical. Delivery `6207454` independently rebuilds the
extracted ZIP with identical text and rendered pixels on every page. Official
style files, frozen analysis and both input hashes pass. See
[UTC job accounting](artifacts/JOB_ACCOUNTING_UTC.txt).
