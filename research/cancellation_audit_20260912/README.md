# Full-cohort cancellation audit and independent input-by-operator test

Parent `b96e36e`. The user renewed the request to address unresolved scientific
objections after the matched-risk study had stopped. That old stopping result
is preserved. The new work first audits full denominators and residual hybrids,
then tests an actual input-field by aggregator interaction under a separate
frozen protocol. It does not claim the old superiority gate passed.

Read [audit protocol](protocol.json), [factorial addendum](factorial_addendum.json),
[identification limits](IDENTIFICATION.md), [independent protocol](independent_protocol.json),
[independent result](artifacts/independent/PRIMARY.json) and
[verification](artifacts/independent_verification/VERIFICATION.json).

## Scientific outcome

On consumed appended data, 580/2335 entries improve both residual components yet
worsen total absolute error; 25 become newly unsafe. Baseline opposition among
original rejections gives 38/1025 new harms versus 5/1192 without the flag.
Overlap-standardized differences are +.35 pp [-.36,1.15] on development and
+1.66 pp [.34,2.33] on appended data. The flag uses reference labels and does not
define a deployment predictor or a causal share.

Algebraic measurement-only substitution gives 113 new harms versus 43 for full
PLIC; the model-relative term removes 71 and one full harm requires both changes.
Such hybrids may leave [0,1]; they are not actual candidate methods.

The actual four-cell intervention passes both valid inputs (exact coarse target
and fixed learned field) through both unchanged operators, at identical learned-
uniform-selected actions. Consumed safe-accept interactions are -1.88/-1.45 pp.
A separate 2,000-scene independent bank then tests the frozen mean interaction
below -.005. It finds **-2.16 pp**, one-sided 95% upper **-1.64 pp**; the finer-
property sensitivity gives -2.07 pp, upper -1.55 pp. Target/learned reconstruction
gains are 3.01/.85 pp of all scenes. The primary fixed-model-bank prediction is
supported. It is not a matched-risk superiority result or new safety certificate.

The independent secondary oracle-cohort difference is +1.35 pp [0,2.01], with
47/1085 versus 3/1171 newly unsafe original rejections. It does not independently
confirm a label-free warning condition. All secondary outcomes remain secondary.

## Data sequence and reproduction

All computation is on Quest under
`/projects/p33100/siosio/hazard_cancellation_audit_20260912`.

1. `test_audit.py` and `audit.py --root ROOT`: source freeze, full-cohort accounting,
   coarsened overlap standardization and scalar residual hybrids on the two
   already consumed 2,000-scene banks.
2. `verify_publish.py --root ROOT`: independent scalar/count verification and
   the subsequently recorded actual input-by-operator factorial addendum.
3. `independent.py --root ROOT`: separate protocol/checkpoint/source freeze before
   one new 2,000-scene bank (seed2612091701), geometry identity check against12,200
   previous scenes, CPU replay of all five models, baseline flags frozen before
   PLIC, the one primary contrast and frozen secondary accounting. No training.
4. `verify_independent.py --root ROOT`: scalar score and primary bootstrap replay,
   reference/PLIC recomputation, fixed-model RGB replay, and paper-ready numbers.
5. `build_paper.sbatch` and `package_paper.sbatch`: source snapshot, nine-page main
   text, all-page render/visual audit and isolated anonymous-source ZIP rebuild.

There are four new tests, 20,000 consumed scalar records/100,000 score checks,
10,000 independent records/50,000 scores, and80 independent RGB replay images.
Model, reference and PLIC replay discrepancies are zero. Original sources and
checkpoints are retained unchanged. Exact anonymous geometry overlap is zero;
this is not equivalence under arbitrary geometric symmetries.

`artifacts/` includes complete consumed-event CSV data, compact results, protocols,
freezes and the new continuous records. Large scene arrays, logits and RGB stay
on Quest; stage manifests name those files without claiming they are in Git.
Bootstrap inference is conditional on the fixed five-model bank and this normal
generator, with shared scenes. The independent test is now closed regardless of
success. No additional bank, model, reconstruction or subgroup search follows.

The paper remains a controlled diagnostic study, now with a prospectively tested
computational interaction. Venue-level novelty and general deployment value are
not decided by the computational checks. Author review and the complete anonymous
experimental supplement remain pending; the ZIP contains manuscript source only.

Final PDF: nine main pages, sixteen total, all visually inspected. The extracted
ZIP independently rebuilds identical page text and110-dpi rasters. The
[delivery record](artifacts/delivery/DELIVERY.json) verifies unchanged source,
input and checkpoint hashes, official style files, and freeze/identity/baseline/
result ordering. The pre-specified independent per-model and per-card factorial
results are in [the secondary table](artifacts/independent_factorial/TABLES.md)
and its JSON; all group partitions reproduce the aggregate primary. They remain
descriptive rather than additional confirmed claims. All seven jobs are recorded in
[UTC accounting](artifacts/JOB_ACCOUNTING_UTC.txt).
