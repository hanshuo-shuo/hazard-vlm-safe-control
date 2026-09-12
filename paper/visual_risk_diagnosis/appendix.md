# Supplementary scientific and reproducibility details

The external experiment plan below is retained for provenance. It is closed,
including its conditional future tense; none of those branches is an active
research commitment. The main scientific argument is the synthetic oracle
decomposition and prospective confirmation in the manuscript.

## Appendix A. Closed external measurement-interface qualification attempt

RELLIS-3D provides an external image-generation and annotation process [5].
The proposed task is approval of four metric, visible corridor segments using
predicted semantics and a public prohibited-class rule. It is not a collision
experiment: compliant visible semantics do not establish traction, support,
vehicle clearance, or physical feasibility.

The extension first audits official timestamps, calibration, poses, label
mapping and model provenance. A fixed small pilot must establish evaluable
geometric support before training. Corridors have matched progress and width;
semantic labels cannot select their geometry. Missing depth, occlusion,
off-image regions and void labels remain unknown. A release checkpoint is not
eligible merely because exact test filenames are absent from its training list:
nearby observations and revisited locations must be checked.

The now-closed plan specified six controls conditional on continuous-cost qualification: normal segmentation plus the
formula, one development-selected simple postprocessing rule, all-candidate
residual protection, fixed-selector risk control, a no-RGB fixed/spatial prior,
and human-label oracle evaluation. Only one prescribed coarse reference is
compared with native labels. No artificial −2 shift or candidate-set inflation
is allowed. Main results preserve natural frequencies and separately report
all-compliant, mixed, no-compliant, and unevaluable scenes.

The closed plan did not treat sequence frames as independent Bernoulli trials. Route or
spatial groups, including repeated visits, determine splits and uncertainty.
The default external claim is empirical. A mechanism prediction will be locked
after development and before calibration/test. If the normal baseline is
already reliable, or geometry/perception dominates the result, the external
study will narrow the claim instead of searching for a manufactured failure.

### A.1 Outcome of the bounded qualification pilot

We freeze 24 timestamp-ordered, evenly spaced images from the official training
list of sequence 00000, before inspecting their labels. Public archive byte
ranges retrieve only the selected RGB, native labels and timestamp-matched PLY
scans. All 24 triples are available within the prescribed 50 ms synchronization
tolerance. Official calibration and the URDF transform chain determine a
LiDAR-centred, body-oriented coordinate frame. The numerical projection agrees
with an independent OpenCV implementation within 1.82×10⁻¹² pixels; this validates
the implemented formula, not the accuracy of the physical calibration.

Four fixed 1.2 m-wide corridor segments span 6–12 m longitudinal range at headings
−12°, −4°, 4°, and 12°. Each has 720 equal-area samples. A nonsemantic local
surface fit and explicit depth/occlusion checks mark unsupported samples unknown.
The implemented screen uses 0.35 m nearest surface support, an 8-pixel projected
depth bin and a 0.35 m foreground-depth margin. These are construction choices,
not universal sensor-error guarantees. Full gravity alignment and independent
registration-error bounds remain unverified.

| Qualification outcome | Fixed pilot result |
|---|---:|
| Scenes with all four corridors at most 5% unknown | 0/24 |
| Required minimum at the registered 80% screen | 20/24 |
| Candidate unknown-area range | 45.97%–88.47% |
| Optimistic all-four support ceiling, dropping depth/occlusion checks | 3/24 |

The optimistic ceiling is a diagnostic, not an admissible acceptance rule.
Sparse projected depth is the largest contributor to unknown support in the
implemented screen. Even the ceiling remains below the pilot requirement. The
result concerns this single-scan, surface-fit corridor interface; it is not
evidence that RELLIS-3D cannot support a different validated geometric construction.
Unknown-area cost intervals condition on the projection and do not measure
physical calibration error or perceptual failure.

Separately, metadata show that portions of the released validation and test
lists are only 0.2 seconds from training frames within the same sequence.
Released poses lack timestamps and sequence coordinate registration is not
established by the metadata audit. The benchmark split therefore cannot simply
be inherited as a spatial-independence certificate for this study. A training
recipe and checkpoint would need a qualified grouped partition.

We stop before external model training, mechanism selection, risk calibration
or final-test evaluation. The result documents an evaluation-interface boundary
and does not confirm transfer of the synthetic diagnosis. Six distinct geometry
tests pass; an independent scalar replay verifies all 96 candidate counts and
69,120 sampled labels. The subsequently authorized revision is recorded below; that revision also
stopped, closing the external extension without a third round.

![Geometry qualification outcome](../../research/rellis_external_20260912/artifacts/publication/qualification_overview.png)

### A.2 Registered distinction between decision and cost qualification

The pilot protocol described cost using evaluable ground area, whereas its
implementation divided by the **full requested corridor area**. We retain the
implemented denominator and explicitly correct the definition in a separately
registered revision. The original pilot and its frozen protocol are preserved.
Removing unknown area from the denominator would change the estimand and could
make a small visible patch appear representative of a mostly unobserved corridor.

For geometric realization θ, let bθ and uθ be the known-prohibited and unknown
fractions of the full corridor. An empirical range Θ, if independently supported
by development geometry checks, induces

\[
c_L=\min_{\theta\in\Theta}b_\theta,\qquad
c_U=\max_{\theta\in\Theta}(b_\theta+u_\theta).
\]

A candidate is determinately compliant when cU≤0.02, determinately noncompliant
when cL>0.02, and otherwise undetermined. For example, 1% known-prohibited area
and 4% unknown area give [0.01,0.05], which cannot determine compliance despite
meeting a 5% unknown-area screen. Conversely, 10% known-prohibited area already
establishes noncompliance even with substantial unknown area. A narrower support
interval is not automatically more accurate: timing, registration, gravity,
surface and projection errors must also be assessed. These empirical intervals
would remain conditional sensitivity analyses without a separate coverage proof.

The revision is limited to two engineering working days, the consumed 24 frames
and necessary neighbouring sensor data, a corrected single-scan construction and
one motion-compensated window of at most one second and 11 scans. Time/pose and
coordinate provenance, independent physical-error checks and spatial isolation
are prerequisites to opening fresh qualification data. The planned fresh sample
is 48 frames from at least 12 frozen spatial/route blocks, with all supporting
scans included in the isolation audit. The decision screen requires at least
39 frames with all four candidate labels determined, and at least eight mixed
compliant/noncompliant scenes spanning four blocks. Knowing all four outcomes
is a requirement for evaluating candidate selection, not a deployment requirement
that all four actions be safe. At least 12 predetermined frames in six blocks
require independent human geometry review; systematic false support that changes
labels fails the screen regardless of average coverage.

A binary-only continuation is registered in advance: if the decision conditions
pass but p95 full cost-interval width exceeds 0.005, subsequent external research
may address accepted danger rate and availability, but not continuous regret or
residual-cost calibration. Missing samples and failed prerequisites remain in the
record; uncertainty is a stopping outcome. No fresh-sample result, human-review
completion or mechanism transfer is implied by this registration.

### A.3 Outcome of the bounded revision: a prerequisite stop

The revision retrieves 240 timestamped raw Ouster scans, ten in each of the
24 consumed development windows, together with IMU, odometry and transform
records. The central raw scans match all released PLY x/y/z/t/ring fields in
24/24 cases. The static LiDAR/body chain agrees numerically with the pinned
URDF. Under the scan-order pose association supported by the linked export code,
a fixed per-point motion-compensation prototype runs on all 24 development
targets. Central-scan point times lie approximately 16–121 ms after the RGB
filename timestamp; the per-frame p95 correction displacement is 0.101–0.337 m.
These are timing and conditional correction measurements, not independently
measured physical registration errors.

Median end-scan nearest-neighbour consistency improves in 21/24 frames, with
the median over frames changing from 0.132 m to 0.130 m. Those scans may have
participated in upstream SLAM, and the linked Cartographer recipe already uses
VectorNav IMU. Neither source can be repurposed as an independent pose-accuracy
test. Independent Ouster specific-force observations are available, but dynamic
acceleration and mounting/attitude error have not been separated into a justified
gravity-error range. A release-specific timestamp/pose record, independent
physical camera-alignment and surface/visibility error ranges, and complete
support-aware spatial isolation also remain unestablished.

We therefore take the preregistered uncertainty branch and stop **before**
instantiating the new 48-frame qualification set. This is an early evidence-based
stop within the two-day upper budget, not budget exhaustion or a 0/48 result.
The new decision-information screen and continuous-cost precision screen are
not evaluated. No independent human review on the uninstantiated set is claimed.
The motion prototype is not promoted into an admissible corridor-reference
interface; no external model or mechanism experiment follows. This outcome
narrows the external evidence available for the controlled-case paper without
supporting or refuting transfer of its mechanism diagnosis.


## Appendix B. Earlier three-experiment computational verification

Independent verification checks 257 manifest-listed artifacts, reconstructs
scalar choices and acceptance counts, and checks exact-binomial certificate
p-values. CPU replay of 80 images across all 20 new checkpoints has 100% binary
field agreement with stored GPU predictions; the maximum probability difference
is 0.00635160, within the registered 0.01 tolerance. Sixteen distinct numerical
and protocol tests pass. A separate identity audit finds 10,200 distinct
continuous field and complete-scene geometries across the new banks after
excluding IDs, seeds, appearance labels and candidate order.

## Appendix C. Oracle decomposition: controls, inference and provenance

The [discovery protocol](../../research/oracle_aggregation_20260912/protocol.json)
uses consumed E3 IID and appearance test banks, 2,000 scenes each, and stored
predictions from all five E2 independent/source-64 models. There is no training,
retrospective certification, resolution sweep or outcome-based replacement.
Direct exposure integrates P512×F512 before pooling. The pooled oracle uses exact
means of nonoverlapping 32×32 cells. The target oracle uses the actual source64-to16
property targets under the same pooled F512. Learned arms use restored output
or the explicitly secondary −2 shift. C is recomputed at every level. Full
channel exposures, signed terms, cancellation statistics, danger-pattern
transitions and per-model counts are in the [JSON report](../../research/oracle_aggregation_20260912/artifacts/report/SUMMARY.json).

The precision check rasterizes properties at 1024 and pools them to 512 while
holding F512 unchanged. Earlier full 256/direct512 sensitivity is separate. Neither
is a continuum theorem. For binary fields, the per-channel envelope
∑sqrt(p(1−p)f(1−f))/∑f bounds discarded covariance. C is bounded by the sum of
channel envelopes since its partial derivatives lie in [0,1]. No label change
outside the envelope or without joint boundary overlap is expected algebraically;
these are implementation checks, not novel empirical predictions.

The [confirmation protocol](../../research/oracle_aggregation_20260912/confirmation_protocol.json)
fixes one new 2,000-scene IID bank, geometry seed 2612091301 and bootstrap seed 2612091302.
Checkpoint and source hashes are persisted before record generation. Existing
continuous sampler constraints remain; no added condition uses safety, margin or
inference outcome to admit a scene. Anonymous shapes and complete-scene identities
are exactly disjoint from all 10,200 E2/E3 records, including training/validation.
This does not establish disjointness under arbitrary geometric symmetries.
New confirmation uses the same weights on CPU; discovery uses stored GPU costs.
The fixed illustration's CPU cost replay differs by at most .000012707. This limited
check does not prove exact agreement on every input. Neither the oracle decision
effect nor its high-resolution reference is learned from those outputs.

### C.1 Prediction thresholds and the direction addendum

The three predictions require stable-reference lost-opportunity probability > .01;
conservative fraction > .90 among pooling label changes at model-selected actions;
and small-versus-larger margin enrichment > .10 conditional on boundary overlap.
Small margin is fixed at .005. Each one-sided comparison uses γ=.05/3. H1 uses exact
binomial inversion; H2/H3 initially use crossed bootstrap draws (5,000 resamples).
Table/figure 95% intervals are descriptive. Training-seed inference is approximate
with only five model seeds.

H2's bootstrap degenerates when every observed flip has one direction. A
[direction addendum](../../research/oracle_aggregation_20260912/confirmation_direction_addendum.json)
was frozen at **20:57:54 UTC**, after confirmation job submission but before the
analyst inspected fresh outcomes and before the hypothesis output existed
(created **21:00:24 UTC**). This is not represented as part of the original freeze.
It additionally requires an exact binomial check with the same H2 budget: among
independent scenes with any pooling flip across the fixed five-model bank, every
flip must be conservative. It does not change the data or offer another route
to passing H2.

All 68 changed scenes have only conservative changes, with exact one-sided lower
94.1566%, rather than the original bootstrap's degenerate 100%. This conditions on
the fixed model bank and a scene having a flip, a different estimand from the
weighted model–scene proportion. H1's exact lower is 2.2869%; H3's crossed-bootstrap
lower is 26.0586 percentage points. All criteria pass, including the extra direction
check. Numerical screening also passes: all-candidate property-check label
change .10417%, p95 cost difference .0005088, covariance residual≤5.04×10⁻¹⁷.

### C.2 Independent computation checks and artifacts

Six distinct tests cover equal coarse means with opposite overlap, block axes
and mass preservation, nonlinear C telescoping, zero covariance without joint
boundary, fixed selection versus reselection, and thresholds/cancellation.
Discovery verification replays 52,000 scalar decisions and 768 blockwise candidate
costs, checking all 14 old input hashes. Fresh verification replays 26,000 scalar
decisions and 384 blockwise costs, checks all five checkpoint hashes and frozen
sources, and independently inverts H1 with SciPy beta quantiles. Blockwise cost
differences are zero. These are computational checks, not cross-task validation.

[Publication tables](../../research/oracle_aggregation_20260912/artifacts/final_publication/TABLES.md),
[original confirmation output](../../research/oracle_aggregation_20260912/artifacts/confirmation/HYPOTHESES.json),
and [independent verification](../../research/oracle_aggregation_20260912/artifacts/confirmation_verification/VERIFICATION.json)
retain the distinct statistics. Initial figures are archived; later changes
repair category colors and legends without changing scientific arrays or reports.
All computation and plotting run on Quest. The one new confirmation is now
consumed and will not be enlarged or relabelled as fresh evidence.

## Appendix D. Retained earlier measurement, postprocessing and certification details

The following material retains the earlier study structure and original section labels for provenance; it now supports the main mechanism rather than defining three equal paper themes.

### 3.4 Earlier measurement and geometry evidence

The historical interface pools a low-resolution property field along candidate
footprints. We reconstruct direct 256 and 512 raster references from the saved
continuous shapes and paths, holding the definition at spatial mean exposure.
The qualification screen requires at most 1% dangerous-label disagreement and
a 95th-percentile cost difference at most 0.002 between these references.

All diagnostic splits pass: dangerous-label disagreement is at most 0.0972%,
and the 95th-percentile cost difference is below 0.00076. This is empirical
raster convergence, not an exact continuum integration guarantee. In contrast,
historical low16 and direct-512 danger labels differ on approximately
2.6%–3.3% of entries. Relative to the earlier low16 definition, these are not
automatically incorrectly labeled examples. They are different discretizations
of the intended cost. Their effects must be reported separately from changes
to the predictor.

| Consumed diagnostic split | Oracle feasible, low16 | Oracle feasible, direct 512 |
|---|---:|---:|
| IID | 76.17% | 79.72% |
| New appearance | 76.33% | 80.78% |
| Shape and appearance | 82.22% | 85.39% |

These oracle values enumerate both image versions and all three nontrivial
cards, giving 3,600 decisions for each 600-pair split. Acceptance diagnostics
instead sample one actual version/card per independent scene; their denominators
are reported separately. The high oracle supply rejects universal task
infeasibility as the explanation for zero coverage on these data.

To separate a prior geometry improvement from a source-raster change, we train
a matched 2×2×5 design: balanced-anchor versus independent geometry, each with
48 or 64 source rasterization. Within a geometry condition, continuous scenes,
property assignment, candidate paths and 64×64 RGB are shared across the two
source rasters. Sampling constraints operate on continuous geometry. Every
model receives 800 training images, 24 epochs and exactly 960 optimizer updates;
the final checkpoint is used without validation-based selection.

Common-reference scoring uses the same 512-derived footprints and truth for
every model. Native-reference results are retained in the supplement.

| New-appearance contrast: balanced anchor minus independent | Regret reduction | Crossed seed/scene 95% interval |
|---|---:|---|
| Source raster 48 | 0.004870 | [0.003101, 0.006853] |
| Source raster 64 | 0.008249 | [0.002850, 0.017690] |

Both intervals exceed the predeclared 0.001 meaningful-effect threshold.
The independent scheme's raster contrast interval lies within the registered
±0.0005 equivalence band. The anchor scheme's much wider interval is
inconclusive; its unfavorable seed is retained. Geometry jointly changes
position, size, shape and rotation, so this comparison supports the geometry
scheme and does not identify position alone as the causal factor.

![Matched geometry and source-raster evidence](../../research/pro_decision_round_20260912/artifacts/final_report/e2_factorial.png)

## 4. How much does postprocessing matter without retraining?

The inherited inference recipe reduces the output logit bias by two after
checkpoint selection. We compare the inherited output with restoration of that
bias using the same weights and forward-pass logits. Image, path, card, truth
reference and danger threshold are held fixed. Residual calibration is repeated
for each postprocessing arm. A monotone pixel-probability transformation need
not preserve ranking after spatial averaging, so actual selections are also
recorded rather than assumed identical.

On the consumed new-appearance diagnostic, unsafe rate among accepted decisions
falls from 4.70% to 1.73%. Acceptance falls from 85.03% to 80.97%, and safe
opportunity efficiency changes from 99.63% to 97.83%. This is a risk–acceptance
tradeoff produced without learning new features. Boundary/interior deficits
and an oracle replacement restricted to the boundary band are saved as
privileged diagnostics, not deployable methods.

![Fixed weights and postprocessing](../../research/pro_decision_round_20260912/artifacts/publication/e1_postprocessing.png)

*E1: five historical checkpoints and consumed diagnostic data; these are not
the E2 checkpoints or E3 certification/test banks.*

Restoring the bias is not the sole explanation for every observed failure.
In the fresh confirmation, the inherited −2 arm itself has an empirical unsafe
rate of 2.85%, below 5%. That table does not give it the same certificate as a
registered restored-output rule. Meanwhile, the restored model still cannot
accept any action using the joint residual bound. We therefore distinguish
one-sided postprocessing effects from the separate cost of protection scope.

## 5. Does an unusable joint bound imply an unusable selector?

### 5.1 A deterministic explanation for zero acceptance

For the five historical E1 models on the consumed 300-scene calibration bank
(not the E2 models or E3 certification bank), the restored-output
maximum-positive-residual quantile is
0.02797–0.03287 over the complete pair, 0.02201–0.02767 over the current four
candidates, and 0.00501–0.00790 for the frozen selected action. Each scope uses
300 independent scene scores; reducing scope does not manufacture more samples
by flattening correlated entries.

If estimated costs are nonnegative and q>τ, the rule
min(1,estimated cost+q)≤τ necessarily rejects all candidates. This explains
zero acceptance even before inspecting any appearance shift. The current-four
quantiles also exceed τ: the difference is not merely the inclusion of
unrealized paired versions and capability cards. Unselected candidates within
the current decision matter as well. The selected-action bound is a mechanism
diagnostic and does not itself certify conditional risk among accepted actions.

### 5.2 Independent risk and usefulness confirmation

This E3 experiment uses all five E2 independent-geometry/source-64 models and
new calibration/test banks. It is not a within-model extension of E1 §5.1.

We freeze the restored-output selector, resolve ties using stored independent
uniform draws at tolerance 1e−9, and accept or reject only after selection.
Rejection never causes reselection. Each independent synthetic scene contributes
one image version and one A/C/D card. Source and labeled-target domains each
provide 1,600 calibration scenes and 2,000 untouched test scenes.

The 12-rule threshold bank is fixed before calibration and includes the original
acceptance threshold 0.02. Exact one-sided binomial upper bounds use
δ/(5 models × 2 domains × 12 rules)=0.05/120. We choose the certified rule with
highest calibration acceptance. No valid rule means rejection and a failed
certificate, not zero-risk success. Certificates are saved before test loading.
This is an application of established selective-risk control [3].

| E3 IID arm: five E2 models, new E3 banks | Acceptance | Unsafe among accepted | Safe opportunity efficiency η |
|---|---:|---:|---:|
| Inherited −2, fixed threshold | 81.87% | 2.85% | 99.18% |
| **Restored output, fixed 0.02 threshold** | **78.58%** | **1.07%** | **96.93%** |
| Restored output, pair residual upper bound | 0% | Undefined | 0% |
| Restored output, selected certified rule | 81.29% | 2.51% | 98.82% |
| Four-candidate oracle | 80.20% | 0% | 100% |

**A more complex model is not required to pass this particular risk and
usefulness screen.** The simple fixed-threshold baseline already has
valid certificates for all five source models, with simultaneous calibration
upper bounds between 1.66% and 2.54%. Its test η interval is [95.93%,97.82%],
far above the predeclared 50% usefulness threshold. Selecting among rules is
unnecessary to pass the screen. The selected-rule procedure gains about 1.88
percentage points of opportunity efficiency while using more of the risk budget;
it is not an improvement without cost.

The selected-rule η interval is [97.96%,99.47%]. Crossed intervals retain shared
scenes and model seeds and condition on the fixed calibration banks. The exact
calibration procedure, rather than a bootstrap test interval, supplies the
conditional-risk certificate. Five seeds provide limited information about
training variability.

![Acceptance, empirical conditional risk, and safe opportunities](../../research/pro_decision_round_20260912/artifacts/publication/e3_acceptance.png)

*E3: the five preselected E2 independent-geometry/source-64 models, each with
1,600 calibration and 2,000 test scenes per domain. E1 residual quantiles in
§5.1 come from different checkpoints and data.*

On new appearance, source-certified rules have empirical unsafe rate 3.58%
and η=98.62%. This is a shift stress test, not an unknown-distribution guarantee.
With independently labeled target calibration, the corresponding values are
2.00% and 97.34%. We neither claim that adaptation is necessary on this particular
test nor interpret successful source transfer as certified OOD safety. The
certificates concern the registered mixture, not each subgroup or an episode.
