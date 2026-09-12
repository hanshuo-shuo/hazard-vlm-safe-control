# From Spatial Attribute Targets to Action Exposure: Diagnosing a Visual Risk Interface

**Working manuscript, 12 September 2026.** Same-reference oracle discovery and
one prospective IID confirmation are complete. This remains a controlled study,
not a claim of universal prevalence or an ICLR-ready submission.
The initial RELLIS-3D corridor construction does not pass its geometry qualification
screen and contributes no confirmed external mechanism result. Authorship and submission venue are
not assigned here. This draft supersedes the earlier method-centered narrative;
the original manuscript and frozen reports remain unchanged.

## Abstract

Does accurate prediction of a spatial supervision target ensure accurate
action-conditioned exposure? We investigate this question in a procedural visual
risk task with two attribute fields, four candidate footprints and a public cost
law. In our motivating pipeline, zero calibrated acceptance left perception,
measurement, postprocessing and protection scope as competing explanations.
We hold the reference footprint fixed and decompose signed cost error into
model error relative to supervision, source-raster difference and separate-pooling
error. The last term discards within-cell attribute–footprint covariance, even
for perfect coarse attributes. Discovery on consumed data motivates one frozen
2,000-scene IID confirmation without new training. The coarse oracle loses safe
opportunities in 63 scenes; 61 remain under a finer-property check, with an exact
one-sided lower bound of 2.29% of scenes. At fixed model-selected actions, pooling
changes 3.22% of danger labels, all conservatively in the observed sample; changes
concentrate at overlapping boundaries and small cost margins. Perfect prediction
of the actual supervision target still loses 2.70 percentage points of safe
acceptance. Signed model and measurement errors can partially cancel. Earlier
matched training and independent risk certification distinguish genuine learning
effects from this interface error: a simple restored-output rule was certified
with empirical conditional unsafe rate 1.07% and safe opportunity use 96.93% on
its separate test bank. The contribution is a tested attribution of decision
error in this controlled task, not a new pooling identity or calibration method.
Cross-task transfer and physical robot safety remain unestablished.

## 1. Introduction

An action-conditioned visual cost is an interface between perception and a
decision rule. Its usefulness depends on more than average prediction error:
which actions are considered, which target defines danger, which errors a bound
must cover, and when a decision is accepted all matter. If these choices remain
implicit, a failure at the interface can be attributed to the learning component
that happens to precede it.

Our motivating case arose in a procedural visual risk study. Repairing a
position–attribute shortcut substantially improved decision ranking. Nevertheless,
adding a scene-level calibrated upper bound yielded zero action acceptance.
Possible explanations included a lack of feasible candidates, persistent
one-sided perception error, discretization differences, and protection of
outcomes irrelevant to the deployed selector. They imply different research
decisions. Only some would motivate a more complex perceptual or learned cost
model; none can be identified from a ranking score alone.

The central question is whether the spatial supervision target retains the
information needed by the action-exposure calculation. We use oracle
interventions to separate this question from how well a model learned its target:

1. **Can perfect coarse attributes still change usable decisions under a fixed reference footprint?**
2. **How much does output postprocessing change decisions with perception weights fixed?**
3. **Does an unusable joint upper bound imply that the fixed selector cannot operate reliably?**

The positive empirical finding is a spatial-interface error that survives
perfect supervision-target prediction, affects actual decisions, and follows
predictions tested on a new sample. This changes what a component-accuracy
improvement would establish: it would not, by itself, validate the exposure
measurement. Existing certification can nevertheless make an imperfect pipeline
useful under a stated distribution and rule. These distinctions support retaining
a simple model while diagnosing its measurement interface; they do not show that
a richer model could never compensate. Cross-task transfer remains untested after
the external measurement-interface qualification attempt was stopped.

## 2. Decision objects and the audit design

### 2.1 Pipeline and target

Let an image be x, a capability/rule card be k, and the four fixed candidate
footprints be F_a. A visual predictor produces two spatial property fields.
Their mean exposure along a footprint is e=(e_w,e_f). The public synthetic cost
is g(e,k): e_w for card A, zero for B, e_w+e_f−e_we_f for C, and e_f for D.
The nontrivial deployment mixture uses A/C/D. Ground truth follows this same
equation. A learned relation cannot discover an unknown law from labels defined
by the public equation; its possible contribution would instead be error
compensation under predicted exposure.

For a frozen selector aπ(x,k), let A indicate acceptance and let cπ be the true
selected cost. We report

\[
C=\Pr(A=1),\quad J=\Pr(A=1,c_\pi>\tau),\quad
R_{\rm sel}=J/C,
\]

and

\[
C_{\rm oracle}=\Pr(\min_a c(x,a,k)\leq\tau),\qquad
\eta=\frac{\Pr(A=1,c_\pi\leq\tau)}{C_{\rm oracle}}.
\]

Unsafe acceptance never contributes to η. Conditional risk is undefined when
C=0. The oracle is restricted to the supplied four candidates, and acceptance
is a static decision, not successful physical execution.

Four parameters have different meanings: τ=0.02 defines dangerous synthetic
cost; ε=0.05 bounds the desired accepted-decision unsafe frequency; δ=0.05 is the
family certification error budget; α=0.05 is the residual-bound miscoverage
level. Conflating these quantities changes the claimed guarantee.

### 2.2 Visual task and learning setup

The task is a top-down procedural image with two nonoverlapping patches,
assigned water and fragile-surface properties. Appearance families render water
with wave texture and fragile surfaces with crack texture while varying color,
floor appearance and illumination. Each continuous scene supplies two images
with the property assignments exchanged; shapes and candidate paths stay fixed.
This is a controlled visual-attribute task, not a natural-image hazard detector.
Four randomly drawn polylines share a start and goal on opposite sides of the
scene, with two interior waypoints each. Their swept-radius masks define the
spatial union to average over; repeated traversal of a pixel is not counted twice.
Candidates are generated independently of property values and are not guaranteed
to include a safe action.

| Component | Fixed implementation and available supervision |
|---|---|
| Image | 64×64 RGB, two visible property patches, rendered from the source-64 fields |
| Predictor | Randomly initialized ResNet-18; four lateral 1×1 projections to 64 channels, bilinear top-down fusion, 3×3 refinement and a two-channel 1×1 output |
| Output / target | Two 16×16 sigmoid fields; area-pooled binary property masks from the training source raster |
| Loss | Mean binary cross entropy with logits plus soft Dice loss, on spatial properties only |
| Training | 800 images (one variant per scene), 24 epochs, batch 20, 960 AdamW steps, learning rate .001 and weight decay .0001; final checkpoint |
| Augmentation | Saturation, polarity, contrast, per-channel gain, brightness and channel permutations; geometry unchanged |
| Action input | Four supplied continuous polylines with radius .05–.10 in normalized scene coordinates; no learned planner |
| Cost / supervision access | Public A/C/D formula; model learns property fields, without high-resolution action-cost labels or candidate-conditioned training |
| Evaluation reference | Direct mean of property × footprint at 512; each coarse evaluation uses the same reference-derived footprint |

The geometry experiment varies source-48 versus source-64 targets with identical
RGB. The new oracle decomposition uses the five fixed source-64 models. High-
resolution properties and oracle costs are privileged diagnostic information.

![One complete scene and its swapped property version](../../research/oracle_aggregation_20260912/artifacts/publication_v3/task_scene.png)

*Figure 1. The first IID discovery scene, fixed before result inspection, and
model seed 0. Left: the two rendered versions and the four candidate centerlines.
Middle: source64-to16 targets (cyan: water; magenta: fragile), coarse footprint
contours and dashed predicted .5 contours. Right: direct512 exposures and stored
model exposures for all candidates. Contours use a verified CPU replay; exposure
markers use the original stored predictions. This example is illustrative, not
selected for a favorable outcome.*

### 2.3 Guarantee objects

| Object | Score or event | What it protects |
|---|---|---|
| Entire matched scene pair | Maximum positive underestimation over two versions, three cards, four candidates | All 24 entries jointly |
| Current observation and card | Maximum over the four current candidates | Subsequent choice among those candidates |
| Fixed selected action | Positive underestimation of the actual selected candidate | A marginal selected-action upper bound |
| Accepted decisions | True danger among decisions accepted by a fixed rule | Conditional unsafe rate, given valid calibration assumptions |

A valid joint upper bound can protect a later choice among its candidates.
That flexibility has value. A smaller bound for a frozen selector has a narrower
purpose. Neither the failure of joint bounds to allow action nor the success of
selective risk certification is evidence that conformal prediction is incorrect.
Ordinary marginal coverage also does not automatically control conditional risk
after acceptance.

### 2.4 Evidence separation

The no-training audit uses five previously frozen independent-geometry models
and already consumed datasets; it is diagnostic. The geometry experiment uses
five new paired seeds in each of four conditions and fixed final checkpoints.
The acceptance confirmation then fixes all five independent-geometry/source-64
models before calibration. No best-performing seed is chosen. Fresh development,
calibration, and test scenes are disjoint, including an additional audit of
anonymous continuous geometry rather than scene IDs alone.

All experimental computation, checkpoint replays and figures run on Quest.
Source freezes and stage manifests preserve the distinction between prospective
comparisons and later presentation refinements. No new relation head, VLM
teacher, actor or critic is trained in this round.

## 3. How much risk change is a measurement change?

### 3.1 What separate pooling discards

For one property field P and footprint F, let a coarse cell B have weight w_B.
Write cell means as p_B and f_B. Direct exposure and separate-pooling exposure
use the same denominator D=∑_B w_B f_B:

\[
e_{\rm ref}=D^{-1}\sum_B w_B\overline{PF}_B,\qquad
e_{\rm pool}=D^{-1}\sum_B w_Bp_Bf_B.
\]

Subtracting gives the elementary identity

\[
e_{\rm pool}-e_{\rm ref}
=-D^{-1}\sum_B w_B\operatorname{Cov}_B(P,F).
\]

This is within-cell spatial covariance, not correlation across training scenes.
It vanishes when either binary property or footprint is constant in each cell.
With half-cell property and footprint occupancy, overlap and disjointness yield
true exposures 1 and 0 respectively, whereas separate means give .5 in both
cases. This illustration is not a frequency estimate or an impossibility result
for the restricted procedural generator. A model with additional geometric
information or useful shape priors might compensate for this error.

For a fixed candidate we distinguish direct512 exposure, separately pooled512
exposure, the actual source64-to16 supervision target under the common footprint,
and the learned prediction under that footprint. Their signed differences obey

\[
\hat e-e_{\rm ref}=(\hat e-e_{\rm target64})
+(e_{\rm target64}-e_{\rm pool512})
+(e_{\rm pool512}-e_{\rm ref}).
\]

The three terms are model error relative to supervision, source-raster difference,
and loss of within-cell spatial coincidence. We apply g separately at each level
before telescoping cost differences; for C, adding channel exposure errors is
not the nonlinear cost difference. Signed terms may cancel. Summing absolute
terms cannot attribute a percentage of the final error to each component.

The identity is basic mathematics. The empirical question is whether this loss
changes actual selection and acceptance under the normal generator and restored
outputs, beyond the reference's numerical sensitivity. Historical native/common
tables cannot answer it because they changed the footprints as well as fields.

### 3.2 Same-reference oracle decomposition on consumed data

We reconstruct the two already consumed E3 test banks (2,000 scenes each) and
use stored predictions from all five E2 independent/source-64 checkpoints. Each
scene retains its original deployed variant, A/C/D card and tie variate. No new
training, threshold tuning or recertification is performed. Every attribution
arm uses the same 512 footprint or its exact 32×32 block averages.

First we freeze each restored model's actual selected action and evaluate that
action at every level. Separately pooled512 oracle fields change its danger
label in 3.68% of model–scene decisions on IID and 3.83% on new appearance.
All observed changes at these selected actions are reference-safe to pool-danger.
Crossed model/scene discovery intervals are [2.89%,4.53%] and [3.03%,4.70%].
They preserve correlation across the five models; there are 2,000 independent
scenes per domain, not 10,000 independent trials.

The precision check holds the footprint fixed, rasterizes the properties at
1024 and pools them to 512 before integration. It changes the fixed-action
reference label in 0.05% of IID decisions and none on new appearance. Pooling
changes stable to this check remain 3.63% and 3.83%. This is a sensitivity check
for property discretization conditional on the fixed footprint, not an exact
continuum or footprint-accuracy guarantee.

Then each level chooses and accepts independently at .02; direct512 always
evaluates the chosen action's true cost. The coarse oracles still discard usable
opportunities after this permitted reselection:

| IID discovery: each level selects | Acceptance | Unsafe among accepted | Safe opportunity use η |
|---|---:|---:|---:|
| Direct512 oracle | 80.20% | 0% | 100% |
| Separately pooled512 oracle | 76.60% | 0% observed | 95.51% |
| Perfect source64 supervision-target oracle | 77.25% | 0.06% | 96.26% |
| Restored learned model | 78.58% | 1.07% | 96.93% |

The pooled oracle loses 3.60 percentage points of all-scene safe acceptance on
IID and 3.85 on new appearance. Perfect prediction of the actual supervision
target still loses 3.00 and 3.35 points. The learned model accepts more safe
opportunities but also more unsafe actions: this is not an unconditional
performance improvement over the oracle representation.

Signed cost terms on fixed selected actions are +.001807 (pooling), −.000653
(source raster), and −.000292 (model) on IID. On new appearance they are +.002060,
−.000714 and −.001453, giving mean total error −.000108 despite mean absolute
error .003102. Model and net measurement errors oppose one another in 31.52%
and 32.49% of decisions. These signed cancellations preclude interpreting
absolute component magnitudes as additive shares of decision error.

Before opening these discovery results, we define a small margin as pooled cost
within .005 of τ and boundary overlap as simultaneous fractional property and
footprint occupancy in a coarse cell relevant to the card. Among overlapping
selected actions, label-change rates are 37.84% versus 3.87% for small versus
larger margins on IID; on new appearance they are 37.45% versus 4.45%. No change
occurs without joint boundary overlap, as the covariance identity requires for
binary fields. That zero is an algebraic check, not an independent discovery.
The empirical concentration and conservative direction motivate a prospective
test, rather than a claim that the elementary identity is novel theory.

![Signed oracle decomposition and its decision consequences](../../research/oracle_aggregation_20260912/artifacts/publication_v3/oracle_discovery.png)

*Discovery on consumed E3 test data, using the five E2 models. Fixed-selection
label comparisons and each-level reselection metrics are distinct analyses.
Exact per-model counts and signed exposure/channel decompositions are retained
in the supplement.*

### 3.3 One prospective confirmation of the predicted decision effect

After discovery, we freeze a single new IID bank of 2,000 scenes, the five
existing source-64 checkpoints, the original candidate generator and tie rule,
and three predictions. There is no new training. Anonymous continuous shape
and complete-scene identities have no exact overlap with the 10,200 E2/E3 scenes.
This is new data under the same normal generator, not an external-task test.

The predictions require (i) a stable-reference lost-opportunity probability
above .01 for the pooled oracle; (ii) more than .90 of fixed-selected pooling
label changes to be conservative; and (iii) more than .10 enrichment of label
changes in small- versus larger-margin overlapping-boundary actions. Each
one-sided claim uses γ=.05/3 (distinct from residual miscoverage α). The first uses an exact binomial bound; the third
uses an approximate crossed model/scene bootstrap with five model seeds.

A recorded strengthening after job submission but before any fresh outcome was
inspected adds an exact scene-level check for direction: conditional on a scene
having any selected-action change across the five models, every change in that
scene must be conservative. This avoids treating a degenerate zero-counterexample
bootstrap as proof of a population fraction of one. The original protocol and
outputs remain available; Appendix C records the timing and different estimands.

| Prospective prediction | Fresh observation | One-sided lower bound | Frozen criterion |
|---|---|---:|---:|
| Stable-reference lost safe opportunities | 61/2,000 scenes | 2.29% | >1% |
| All selected-action changes conservative within a changed scene | 68/68 changed scenes | 94.16% | >90% |
| Small-minus-larger margin label-change rate | 40.57% − 3.26% = 37.31 pp | 26.06 pp | >10 pp |

The pooled oracle loses 63/2,000 safe opportunities before the precision
restriction (3.15 pp); perfect source64 supervision-target prediction loses
54/2,000 (2.70 pp). Direct512 accepts 1,592 safe scenes; the pooled oracle accepts
1,529 with no unsafe outcome observed. The target oracle accepts 1,540, of which
two are unsafe. These observed zero/two counts are not new risk certificates.
The learned restored model has empirical unsafe rate 1.03% and η=96.77% in this
new bank; the separately established E3 certificates are not recalibrated here.

At fixed restored-model actions, 322/10,000 model–scene decisions change label
under pooling (3.22%, descriptive crossed 95% interval [2.47%,4.02%]). The 10,000
decisions share only 2,000 scenes. All observed changes are conservative; 312
remain when the finer-property check agrees with the reference label. The same
check changes only 15 fixed-action labels (0.15%). Across all nontrivial candidate
entries its p95 cost difference is .000509, passing the frozen numerical screen.

![Discovery and prospective oracle confirmation](../../research/oracle_aggregation_20260912/artifacts/final_publication/oracle_confirmation.png)

*The last bank was generated after the hypothesis freeze. All arms use a common
512 footprint; A permits each oracle to select anew, whereas B–C hold each
restored model's selection fixed. C conditions on joint boundary overlap. Figure
intervals are descriptive; the table uses the registered one-sided comparisons.*

This supports the predicted mechanism in the normal procedural distribution.
It does not isolate a universal direction of pooling error: the algebra permits
both signs, and selection changes which spatial configurations are observed.
It also does not establish a general information-theoretic impossibility for
models receiving richer inputs. The practical conclusion is narrower: making
this coarse supervision target exact does not remove its downstream decision
error, so component fidelity alone is an inadequate diagnosis of the bottleneck.

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

## 6. External evidence boundary

An attempted real-image extension was stopped before mechanism evaluation
because the proposed geometric reference lacked independently validated error
bounds. It therefore supplies neither supporting nor refuting evidence for
cross-task transfer. The initial 24-frame interface screen and the subsequent
early prerequisite stop are documented in [Appendix A](appendix.md). The planned
new 48-frame set was never instantiated; this is not a 0/48 mechanism result.
The external extension is closed.

## 7. Related work and contribution boundary

Egocentric conformal prediction already explains how safety-irrelevant forecast
errors can produce conservative or immobilized navigation. It constructs
state-dependent scores and integrates them with MPC; its formulation assumes
accurate obstacle-position perception [1]. We therefore do not claim that
decision relevance or freezing from irrelevant errors is new. Our different
object is the attribution of behavior to spatial visual perception,
postprocessing, cost measurement, and the scope of a static acceptance rule.

Conformal Decision Theory directly controls decision loss, without first
constructing prediction sets, including online guarantees for average realized
loss under its conditions [2]. This differs from our IID certificate for
conditional unsafe frequency after acceptance. We claim neither the move from
prediction to decision nor a new general decision-calibration theory.

Learn then Test supplies a framework for selecting rules under simultaneous
risk control, including selective classification [3]. Our finite-bank
exact-binomial construction applies that established framework. The fact that
an already registered fixed rule suffices is evidence for simplifying the
pipeline, not an argument for renaming the procedure.

Perceive with Confidence studies learned perception combined with safe planning,
including the effect of planning on visited states and assurances in new
environments under its assumptions [4]. Our static synthetic certification
does not inherit these closed-loop results. Concept bottlenecks and shortcut
learning also predate our spatial decomposition and intervention-based
diagnosis [6,7].

Smart "Predict, then Optimize" explicitly distinguishes prediction error from
downstream decision error and develops a decision-aware loss and surrogate [8].
We do not claim this general distinction. Our narrower investigation tests a
specific spatial interface: separately pooled property and footprint fields can
lose the within-cell coincidence required by the published exposure calculation,
even when the supervised coarse property target is predicted perfectly.

The contribution is a same-reference oracle attribution that predicts and
confirms actual decision effects of a particular spatial measurement interface.
The earlier geometry, postprocessing and risk-control experiments locate this
effect among distinct learning and decision mechanisms. Neither the covariance
identity nor decision-aware evaluation is new; the evidence concerns the
measured effect and its operating conditions in the specified task. Its external
prevalence remains outside the established claim.

## 8. Limitations, reproducibility, and conclusion

The controlled task uses synthetic properties and a known cost law. Numerical
reference stability is empirical, and the finer-property comparison conditions
on the same footprint. Oracle boundary/margin features use privileged labels
and are not a deployable image-only detector. The direction check conditions on
five fixed models; crossed bootstrap inference has only five training seeds.
The four-candidate oracle is restricted,
and no static acceptance result measures physical task completion. The
geometry intervention changes multiple geometric factors. Diagnostic datasets
have been consumed; fresh tests supply confirmation only for their frozen
protocols. Statistical certificates depend on the stated distribution and
sampling assumptions. The external construction fails its initial observability
screen and supplies no confirmed cross-task mechanism result.

Reproduction details and independent numerical checks appear in
[Appendix B](appendix.md); the frozen experiment reports are retained.

In this controlled case, making a spatial supervision target exact does not
make the downstream exposure exact. A common-footprint oracle decomposition
identifies a separate-pooling effect that changes usable decisions, predicts
conservative boundary/margin failures, and survives one prospective confirmation.
Learning and source-raster errors can offset part of this measurement bias,
which makes aggregate accuracy an unreliable attribution tool. Earlier
experiments show that geometric learning improvements, fixed postprocessing and
the scope of risk protection also affect behavior. An imperfect component can
still be useful under an established risk-control procedure; that does not
validate every upstream measurement choice. Cross-task mechanism transfer
remains unestablished, and the external extension is closed.

## References

1. Jaeuk Shin, Jungjin Lee, and Insoon Yang. *Egocentric Conformal Prediction for Safe and Efficient Navigation in Dynamic Cluttered Environments.* 2025. [Primary paper](https://arxiv.org/html/2504.00447v1).
2. Jordan Lekeufack, Anastasios N. Angelopoulos, Andrea Bajcsy, Michael I. Jordan, and Jitendra Malik. *Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions.* ICRA, 2024. [Primary paper](https://arxiv.org/html/2310.05921v3).
3. Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* [Primary paper, Section 3.2](https://arxiv.org/html/2110.01052v5).
4. Anushri Dixit, Zhiting Mei, Meghan Booker, Mariko Storey-Matsutani, Allen Z. Ren, and Anirudha Majumdar. *Perceive With Confidence: Statistical Safety Assurances for Navigation with Learning-Based Perception.* [Primary paper](https://arxiv.org/html/2403.08185v2).
5. Peng Jiang, Philip Osteen, Maggie Wigness, and Srikanth Saripalli. *RELLIS-3D Dataset: Data, Benchmarks and Analysis.* [Paper](https://arxiv.org/abs/2011.12954), [official data and baseline implementation](https://github.com/unmannedlab/RELLIS-3D/tree/c17a118fcaed1559f03cc32cc3a91dedc557f8b8).
6. Pang Wei Koh et al. *Concept Bottleneck Models.* ICML, 2020. [Primary paper](https://proceedings.mlr.press/v119/koh20a.html).
7. Robert Geirhos et al. *Shortcut learning in deep neural networks.* Nature Machine Intelligence, 2020. [Primary article](https://www.nature.com/articles/s42256-020-00257-z).
8. Adam N. Elmachtoub and Paul Grigas. *Smart "Predict, then Optimize".* [Primary paper](https://arxiv.org/abs/1710.08005v5).

## Supplement and artifact entry points

- [New oracle study: protocols and reproduction](../../research/oracle_aggregation_20260912/README.md).
- [Discovery and prospective confirmation tables](../../research/oracle_aggregation_20260912/artifacts/final_publication/TABLES.md).
- [Supplementary scientific details](appendix.md).

- [Frozen three-experiment protocol and reproduction](../../research/pro_decision_round_20260912/README.md).
- [Detailed tables, certificates and supplementary figures](../../research/pro_decision_round_20260912/artifacts/publication/PAPER_ADDENDUM.md).
- [Machine-readable decision](../../research/pro_decision_round_20260912/artifacts/final_report/DECISION.json).
- [Independent verification](../../research/pro_decision_round_20260912/artifacts/final_report/VERIFICATION.json).
- [RELLIS-3D qualification protocol](../../research/rellis_external_20260912/qualification_protocol.json).
- [Bounded measurement-interface revision](../../research/rellis_revision_20260912/revision_protocol.json).
