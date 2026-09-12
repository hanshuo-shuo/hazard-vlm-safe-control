# Compositional Visual Risk Under Property Interventions

Separating spatial recognition from capability–rule composition

Research working paper · Internal draft 0.1 · 12 September 2026

## Abstract

A robot's semantic risk depends jointly on what is visible, where it will move, what it can tolerate, and which task rules apply. Predicting a spatial property field and composing motion exposure with capability and rule variables makes these dependencies explicit. However, strong segmentation and decision scores on conventional splits do not establish that the visual model has learned the intended properties. We audit a two-property simulator benchmark using matched visual interventions and a missing equal-perception control: the public semantic-cost equation applied directly to the same predicted exposures as the learned relation models. Across three newly trained visual models, changing property identities while holding candidate trajectories fixed reduces field mIoU from {{base_iou_original}} to {{base_iou_swapped}} on an appearance-shift probe. None of the three models simultaneously selects the correct original and intervened action on the eligible changed-optimum pairs. Replacing half of the training and validation images with property-swapped versions, without increasing their counts or changing the architecture or epoch limit, raises paired decision accuracy to {{repair_pair}}. {{abstract_formula}} These are simulator-supervised, post-development diagnostics; the data repair was chosen after observing an initial failure. A separate frozen vision-language teacher produced no positive property labels on 126 developer images. The results identify a concrete spatial shortcut, demonstrate a practical data repair, and delimit what the present evidence supports about learned composition and offline visual teaching.

## 1. Introduction

Consider the same path through water for two robots: water exposure can be costly for one and acceptable for a waterproof robot. A fragile surface introduces a different dependency: a task rule may prohibit crossing it even when the robot is physically capable. An image-level hazard label conflates these relations and cannot distinguish entering a region from merely observing it. A useful interface therefore separates local properties, motion exposure, and capability- and rule-conditioned cost.

This decomposition also creates an evaluation obligation. A visual field can appear accurate when property identity is correlated with location. A relation network can appear effective when the benchmark's candidate generator assigns a nearly fixed optimal candidate role to each card. Neither observation alone demonstrates correct reasoning under an intervention that changes the relevant visual property while preserving the action geometry.

We investigate these issues within the C³-Safe research project. Our current experiments isolate its spatial-field and composition components. They do not evaluate the planned semantic and physical critics, SAC actor, or a deployed safety policy. The contribution of this working paper is a reproducible diagnostic result: a matched property-intervention protocol, an equal-perception analytic-composition control, and a controlled repair of a location-confounded training distribution. We make no claim that a factorized architecture, a concept intermediate representation, or an extra counterfactual loss is novel or already necessary.

<!-- PAGE -->

## 2. Spatial properties, exposure, and composition

Let an RGB observation be I and its predicted spatial field be Rθ(I) ∈ [0,1]^(2×H×W), with water and fragile-surface channels. Each candidate trajectory a has a projected, nonnegative footprint mask Fₐ. In the legacy protocol studied here, channel j exposure is a weighted mean:

eⱼ(a,I) = Σₚ Fₐ(p) Rθ,j(I,p) / Σₚ Fₐ(p).

The 48×48 simulator field and footprint are area-downsampled to 16×16 before this pooling. The evaluation target uses this same low-resolution convention. It is not the 48×48 contact measurement used to construct candidate sets. RGB input is 64×64, and candidates are not drawn into the model input.

A binary capability κ denotes waterproofing, and a binary rule q denotes protection of fragile surfaces. The public benchmark cost is

g(e,κ,q) = 1 − [1 − (1−κ)e_water] [1 − q e_fragile].

Cards A, B, C, D correspond to (κ,q) = (0,0), (1,0), (0,1), (1,1). Card B has identically zero semantic cost and supplies no nontrivial ranking target. These are explicit synthetic semantics, not an empirically estimated probability of physical damage.

{{architecture_figure}}

The analytic control computes g from predicted exposure without a trained relation head. The learned factorized arms receive those exact same exposures and cards; the dense arm receives predicted fields, footprints, and cards. Existing relation checkpoints are frozen throughout both new experiments. An oracle-formula arm uses target exposures, and an oracle-learned arm uses target exposures with the frozen relation network. The latter may retain relation approximation error and must not be described as perfect ground truth.

### 2.1 What the decomposition guarantees

For fixed binary κ and q, each partial derivative of g with respect to exposure lies in [0,1]. Therefore |g(e,κ,q)−g(e′,κ,q)| ≤ ‖e−e′‖₁. If every candidate's predicted cost differs from its target by at most δ, selecting a minimum predicted-cost candidate incurs semantic regret at most 2δ. The same bound holds for uniform tie resolution by averaging over tied choices. These are algebraic properties of the synthetic cost, not safety guarantees for a physical robot.

A learned composition function h introduces an additional discrepancy |h(ê,κ,q)−g(ê,κ,q)| beyond exposure error. It may empirically compensate for perception bias, but that possibility must be measured against g(ê,κ,q). Beating a different perceptual baseline does not isolate the contribution of learning this relation.

### 2.2 Why matched visual interventions matter

For two images with identical candidate geometry and cards but different unique optimal actions, any deterministic policy with identical nonvisual inputs makes the same choice in both. It cannot solve both decisions. More generally, an unchanged distribution over actions has average correctness at most one half on such a pair. This motivates evaluating both decisions together, alongside ordinary single-scene regret. Candidate-order randomization alone cannot break a correlation between property identity and absolute position.

<!-- PAGE -->

## 3. Experimental protocol

We recovered the deployed Quest reference snapshot and verified all 15 protected file and relation-checkpoint hashes against its historical manifest. It contains the final channel-permutation training recipe, with SHA-256 beginning 838f49158b7c. The historical best visual checkpoints were not exported; our three new models are consequently not presented as reproductions of those runs. We preserve the recovered files and execute the diagnostic in an isolated namespace.

{{protocol_table}}

### 3.1 Models and training

The visual predictor is a randomly initialized ResNet-18 with a skip decoder and 11,275,266 trainable parameters. It predicts two 16×16 logit maps. Training uses pixelwise binary cross-entropy plus soft Dice, AdamW with learning rate 0.001 and weight decay 0.0001, batches of 20, and at most 24 epochs with validation-loss early stopping after six stale epochs. The captured recipe includes saturation, contrast, brightness, channel-gain, luminance-polarity and RGB-channel-permutation augmentations. It subtracts 2.0 from the final output bias after selecting the validation checkpoint. This calibration rule is inherited and fixed, not selected on our new probes. Runs use PyTorch 2.5.1 with CUDA 12.4 on A100 40 GB or 80 GB GPUs; hardware variants are recorded and no runtime-superiority claim is made.

### 3.2 Original audit and interventions

The original generator places water near x = −0.34 and fragile terrain near x = +0.34, with small jitter. Its high-resolution candidate audit fixes the optimal role for cards A, C, and D. We generate 200 new scene pairs for each of three appearance conditions. In each pair we swap the two physical property channels before rendering, while keeping trajectories, footprint masks, render seed, appearance family and cards fixed. Thus the pixels and target property channels change together. The same random candidate permutation is applied to both members. These are new scene seeds from already studied procedural families, not an independently administered formal holdout.

The no-RGB spatial-prior control pools the original training mean field along each supplied footprint and applies g. A separate role-metadata diagnostic uses mean training risk by generator role and card. Role metadata is privileged diagnostic information; we do not assert that the actual learned methods received it. A cards-only baseline predicts the same per-card cost for all candidates.

### 3.3 Adaptive data repair

After observing failure in the first audit seed, we froze a second protocol. It swaps property channels in exactly 400 of the 800 training scenes and 50 of the 100 validation scenes before rerendering. Selection uses fixed seeds; image counts, geometry identities, model seeds, model code and maximum training budget remain unchanged. Both training and validation distributions are changed, so the experiment estimates the effect of this combined data repair. Test probes are reused and explicitly considered consumed. Fixed no-RGB controls continue to use the original reference training distribution. No new counterfactual loss is introduced.

### 3.4 Metrics and uncertainty

Regret is target cost of the selected candidate minus the lowest target cost, averaged over cards and scenes. Predicted ties are resolved uniformly in expectation. Pair ranking gives half credit to predicted ties. False-safe is the fraction of entries whose target cost exceeds 0.02 but predicted cost is at most 0.02; we preserve both scene-macro rates and eligible-entry counts. Dangerous selection uses excess target cost greater than 0.01. The joint condition additionally reports card D alone.

For paired correctness we retain cards with different unique target optima across the two images and compute the probability that both choices are correct under independent uniform tie resolution. Each probe contains 200 scene pairs and 400 eligible changed-card pairs. Confidence intervals resample scene pairs 10,000 times after averaging the three training seeds, preserving the original/intervened block. They describe scene uncertainty conditional on these three seeds, not the full uncertainty of training a new model. Secondary comparisons are descriptive and are not corrected for multiplicity.

<!-- PAGE -->

## 4. Results: conventional visual scores hide a location shortcut

{{audit_results_text}}

{{audit_table}}

The exposure and cost modules use the same footprints in each image pair, and candidate order is randomized consistently. Consequently, the large intervention degradation cannot be attributed to a different action set or a convenient tie-breaking order. It is consistent with reliance on the fixed relationship between terrain identity and location. On original appearance scenes the no-RGB spatial prior has regret 0.011207, compared with 0.026359 for cards only; its swapped regret rises to 0.027526. It is weaker than the neural predictor on original scenes, but illustrates that conventional testing rewards useful location information even without RGB.

{{intervention_figure}}

The role diagnostic is an additional warning about the candidate generator. Its success would demonstrate that generator metadata can be sufficient for decision selection; it would not establish that such metadata leaked into the neural model. Our main evidence for visual failure comes from the same trained models evaluated on matched pixels and targets, with their legitimate interfaces unchanged.

### 4.1 Equal-perception composition

{{composition_results_text}}

{{composition_table}}

All these arms receive either identical predicted exposures or, for the dense model, the same predicted field and footprints. The analytic arm therefore controls for the quality of the upstream visual representation. Its comparison with a learned relation is specific to this public synthetic equation. A different task with unknown interactions would require a new argument and experiment; it would not justify omitting this baseline here.

<!-- PAGE -->

## 5. Results: balancing property locations repairs much of the failure

{{repair_results_text}}

{{repair_table}}

{{repair_figure}}

The repair changes the location–property association while leaving the candidate geometry inventory fixed. Its success supports the interpretation that this association was a major source of the original failure. It does not isolate every feature used by the repaired network: texture, region shape, local context and other procedural cues remain possible signals. Nor does it prove that a new counterfactual objective is required, because the improvement here comes from matched changes to input images and spatial targets using the existing loss.

{{repair_composition_text}}

The repair is useful progress toward a field predictor that responds to visual properties. It remains an adaptive diagnostic, chosen after seeing a failure, and must be evaluated on separately frozen scene families before being promoted to a main empirical claim. In particular, balanced left/right assignments do not establish robustness to arbitrary region locations, unseen shapes, occlusion, perspective projection or real surfaces.

<!-- PAGE -->

## 6. Historical evidence and the offline-teacher boundary

### 6.1 Recovered development results

We separately re-executed the unmodified historical comparison rule on the saved per-scene arrays. The calibrated-polarity and cosine-schedule confirm runs improve regret over the historical baseline but fail its irrelevant-property invariance guard. The runtime-margin confirm run satisfies the recorded development KEEP rule. A KEEP decision is a selection within a repeatedly used development distribution, not an approval to claim generalization or safe control.

{{history_table}}

These numbers explain why the earlier branch appeared mature: its ordinary appearance-shift results are much stronger than its baseline. However, they do not substitute for the new property-swap test, and we do not retroactively infer the swapped performance of unavailable historical visual checkpoints. The copied reference recipe and the three fresh training seeds are identified separately throughout this paper.

### 6.2 Frozen vision-language teacher diagnostic

A separate C³-Safe developer run used Qwen3-VL-8B-Instruct at frozen revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b. The model received only RGB and an action-free spatial-property prompt. All 126 raw responses were retained. None contained a positive property region; all reported unknown regions. The original parser rejected an optional confidence field, and a subsequent offline compatibility repair accepted the responses without adding any property labels or rerunning the teacher.

Unknown regions covered enough motion projections that only 39 of 147 motion samples remained valid. Of 16 high-exposure samples, 15 were excluded as unknown; the remaining sample had predicted zero exposure. Reported zero error on a tiny valid subset therefore cannot establish successful teaching. We retain coverage, eligible counts and raw response status rather than converting unknown regions to negative supervision. This motivates coverage-aware evaluation, related to the risk–coverage distinction in selective prediction [7].

Five additional image-ingestion probes correctly distinguished flipped red/blue halves and produced scene-dependent descriptions with distinct processed-image hashes. This supports that images reached the model, but does not establish accurate spatial grounding. The negative finding is limited to the tested renderer, prompt, model and developer images. It is not a general claim that vision-language models cannot label terrain.

### 6.3 Protocol separation

The teacher diagnostic and local foundation use three property channels, high-quantile motion exposure and a different capability/rule cost equation. The present composition audit uses two channels, mean exposure and the union-style equation in Section 2. We do not pool their numerical errors, describe simulator labels as teacher labels, or attach the teacher to the six visual models evaluated here. No critic or actor was trained in these experiments.

<!-- PAGE -->

## 7. Related work and scope of the claim

Concept bottleneck models predict intermediate concepts that support interventions before the final prediction [1]. Our spatial property field shares this motivation, while pooling along a candidate footprint makes the downstream input depend on motion. The intermediate representation alone is not a novelty claim. Geirhos et al. [2] describe decision rules that perform well on standard tests but fail under distribution changes. Our matched property swap supplies a concrete diagnostic of such behavior in a compositional risk pipeline.

Vision-language safety signals also predate this study. VLM-SAFE combines semantic guidance with imagined trajectories and actor–critic learning for driving [3]. PROCO uses language-grounded costs and learned dynamics to synthesize unsafe examples from largely safe offline data [4]. Tetteh and Fleming integrate frozen VLM signals into an anticipatory constrained-RL update [5]. These approaches address policy learning and temporal risk. Our present measurements concern spatial recognition and one-step candidate evaluation; they neither reproduce nor compare closed-loop performance against those methods.

Safety-Gymnasium provides a broader safe-RL environment and algorithm ecosystem [6]. Moving to that scale requires distinguishing its native physical cost from the additional semantic rule costs investigated here. Our procedural decision task is a component diagnostic, not a replacement for such environment-level evaluation. We report fixed-seed scene intervals and individual training seeds because few-run aggregate estimates can be misleading, as emphasized by Agarwal et al. [8].

## 8. Limitations and next validation

The current evidence is narrow in five ways. First, the two terrain properties, four cards and procedural palettes are simple synthetic semantics. Second, the analytic formula is public and exact for this task; a learned head needs evidence of benefit under equal perception before being treated as essential. Third, the repaired training data and validation data both change, and the repair is chosen adaptively. Fourth, scene bootstrap intervals condition on only three training seeds. Fifth, the true offline teacher produced no usable positive spatial supervision, and the desired actor with separately constrained semantic and physical critics remains unevaluated.

The next validation should freeze a new generator that randomizes property identity, absolute position, region shape and candidate ordering independently, and verify unique-optimum changes from target exposure rather than candidate names. A confirmatory run should then compare the captured and balanced recipes at a predeclared training budget on untouched families, with the analytic composition control and explicit risk–coverage reporting. Human review of a bounded teacher-label qualification set remains necessary before interpreting VLM-derived fields as supervision. Closed-loop claims additionally require action-generation experiments, native-cost accounting, task success, latency and appropriate baselines under the same observation permissions.

## 9. Conclusion

{{conclusion_text}}

## Reproducibility and artifact status

The two new protocols, isolated runner, preserved reference hashes, all six exported visual checkpoints, exact predictions, target fields, candidate footprints and training histories are retained. The local aggregation verifies checkpoint and artifact hashes and identical probe fingerprints across all six models. Source and summary artifacts are in the repository; large checkpoints and arrays are kept in the result directory and Quest project storage. Full local and remote paths and replay commands are documented in the accompanying artifact README. Historical files are not overwritten. This draft reports development and diagnostic evidence; no result has been promoted to the project's formal VALIDATED status.

<!-- PAGE -->

## References

[1] Pang Wei Koh et al. Concept Bottleneck Models. ICML, PMLR 119:5338–5348, 2020. https://proceedings.mlr.press/v119/koh20a.html

[2] Robert Geirhos et al. Shortcut learning in deep neural networks. Nature Machine Intelligence 2:665–673, 2020. https://doi.org/10.1038/s42256-020-00257-z

[3] Yansong Qu et al. VLM-SAFE: Vision-Language Model-Guided Safety-Aware Reinforcement Learning with World Models for Autonomous Driving. arXiv:2505.16377v2, 2026; first version 2025. https://arxiv.org/abs/2505.16377v2

[4] Ruiqi Xue et al. Model-Based Proactive Cost Generation for Learning Safe Policies Offline with Limited Violation Data. arXiv:2605.01356, 2026. https://arxiv.org/abs/2605.01356

[5] Samuel Tetteh and Cody Fleming. Seeing Before Colliding: Anticipatory Safe RL with Frozen Vision-Language Models. arXiv:2606.11266, 2026. https://arxiv.org/abs/2606.11266

[6] Jiaming Ji et al. Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark. NeurIPS Datasets and Benchmarks, 2023. https://arxiv.org/abs/2310.12567

[7] Yonatan Geifman and Ran El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS, 2017. https://arxiv.org/abs/1705.08500

[8] Rishabh Agarwal et al. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 34, 2021. https://arxiv.org/abs/2108.13264

### Artifact identifiers

{{artifact_table}}

### Interpretation checklist

The six new visual models use simulator supervision. The repaired data are a response to an observed failure. Historical confirm means a development tier. The 126 teacher responses contain no positive property annotations. Mean-exposure costs and q95 C³-Safe costs are different protocols. A correct candidate selection is not a measured closed-loop safety guarantee.
