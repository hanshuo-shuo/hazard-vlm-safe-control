# From visual ranking to safe acceptance: a three-experiment decision round

Pro-directed follow-up, executed on Quest on 2026-09-12. This addendum tests measurement and deployment alignment; it proposes no new risk-control algorithm.

## Decision

Keep a measurement/postprocessing/selection-unit diagnostic study. No novel calibration or public-relation learning claim; require independent-task evidence before further method expansion.

The no-training audit's simple-postprocessing flag is **True**. The matched geometry claim passes its predeclared effect criterion: **True**. The IID risk/usefulness screening gate passes: **True**.

The raw restored-bias model is retained as a strong baseline. Any improvement supplied by known exact-binomial selective risk certification is an application of existing methods, not evidence of a new calibration mechanism. No VLM call, relation-head training, actor or critic training was performed.

## E1: oracle feasibility, postprocessing and calibration scope

Five previously frozen independent-geometry checkpoints are reevaluated without training. Inherited minus-two logits and restored-zero logits come from the same forward pass. We keep the old low16 costs and reconstruct direct 256/512 mean-exposure references from saved continuous shapes and paths. This remains spatial mean exposure, not contact or q95. The reference gate requires at most 1% dangerous-label disagreement and 95th-percentile cost difference at most 0.002 between 256 and 512. All diagnostic splits pass.

The old low16 versus direct-reference danger labels differ in roughly 2.6%-3.3% of entries. This measurement change must not be described as a model improvement or silently pooled with old scores. The exact oracle table enumerates both variants and all A/C/D cards; the acceptance table uses one preregistered deployment draw per scene, shared across models and postprocessing arms.

| Split | Oracle low16 | Oracle direct 512 | Direct-512 oracle margin >=0.005 |
|---|---:|---:|---:|
| random_iid | 76.17% | 79.72% | 77.61% |
| random_appearance | 76.33% | 80.78% | 78.39% |
| shape_appearance | 82.22% | 85.39% | 83.67% |

| Split | Postprocessing | Acceptance | Unsafe among accepted | Safe opportunity efficiency |
|---|---|---:|---:|---:|
| random_iid | inherited_minus2 | 80.23% | 1.99% | 98.70% |
| random_iid | restored_zero | 76.77% | 0.22% | 96.15% |
| random_appearance | inherited_minus2 | 85.03% | 4.70% | 99.63% |
| random_appearance | restored_zero | 80.97% | 1.73% | 97.83% |
| shape_appearance | inherited_minus2 | 88.17% | 3.82% | 99.38% |
| shape_appearance | restored_zero | 83.77% | 1.07% | 97.11% |

![Same weights, changed postprocessing](e1_postprocessing.png)

Nested residual calibration uses exactly 300 scene scores for each scope: the pair's 24 nontrivial entries, the current image/card's four candidates, or the fixed selected action. The current image/card is sampled once per independent scene; the smaller scopes do not gain sample size by flattening correlated entries. Separate scores are recomputed for each postprocessing and truth convention. Selected-action conformal bounds remain a diagnostic and do not automatically certify conditional unsafe rate.

| Restored model seed index | Pair q | Current four-candidate q | Selected-action q |
|---|---:|---:|---:|
| 0 | 0.032868 | 0.027670 | 0.007633 |
| 1 | 0.030539 | 0.024209 | 0.007062 |
| 2 | 0.027991 | 0.022183 | 0.005013 |
| 3 | 0.028454 | 0.024369 | 0.006618 |
| 4 | 0.027968 | 0.022005 | 0.007897 |

A one-pixel boundary-band oracle replacement is retained as a privileged mechanism diagnostic, with boundary/interior positive deficits in E1 raw records. It is not a deployment method. E1 uses consumed scenes and supplies diagnostic evidence only.

## E2: matched geometry by source-raster experiment

The 2x2 design uses balanced-anchor versus independent geometry and 48 versus 64 source supervision. Each continuous scene, property assignment, path and 64x64 RGB is fixed before source rasterization. Geometry acceptance uses continuous bounding-circle/in-frame checks, not minimum pixel counts or cost/optimum rejection. Both schemes use anonymous independent paths. Exactly half the train/validation property assignments are swapped. Each model trains on variant zero only: 800 images, 24 epochs, batches of 20, exactly 960 optimizer updates, and the final checkpoint. Validation is diagnostic and never selects a checkpoint. All five seeds are paired; 20 models are retained.

Common scoring uses the same 512-derived query footprints for every arm and direct-512 mean-exposure truth. Native scoring uses each source raster's pooled masks and footprints. The main geometry effect must exceed 0.001 regret reduction at both source rasters, using crossed seed/scene 95% intervals. Raster equivalence requires the entire interval inside +/-0.0005; failure to reject zero is not equivalence.

| Geometry/source | Common-reference appearance regret | Native-reference appearance regret | Common-reference false-safe |
|---|---:|---:|---:|
| balanced_anchor_48 | 0.005154 | 0.004972 | 18.25% |
| balanced_anchor_64 | 0.008503 | 0.008207 | 23.38% |
| independent_48 | 0.000285 | 0.000261 | 2.43% |
| independent_64 | 0.000254 | 0.000228 | 1.87% |

| Appearance contrast | Mean regret difference | Crossed 95% interval |
|---|---:|---|
| geometry_at_48 | 0.004870 | [0.003101, 0.006853] |
| geometry_at_64 | 0.008249 | [0.002850, 0.017690] |
| raster_48_minus_64_balanced_anchor | -0.003349 | [-0.012681, 0.003019] |
| raster_48_minus_64_independent | 0.000031 | [-0.000033, 0.000094] |
| interaction | -0.003380 | [-0.012946, 0.003022] |

![Matched factorial](../final_report/e2_factorial.png)

The geometry scheme still jointly changes position, size, rotation and shape. This matched experiment removes the RGB/source-raster coupling; it does not isolate position as a single causal factor. The balanced-anchor generator is a continuous legacy-like layout with anonymous paths, not an exact rerun of the earlier raster-rejection generator.

## E3: fixed-selector risk certification

The five independent/source-64 final checkpoints are chosen a priori. Restored-zero formula scores select an action first; stored independent uniforms resolve ties at 1e-9. Acceptance follows selection, and rejection never triggers reselection. Each independent certification scene draws one of two image variants and one of A/C/D, supplying one Bernoulli outcome. The guarantee concerns that mixture, not each card or appearance subgroup.

Danger remains true mean-exposure cost >tau=0.02. Epsilon=0.05 is the maximum conditional unsafe rate; delta=0.05 is the family certification failure probability; alpha=0.05 is used separately for residual upper bounds. The development, 1,600-scene calibration and 2,000-scene test banks are disjoint. A finite 12-threshold acceptance bank is frozen before calibration. Exact one-sided binomial bounds use delta/(5 models x 2 calibration domains x 12 rules)=0.05/120. The highest-coverage certified rule is selected; an empty valid set rejects all and is not certified. Certificates are persisted before opening final tests.

This finite-bank selective-risk construction applies the exact-binomial and multiple-testing ideas in [Learn then Test, Section 3.2](https://arxiv.org/html/2110.01052v5). It is not a new algorithm. A source-certified rule tested on new appearance is a shift stress test. Independently labeled target calibration is explicit adaptation. Neither provides a guarantee on unknown future shifts.

| Scenario | Arm | Acceptance | Unsafe among accepted | Eta | Eta crossed 95% interval |
|---|---|---:|---:|---:|---|
| iid__source | raw_minus2 | 81.87% | 2.85% | 99.18% | [0.9860, 0.9961] |
| iid__source | raw_restored | 78.58% | 1.07% | 96.93% | [0.9593, 0.9782] |
| iid__source | pair_conformal_restored | 0.00% | undefined | 0.00% | [0.0000, 0.0000] |
| iid__source | selective_certified | 81.29% | 2.51% | 98.82% | [0.9796, 0.9947] |
| iid__source | oracle | 80.20% | 0.00% | 100.00% | [1.0000, 1.0000] |
| target__source | raw_minus2 | 84.26% | 3.98% | 99.15% | [0.9862, 0.9958] |
| target__source | raw_restored | 80.31% | 1.54% | 96.90% | [0.9586, 0.9783] |
| target__source | pair_conformal_restored | 0.00% | undefined | 0.00% | [0.0000, 0.0000] |
| target__source | selective_certified | 83.46% | 3.58% | 98.62% | [0.9762, 0.9939] |
| target__source | oracle | 81.60% | 0.00% | 100.00% | [1.0000, 1.0000] |
| target__target_adapted | raw_minus2 | 84.26% | 3.98% | 99.15% | [0.9862, 0.9958] |
| target__target_adapted | raw_restored | 80.31% | 1.54% | 96.90% | [0.9586, 0.9783] |
| target__target_adapted | pair_conformal_restored | 0.00% | undefined | 0.00% | [0.0000, 0.0000] |
| target__target_adapted | selective_certified | 81.05% | 2.00% | 97.34% | [0.9620, 0.9836] |
| target__target_adapted | oracle | 81.60% | 0.00% | 100.00% | [1.0000, 1.0000] |

![Risk and safe acceptance](e3_acceptance.png)

Eta counts accepted AND actually safe decisions divided by oracle-safe opportunities; unsafe accepts never contribute. Zero coverage has undefined conditional risk. Crossed intervals resample model seeds and independent scenes, conditional on the fixed calibration banks. Exact calibration certificates, rather than a potentially degenerate zero-error bootstrap interval, control conditional risk. Pointwise exact-binomial test upper bounds and raw counts are retained in DECISION.json.

| Seed | Calibration domain | Chosen threshold | Accepted calibration samples | Unsafe | Simultaneous upper bound |
|---|---|---:|---:|---:|---:|
| 20261117 | iid | 0.030 | 1301 | 21 | 0.0314 |
| 20261117 | target | 0.020 | 1245 | 16 | 0.0273 |
| 20261129 | iid | 0.020 | 1282 | 15 | 0.0254 |
| 20261129 | target | 0.020 | 1271 | 30 | 0.0414 |
| 20261143 | iid | 0.030 | 1324 | 40 | 0.0492 |
| 20261143 | target | 0.020 | 1275 | 27 | 0.0382 |
| 20261157 | iid | 0.030 | 1321 | 37 | 0.0465 |
| 20261157 | target | 0.020 | 1275 | 28 | 0.0392 |
| 20261171 | iid | 0.030 | 1310 | 30 | 0.0401 |
| 20261171 | target | 0.030 | 1304 | 40 | 0.0500 |

## The predeclared fixed baseline already suffices

The restored-bias rule at the original fixed acceptance threshold 0.02 is already
one of the 12 registered candidates. All five source certificates and all five
explicit target-adaptation certificates mark this fixed rule valid under the
same 120-hypothesis correction. Its IID test unsafe rate is 1.07%, with eta
96.93% (crossed 95% interval 95.93%-97.82%). The selected-rule method has eta
98.82% and unsafe rate 2.51%; its extra acceptance uses more of the allowed
risk budget. Threshold search is therefore not necessary to pass this round's
screen. This observation reads existing predeclared-control certificates; it
does not introduce a new threshold, certification family or test evaluation.
It strengthens the decision to report a simple repair and existing-method
application instead of claiming a new safety algorithm.

## Reproducibility and next decision

All three source freezes, 20 new checkpoints, five old checkpoint identities, dataset/certificate hashes and temporal load records are preserved under `/projects/p33100/siosio/hazard_pro_decision_round_20260912`. Independent verification checked 257 manifest-listed files and 80 CPU replay images across 20 new models. It reconstructed scalar choices, mean exposures, regrets, false-safe/acceptance counts and certificate p-values. GPU predictions remain authoritative. Numerical reference precision is an empirical raster-convergence check, not an exact continuous integration theorem.

The screening thresholds are research decisions, not robot safety standards. Static certification says nothing by itself about episode risk, task completion, dynamics, occlusion or physical damage. The planned round ends here: no additional test-set tuning, grid/checkpoint search, relation/VLM/RL expansion or claimed novel calibration. If pursued as a paper, the defensible scope is measurement and decision-object alignment with simple postprocessing controls; independent-task or physical-outcome evidence is still required for broader claims.
