# Next five experiments — pilot report

Generated 2026-07-24. These are five-seed go/no-go pilots, not formal
paper results. All provider conditions use the matched PointHazard scene seeds
`0–4`. The two VLMs are `google/gemini-2.5-flash-lite` and the open-weight
`qwen/qwen3-vl-30b-a3b-instruct`.

## Executive decision

**Terminate the marker-based PointHazard ICLR mainline.**

The predeclared marker/interface kill condition fired. Under marker-ID
permutation, only 20% of model/seed pairs preserved the same physical choice.
Candidate-coordinate order permutation preserved only 30%. This is too
interface-dependent to support a claim that the current result reflects stable
semantic spatial reasoning.

The most useful path forward is now the detector/geometry route plus a
marker-free interface, with PointHazard retained as a causal unit test and the
Safety-Gymnasium port continued independently.

## Findings

### 1. Core information audit

- P0/P1/P2 selected-safe rates were 50% / 60% / 40% across both models.
- No model rank reversal occurred: Qwen stayed above Gemini at all three levels.
- Safety was strongly conditional on correct recognition: 71.4% selected-safe
  when recognition was correct versus 3.6% when it was incorrect.
- Same-length irrelevant and wrong-class controls reduced selected-safe rate to
  30%. Geometry perturbations did not produce a monotonic response.
- P2 did not improve over P0. This rejects a simple “more privileged geometry
  monotonically improves behavior” story in this pilot.

See [experiment 1](01_core_information_audit/REPORT.md).

### 2. Marker/interface robustness

- Marker-ID permutation physical-choice consistency: 20%.
- Candidate-order consistency against the canonical coordinate interface: 30%.
- Direct-coordinate and unmarked-image controls preserved the original
  physical choice in 40% and 10% of cases, respectively.
- Safe-choice rates varied materially with candidate count: 4/8/16 candidates
  yielded 80% / 50% / 60%.

This triggers the predeclared `TERMINATE` decision.

See [experiment 2](02_marker_interface_robustness/REPORT.md).

### 3. Modern perception substitution

The open-weight Qwen detector found the water region in 5/5 scenes. Its native
0–1000 coordinates were converted through a frozen adapter to the shared world
disk schema; mean center error was 6.3 pixels.

With the same MPC planner/executor:

| Arm | Success | Semantic violation | STC |
|---|---:|---:|---:|
| Blind | 100% | 100% | 0% |
| Detector | 40% | 0% | 40% |
| VLM waypoint | 100% | 80% | 20% |
| Oracle | 80% | 0% | 80% |

The detector is safe but over-conservative at this operating point. It is the
strongest non-oracle direction, but needs geometry calibration or planner
weight tuning before scaling.

See [experiment 3](03_modern_perception_substitution/REPORT.md).

### 4. Five-stage outputs and twins

- Capability-twin applicability accuracy: 0%.
- Only 30% of capability twins changed routing.
- Appearance-twin applicability accuracy: 40%.
- Appearance recognition accuracy: 80%; routing changed in 80%.

Both models often recognized water yet continued to treat it as incompatible
for an amphibious robot. The bottleneck is therefore norm applicability, not
only recognition. This is the cleanest positive justification for retaining
the five-stage audit protocol.

See [experiment 4](04_five_stage_twins/REPORT.md).

### 5. Second environment

The real `SafetyPointGoal1-v0` dynamics ran headlessly for five matched
capability-twin seeds. Semantic geometry matched across twins and native cost
remained separate from semantic violation accounting. The minimal port is a
**GO**.

This is an integration go/no-go only; the current zero-action smoke policy is
not a performance baseline.

See [experiment 5](05_second_environment/REPORT.md).

## Reproducibility and cost

- Provider requests in the evidence set: 180.
- Total tokens: 87,636.
- Recorded OpenRouter cost: approximately USD 0.014.
- API keys and authorization headers are absent from artifacts.
- Exact request hashes, returned model IDs, usage and raw responses are stored
  in the row-level results and cache.
- Every experiment directory includes `results.json`, `REPORT.md`,
  `summary.png`, `example.png`, and `audit.gif`.
- Verification: 32 relevant tests passed; the GUI-dependent macOS render smoke
  was deselected because the real rollout uses a headless MuJoCo path.

The complete run manifest is [MANIFEST.json](MANIFEST.json). The reusable runner
is [`scripts/run_next_five_experiments.py`](../../scripts/run_next_five_experiments.py).

## Recommended next action

Do not increase seeds on the marker-based VLM waypoint line. Replace the main
interface with detector/segmenter-derived geometry or a marker-free coordinate
contract, calibrate the detector-to-planner geometry on a fresh dev split, then
run the same five-stage capability twins in Safety-Gymnasium.
