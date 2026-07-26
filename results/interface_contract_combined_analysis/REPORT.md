# Interface-conditioned embodied safety: combined five-seed assessment

- Status: exploratory go/no-go evidence; not a formal paper result
- Valid models: google/gemini-2.5-flash-lite, mistralai/mistral-small-3.2-24b-instruct, qwen/qwen3-vl-30b-a3b-instruct
- Tasks: semantic route decision and candidate waypoint selection
- Environments: PointHazard and Safety-Gym headless contract render
- Seeds per model: 5 (`20–24`)
- Valid rows: 330
- Total calls including retained invalid GPT arm: 440
- Total spend: $0.051349

## Verdict

**Promising method and phenomenon, but the ICLR main result is not recovered yet.**

The strongest current result is not that every contract perturbation breaks every
model. It is that physically identical embodied-safety evaluations have a
measurable *interface-conditioned uncertainty envelope*: equivalent output
contracts, marker identities, candidate serialization, and downstream label
mappings can change actions, physical choices, STC, and one pairwise model
ranking. The effect is heterogeneous across model and renderer, which is itself
important but fails the preregistered universal two-environment contract gate.

## Exact paired evidence

| Intervention | Consistent / total | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| field_order | 23 / 30 | 0.767 | [0.591, 0.882] |
| structured_vs_free_text | 21 / 30 | 0.700 | [0.521, 0.833] |
| constraint_label_polarity | 27 / 30 | 0.900 | [0.744, 0.965] |
| compatibility_label_polarity | 28 / 30 | 0.933 | [0.787, 0.982] |
| constraint_vs_compatibility_wording | 29 / 30 | 0.967 | [0.833, 0.994] |
| explicit_vs_ambiguous | 29 / 30 | 0.967 | [0.833, 0.994] |

| Waypoint intervention | Same physical choice / total | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| candidate_baseline | 30 / 30 | 1.000 | [0.886, 1.000] |
| marker_id_permutation | 23 / 30 | 0.767 | [0.591, 0.882] |
| candidate_order_permutation | 18 / 30 | 0.600 | [0.423, 0.754] |


## Downstream planner mapping

For the same ambiguous-applicability model outputs replayed through executable
controllers, aggregate STC changes from
`0.500` when
`applicable` is interpreted as terrain-compatible, to
`0.800` when the explicit action is
authoritative. The constraint interpretation reaches
`0.700`. PointHazard uses
the repository fixed CEM-MPC; the second arm uses deterministic Safety-Gym
six-value headless dynamics. `unknown` is an explicit no-op task failure.

## Ranking stability

Pairwise reversals: `[["google/gemini-2.5-flash-lite", "qwen/qwen3-vl-30b-a3b-instruct"]]`.
Mistral is stable across all matched contract pairs; the observed reversal is
between Gemini Flash-Lite and Qwen3-VL. This supports a ranking-instability
claim for the evaluated pair, not a claim that every leaderboard must reverse.

## Invalid arm handling

GPT-5-mini is excluded from capability ranking: only
`8/110` rows parsed,
with `87` empty contents. The raw arm is
retained as an API/request-interface compatibility failure. Its replacement
changed only model identity; all 110 prompt and image hashes match the parent
matrix.

## Required next gate

1. Replace the headless Safety-Gym render with native RGB and actual dynamics.
2. Execute candidate selections, not only route labels, through the same
   controller; retain the explicit `unknown → stop` policy.
3. Add one manipulation/contact task in a genuinely independent environment.
4. Freeze a larger formal split only after the five-seed effect survives those
   bridges; do not expand the terminated marker-only story by itself.
5. Report contract envelopes and rank intervals, not a single privileged score.
