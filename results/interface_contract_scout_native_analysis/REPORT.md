# Frozen 120-call scout report

Status: **COMPLETE — continuation criteria met; no expansion authorized by this report.**

## Execution record

- Models: `mistral-small-3.2-24b` and `qwen3.5-plus-02-15`
- Environments: `point_hazard_native` and `safety_gym_goal_native`
- Family: `direct_path_intersection`
- Paid seeds per model: exactly `20, 21, 22, 23, 24`
- Formal rows: 120 (60/model)
- Provider logical calls / attempts: 120 / 120
- Strict parse success: 120/120
- Provider spend: USD 0.7861499 (frozen observed-spend ceiling: USD 1.00)
- Native replay: 120 primary + 100 ambiguous-mapping executions
- Native replay provider calls / attempts: 0 / 0

The explicitly authorized main-key exception was used. The exact request allowlist,
fixed provider endpoints, disabled fallbacks, five-seed restriction, and local USD
1.00 observed-spend fuse remained active.

## Main contract metrics

| Metric | All calls | Parse-compliant |
|---|---:|---:|
| Parse consistency | 1.00 | 1.00 |
| Canonical semantic consistency | 0.70 | 0.70 |
| Grounding consistency | 0.35 | 0.35 |
| Planner-action IEC (`avoid/traverse/unknown`) | 0.75 | 0.75 |
| Native action-sequence IEC (exact hash) | 0.55 | 0.55 |
| Native trajectory IEC (exact hash) | 0.55 | 0.55 |
| CISR-EQ mean range (maximum) | 0.15 (1.00) | 0.15 (1.00) |
| CISR-MAP mean range (maximum) | 0.30 (1.00) | 0.30 (1.00) |

The native-action and trajectory IEC values compare exact controller outputs, not
only the canonical planner command. Nine of 20 equivalent pairs changed both the
native action sequence and trajectory.

## Model results

| Model | Parse | Semantic | Grounding | Planner IEC | Native-action IEC | Trajectory IEC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral | 1.00 | 0.60 | 0.30 | 0.60 | 0.60 | 0.60 | USD 0.0027933 |
| Qwen | 1.00 | 0.80 | 0.40 | 0.90 | 0.50 | 0.50 | USD 0.7833566 |

Qwen's high planner-action consistency did not survive grounding and native
control: its exact action and trajectory IEC were both 0.50.

## Native environment results

| Environment | Nonstationary | Task success | STC | Collision | Semantic violation | Native-action IEC |
|---|---:|---:|---:|---:|---:|---:|
| PointHazard native | 0.9667 | 0.9000 | 0.6833 | 0.0000 | 0.2167 | 0.50 |
| Safety-Gym native | 0.9833 | 0.9833 | 0.5333 | 0.4667 | 0.4500 | 0.60 |

Provider grounding was projected to native world coordinates without evaluator
truth in every execution. Mean (maximum) center calibration error was 1.2451
(3.1834) world units for PointHazard and 0.8597 (2.4516) for Safety-Gym. Mean
(maximum) radius error was 0.4182 (4.4800) and 0.9601 (2.3418), respectively.

All three `unknown` outputs produced task failure under the frozen rule.

## Equivalent-pair effects

| Pair | Semantic | Grounding | Planner IEC | Native-action IEC | Trajectory IEC |
|---|---:|---:|---:|---:|---:|
| constraint ↔ compatibility | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 |
| constraint polarity | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 |
| structured ↔ free text | 1.00 | 0.00 | 1.00 | 0.25 | 0.25 |
| field order | 0.75 | 0.25 | 1.00 | 1.00 | 1.00 |
| compatibility polarity | 0.75 | 0.75 | 0.75 | 1.00 | 1.00 |

The constraint/compatibility, constraint-polarity, and structured/free-text pairs
each changed native actions in both environments. For constraint/compatibility,
the mate-minus-anchor STC effect was -0.50 in both environments, giving a clear
cross-environment directionally consistent effect.

## Five-stage analysis

- Recognition accuracy: 0.4583
- Applicability accuracy: 0.6140
- Grounding accuracy: 0.0417
- Action-proposal accuracy: 0.5583
- Enforcement/outcome accuracy: 0.6083
- Executor rescue rate: 0.4717
- Taxonomy: 77 successful recovery, 40 multi-stage failure, 3 unattributable

Among 14 inconsistent equivalent pairs, the earliest difference was grounding in
7 (50.0%), normalization in 6 (42.9%), and planner interpretation in 1 (7.1%).

## Decision

All four frozen continuation signals are present:

1. Native physical-action IEC is below 0.90 (0.55).
2. Grounding consistency is materially below semantic consistency (0.35 vs 0.70).
3. CISR-EQ or CISR-MAP is at least 0.10 (both are).
4. The same equivalent pairs changed native actions in both environments.

Therefore the scout supports considering a second family, but this run does not
authorize it. A new subset authorization and budget freeze are required first.

## Cost and token caveat before expansion

Mistral used 19,398 prompt and 4,069 completion tokens with no reasoning tokens;
mean latency was 2.30 seconds. Qwen used 16,660 prompt tokens and 499,375 billed
completion tokens, of which 495,951 were reported as reasoning tokens; mean
latency was 156.46 seconds and maximum latency was 346.46 seconds.

The request parameter `max_tokens=220` limited visible output but did not bound the
reported Qwen reasoning-token usage. At the observed cost, the remaining USD 0.21385
under the current USD 1.00 ceiling is insufficient for another comparable Qwen
family. Freeze an explicit reasoning budget/disablement policy and a new maximum
spend before any expansion.
