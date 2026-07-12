# Results

> **ARCHIVED / INVALIDATED AS PAPER EVIDENCE (reviewed 2026-07-11).** Numbers remain
> exactly as recorded for provenance. Do not cite them as clean benchmark results; see
> [`../../docs/RESULTS_REGISTRY.md`](../../docs/RESULTS_REGISTRY.md).

> **Read this with [`LEAKAGE_AND_LIMITATIONS.md`](LEAKAGE_AND_LIMITATIONS.md).** The strongest
> configurations had safety information leaking into the prompt; the numbers below are directional
> evidence for a *design principle*, not a clean benchmark. They are reported as-run, including the
> deflated post-audit figures.

---

## 1. Prompt-information ablation

Matched model, seeds, and episode budget; only the prompt design changes. This isolates *what kind
of physical information* the VLM needs.

| Variant | Control signal | Success | Hazard hit | Timeout | Return | Reading |
|---|---|---:|---:|---:|---:|---|
| Safety-filtered primitive PIVOT | "which actions definitely won't work" | 4/10 | 0/10 | 6/10 | 37.7 | Safe but stalls; many timeouts |
| Momentum / rollout-rendered PIVOT | each action's dynamics consequence | 5/10 | 0/10 | 5/10 | 48.0 | Better progress, still safe, lower clearance |
| Self-iterating PIVOT | last-3-step history, no future rollout | 5/10 | 5/10 | 0/10 | 24.5 | Efficient and aggressive, but **unsafe** |
| Frontier VLM + physics trajectory | sees the physical trajectory | 10/10 | 0/10 | 0/10 | — | Upper bound: with explicit physics, solvable |

**Takeaway.** The three ablations cleanly separate "safety knowledge" (*which actions are
forbidden?*), "dynamics knowledge" (*what will this action cause?*), and "history" (*what happened
recently?*). The decisive ingredient is **current, counterfactual dynamics** — not constraints and
not retrospective feedback.

---

## 2. Learned-physics PIVOT (demo-free)

A small MLP learns one-step dynamics online and renders candidate rollouts. No oracle simulator, no
expert demonstrations.

| Variant | Model | Episodes | Success | Hazard hit | Timeout | Min clearance | VLM calls/ep |
|---|---|---:|---:|---:|---:|---:|---:|
| Plain PIVOT baseline (no physics) | Qwen3-32B | 100 | 28% | 72% | 6% | — | — |
| Learned-physics PIVOT | Qwen3-32B | 100 | **85%** | 14% | 1% | 0.52 | 22.6 |
| Learned-physics PIVOT | ~7B-scale | 200 | 74% | 26% | 0% | 0.31 | 46.0 |
| Learned-physics PIVOT + LoRA | 7B + teacher distill | 1000 | **87.8%** | 12.1% | 0.1% | 0.53 | 25.5 |

Plain PIVOT without physics collapses (28%, mostly hazard hits). Online learned rollouts recover most
of the frontier-model benefit **without** an oracle. The key systems point: even a small VLM becomes
useful once the controller externalizes the hard dynamics reasoning into a learned physics module.

### LoRA self-improvement (1000-episode run)

| Metric | First 100 eps | Last 100 eps |
|---|---:|---:|
| Success | 68.0% | 90.0% |
| Hazard hit | 32.0% | 10.0% |
| VLM–teacher agreement | 31.3% | 84.4% |

Predicted-vs-real correlation after warm-up: one-step progress ≈ **0.948**, next-step clearance ≈
**0.980** — the learned physics genuinely tracks the environment, and the small VLM becomes more
aligned with the score-function teacher over time.

---

## 3. Demo-free VLM control vs. fine-tuned VLA (same Qwen2-VL-7B backbone)

The advisor's question: *why not just convert the same backbone into a VLA?* Answer: you can, but it
is far more tuning- and data-sensitive, and it loses.

![main comparison](../assets/results_main_comparison.png)

| Variant | Train data | Eval | Success | Hazard hit | Timeout |
|---|---:|---|---:|---:|---:|
| Expert controller (data source) | 1000 eps | collection | 98.1% | 1.9% | 0.0% |
| Direct VLA + regression head | 1000 | heldout 1k | 36.9% | 60.4% | 2.7% |
| Direct VLA + regression head | 3000 | heldout 1k | 40.6% | 57.4% | 2.0% |
| Direct VLA + regression head | 5000 | heldout 1k | 36.0% | 62.8% | 1.2% |
| Direct VLA + regression + LoRA | 1000 | heldout 1k | 63.4% | 36.6% | 0.0% |
| Direct VLA + flow head | 1000 | heldout 1k | 10.0% | 67.5% | 22.5% |
| Direct VLA + flow head + LoRA | 1000 | heldout 1k | 66.1% | 33.9% | 0.0% |
| **Learned-physics PIVOT + LoRA** | **0 demos** | 1000 eps | **87.8%** | **12.1%** | 0.1% |

The expert data itself is strong (98%), so poor VLA performance is not bad demonstrations — it is the
difficulty of learning inertial dynamics + safety + action decoding in one network. The single
interpretable knob for the PIVOT method is **rollout horizon**: horizon-1 gives 60.1% / 39.9%,
horizon-3 gives 87.8% / 12.1%.

### Data scaling

![data scaling](../assets/results_data_scaling.png)

More expert demos barely move the VLA (1k→5k essentially flat). The hard part is dynamics, and
imitation data does not teach it cleanly. The demo-free method sits above the entire VLA curve.

### Full table

![full results table](../assets/results_table.png)

---

## Related work positioning

| Direction | Representative work | Relation to this project |
|---|---|---|
| Visual prompting / action selection | PIVOT, MOKA, RoboPoint | We extend "draw the action" → "draw the action's **consequence**" |
| LLM/VLM planner + affordance | SayCan, Inner Monologue, Code-as-Policies, VoxPoser | Same modular split: VLM = semantics, physics module = executability/safety |
| VLM + MPC / world model | VLM-MPC, Traj-VLMPC, SIMPACT | Closest line; our edge is a **lightweight, online** physics model rendered directly for the VLM |
| End-to-end VLA | RT-2, OpenVLA, Octo, π0 | Our VLA baselines show the data/tuning cost of the end-to-end route |
| Safe-control environments | Safety Gymnasium, ManiSkill3, Meta-World | PointHazard is a controlled diagnostic for inertial safety |

The sharpest one-line positioning: **online-learned local physics, with the counterfactual
consequence of each high-level choice rendered for the VLM**, aimed at *safety-critical, low-data,
no-full-simulator* control.
