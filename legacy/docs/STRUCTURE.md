# File-by-file map

> **ARCHIVED (reviewed 2026-07-11).** This map describes only the historical files in
> `legacy/`; it is not the current repository structure. See
> [`../../STRUCTURE.md`](../../STRUCTURE.md) for the live map.

Everything in this project sits in one flat directory (all imports are top-level), grouped here by
role. Two backends recur: a **local Qwen2-VL** path and an **OpenRouter API** path (`*_openrouter.py`).

## Entry points — PIVOT family (VLM picks a numbered candidate)

| File | Role |
|---|---|
| `shared_autonomy_hazard_learned_physics_pivot.py` | ★ **Main method.** Online MLP learns one-step dynamics from real transitions, recursively renders each candidate's predicted trajectory, VLM chooses. Demo-free. |
| `shared_autonomy_hazard_dual_mlp_pivot_openrouter.py` | Two MLPs — a **PhysicsMLP** (trajectory) and a **SafetyMLP** (P(hazard hit)) — each independently toggleable. The safety head is exactly the kind of explicit signal the leakage audit flags. |
| `shared_autonomy_hazard_momentum_pivot.py` | Momentum-aware PIVOT: candidates rendered as short dynamics rollouts instead of static force arrows. |
| `shared_autonomy_hazard_primitive_pivot.py` | Safety-filtered motion-primitive PIVOT ("which actions are unsafe") — the constraint ablation. |
| `shared_autonomy_hazard_self_iterating_pivot.py` | History-only PIVOT: last-3-step action→result feedback, no future rollout — the history ablation. |
| `shared_autonomy_hazard_learned_physics_pivot_openrouter.py` | API version of the main method. |
| `shared_autonomy_hazard_primitive_pivot_openrouter.py` | API version of the primitive/safety-filtered variant. |
| `shared_autonomy_hazard_pivot.py` / `_pivot_openrouter.py` | Original plain PIVOT (force-arrow) baseline, local + API. |
| `shared_autonomy_hazard_openrouter.py` | API shared-autonomy entry point. |
| `pivot_primitive.py` | Library: builds safety-filtered motion-primitive candidates, rolls them out under the env dynamics, draws numbered trajectory candidates. |

## Entry points — VLM-guided control with a diffusion safety copilot

| File | Role |
|---|---|
| `shared_autonomy_hazard.py` | VLM emits a **force intent**; a conditional DDPM **safety prior** (knows hazards, not the goal) corrects it. VLM uncertainty sets the guidance scale. |
| `test.py` | Same, but copilot strength is **proximity-adaptive**: relaxed far from hazards, strong when close. |

## Training & data

| File | Role |
|---|---|
| `train_sac_hazard.py` | Train a SAC expert (Stable-Baselines3); collect **safe-only** demonstrations (episodes that hit a hazard are filtered out). |
| `train_diffusion_hazard.py` | Train the conditional DDPM as a **hazard-aware safety prior** — conditions on position + hazards but never sees the goal. |
| `ddpm.py` | Minimal, self-contained conditional DDPM for vector data. |
| `augment_safe_actions.py` | For each expert `(obs, action)`, enumerate a grid of candidate actions, simulate lookahead, and keep all that stay clear — expands the dataset with every safe action per state. |

## Misc

| File | Role |
|---|---|
| `env_hazard_gym.py` | Gym wrapper (archived — depends on an external package; kept for reference). |

## Docs & assets

| Path | Contents |
|---|---|
| `README.md` | Project overview, key results, honesty note. |
| `docs/METHOD.md` | Counterfactual physics prompting, the control loop, LoRA distillation. |
| `docs/RESULTS.md` | Full result tables + related-work positioning. |
| `docs/LEAKAGE_AND_LIMITATIONS.md` | The leakage audit, the L0–L6 taxonomy, limitations. |
| `docs/STRUCTURE.md` | This file. |
| `assets/` | Result figures + episode GIFs used in the README. |

## A note on provenance

This directory is the **original research line** (learned-physics PIVOT, diffusion copilots, VLA
baselines). Its strongest numbers were later found to be leakage-contaminated — see
[`LEAKAGE_AND_LIMITATIONS.md`](LEAKAGE_AND_LIMITATIONS.md) — which is exactly why it is presented as an
*analysis of how to connect VLMs to safe physical control*, not as a clean benchmark. The corrected
conclusion motivated a separate follow-up project (VLM high-level planner + safety-oriented
sampling MPC, evaluated empirically).
