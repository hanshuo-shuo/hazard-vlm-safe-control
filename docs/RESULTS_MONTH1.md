# Month-1 Confidence Experiment — Results

Date: 2026-06-21
Script: [`../subgoal_pivot_hazard.py`](../subgoal_pivot_hazard.py)
Plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §4 (Month 1), motivation in
[`FAILURE_MODE_ANALYSIS.md`](FAILURE_MODE_ANALYSIS.md).

## Question

On the **pure-geometry** PointHazard task, does using a VLM as a *low-level*
controller buy anything over a classical safe controller — and is it even safe?
This is the deliberately-negative first figure of the Path B story.

## Setup (leakage-free, matched-seed)

Three policies, **same seeds**, same env:

- `mpc` — CEM-MPC safe controller, **no VLM**.
- `subgoal` — VLM picks a high-level waypoint → MPC executes it safely.
- `direct` — VLM picks the low-level force directly (de-leaked PIVOT).

Anti-leakage contract (verified): VLM prompts + image annotations contain **no
clearance, no safe/unsafe label, no score** — only the rendered scene and
numbered candidates. Every raw VLM response is logged for audit.

Run config: `model=google/gemini-3-flash-preview`, `temperature=0`,
`vlm_fallback=hold` (on an unusable VLM reply the policy does a neutral no-op —
the classical planner never silently rescues the VLM), retries=1.

## Result (n=5 — pilot / pipeline validation only)

| policy | success [95% CI] | hazard [95% CI] | mean min-clearance | VLM calls/ep | fallback % |
|---|---|---|---|---|---|
| `mpc` (controller only) | 100% [56.6, 100] | 0% [0, 43.4] | +0.534 | 0 | – |
| `subgoal` (VLM→controller) | 100% [56.6, 100] | 0% [0, 43.4] | +0.634 | 3.0 | 0% |
| `direct` (VLM low-level) | 80% [37.6, 96.4] | **20%** [3.6, 62.4] | +0.413 | 20.2 | 0% |

`fallback % = 0` everywhere ⇒ every decision was a genuine VLM choice, not the
classical pilot. (116 VLM transcripts saved alongside the run.)

## Reading

The qualitative pattern matches the Path B thesis:

- The classical controller is already perfect on this geometric toy — **a VLM
  low-level is redundant here.**
- Routing the controller with a VLM high-level (`subgoal`) **keeps** that
  perfect safety while adding the VLM's semantic layer for free.
- The VLM-as-low-level (`direct`) is the one that **collides** (seed 43, min
  clearance −0.144). Its transcript shows it stating it was *"safely avoiding
  the red hazards"* on the very steps before the collision — i.e. confident but
  unable to estimate precise clearance. This is the failure mode
  `FAILURE_MODE_ANALYSIS.md` predicts.

## Honest caveats

- **n=5 is a pipeline validation, not a paper figure.** The CIs are very wide;
  `direct` at 80% is not yet statistically separable from 100%. A real figure
  needs ~100 matched seeds (and ideally ≥2 VLM backbones for robustness).
- Single model, single arena difficulty (`n_hazards=8`). Scale both before any
  claim goes in the paper.

## Reproduce

```bash
# offline sanity (no API)
python subgoal_pivot_hazard.py --pilot_mode heuristic --episodes 30

# the run above
OPENROUTER_API_KEY=... python subgoal_pivot_hazard.py \
  --pilot_mode vlm --episodes 5 --seed 43 \
  --model google/gemini-3-flash-preview --temperature 0 \
  --vlm_fallback hold --log_transcripts --out outputs/month1_confidence.json
```
