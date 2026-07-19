# The Leakage Audit & Limitations

> **ARCHIVED (reviewed 2026-07-11).** Historical audit from the learned-physics PIVOT
> phase. Keep it for provenance; use [`../../docs/RESEARCH_REVIEW_COMMENTS.md`](../../docs/RESEARCH_REVIEW_COMMENTS.md)
> and [`../../docs/RESULTS_REGISTRY.md`](../../docs/RESULTS_REGISTRY.md) for current status.

This is the most important document in the repo. It explains why the headline numbers in
[`RESULTS.md`](RESULTS.md) are reported as **directional evidence for a design principle**, not as a
clean benchmark — and why the negative finding here is more valuable than the positive ones.

## What happened

The early story was optimistic:

> *A VLM can read visualized physics trajectories and choose safe actions.*

A later audit of the prompts showed this claim was **too strong**. In some of the strongest
configurations, the prompt did not only contain the rendered trajectory. It also exposed explicit
**safety quantities**:

- candidate **clearance** (distance to nearest hazard),
- **safe / unsafe labels**, or
- score-like **ranking** hints.

When those are present, the VLM can pick the best safety/progress tradeoff **without computing hazard
distance from the image at all.** That turns the task into "select from a learned safety classifier,"
not "do visual physics reasoning." The reported success was partly an **assisted-safety upper bound**,
not clean VLM inference.

## The corrected numbers

Removing the leaked quantities weakens the system, exactly as you'd expect if the VLM had been leaning
on them:

| Task | With leaked safety info | Leakage-free (de-leaked) |
|---|---|---|
| PointHazard | ≈ 100% | **≈ 90%** (informal diagnostic) |
| Reacher-style | strong | needs progressively more explicit physics to work |
| LunarLander-style | — | **≈ 20%** even with trajectory + velocity + progress |

A representative VLM rationale — *"Candidate 4 provides the best balance of moving toward the green
goal while maintaining a safe distance from the red hazards…"* — sounds like reasoning, but it is
impossible to tell whether the model computed clearance from pixels or **copied** the safety structure
already present in the prompt.

## The information-level taxonomy it produced

The audit turned a confound into a clean experimental axis: *which abstraction level does the VLM
actually need before it can choose well?*

| Level | Information shown | What it tests | Clean for a "VLM-only" claim? |
|---|---|---|---|
| L0 | image + action arrows | choose from directions | ✅ |
| L1 | predicted trajectory only | infer safety from path shape | ✅ |
| L2 | trajectory + final state | does final pos/vel help? | ✅ |
| L3 | trajectory + velocity | needs explicit speed? | ✅ |
| L4 | trajectory + progress-to-goal | needs explicit task progress? | mostly |
| L5 | trajectory + safe/unsafe label | how much does a safety classifier help? | ❌ moves safety out of the VLM |
| L6 | trajectory + full score/rank | upper bound, near-oracle | ❌ basically gives the answer |

The leaked runs were effectively operating at **L5–L6** while being interpreted as L1–L2.

The Reacher-style ablation shows the gradient directly:

| Prompt | Success | Crash | Out of bounds | Timeout |
|---|---:|---:|---:|---:|
| Predicted trajectory only | 0.0 | 0.2 | 0.8 | 0.0 |
| + predicted velocity | 0.4 | 0.0 | 0.4 | 0.2 |
| + progress-to-goal | 0.8 | 0.0 | 0.2 | 0.0 |

Path *shape* alone is not enough; the VLM gets useful only as task-relevant physical quantities are
made explicit.

## The honest conclusion

> A VLM does **not** reliably extract safety-critical physical quantities (e.g. minimum hazard
> clearance) from raw trajectory images. It needs either explicit physical abstractions, or a
> **separate low-level controller** that owns safety while the VLM handles high-level strategy.

This matches the field's consensus design (SayCan, VoxPoser, VLM-MPC, SIMPACT): *VLM for semantics,
physics/MPC for low-level safety.* So the leakage finding is not a dead end — it is a **signpost.**

## What is kept vs. reframed

**Kept (genuinely valid):**

- the prompt audit and the L0–L6 information-level taxonomy;
- the leakage-free ablations showing where VLMs fail (hazard distance, velocity, terminal constraints);
- PointHazard as a controlled diagnostic environment;
- the online learned-physics module — as a tool to *compute* physical quantities / support MPC, not as
  proof the VLM "understands physics";
- the design rule: use VLMs for semantic comparison and high-level choice; expose computed physical
  quantities (or hand safety to a controller) when exact margins matter.

**Reframed / dropped:**

- the strong claim that trajectory visualization *alone* yields strong VLM physical reasoning;
- any result where clearance / score / rank was inside the prompt but interpreted as VLM inference;
- the near-100% PointHazard number → relabeled an **assisted-safety upper bound**.

## Other limitations

- **Single task family.** PointHazard is a 2D toy; Reacher/Lander results are preliminary diagnostics.
- **Episode budgets vary** across configs; some ablation tables are 10-episode reads, not paper-grade
  statistics. Treat small-n tables as qualitative.
- **No confidence intervals** on the original runs. The follow-up project re-runs the clean
  comparisons with matched seeds, Wilson CIs, transcript audits, and a neutral no-op fallback so the
  classical planner can never silently rescue the VLM.

## Where this led

The corrected conclusion directly motivated a follow-up: promote the VLM to a **high-level planner**
that picks a semantic subgoal, and let a **safety-oriented sampling MPC controller (empirical)**
(CEM-MPC over the exact dynamics) own low-level hazard avoidance — the leakage-free design this
audit pointed to. Reporting the
deflated numbers and pivoting on them is the point: *the negative result is the contribution.*
