# Semantic-Constraint Experiment — Results (Path-B main result, pilot)

Date: 2026-06-22
Script: [`../subgoal_pivot_hazard.py`](../subgoal_pivot_hazard.py)
Plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §4 (Month 2–3, the main result).
Builds on the Month-1 negative figure in [`RESULTS_MONTH1.md`](RESULTS_MONTH1.md).

## Question

Month-1 showed that on the *pure-geometry* PointHazard task a classical planner
is already near-perfect, so a VLM buys nothing. This experiment asks the
opposite, deliberately-constructed question: when the task carries a constraint
a geometric cost **cannot express** — a semantic "keep-out" zone — does a VLM
high-level finally beat pure classical planning?

## Task (semantic keep-out zone)

The arena gains one off-limits zone (amber ✕ disk, `--n_semantic_zones 1`). It is
deliberately **not** a geometric hazard:

- it **never terminates** the episode (entering it is a soft "violation"),
- it is **not in the obs vector**,

so a geometry-only controller is structurally blind to it. The zone is sampled
to straddle the straight start→goal corridor, so a planner that heads straight at
the goal will plough through it unless something tells it not to.

## Setup (leakage-free, matched-seed)

Three policies, **same seeds**, same env:

- `mpc` — **C1**, geometry-only controller. Blind to the zone.
- `mpc_oracle` — **C2**, the same controller but **handed the zone as an extra
  obstacle** (a human hand-coded the keep-out cost). The upper bound.
- `subgoal` — **B**, the VLM **sees** the zone in the rendered image and picks
  subgoals that detour; the underlying controller stays geometry-only, so the
  VLM is the **only** zone-aware component.

Anti-leakage contract (verified): the VLM is given only the rendered image plus a
generic **language** constraint ("stay out of the amber restricted zone" — task
spec, like "reach the green goal"). It is never told the zone's coordinates,
which numbered waypoint enters it, any clearance, label, or score. Raw responses
saved for audit.

Run config: `model=google/gemini-3-flash-preview`, `temperature=0`,
`vlm_fallback=hold`, retries=1, `n_hazards=8`.

## Result (n=5 — pilot / "can it win?" check only)

| policy | success [95% CI] | hazard [95% CI] | **sem_viol [95% CI]** | mean min-clr | VLM calls/ep | fb% |
|---|---|---|---|---|---|---|
| `mpc` (C1, geometry-blind) | 100% [56.6, 100] | 0% [0, 43.4] | **80%** [37.6, 96.4] | +0.533 | 0 | – |
| `mpc_oracle` (C2, hand-coded) | 100% [56.6, 100] | 0% [0, 43.4] | **0%** [0, 43.4] | +0.566 | 0 | – |
| `subgoal` (B, VLM) | 100% [56.6, 100] | 0% [0, 43.4] | **20%** [3.6, 62.4] | +0.283 | 3.2 | 0% |

`sem_viol` = fraction of episodes that ever entered the keep-out zone.
`fb% = 0` ⇒ every subgoal decision was a genuine VLM choice (16 transcripts saved).

## Reading

The qualitative pattern is exactly the Path-B main-result thesis:

- **C1 (pure classical) violates the semantic constraint 80% of the time** — it
  cannot see the zone and ploughs straight through, just as designed.
- **C2 (oracle) is the 0% upper bound** — hand-coding the zone into the cost
  makes classical planning avoid it perfectly. (This is the honest answer to
  "why not classical?": classical wins *if* a human labels the semantic.)
- **B (VLM) cuts violations 4× vs C1 (80% → 20%), approaching the oracle**, with
  **zero hand-coded perception** — it reads the zone from pixels. That is the win.

So on this leakage-clean toy the VLM high-level does what Month-1 showed it could
not do in pure geometry: it adds value a classical planner can't get for free.

## Figures

**What the VLM is given** (its full input on seed 44). Only the rendered scene
plus numbered candidate waypoints — no clearance, label, or score. Note that
candidate **7 sits inside the amber keep-out zone**, directly on the way to the
goal "G": picking it would be a violation, so the VLM must choose a skirting
waypoint instead.

![What the subgoal-VLM sees](../outputs/fig_vlm_view_seed44.png)

**The win** (seed 44, blue = trajectory). The geometry-blind classical planner
(C1) drives its trail **straight through** the zone; the oracle (C2) and the VLM
(B) both **curve around** it — and B does so having only *seen* the zone, with no
hand-coded keep-out cost.

![Trajectory contrast, seed 44 (C1 through, C2/B around)](../outputs/fig_traj_seed44.png)

(Figures regenerated with the MPC RNG seeded for determinism — see the
reproducibility caveat below — so per-panel in-zone counts are illustrative of
the seed and can differ slightly from the run-of-record table.)

## The one VLM failure (honest)

Seed 47 is the only `subgoal` violation (7 steps in-zone). Its transcript shows
the VLM confidently picking waypoint 8 **twice while stating** it was
"navigating around the amber restricted zone" — yet it still grazed the zone.
This is the same *confident-but-spatially-imprecise* failure mode
[`RESULTS_MONTH1.md`](RESULTS_MONTH1.md) found for the low-level VLM, now at the
high level. Contributing factor to rule out at scale: subgoal **granularity** —
between subgoal queries the geometry-only controller can cut a corner through the
zone even when the chosen waypoint is on the safe side (knobs: `--subgoal_radius`
smaller = finer, `--subgoal_horizon` smaller = re-query more often).
B's lower mean min-clearance (+0.283 vs C1 +0.533) reflects these tighter
detours hugging hazards/zone edges.

![Trajectory contrast, seed 47 (B grazes the zone near the goal)](../outputs/fig_traj_seed47.png)

Seed 47: C1 clips the zone; C2 stays clear; **B detours right but cuts back
toward the goal too early and grazes the zone's lower edge** — the corner-cut
granularity failure described above.

## Honest caveats

- **n=5 is a "can it win?" pilot, not a paper figure.** The CIs overlap heavily;
  B at 20% is not yet statistically separable from C1 at 80% or C2 at 0%. A real
  figure needs ~100 matched seeds (and ideally ≥2 VLM backbones).
- Single model, single zone, single arena difficulty. Scale zone count/placement
  and `n_hazards` before any claim goes in the paper.
- The constraint here is a **language** constraint (the VLM is told to avoid the
  marked zone). A stronger, future version makes the zone's meaning *implicit*
  (appearance only, no naming) to test commonsense rather than instruction-
  following.
- **Reproducibility gap (fix before the 100-seed run):** `MPCExpert`'s CEM RNG
  is currently unseeded (`np.random.default_rng()` with no seed), so the low-
  level controller is not bit-reproducible run-to-run. It shows up as the same
  seed giving slightly different in-zone counts on re-runs (e.g. seed 47 `B`:
  7 steps in the table vs 5 in the seeded figure). The qualitative result is
  stable, but the paper run must seed the controller RNG per episode (e.g. from
  the episode seed) so the matched-seed table is exactly reproducible.

## Reproduce

```bash
# offline sanity (no API; B looks like C1 because the stand-in is zone-blind)
python subgoal_pivot_hazard.py --pilot_mode heuristic --n_semantic_zones 1 --episodes 20

# the run above
OPENROUTER_API_KEY=... python subgoal_pivot_hazard.py \
  --pilot_mode vlm --n_semantic_zones 1 --episodes 5 --seed 43 \
  --model google/gemini-3-flash-preview --temperature 0 \
  --vlm_fallback hold --log_transcripts --out outputs/semantic_pilot.json
```
