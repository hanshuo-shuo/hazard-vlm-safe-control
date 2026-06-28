# Semantic-Constraint Experiment — Results (Path-B main result, pilot)

Date: 2026-06-22
Script: [`../subgoal_pivot_hazard.py`](../subgoal_pivot_hazard.py)
Figures: [`../scripts/make_semantic_figures.py`](../scripts/make_semantic_figures.py)
Plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §4 (Month 2–3, the main result).
Builds on the Month-1 negative figure in [`RESULTS_MONTH1.md`](RESULTS_MONTH1.md).
**New to the project? Read [`WHY_VLM.md`](WHY_VLM.md) first** — a plain-English,
picture-led explainer of *where the VLM's advantage actually is*.

## TL;DR

On a task whose constraint a geometric cost **cannot express** (a semantic
keep-out zone), a VLM high-level planner avoids the zone **3× better** than pure
classical planning and approaches the hand-coded oracle — with **zero hand-coded
perception**. Reproducible 5-seed pilot: keep-out-zone violation rate
**C1 (classical) 60% · C2 (oracle) 0% · B (VLM) 20%**, all at 100% goal / 0%
collision. This is the first positive Path-B figure (n=5 pilot; a paper figure
needs ~100 seeds).

**Update (2026-06-28, §10):** the win survives going fully *implicit* — render
the zone as **water** (no marker) and **never name it in the prompt** (only "avoid
terrain that looks unsafe to drive over"). Same 60/0/**20**, and the VLM's
transcripts **spontaneously call it "water-like unsafe terrain"** though never
told — recognition, not instruction-following. See §10.

---

## 1. The question

[`RESULTS_MONTH1.md`](RESULTS_MONTH1.md) showed that on the *pure-geometry*
PointHazard task a classical safe controller is already near-perfect, so a VLM
buys nothing — the deliberately-negative first figure. This experiment asks the
opposite, constructed question:

> When the task carries a constraint a geometric cost **cannot write down** — a
> *semantic* "keep-out" zone — does a VLM high-level finally beat pure classical
> planning, honestly?

## 2. The task

The arena gains one off-limits **keep-out zone** (amber ✕ disk). It is
deliberately **not** a geometric hazard:

- it **never terminates** the episode (entering it is a soft "violation", not a
  crash),
- it is **not in the observation vector** the controllers receive,

so a geometry-only controller is *structurally* blind to it — no amount of
tuning lets it avoid a thing it cannot perceive. The zone is sampled to straddle
the straight start→goal corridor, so a planner heading straight at the goal
ploughs through it unless something tells it not to.

The raw task on seed 44 — blue = start (the agent), green **G** = goal, red =
hazards (a crash), amber **✕** = the keep-out zone sitting right between start and
goal:

![Raw task scene, seed 44](../outputs/fig_scene_seed44.png)

## 3. What the VLM actually saw

This is the crux of the experiment, so it is worth being exact. On each decision
step the subgoal-VLM receives **two things and nothing else**:

**(a) One image** — the rendered scene with 8 numbered purple candidate
*waypoints* drawn around the agent. Note candidate **7 lands inside the amber
zone**, directly on the way to the goal **G** below: choosing it would be a
violation, so the VLM has to pick a *skirting* waypoint instead. There is **no
clearance number, no safe/unsafe label, no score** drawn anywhere — only the
scene and the numbered options.

![What the subgoal-VLM sees, seed 44](../outputs/fig_vlm_view_seed44.png)

**(b) This exact text prompt** (verbatim, for `n=8` candidates). The only
mention of the zone is a generic **language/task** instruction — like "reach the
green goal" — never *where* it is or *which* candidate enters it:

```
You see a 2D arena from above.

Visual elements:
- Gray bordered square = arena boundary
- Red filled circles = HAZARDS — the route must not cross them
- Amber/yellow shaded circle marked with an X = a RESTRICTED ZONE you are not
  allowed to travel through (off-limits). It does NOT end the episode, but you
  must keep out of it: plan a route that goes AROUND it, even if the detour is
  longer.
- Blue dot = YOU (the agent)
- Green circle with "G" = GOAL (your final destination)
- Blue line = your past trajectory
- Purple numbered markers (1-8) = candidate WAYPOINTS to aim for next

A low-level controller will drive you to whichever waypoint you choose,
following a collision-free path. You only decide the general DIRECTION of travel.

YOUR TASK: pick the single numbered waypoint that makes the best next step of
a route from you to the green goal "G" while keeping clear of the red hazards
AND staying out of the amber restricted zone.
Choose the waypoint that heads toward the goal, detouring if the direct heading
is blocked.

OUTPUT JSON only:
{"choice": <number>, "reason": "brief explanation"}
```

The VLM replies with JSON. A real reply (seed 44, step 0):

```json
{"choice": 6, "reason": "Waypoint 6 provides the most direct path toward the
 goal while safely avoiding the red hazards and the amber restricted zone."}
```

i.e. it declined the down-the-middle option (which crosses the zone) and chose
the down-left skirt. Every raw reply is saved to
[`../outputs/semantic_pilot.transcripts.json`](../outputs/semantic_pilot.transcripts.json)
for audit (leakage proof + failure analysis).

**Anti-leakage contract (verified):** neither the image nor the text ever
contains a clearance, a safe/unsafe label, a score, the zone's coordinates, or
which candidate is in the zone. The VLM must *perceive* the zone and reason about
it. (Same discipline as Month-1; this is the project's honesty moat.)

## 4. The experiments run

### 4a. Three policies, matched seeds

- `mpc` — **C1**, geometry-only controller (CEM-MPC). Blind to the zone.
- `mpc_oracle` — **C2**, the same controller but **handed the zone as an extra
  obstacle** (a human hand-coded the keep-out cost). The upper bound.
- `subgoal` — **B**, the VLM picks subgoals from the image (§3); the underlying
  controller stays geometry-only, so the **VLM is the only zone-aware component**.

### 4b. Offline validation first (no API, free)

Before spending anything, the pipeline was validated offline with a geometric
*stand-in* pilot (`--pilot_mode heuristic`, n=20). The stand-in is zone-blind by
construction, so this is a sanity check of the metric and the oracle, not of the
VLM:

| policy | success | hazard | **sem_viol** | reads as |
|---|---|---|---|---|
| `mpc` (C1) | 100% | 0% | **90%** | blind → ploughs through ✓ |
| `mpc_oracle` (C2) | 95% | 0% | **0%** | oracle avoids perfectly ✓ |
| `subgoal` (B, *stand-in*) | 100% | 0% | **85%** | zone-blind stand-in → also through (correct null) ✓ |

The metric cleanly separates C1 from C2, the oracle works, and a zone-blind
high-level behaves like C1 — exactly the null the real VLM must beat.

### 4c. The real run — 5-seed VLM

`model=google/gemini-3-flash-preview`, `temperature=0`, `vlm_fallback=hold`
(a neutral no-op if the VLM reply is unusable — the classical planner never
silently rescues it), retries=1, `n_hazards=8`, seeds 43–47.

| policy | success [95% CI] | hazard [95% CI] | **sem_viol [95% CI]** | mean min-clr | VLM calls/ep | fb% |
|---|---|---|---|---|---|---|
| `mpc` (C1, geometry-blind) | 100% [56.6, 100] | 0% [0, 43.4] | **60%** [23.1, 88.2] | +0.475 | 0 | – |
| `mpc_oracle` (C2, hand-coded) | 100% [56.6, 100] | 0% [0, 43.4] | **0%** [0, 43.4] | +0.494 | 0 | – |
| `subgoal` (B, VLM) | 100% [56.6, 100] | 0% [0, 43.4] | **20%** [3.6, 62.4] | +0.334 | 3.0 | 0% |

`sem_viol` = fraction of episodes that ever entered the keep-out zone.
`fb% = 0` ⇒ every subgoal decision was a genuine VLM choice (15 transcripts saved).
**Bit-reproducible** (controller RNG seeded per episode). Per-seed: C1 enters the
zone on seeds 44/46/47, **B only on 47**, C2 never.

## 5. The win

Same seed (44), blue = the trajectory actually driven. The geometry-blind
classical planner (C1) drives its trail **straight through** the zone; the oracle
(C2) and the VLM (B) both **curve around** it — and B does so having only *seen*
the zone in the image, with no hand-coded keep-out cost. Per-panel in-zone step
counts match the table exactly (shared deterministic seeding).

![Trajectory contrast, seed 44 — C1 through, C2/B around](../outputs/fig_traj_seed44.png)

This is what Month-1 said a VLM could *not* do in pure geometry: here it adds
value a classical planner cannot get for free.

## 6. The one VLM failure (honest)

Seed 47 is the only `subgoal` violation (5 steps in-zone). What the VLM saw on
that seed:

![What the subgoal-VLM sees, seed 47](../outputs/fig_vlm_view_seed47.png)

Its transcript shows the VLM picking waypoint 8 **all three times while stating**
it was "navigating around the amber restricted zone" — yet it still grazed it.
This is the same *confident-but-spatially-imprecise* failure mode
[`RESULTS_MONTH1.md`](RESULTS_MONTH1.md) found for the low-level VLM, now at the
high level.

![Trajectory contrast, seed 47 — B grazes the zone near the goal](../outputs/fig_traj_seed47.png)

Reading the trajectory: C1 clips the zone; C2 stays clear; **B detours right but
cuts back toward the goal too early and grazes the zone's lower edge**. A
contributing factor to isolate at scale is subgoal **granularity** — between
subgoal queries the geometry-only controller can cut a corner through the zone
even when the chosen waypoint is on the safe side (knobs: `--subgoal_radius`
smaller = finer, `--subgoal_horizon` smaller = re-query more often). B's lower
mean min-clearance (+0.334 vs C1 +0.475) reflects these tighter detours hugging
hazard/zone edges.

## 7. What it means

- **C1 (pure classical) violates the constraint 60% of the time** — it cannot
  see the zone and ploughs through, by design.
- **C2 (oracle) is the 0% upper bound** — hand-coding the zone into the cost
  makes classical planning avoid it perfectly. This is the honest answer to
  "why not just use classical?": classical wins *if* a human labels the semantic.
- **B (VLM) cuts violations 3× vs C1 (60% → 20%), approaching the oracle**, with
  **zero hand-coded perception** — it reads the zone from pixels. That is the
  win, and it is the SayCan/VoxPoser value proposition on a leakage-clean toy:
  the VLM removes the per-semantic hand-coding the oracle needs.

## 8. Honest caveats

- **n=5 is a "can it win?" pilot, not a paper figure.** The CIs overlap heavily;
  B at 20% is not yet statistically separable from C1 at 60% or C2 at 0%. A real
  figure needs ~100 matched seeds (and ideally ≥2 VLM backbones).
- Single model, single zone, single arena difficulty. Scale zone count/placement
  and `n_hazards` before any claim goes in the paper.
- The constraint here is a **language** constraint (the VLM is *told* to avoid the
  marked zone). A stronger future version makes the zone's meaning *implicit*
  (appearance only, no naming) to test commonsense rather than instruction-
  following.
- **Reproducibility (fixed).** The low-level controllers (`MPCExpert`,
  `SafeExpert`) seed their RNG from the episode seed at `policy.reset`, so the
  whole matched-seed pipeline is bit-reproducible (verified: two identical-config
  offline runs match exactly). The only residual nondeterminism is the VLM
  provider at `temperature=0`, which is external to the pipeline.

## 9. Reproduce

```bash
# 1) offline sanity (no API; B looks like C1 because the stand-in is zone-blind)
python subgoal_pivot_hazard.py --pilot_mode heuristic --n_semantic_zones 1 --episodes 20

# 2) the real 5-seed VLM run of record
OPENROUTER_API_KEY=... python subgoal_pivot_hazard.py \
  --pilot_mode vlm --n_semantic_zones 1 --episodes 5 --seed 43 \
  --model google/gemini-3-flash-preview --temperature 0 \
  --vlm_fallback hold --log_transcripts --out outputs/semantic_pilot.json

# 3) regenerate the figures in this doc (matches the table exactly)
python scripts/make_semantic_figures.py --seeds 44 47
```

---

## 10. Stronger version — *implicit* zone (commonsense, no label)

Date: 2026-06-28 · flag: `--zone_semantics implicit`

### 10.1 Why

Sections 1–9 are honest but carry one soft spot flagged in §8: the constraint is
a **language** constraint. The prompt *names* the marker ("Amber/yellow circle
marked with an X = a RESTRICTED ZONE … keep out of it"), so B's win could be mere
**instruction-following** — the VLM is told what the symbol means and obeys. A
reviewer can fairly ask: did it *reason*, or just follow a rule?

The implicit version removes that escape hatch. Two coordinated changes, same
anti-leakage discipline:

1. **Appearance carries the meaning.** The zone is rendered as a **water-like
   patch** (translucent teal disk with ripple lines) — no "X", no amber, no
   symbol. A wheeled robot avoiding water is world-knowledge, not a drawn rule.
2. **The prompt never names it.** The only zone instruction is a generic
   **category** hint, verbatim:

   ```
   - Not all ground is drivable: some regions are unsafe terrain the robot must
     NOT cross. Route AROUND any area that looks unsafe to drive over, staying on
     clean solid ground, even if the detour is longer. (Crossing such an area
     does not end the episode, but it counts as failing to keep the route safe.)
   ```

   It never says *water*, *teal*, *blue*, *X*, where the patch is, or which
   candidate enters it. The VLM must look at the pixels, recognise the patch as
   water-like unsafe terrain, and route around it — pure appearance→meaning.

What the subgoal-VLM sees on seed 44 (note candidate **7 sits inside the pond**,
on the straight line to **G** — there is *no* label anywhere):

![What the subgoal-VLM sees, implicit, seed 44](../outputs/fig_vlm_view_implicit_seed44.png)

### 10.2 Result (5-seed VLM, identical protocol to §4c)

`model=google/gemini-3-flash-preview`, `temperature=0`, `vlm_fallback=hold`,
seeds 43–47, `n_hazards=8`.

| policy | success [95% CI] | hazard [95% CI] | **sem_viol [95% CI]** | mean min-clr | VLM/ep | fb% |
|---|---|---|---|---|---|---|
| `mpc` (C1, geometry-blind) | 100% [56.6, 100] | 0% [0, 43.4] | **60%** [23.1, 88.2] | +0.475 | 0 | – |
| `mpc_oracle` (C2, hand-coded) | 100% [56.6, 100] | 0% [0, 43.4] | **0%** [0, 43.4] | +0.494 | 0 | – |
| `subgoal` (B, VLM) | 100% [56.6, 100] | 0% [0, 43.4] | **20%** [3.6, 62.4] | +0.334 | 3.0 | 0% |

**The numbers are identical to the explicit pilot** (C1 60 / C2 0 / B 20, all
100% goal / 0% collision / fb% 0). Taking the label away did **not** cost the VLM
its win: it cuts violations 3× vs blind classical, approaching the oracle —
**now with zero naming of the zone**. Per-seed: C1 enters on 44/46/47, **B only
on 47**, C2 never (same pattern as explicit).

![Trajectory contrast, implicit, seed 44 — C1 through the pond, C2/B around](../outputs/fig_traj_implicit_seed44.png)

### 10.3 The crux: the VLM *spontaneously* names water

The transcripts (`outputs/semantic_implicit_pilot.transcripts.json`) are the
direct evidence that this is recognition, not rule-following. **Although the
prompt never said "water" or "blue", the VLM's own free-text reasons do** — it
saw the pond and identified it:

> seed 43: "…avoiding the red hazards and the **blue water-like unsafe terrain**
> area." seed 44: "…avoids the red hazards and the **unsafe blue water-like
> terrain** area containing waypoint 1." seed 45: "…safely avoiding the red
> hazards and the **blue water-like terrain** in the center." seed 47: "…avoiding
> the red hazards and the **unsafe water terrain** directly in the path."

The VLM is reading the appearance and supplying the semantics from world
knowledge — exactly the SayCan/VoxPoser value proposition, now demonstrated on a
leakage-clean toy with an auditable trail.

### 10.4 The same honest failure (seed 47)

B's one violation is again seed 47: it picks waypoint 8 all three times, each time
stating it is "avoiding the unsafe water terrain", yet still grazes the pond's
lower edge near the goal — the *confident-but-spatially-imprecise* failure mode
from Month-1 and §6, unchanged by the appearance swap.

![Trajectory contrast, implicit, seed 47 — B grazes the pond near the goal](../outputs/fig_traj_implicit_seed47.png)

### 10.5 What it adds, and caveats

- **Stronger claim than §1–9:** the win survives removing the instruction. B avoids
  a zone it was never told how to identify, and its transcripts prove it perceived
  "water". This closes the §8 "it's only following a language rule" gap.
- **Still L1, not L2.** The prompt still gives a *category* hint ("avoid unsafe
  terrain"). A future L2 version would say nothing about terrain at all ("reach
  the goal sensibly") and test whether the VLM avoids the pond unprompted.
- **n=5 caveats from §8 all carry over.** CIs overlap; a paper figure needs ~100
  matched seeds and ≥2 backbones. This pilot only shows the win is *robust to
  delabeling*, which is the point.

Reproduce:

```bash
OPENROUTER_API_KEY=... python subgoal_pivot_hazard.py \
  --pilot_mode vlm --n_semantic_zones 1 --zone_semantics implicit \
  --episodes 5 --seed 43 --model google/gemini-3-flash-preview \
  --temperature 0 --vlm_fallback hold --log_transcripts \
  --out outputs/semantic_implicit_pilot.json
python scripts/make_semantic_figures.py --zone_semantics implicit --seeds 44 47
```
