# Amplifying the VLM advantage — detailed, image-rich results

> ⚠️ **Superseded (2026-06-29).** The "B+ matches / out-robusts the hand-coded
> oracle" claim below was measured against a *hard* oracle that over-constrains.
> Against a **fair soft oracle** (true geometry, B+'s own soft mechanism) **and** a
> trivial **CV detector**, B+ only **ties** — it does not beat them. See the
> correction + 5-seed VLM rerun in [`RESULTS_FAIR_BASELINES.md`](RESULTS_FAIR_BASELINES.md)
> and the repositioning in [`ICLR_PLAN.md`](ICLR_PLAN.md). This doc is kept for the
> mechanism walkthrough and figures, not its headline comparison.

*How the Path-B win went from "3× better than blind classical" to "matches (and
on multi-constraint scenes out-robusts) the hand-coded oracle, with zero
hand-coding".*

Companion to [`WHY_VLM.md`](WHY_VLM.md) (the plain-English "where is the
advantage") and [`RESULTS_SEMANTIC.md`](RESULTS_SEMANTIC.md) (the results of
record; §11 = lever 1, §12 = lever 2). This doc is the **walkthrough with
pictures**. Date: 2026-06-28. Model: `google/gemini-3-flash-preview`,
`temperature=0`, `vlm_fallback=hold`, 5 matched seeds (43–47), `n_hazards=8`.

All figures regenerate **offline, with no API calls**, by replaying the recorded
VLM choices:

```bash
python scripts/make_amplify_figures.py
```

---

## 0. The starting point (recap)

The semantic task adds an off-limits **keep-out zone** that is *not* a geometric
hazard — it never ends the episode and is not in the controller's observation, so
a geometry-only planner is structurally blind to it. Four "drivers", matched
seeds:

| | what it is | knows the zone how |
|---|---|---|
| **C1** `mpc` | pure classical planner | never (blind) |
| **C2** `mpc_oracle` | classical + the zone hand-coded as an obstacle | a human labels it, every scene |
| **B** `subgoal` | VLM picks waypoints; controller stays zone-blind | VLM sees it in the image |
| **B+** `subgoal_perceive` | B **plus** the VLM's perceived keep-out fed to the controller | VLM sees it *and tells the controller* |

The original result: B cuts violations 3× vs C1 (60% → 20%) but is capped below
the oracle. The two levers below remove that cap and then pile on pressure.

---

## 1. Lever 1 — feed the VLM's perception to the controller (B → oracle)

### 1.1 Why B was stuck at 20%

In B the VLM picks a good waypoint, but the low-level controller is **blind** to
the zone, so between waypoint queries it can **cut a corner** through a zone it
cannot see. The residual 20% is a *controller* failure, not a perception failure.

### 1.2 The fix

In the **same** VLM call, the model also reports which numbered markers sit on
the off-limits terrain — a leakage-clean `"avoid": [...]` array (we never tell it
which are unsafe; the answer flows *out* of the VLM). We cluster those flagged
markers into a centroid estimate of the zone and hand it to the controller as a
**hard core + soft halo**: a small hard disk (≈ true zone scale) blocks driving
through the centre, a wider soft cost adds avoidance pressure but is never a wall.
Real sensor hazards stay hard; the *perceived* zone is the soft/uncertain part.
The controller now learns the zone **only** through the VLM's own perception —
still zero hand-coding.

### 1.3 The picture (single water zone, seed 47)

![Lever 1: C1 through, C2 around, B grazes, B+ around](../outputs/fig_bplus_implicit_traj_seed47.png)

Read the blue trail left to right:

- **C1** drives through the pond's lower edge (4 steps in-zone) — blind.
- **C2** (oracle) curves around — *because a human hand-coded the pond*.
- **B** picks correct waypoints but the **zone-blind controller cuts the corner**
  on final approach to the goal (5 steps in-zone) — the documented
  "confident-but-imprecise" failure.
- **B+** swings **wide around the pond** (0 steps in-zone) — the perceived
  keep-out, fed to the controller, removes the corner-cut. It matches the oracle.

### 1.4 Result (single implicit/water zone)

| policy | success | **sem_viol** | VLM/ep |
|---|---|---|---|
| C1 blind | 100% | **60%** | 0 |
| C2 oracle | 100% | **0%** | 0 |
| B | 100% | **20%** | 3.0 |
| **B+** | **100%** | **0%** | 3.4 |

**B+ reaches the oracle (0% / 100%), beating B, for ~0.4 extra calls/ep and zero
hand-coded perception.** Honest boundary: B+ removes *controller corner-cuts*, not
*VLM mis-choices* — on the explicit framing of seed 47 the VLM confidently picks a
waypoint heading into the zone, and no perception-feed can rescue a wrong
high-level choice (there B+ ties B). See `RESULTS_SEMANTIC.md` §11.

Three design pitfalls, each forced by an observed failure (details in §11.2):
scatter→**centroid clustering** (else a naïve estimate sealed a tight gauntlet and
live-locked seed 45); **hard core + soft halo** (a soft cost can never trap the
agent); **flag terrain, not hazards** (else the VLM flagged hazard-adjacent
markers and success collapsed to 40%).

---

## 2. Lever 2 — heterogeneous terrains (one VLM vs N hand-coded detectors)

### 2.1 Why

A fair sceptic says: "so hand-code the one zone (C2) — why a VLM?" The honest
answer is **scale across semantics**: real keep-outs are many and varied, and each
visual category would need its **own** hand-coded detector. A VLM recognises all
of them zero-shot from one generic instruction. So we put **three visually
distinct** unsafe terrains in one scene — **water, mud, grass** — with the prompt
unchanged ("avoid terrain that looks unsafe to drive over", never naming any).

### 2.2 What the VLM sees (seed 44)

![What the VLM sees — three terrains, no labels](../outputs/fig_hetero_view_seed44.png)

Between the agent (blue, centre) and goal **G** (green, below) sit three stacked
terrains — water (teal ripples), mud (brown blotches), grass (green blades) — plus
red hazards and 8 candidate waypoints. **No terrain carries a label.** The VLM must
recognise each from appearance.

### 2.3 The picture (seed 46)

![Lever 2: C1 through all three, C2 times out, B grazes, B+ around all](../outputs/fig_hetero_traj_seed46.png)

- **C1** drives the straight diagonal through grass + mud + water (**11** steps
  in-zone) — blind.
- **C2** (oracle), forced to *hard*-avoid all three zones **and** eight hazards,
  is **over-constrained and times out** — it never reaches the goal.
- **B** routes around but still grazes (2 steps in-zone).
- **B+** takes a wide detour **around all three terrains** and **reaches the goal,
  0 steps in-zone**.

### 2.4 Result (three terrains)

| policy | success | **sem_viol** | VLM/ep |
|---|---|---|---|
| C1 blind | 100% | **100%** | 0 |
| C2 oracle | **60%** | **0%** | 0 |
| B | 100% | **40%** | 3.6 |
| **B+** | **100%** | **20%** | 4.4 |

Two findings, both stronger than the single-zone case:

1. **B+ halves B's violations (40% → 20%) and cuts blind classical 5× (100% →
   20%).** Three zones make C1 plough through *every* episode.
2. **B+ (100% success) beats the hand-coded oracle (60%).** Hard-avoiding
   everything over-constrains the planner; B+'s *soft* perceived zones always keep
   a feasible path. On a multi-constraint scene the VLM-perception route is **more
   robust than hand-coding**, not merely cheaper.

### 2.5 The decisive evidence — the VLM names all three terrains, unprompted

The prompt never wrote "water", "mud", "grass", "brown", or "green". Across the
saved transcripts the VLM's own free-text reasons use **water 28× · mud 18× ·
grass 9×** (plus "brown cratered area", "striped green", "swamp", "muddy"):

> seed 44: *"Waypoints 6, 7, and 8 lead directly into unsafe terrain (**water,
> mud, and tall grass**). Waypoint 5 provides a safe path around these obstacles."*
>
> seed 45: *"Waypoints 3, 4, and 5 are blocked by red hazards or unsafe terrain
> (**the brown cratered area**)."*

One model, one generic instruction, recognising three distinct terrains it was
never told about — the per-semantic hand-coding a classical detector stack would
need, removed. This is the sharpest form of the SayCan/VoxPoser claim the project
has shown, on a leakage-clean toy with an auditable trail.

---

## 3. Where this leaves the story

| claim | before | after levers 1–2 |
|---|---|---|
| vs blind classical | 3× fewer violations | **5× fewer** (multi-zone) |
| vs hand-coded oracle | strictly worse (20% vs 0%) | **matches** it (single zone); **out-robusts** it on success (multi-zone) |
| hand-coding required | none (B) | **none** (B+) — and it now reaches oracle-level |
| recognition vs rule-following | implicit single zone | **multiple distinct terrains named unprompted** |

The one-line summary is unchanged but now much better supported: *classical
planning wins on what you can measure or hand-code; the VLM wins on what you can
only recognise — and it removes the human who would otherwise have to label every
such case by hand.*

## 4. Honest caveats (carried from `RESULTS_SEMANTIC.md` §8)

- **n=5 is a "can it win?" pilot, not a paper figure.** CIs overlap; B+ vs B is
  not yet statistically separated. A real figure needs ~100 matched seeds and
  ≥2 VLM backbones (the deferred "lever 4").
- The oracle's 60% multi-zone success is a consequence of *hard*-avoiding
  everything; a soft-cost oracle would trade some 0%-violation back for success.
  The point is not "B+ dominates a tuned oracle" but "B+ needs **zero** hand-coding
  and is already on the robust side of the trade-off".
- Single model, single arena difficulty, three terrains. Scale before any claim
  goes in the paper.

## 5. Reproduce

```bash
# figures in this doc (offline, replays recorded choices — no API)
python scripts/make_amplify_figures.py

# lever 1 — single implicit zone, four arms C1/C2/B/B+ (run of record)
OPENROUTER_API_KEY=... python subgoal_pivot_hazard.py \
  --pilot_mode vlm --n_semantic_zones 1 --zone_semantics implicit \
  --episodes 5 --seed 43 --model google/gemini-3-flash-preview \
  --temperature 0 --vlm_fallback hold --log_transcripts \
  --out outputs/semantic_bplus_implicit.json

# lever 2 — three heterogeneous terrains
OPENROUTER_API_KEY=... python subgoal_pivot_hazard.py \
  --pilot_mode vlm --n_semantic_zones 3 --zone_semantics implicit \
  --semantic_styles water,mud,grass --episodes 5 --seed 43 \
  --model google/gemini-3-flash-preview --temperature 0 \
  --vlm_fallback hold --log_transcripts --out outputs/semantic_hetero.json
```
