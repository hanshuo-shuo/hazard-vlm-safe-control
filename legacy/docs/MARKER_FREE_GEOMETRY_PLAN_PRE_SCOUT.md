# Marker-free semantic geometry plan

> **ARCHIVED PRE-SCOUT PLAN.** The infrastructure described here was completed
> and then exercised by Experiment 0 and the frozen 120-call scout. Use
> [`../../docs/INTERFACE_CONTRACT_HANDOFF.md`](../../docs/INTERFACE_CONTRACT_HANDOFF.md)
> for current execution state and
> [`../../docs/INTERFACE_CONTRACT_SCOUT_REPORT.md`](../../docs/INTERFACE_CONTRACT_SCOUT_REPORT.md)
> for the resulting evidence.

Status: infrastructure implemented; scientific calibration results `TBD`.

## 1. Why the marker mainline ended

The five-seed pilot generated on 2026-07-24 triggered the predeclared
interface-stability kill condition. Marker-ID permutation preserved the same
physical choice in 20% of model/seed pairs; candidate-coordinate ordering
preserved 30%. Direct coordinates and unmarked images preserved 40% and 10%.
P0/P1/P2 selected-safe rates were 50%/60%/40%, P2 did not improve over P0, and
no model rank reversal appeared.

The marker-based PointHazard VLM waypoint ICLR line is therefore `TERMINATED`.
It may be used only as negative interface evidence, historical pilot,
regression test, or case study. It must not be rescued by adding seeds,
changing prompts, renumbering markers, changing candidate count, or selecting a
favorable configuration.

## 2. Current hypotheses

The active, falsifiable hypotheses are:

1. recognition and capability-conditioned applicability can fail separately;
2. detector geometry error and planner enforcement interact, so perception
   cannot be ranked without a fixed operating curve;
3. capability twins should reverse applicability and, where feasible, route/STC
   while preserving scene, rendering, native cost and planner;
4. downstream enforcement may absorb, amplify, or introduce upstream errors.

These are research questions, not established results.

## 3. Architecture

```text
public RGB/task/capability
  → recognition
  → applicability
  → marker-free SemanticGeometryPayload
  → fixed planner and enforcement
  → trajectory
  → evaluator-only STC + five-stage attribution
```

`evaluation/semantic_geometry.py` owns the policy-visible contract. Evaluator
truth is admitted only through an explicit oracle arm. Provider-native,
normalized, pixel, world, simulator and planner frames are named, versioned and
never implicitly rescaled.

## 4. Geometry contract

`SemanticRegion` carries region ID, class, geometry kind, geometry, frame,
confidence, source/model/artifact hash, applicability, provenance and adapter
version. Supported representations are disk, AABB, polygon, mask reference and
empty.

`SemanticGeometryPayload` carries request/scene IDs, image hash, transform
metadata, ordered regions, parser/adapter/fallback status, provenance and schema
version. Canonical bytes and SHA-256 are deterministic.

Adapters currently exist for `none`, explicit `oracle`, `fixture` and cached
`detector`. `vlm_grounding` is schema-reserved but deliberately not connected
to a provider.

## 5. Five-stage protocol

The scored stages are:

1. `recognition`: classes, confidence, region references, unknown;
2. `applicability`: applicable/not-applicable/unknown, capability/task basis,
   reason code;
3. `grounding`: region ID, geometry, uncertainty and frame;
4. `action_proposal`: target/route, intended interaction, avoidance and
   feasibility;
5. `enforcement_outcome`: accepted/modified/blocked/fallback/executed,
   trajectory, violation, success and STC.

`evaluation/five_stage_audit.py` preserves earliest detectable failure,
contributing failures and final mode. It supports recognition-correct but
applicability-wrong, unsafe-proposal rescue, over-conservative blocking and
multi-stage failure without forcing a single exclusive cause.

## 6. Detector decomposition

Registered arms are detector full geometry; detector center/detector radius;
detector center/oracle radius; oracle center/detector radius; oracle
center/oracle radius; detector box; detector polygon/convex hull; detector
mask; blind; and oracle.

Existing boxes may be converted to disks, boxes or polygons. A real mask may be
used only when present in an artifact. Synthetic masks must be marked adapter
sensitivity and cannot be represented as measured segmentation.

Disk metrics currently include center/radius/area error, IoU, false-positive
and false-negative area, and boundary distance. Closed-loop rows must add
minimum clearance, path length, success, semantic violation, collision,
timeout, STC, interventions and replans.

## 7. Safety-Gym capability twins

The water family is implemented for non-waterproof versus amphibious profiles.
Mud/rough terrain is a protocol fixture for ordinary-wheeled versus
tracked/all-terrain profiles. Clearance/footprint is protocol-only until robot
size affects dynamics; physical feasibility remains `TBD`.

Capability twins must preserve seed, geometry, RGB, goal, initial state, native
cost, geometry interface and planner. Only capability/applicability may change.
Appearance twins are a separate intervention and may change only declared
appearance and, when registered, evaluator semantics.

## 8. Dev/test discipline

- dev/calibration: seeds 0–42; adapter and planner calibration allowed;
- historical pilot: current seeds 0–4 are never used to choose new parameters;
- future pilot: 200–299, exact subset frozen before inspection;
- future test/formal: 300–499, parameters frozen and one-time sentinel.

Every operating curve changes one parameter family, has a unique config hash
and consumes a declared calibration budget. Test runs are refused after the
sentinel exists. Any future real API key is limited to five distinct scene
seeds; this limit does not grant provider authorization.

## 9. Allowed claims

- the marker interface was unstable in the five-seed pilot and its kill
  condition fired;
- recognition-conditioned safety and applicability failure are pilot signals;
- detector geometry showed a safety/completion trade-off at one pilot point;
- the minimal Safety-Gym port is runnable infrastructure;
- the marker-free schema, adapters, audits and twin invariants pass offline
  tests.

## 10. Forbidden claims

- VLMs generally cannot reason about norms;
- detector is better than VLM;
- applicability is proven to be the dominant bottleneck;
- Safety-Gym reproduces the PointHazard phenomenon;
- calibration will improve STC;
- the system has formal safety guarantees;
- current evidence is paper-grade or generalizes across models/environments.

## 11. Experiment gates

1. schema, permissions, transforms and secret audits pass;
2. cached detector decomposition produces complete non-fabricated metrics;
3. a bounded dev operating curve is frozen;
4. capability-twin invariants pass headlessly;
5. cached/offline stage scoring is stable;
6. only then define a provider pilot, with five seeds maximum per real key;
7. paid execution still requires a separate clean release authorization.

## 12. Venue decision

- ICLR measurement mainline: not ready. Requires cross-model, cross-environment
  marker-free evidence after a new protocol freeze.
- CoRL method line: not ready. Requires a substantive calibrated enforcement
  method and credible dynamics/robot evaluation.
- Workshop/technical report: plausible if the negative marker finding,
  provenance contract and cached geometry analysis remain carefully bounded.
