# Documentation index

Updated: 2026-07-26.

The documentation set is intentionally small. Current experiment semantics,
execution state and result validity each have one owner.

| Read order | Document | Authority |
|---:|---|---|
| 1 | [INTERFACE_CONTRACT_HANDOFF.md](INTERFACE_CONTRACT_HANDOFF.md) | Current execution state, exact artifacts and next authorization boundary |
| 2 | [INTERFACE_CONTRACT_SCOUT_REPORT.md](INTERFACE_CONTRACT_SCOUT_REPORT.md) | Illustrated 120-call scout narrative, native replay evidence and continuation decision |
| 3 | [INTERFACE_CONTRACT_PROTOCOL.md](INTERFACE_CONTRACT_PROTOCOL.md) | Frozen interface-contract experiment semantics |
| 4 | [INTERFACE_CONTRACT_MAINLINE.md](INTERFACE_CONTRACT_MAINLINE.md) | Research question, exploratory evidence and paper gate |
| 5 | [RESULTS_REGISTRY.md](RESULTS_REGISTRY.md) | Only authority for whether an artifact may be cited |
| 6 | [PROTOCOL.md](PROTOCOL.md) | Earlier task, information, evaluator, replay and seed semantics |
| 7 | [review_status_registry.json](review_status_registry.json) | Machine-readable earlier-mainline review status |
| 8 | [RESEARCH_REVIEW_COMMENTS.md](RESEARCH_REVIEW_COMMENTS.md) | Earlier-mainline blockers and rendered status snapshot |
| 9 | [ICLR_PLAN.md](ICLR_PLAN.md) | Broader research strategy and publication gates |
| 10 | [PAID_RUN_RELEASE.md](PAID_RUN_RELEASE.md) | Blocked general release record; separate from the completed scout authorization |

The code map is [../STRUCTURE.md](../STRUCTURE.md).

## Current status

Review-item status is owned by [review_status_registry.json](review_status_registry.json).
The review comments and execution plan may describe evidence and remaining gaps, but do not
define a second status source.

- PointHazard B01 layout validation is complete.
- The condition contract is complete.
- `direct/replay × none/oracle` runs through the unified harness.
- Direct/replay routing uses an immutable, audited minimal-permission input.
- Episode artifacts use the shared STC component reduction as the headline metric.
- Structured VLM output and exact prompt/image/policy bytes have an offline audit gate.
- The public candidate/PNG adapter completes zero-cost structured fixture decisions.
- Protocol 1.2.2 registers the four-field structured output and cumulative P0–P4 prompt ladder.
- The frozen 14-condition zero-provider offline matrix passes.
- The pilot release manifest and portable core dependency lock audit pass, but
  the manifest remains `BLOCKED` with provider calls disabled.
- The provider-free executable bridge passed 640 native executions across
  PointHazard and Safety-Gym with fixture action/trajectory IEC = 1.00.
- The separately authorized frozen scout completed 120/120 paid calls with
  strict parse success, no retries and total recorded cost USD 0.7861499.
- Zero-call native replay produced action IEC = 0.55, trajectory IEC = 0.55,
  CISR-EQ = 0.15 and CISR-MAP = 0.30.
- These results pass the scientific continuation gate for considering a second
  family, but do not authorize it and do not unlock the general release manifest.
- No semantic-safety result is currently `VALIDATED`.
- The completed scout remains `PILOT_ONLY`; no full 1,440-call experiment has
  been authorized.

## Historical evidence

Raw pilot outputs, transcripts and invalidated runs remain in `../outputs/`.
Earlier result narratives and superseded roadmaps were removed because they
duplicated the registry and contained claims that are no longer admissible.
The registry retains their status, invalidation reason and artifact location.
Completed pre-scout implementation plans are preserved under `../legacy/docs/`.
