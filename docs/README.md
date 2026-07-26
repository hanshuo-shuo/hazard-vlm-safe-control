# Documentation index

Updated: 2026-07-26.

The documentation set is intentionally small. Current experiment semantics,
execution state and result validity each have one owner.

| Read order | Document | Authority |
|---:|---|---|
| 1 | [PROTOCOL.md](PROTOCOL.md) | Frozen task, information, evaluator, replay and seed semantics |
| 2 | [review_status_registry.json](review_status_registry.json) | Machine-readable review-item status authority |
| 3 | [RESEARCH_REVIEW_COMMENTS.md](RESEARCH_REVIEW_COMMENTS.md) | Blockers and required fixes; rendered status snapshot |
| 4 | [ICLR_PLAN.md](ICLR_PLAN.md) | Research question, experiment design and publication gates |
| 5 | [CLAUDE_PLAN.md](CLAUDE_PLAN.md) | Current implementation plan and acceptance evidence |
| 6 | [PAID_RUN_RELEASE.md](PAID_RUN_RELEASE.md) | Blocked release-readiness and authorization record |
| 7 | [RESULTS_REGISTRY.md](RESULTS_REGISTRY.md) | Only authority for whether an artifact may be cited |
| 8 | [INTERFACE_CONTRACT_MAINLINE.md](INTERFACE_CONTRACT_MAINLINE.md) | Exploratory interface-contract evidence and ICLR recovery gate |
| 9 | [INTERFACE_CONTRACT_SCOUT_REPORT.md](INTERFACE_CONTRACT_SCOUT_REPORT.md) | Illustrated 120-call scout narrative, native replay evidence and continuation decision |

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
- Detector integration and formal paid API runs remain gated; the separately
  authorized interface-contract micro-pilot does not unlock the release manifest.
- No semantic-safety result is currently `VALIDATED`.
- A five-seed, three-valid-model interface-contract micro-pilot is complete and
  remains `PILOT_ONLY`; it finds action/waypoint/mapping/ranking sensitivity but
  fails the predeclared two-environment semantic-contract replication gate.

## Historical evidence

Raw pilot outputs, transcripts and invalidated runs remain in `../outputs/`.
Earlier result narratives and superseded roadmaps were removed because they
duplicated the registry and contained claims that are no longer admissible.
The registry retains their status, invalidation reason and artifact location.
