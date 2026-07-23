# Documentation index

Updated: 2026-07-23.

The documentation set is intentionally small. Current experiment semantics,
execution state and result validity each have one owner.

| Read order | Document | Authority |
|---:|---|---|
| 1 | [PROTOCOL.md](PROTOCOL.md) | Frozen task, information, evaluator, replay and seed semantics |
| 2 | [RESEARCH_REVIEW_COMMENTS.md](RESEARCH_REVIEW_COMMENTS.md) | Blockers and required fixes |
| 3 | [ICLR_PLAN.md](ICLR_PLAN.md) | Research question, experiment design and publication gates |
| 4 | [CLAUDE_PLAN.md](CLAUDE_PLAN.md) | Current implementation plan and acceptance evidence |
| 5 | [PAID_RUN_RELEASE.md](PAID_RUN_RELEASE.md) | Blocked release-readiness and authorization record |
| 6 | [RESULTS_REGISTRY.md](RESULTS_REGISTRY.md) | Only authority for whether an artifact may be cited |

The code map is [../STRUCTURE.md](../STRUCTURE.md).

## Current status

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
- Detector integration and paid API runs remain gated.
- No semantic-safety result is currently `VALIDATED`.

## Historical evidence

Raw pilot outputs, transcripts and invalidated runs remain in `../outputs/`.
Earlier result narratives and superseded roadmaps were removed because they
duplicated the registry and contained claims that are no longer admissible.
The registry retains their status, invalidation reason and artifact location.
