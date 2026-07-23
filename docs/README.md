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
| 5 | [RESULTS_REGISTRY.md](RESULTS_REGISTRY.md) | Only authority for whether an artifact may be cited |

The code map is [../STRUCTURE.md](../STRUCTURE.md).

## Current status

- PointHazard B01 layout validation is complete.
- The condition contract is complete.
- `direct/replay × none/oracle` runs through the unified harness.
- Detector/VLM integration and paid API runs remain gated.
- No semantic-safety result is currently `VALIDATED`.

## Historical evidence

Raw pilot outputs, transcripts and invalidated runs remain in `../outputs/`.
Earlier result narratives and superseded roadmaps were removed because they
duplicated the registry and contained claims that are no longer admissible.
The registry retains their status, invalidation reason and artifact location.
