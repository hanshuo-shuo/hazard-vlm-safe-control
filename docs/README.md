# Documentation Map

Updated: 2026-07-28.

## For a research discussion

Read [`../RESEARCH_STORY.md`](../RESEARCH_STORY.md). It is the short,
plain-English version with figures, limitations, and literature-supported next
steps.

## Current truth sources

| Document | Authority |
|---|---|
| [`INTERFACE_CONTRACT_PROTOCOL.md`](INTERFACE_CONTRACT_PROTOCOL.md) | Frozen semantics for the current interface-contract experiment |
| [`RESULTS_REGISTRY.md`](RESULTS_REGISTRY.md) | Only authority for artifact validity (`VALIDATED`, `PILOT_ONLY`, `INVALIDATED`, `ARCHIVED`) |
| [`PAID_RUN_RELEASE.md`](PAID_RUN_RELEASE.md) | Blocked general provider-release record |
| [`review_status_registry.json`](review_status_registry.json) | Machine-readable status of earlier review items |

## Provenance-locked internal documents

These files are not recommended as the first reading. They remain here because
checked-in manifests or tests depend on their exact paths and contents.

| Document | Why it remains |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | Hashed by the blocked release manifest and used by release tests |
| [`ICLR_PLAN.md`](ICLR_PLAN.md) | Hashed in the provider-free dry-run provenance |
| [`INTERFACE_CONTRACT_MAINLINE.md`](INTERFACE_CONTRACT_MAINLINE.md) | Hashed in the provider-free dry-run provenance |
| [`RESEARCH_REVIEW_COMMENTS.md`](RESEARCH_REVIEW_COMMENTS.md) | Parsed against the machine-readable review registry by tests |

## Detailed evidence

- Native scout report:
  [`../results/interface_contract_scout_native_analysis/REPORT.md`](../results/interface_contract_scout_native_analysis/REPORT.md)
- Machine-readable scout analysis:
  [`../results/interface_contract_scout_native_analysis/ANALYSIS.json`](../results/interface_contract_scout_native_analysis/ANALYSIS.json)
- Provider-free executable-bridge summary:
  [`../results/interface_contract_experiment_0/SUMMARY.json`](../results/interface_contract_experiment_0/SUMMARY.json)

## Archive

Superseded narratives and completed internal handoffs are indexed in
[`../legacy/docs/README.md`](../legacy/docs/README.md). Raw historical outputs
stay at their registered paths so result hashes and provenance links remain
valid.
