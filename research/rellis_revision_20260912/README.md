# Bounded RELLIS measurement-interface revision

**Closed at `STOP_PREREQUISITES_UNCERTAIN`, before any fresh qualification set.**
The 24 consumed central PLY scans match raw sensor content exactly; a fixed
per-point motion prototype runs, but independent physical error ranges and the
complete spatial partition remain unqualified. This is an early stop within
the upper budget, not a 0/48 result or an external mechanism result.

- [Chinese progress and decision](../../docs/RELLIS_REVISION_PROGRESS_2026-09-12.md)
- [Machine-readable decision](artifacts/REVISION_DECISION.json)
- [24-frame time/coordinate audit](artifacts/DEVELOPMENT_CHAIN.json)
- [Exact source-time correspondence table](artifacts/TIME_POSE_CORRESPONDENCE_EXACT.md)
- [Motion and independent specific-force observations](artifacts/MOTION_AUDIT.json)
- [Verified computation and raw-byte accounting](artifacts/REVISION_VERIFICATION.json)
- [Development ledger](DEVELOPMENT_LOG.md)
- [Upstream versions and file fingerprints](artifacts/UPSTREAM_PROVENANCE.json)

This round implements [Pro's third review](PRO_REVIEW.md), preserving the first
single-scan qualification failure. Read the [new protocol](revision_protocol.json).
The earlier protocol and scientific artifacts are immutable. Its ambiguous
"evaluable area" wording is superseded here by **full corridor area**.

There are two sequential freezes. The first fixes the budget, question, candidates,
two permitted geometry versions and three outcomes. Only if the time/pose,
coordinate, physical-error and spatial-isolation prerequisites are substantiated
does a second freeze name 48 fresh centres, their complete support scans, 12 blocks,
the empirical uncertainty construction and 12 independent human-review frames.
No new qualification sample is opened before that freeze.
Hashes provide integrity fingerprints and a Slurm-recorded chronology; they are
not cryptographic signatures or a public preregistration service. The protocol's
"signed-by-hash" wording refers only to that machine freeze.

All processing runs on Quest, under
`/projects/p33100/siosio/hazard_rellis_revision_20260912`.
Existing development data are read-only inputs from
`/projects/p33100/siosio/hazard_rellis_external_20260912`.
No segmentation, mechanism, VLM or RL computation is permitted in this round.

The two-day budget is an upper bound. An unsupported prerequisite is a stopping
result, not an invitation to invent a correspondence or develop a new SLAM system.
Numerical transform agreement is distinct from physical alignment evidence.

The new binary-only path is registered in advance: a candidate is compliant when
its full empirical upper cost bound is at most 0.02, noncompliant when its lower
bound exceeds 0.02, and otherwise undetermined. Continuous cost/regret/residual
analysis additionally requires p95 interval width at most 0.005. Neither path
certifies physical traversability or the preceding 0-6 m of travel.

The `.sbatch` files record this completed run and contain its actual Quest paths.
Do not blindly resubmit them against the existing output directories. Reproduction
requires a separate root, the recorded source versions and explicit preservation
of the prerequisite stop; this repository does not authorize a third geometry
search or new qualification samples.
