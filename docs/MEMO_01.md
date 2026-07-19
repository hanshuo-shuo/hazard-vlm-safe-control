# Memo #1 — W1 gate

**Date:** 2026-07-19  
**Scope:** WP-1.1–WP-1.5, environment trust, protocol freeze, and factor orthogonality  
**Status:** PASS

## This week’s decision

Proceed to W2. The semantic-zone environment now rejects invalid layouts, the
experimental contract is frozen, and the three B06 factors have independent
control paths:

- `zone_semantics` selects renderer appearance only;
- `prompt_level` selects the privilege section only;
- `capability` selects the public capability card only.

Changing one factor does not silently change the other prompt sections or the
scene layout. The legacy `zone_mode` argument remains only as a compatibility
alias for older figure scripts; the CLI evaluation path passes the independent
`prompt_level` explicitly.

## Gate evidence

| Gate | Evidence | Result |
|---|---|---|
| Snapshot orthogonality | `python -m pytest -q tests/test_factor_orthogonality.py -p no:cacheprovider` | **6 passed** |
| Test suite acceptance sweep | `LAYOUT_TEST_SEEDS=25 python -m pytest -q tests -p no:cacheprovider` | **14 passed** |
| Layout regression | Existing golden layout SHA-256 snapshots remain unchanged | **PASS** |
| Protocol freeze | [`PROTOCOL.md`](PROTOCOL.md), `protocol-v1`, version `1.2.1`, frozen 2026-07-17 | **FROZEN** |
| Static checks | `git diff --check` and Python compilation | **PASS** |

The 25-seed sweep is the bounded W1 acceptance run. The layout test’s default
10,000-seed stress sweep remains available through `LAYOUT_TEST_SEEDS=10000`;
its larger runtime is a stress/performance concern, not a semantic exception.

## Corrective action recorded

The prior sampler could exhaust all 101 complete-layout retries for a narrow
three-zone corridor band (reproduced at seed 21). A checked secondary sampler
now runs only after the final complete-layout retry. It validates every joint
candidate with `zone_layout_valid`, preserves the existing resample sequence and
golden bytes, and resolves the seed-21 failure without introducing an
unchecked fallback.

## Next week

Implement the fair router/zone-source harness, keep the frozen protocol and
factor vector unchanged, and add artifact-level provenance before any formal
run.

