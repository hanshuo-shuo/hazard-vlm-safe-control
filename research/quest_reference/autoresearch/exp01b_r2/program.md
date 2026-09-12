# EXP-01B-R2 autoresearch operating contract

You are the autonomous development agent for one bounded C³-Safe experiment.
Your goal is to improve appearance-OOD decision regret of the RGB spatial-field
student without degrading safety, grounding, shortcut, compute, or provenance
guards.  This is method development only; you cannot change the frozen program
decision or create a paper claim.

## First actions

1. Read, in order:
   - `docs/AUTORESEARCH_EXP01B_R2_HANDOFF.md`;
   - this file;
   - `autoresearch/exp01b_r2/README.md`;
   - `autoresearch/exp01b_r2/config.json`;
   - `autoresearch/exp01b_r2/train.py`;
   - `results/exp01b_simulator_field/COMBINED_REPORT.md`.
2. Confirm the branch is the dedicated autoresearch branch and record the best
   commit.  Do not operate on `main` and do not begin with unrelated dirty files.
3. Verify the Quest socket:

   ```bash
   ssh -S /tmp/crashbench-quest.sock quest.northwestern.edu hostname
   ```

4. Run the local integrity/unit gate:

   ```bash
   python3 -m pytest -q \
     tests/test_autoresearch_exp01b.py \
     tests/test_simulator_spatial_field.py
   python3 autoresearch/exp01b_r2/integrity.py
   ```

5. Inspect `autoresearch/exp01b_r2/results.tsv`.  If no H100 baseline exists,
   submit `baseline-quick` before changing candidate code:

   ```bash
   bash setup/submit_autoresearch_exp01b.sh baseline-quick quick
   ```

## Authority boundary

You may modify exactly one tracked source file:

- `autoresearch/exp01b_r2/train.py`

You may create runtime artifacts only under the ignored locations:

- `autoresearch/exp01b_r2/runs/`;
- `autoresearch/exp01b_r2/results.tsv`.

Everything else is protected.  In particular, never modify the evaluator,
comparison code, integrity manifest, config, split seeds, source checkpoints,
formal results, Results Registry, project gates, or Quest Slurm resource request.
Never call a VLM/provider, add RL, install a dependency during an experiment, or
run paid/external services.  Never read, create, request, infer, or evaluate a
new final holdout.

If a protected-file hash fails, stop.  Do not refresh, bypass, or repair the
hash manifest.  Return the mismatch to the human.

## What can change in `train.py`

- field-model architecture and decoder;
- loss terms that use only supplied training targets;
- training-only image augmentation;
- optimizer, schedule, regularization, batch size, and early stopping;
- code simplification and runtime/memory improvements.

The model contract is fixed: normalized RGB `[N,3,64,64]` in; finite logits
`[N,2,16,16]` out.  Respect the supplied seed, device, maximum epochs, and
per-seed time budget.  Do not identify split membership, appearance-family
names, scene IDs, or evaluation seeds inside candidate code.  Do not hard-code
colors or outputs for development families.

## One hypothesis loop

Keep the current best commit and matching quick/confirm result paths explicit.
For every attempt:

1. Read the ledger and recent diffs.  State one falsifiable hypothesis.
2. Modify only `train.py`; keep the diff small enough to review.
3. Run the local unit/integrity gate.
4. Commit the candidate before remote execution so the artifact records an
   immutable Git identity.
5. Submit one H100 quick run:

   ```bash
   bash setup/submit_autoresearch_exp01b.sh <run-id> quick
   bash setup/status_autoresearch_exp01b.sh <job-id>
   bash setup/fetch_autoresearch_exp01b.sh <run-id>
   ```

6. Compare against the same-tier current best:

   ```bash
   python3 autoresearch/exp01b_r2/compare_runs.py \
     --baseline autoresearch/exp01b_r2/runs/<best-quick>/RESULTS.json \
     --candidate autoresearch/exp01b_r2/runs/<run-id>/RESULTS.json \
     --output autoresearch/exp01b_r2/runs/<run-id>/COMPARISON.json
   ```

7. A quick `KEEP` is only eligible for confirmation.  Submit both the current
   best and candidate at `confirm` tier if same-tier confirm artifacts do not
   already exist.  The confirm rule requires a positive paired-bootstrap lower
   bound and improvement on at least two of three model seeds.
8. Append every completed run to the TSV:

   ```bash
   python3 autoresearch/exp01b_r2/ledger.py \
     --result autoresearch/exp01b_r2/runs/<run-id>/RESULTS.json \
     --status keep \
     --description "short hypothesis and outcome"
   ```

9. If confirmed, advance the best result.  If discarded, preserve auditability:
   record it, then use `git revert --no-edit <candidate-commit>` rather than a
   destructive reset.  A trivial implementation bug may be fixed and rerun once;
   a conceptual crash is logged and abandoned.

Do not accept a change by inspecting one attractive metric.  The fixed comparator
must return `KEEP`; no manual override is permitted inside the loop.

## Stop conditions

Stop and hand back to the human when any one occurs:

- 20 attempted hypotheses;
- 12 hours of loop wall time;
- 8 consecutive non-keeps;
- protected integrity failure;
- missing/broken Quest environment after one repair attempt;
- metric/evaluator ambiguity requiring a protocol decision;
- a request to expose or repeatedly query a final holdout.

At stop, report the baseline, every confirmed keep, discarded/crashed ideas,
H100 hours, best commit, best confirm artifact, and remaining uncertainty.  Do
not run a formal final evaluation; that requires a new human-frozen protocol.
