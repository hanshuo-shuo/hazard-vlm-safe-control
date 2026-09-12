#!/bin/bash
#SBATCH --job-name=rellis-revision-sources
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/sources_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/sources_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_revision_20260912
SRC="$TASK_ROOT/research/rellis_revision_20260912"
sha256sum "$SRC"/revision_protocol.json "$SRC"/PRO_REVIEW.md "$SRC"/audit_sources.py > "$TASK_ROOT/artifacts/FREEZE_SOURCES_${SLURM_JOB_ID}.sha256"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/audit_sources.py"
