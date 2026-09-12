#!/bin/bash
#SBATCH --job-name=rellis-revision-delivery
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/delivery_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/delivery_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_revision_20260912
SRC="$TASK_ROOT/research/rellis_revision_20260912"
sha256sum "$SRC"/collect_delivery.py > "$TASK_ROOT/artifacts/FREEZE_COLLECT_${SLURM_JOB_ID}.sha256"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/collect_delivery.py"
