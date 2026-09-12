#!/bin/bash
#SBATCH --job-name=rellis-dev-chain
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/dev_chain_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/dev_chain_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_revision_20260912
SRC="$TASK_ROOT/research/rellis_revision_20260912"
sha256sum "$SRC"/revision_protocol.json "$SRC"/audit_development_chain.py > "$TASK_ROOT/artifacts/FREEZE_DEV_CHAIN_${SLURM_JOB_ID}.sha256"
export OMP_NUM_THREADS=2
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/audit_development_chain.py"
