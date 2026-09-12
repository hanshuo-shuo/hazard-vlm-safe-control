#!/bin/bash
#SBATCH --job-name=oracle-confirm-verify
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_oracle_aggregation_20260912/logs/confirmation_verify_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_oracle_aggregation_20260912/logs/confirmation_verify_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_oracle_aggregation_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/projects/p33100/siosio/hazard_rellis_external_20260912/qualification_deps
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/oracle_aggregation_20260912/verify_confirmation.py" --root "$TASK_ROOT"
