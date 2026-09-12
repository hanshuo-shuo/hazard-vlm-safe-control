#!/bin/bash
#SBATCH --job-name=oracle-decompose
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/p33100/siosio/hazard_oracle_aggregation_20260912/logs/discovery_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_oracle_aggregation_20260912/logs/discovery_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_oracle_aggregation_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/oracle_aggregation_20260912/decompose.py" --root "$TASK_ROOT"
