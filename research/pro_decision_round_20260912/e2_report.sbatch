#!/bin/bash
#SBATCH --job-name=pro-e2-effect
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e2_report_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e2_report_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_pro_decision_round_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/pro_decision_round_20260912/aggregate_e2.py" --root "$TASK_ROOT"
