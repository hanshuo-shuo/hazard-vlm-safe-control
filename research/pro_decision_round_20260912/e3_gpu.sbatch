#!/bin/bash
#SBATCH --job-name=pro-e3-certify
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --array=0-4%3
#SBATCH --output=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e3_%A_%a.out
#SBATCH --error=/projects/p33100/siosio/hazard_pro_decision_round_20260912/logs/e3_%A_%a.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_pro_decision_round_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/pro_decision_round_20260912/run_e3.py" --root "$TASK_ROOT" --index "$SLURM_ARRAY_TASK_ID"
