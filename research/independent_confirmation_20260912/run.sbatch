#!/bin/bash
#SBATCH --job-name=c3-independent-confirm
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-14%3
#SBATCH --output=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/%A_%a.out
#SBATCH --error=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/%A_%a.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_independent_confirmation_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$TASK_ROOT"
/home/shv7753/envs/hazard-exp01b-r2/bin/python research/independent_confirmation_20260912/run.py --root "$TASK_ROOT" --task-index "$SLURM_ARRAY_TASK_ID"

