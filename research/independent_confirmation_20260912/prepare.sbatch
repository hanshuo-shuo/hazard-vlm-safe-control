#!/bin/bash
#SBATCH --job-name=c3-independent-dataset
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/prepare_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/prepare_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_independent_confirmation_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT"
/home/shv7753/envs/hazard-exp01b-r2/bin/python research/independent_confirmation_20260912/prepare.py --root "$TASK_ROOT"

