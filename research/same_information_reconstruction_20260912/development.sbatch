#!/bin/bash
#SBATCH --job-name=same-info-development
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/development_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/development_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/same_information_reconstruction_20260912/run.py" --root "$TASK_ROOT" --phase development
