#!/bin/bash
#SBATCH --job-name=c3-independent-preflight
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/preflight_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/preflight_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_independent_confirmation_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT"
/home/shv7753/envs/hazard-exp01b-r2/bin/python research/independent_confirmation_20260912/test_study.py
/home/shv7753/envs/hazard-exp01b-r2/bin/python -c 'import importlib.util; print({k: importlib.util.find_spec(k) is not None for k in ["matplotlib", "reportlab", "pytest"]})'
