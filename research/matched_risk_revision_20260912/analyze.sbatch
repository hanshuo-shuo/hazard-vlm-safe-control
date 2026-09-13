#!/bin/bash
#SBATCH --job-name=matched-risk-curves
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/p33100/siosio/hazard_matched_risk_revision_20260912/logs/analysis_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_matched_risk_revision_20260912/logs/analysis_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_matched_risk_revision_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT/research/matched_risk_revision_20260912"
/home/shv7753/envs/hazard-exp01b-r2/bin/python -m unittest -v test_curves
/home/shv7753/envs/hazard-exp01b-r2/bin/python analyze.py --root "$TASK_ROOT"
