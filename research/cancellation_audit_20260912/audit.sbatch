#!/bin/bash
#SBATCH --job-name=cancellation-cohort
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/audit_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/audit_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_cancellation_audit_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$TASK_ROOT/research/cancellation_audit_20260912"
/home/shv7753/envs/hazard-exp01b-r2/bin/python -m unittest -v test_audit
/home/shv7753/envs/hazard-exp01b-r2/bin/python audit.py --root "$TASK_ROOT"
