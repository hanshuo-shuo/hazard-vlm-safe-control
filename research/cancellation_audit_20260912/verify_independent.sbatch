#!/bin/bash
#SBATCH --job-name=interaction-check
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/independent_verify_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/independent_verify_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_cancellation_audit_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/projects/p33100/siosio/hazard_independent_confirmation_20260912/report_deps
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/cancellation_audit_20260912/verify_independent.py" --root "$TASK_ROOT"
