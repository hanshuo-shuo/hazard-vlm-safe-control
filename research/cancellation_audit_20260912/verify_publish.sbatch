#!/bin/bash
#SBATCH --job-name=cancellation-verify
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/verify_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/verify_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_cancellation_audit_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/projects/p33100/siosio/hazard_independent_confirmation_20260912/report_deps
export MPLCONFIGDIR="$TASK_ROOT/matplotlib_cache"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/cancellation_audit_20260912/verify_publish.py" --root "$TASK_ROOT"
