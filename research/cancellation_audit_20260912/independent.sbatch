#!/bin/bash
#SBATCH --job-name=input-operator-confirm
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/independent_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/independent_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_cancellation_audit_20260912
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/cancellation_audit_20260912/independent.py" --root "$TASK_ROOT"
