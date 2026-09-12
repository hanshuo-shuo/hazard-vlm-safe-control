#!/bin/bash
#SBATCH --job-name=c3-cached-verification
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/resume_report_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/resume_report_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_independent_confirmation_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$TASK_ROOT/matplotlib_cache"
cd "$TASK_ROOT"
test -f "$TASK_ROOT/reports/SUMMARY.json"
test ! -e "$TASK_ROOT/reports/VERIFICATION.json"
/home/shv7753/envs/hazard-exp01b-r2/bin/python research/independent_confirmation_20260912/verify_cached.py --root "$TASK_ROOT"
PYTHONPATH="$TASK_ROOT/report_deps" /home/shv7753/envs/hazard-exp01b-r2/bin/python research/independent_confirmation_20260912/report.py --root "$TASK_ROOT"
