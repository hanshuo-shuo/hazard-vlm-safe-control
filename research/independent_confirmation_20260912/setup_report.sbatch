#!/bin/bash
#SBATCH --job-name=c3-report-dependencies
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/report_deps_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_independent_confirmation_20260912/logs/report_deps_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_independent_confirmation_20260912
test ! -e "$TASK_ROOT/report_deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python -m pip install --disable-pip-version-check \
  --target "$TASK_ROOT/report_deps" --report "$TASK_ROOT/REPORT_DEPENDENCIES.json" \
  'matplotlib==3.10.8' 'numpy==1.23.5' 'pillow==9.4.0'

