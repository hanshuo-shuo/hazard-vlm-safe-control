#!/bin/bash
#SBATCH --job-name=rellis-report
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/report_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/report_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_external_20260912
export PYTHONPATH=/projects/p33100/siosio/hazard_independent_confirmation_20260912/report_deps
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/rellis_external_20260912/render_qualification.py" --root "$TASK_ROOT"
