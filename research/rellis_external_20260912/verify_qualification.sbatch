#!/bin/bash
#SBATCH --job-name=rellis-verify
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/verify_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/verify_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_external_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$TASK_ROOT/qualification_deps"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/rellis_external_20260912/verify_qualification.py" --root "$TASK_ROOT"
