#!/bin/bash
#SBATCH --job-name=coarse-paper-package
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/package_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/package_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912
module load texlive/2026
export PYTHONPATH="$TASK_ROOT/paper_deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/same_information_reconstruction_20260912/package_paper.py" \
  --root "$TASK_ROOT" --build "$TASK_ROOT/paper/iclr2027/build_6197893" \
  --out "$TASK_ROOT/delivery_${SLURM_JOB_ID}"
