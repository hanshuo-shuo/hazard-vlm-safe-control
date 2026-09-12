#!/bin/bash
#SBATCH --job-name=hazard-balanced-repair
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --array=0-2%3
#SBATCH --output=/projects/p33100/siosio/hazard_composition_audit_20260912/logs/repair_%A_%a.out
#SBATCH --error=/projects/p33100/siosio/hazard_composition_audit_20260912/logs/repair_%A_%a.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_composition_audit_20260912
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/composition_audit_20260912/run_balanced.py" \
  --seed-index "$SLURM_ARRAY_TASK_ID" --output "$TASK_ROOT/repair_runs/seed_$SLURM_ARRAY_TASK_ID"
