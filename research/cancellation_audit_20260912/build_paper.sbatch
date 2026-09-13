#!/bin/bash
#SBATCH --job-name=interaction-paper
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/paper_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_cancellation_audit_20260912/logs/paper_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_cancellation_audit_20260912
OLD_ROOT=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912
module load texlive/2026
BUILD_DIR="$TASK_ROOT/paper_build_${SLURM_JOB_ID}"
mkdir "$BUILD_DIR"
cp -R "$TASK_ROOT/paper/iclr2027" "$BUILD_DIR/source"
bash "$BUILD_DIR/source/build.sh" > "$BUILD_DIR/compile.log" 2>&1
export PYTHONPATH="$OLD_ROOT/paper_deps"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$OLD_ROOT/research/same_information_reconstruction_20260912/render_paper.py" --build "$BUILD_DIR/source/build"
