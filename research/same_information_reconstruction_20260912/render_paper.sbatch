#!/bin/bash
#SBATCH --job-name=coarse-paper-render
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/render_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912/logs/render_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_same_information_reconstruction_20260912
export PYTHONPATH="$TASK_ROOT/paper_deps" PYTHONDONTWRITEBYTECODE=1
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/same_information_reconstruction_20260912/render_paper.py" --build "$TASK_ROOT/paper/iclr2027/build_6197203"
