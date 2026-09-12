#!/bin/bash
#SBATCH --job-name=rellis-metadata
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/metadata_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/metadata_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_external_20260912
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$TASK_ROOT/research/rellis_external_20260912/acquire_metadata.py" --root "$TASK_ROOT"
