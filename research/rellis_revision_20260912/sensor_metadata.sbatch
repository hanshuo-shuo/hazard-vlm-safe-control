#!/bin/bash
#SBATCH --job-name=rellis-sensor-index
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/sensor_index_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/sensor_index_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_revision_20260912
SRC="$TASK_ROOT/research/rellis_revision_20260912"
sha256sum "$SRC"/revision_protocol.json "$SRC"/sensor_metadata.py > "$TASK_ROOT/artifacts/FREEZE_SENSOR_INDEX_${SLURM_JOB_ID}.sha256"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/sensor_metadata.py"
