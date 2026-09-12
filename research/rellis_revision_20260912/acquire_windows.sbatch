#!/bin/bash
#SBATCH --job-name=rellis-dev-sensors
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/dev_sensors_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_revision_20260912/logs/dev_sensors_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_revision_20260912
SRC="$TASK_ROOT/research/rellis_revision_20260912"
sha256sum "$SRC"/revision_protocol.json "$SRC"/acquire_windows.py "$SRC"/decision_contract.py > "$TASK_ROOT/artifacts/FREEZE_WINDOWS_${SLURM_JOB_ID}.sha256"
cd "$SRC"
/home/shv7753/envs/hazard-exp01b-r2/bin/python -m unittest -v test_decision_contract
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/acquire_windows.py"
