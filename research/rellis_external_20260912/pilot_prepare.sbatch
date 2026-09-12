#!/bin/bash
#SBATCH --job-name=rellis-pilot-data
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/pilot_data_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/pilot_data_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_external_20260912
PY=/home/shv7753/envs/hazard-exp01b-r2/bin/python
SRC="$TASK_ROOT/research/rellis_external_20260912"
sha256sum "$SRC"/qualification_protocol.json "$SRC"/metadata_audit.py "$SRC"/prepare_pilot.py "$SRC"/remote_zip.py > "$TASK_ROOT/artifacts/FREEZE_METADATA_${SLURM_JOB_ID}.sha256"
"$PY" "$SRC/metadata_audit.py" --root "$TASK_ROOT"
"$PY" "$SRC/prepare_pilot.py" --root "$TASK_ROOT"
