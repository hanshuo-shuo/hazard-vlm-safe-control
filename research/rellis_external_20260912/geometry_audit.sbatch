#!/bin/bash
#SBATCH --job-name=rellis-geometry
#SBATCH --account=p33100
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/geometry_%j.out
#SBATCH --error=/projects/p33100/siosio/hazard_rellis_external_20260912/logs/geometry_%j.err
set -euo pipefail
TASK_ROOT=/projects/p33100/siosio/hazard_rellis_external_20260912
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$TASK_ROOT/qualification_deps"
SRC="$TASK_ROOT/research/rellis_external_20260912"
sha256sum "$SRC/geometry_audit.py" "$SRC/qualification_protocol.json" > "$TASK_ROOT/artifacts/FREEZE_GEOMETRY_${SLURM_JOB_ID}.sha256"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/test_geometry.py"
/home/shv7753/envs/hazard-exp01b-r2/bin/python "$SRC/geometry_audit.py" --root "$TASK_ROOT"
