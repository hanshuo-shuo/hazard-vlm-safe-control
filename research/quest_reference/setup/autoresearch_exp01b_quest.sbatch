#!/bin/bash
# One fixed-tier EXP-01B-R2 development run on a Quest H100.
# Submit through setup/submit_autoresearch_exp01b.sh from the local workspace.
#SBATCH --account=p33100
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=sxm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=hazard_ar_exp01b
#SBATCH --output=autoresearch/exp01b_r2/remote_logs/%x_%j.log

set -euo pipefail

ROOT="${AR_REMOTE_ROOT:-$HOME/hazard-autoresearch-exp01b-r2}"
ENV_DIR="${AR_ENV_DIR:-$HOME/envs/hazard-exp01b-r2}"
RUN_ID="${AR_RUN_ID:?AR_RUN_ID is required}"
TIER="${AR_TIER:-quick}"
OUTPUT="$ROOT/autoresearch/exp01b_r2/runs/$RUN_ID"

echo "JOB $SLURM_JOB_ID on $(hostname) @ $(date -Is)"
echo "RUN_ID=$RUN_ID TIER=$TIER ROOT=$ROOT"
nvidia-smi -L

module purge
module load mamba/24.3.0
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing environment: $ENV_DIR" >&2
  echo "Run: bash setup/provision_autoresearch_exp01b.sh" >&2
  exit 2
fi
source activate "$ENV_DIR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$ROOT"
python -m pytest -q tests/test_autoresearch_exp01b.py tests/test_simulator_spatial_field.py
python autoresearch/exp01b_r2/integrity.py

checkpoint_args=()
if [[ "${AR_SAVE_CHECKPOINT:-0}" == "1" ]]; then
  checkpoint_args+=(--save-checkpoint)
fi

timeout --signal=TERM 55m python autoresearch/exp01b_r2/run_dev.py \
  --run-id "$RUN_ID" \
  --tier "$TIER" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --require-cuda \
  "${checkpoint_args[@]}"

echo "DONE $SLURM_JOB_ID @ $(date -Is)"
