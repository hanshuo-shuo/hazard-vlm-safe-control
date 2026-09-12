#!/bin/bash
set -euo pipefail
C3_REMOTE_ROOT="${1:?Pass the isolated C3 remote directory}"
C3_PYTHON="${2:?Pass the verified Qwen Python executable}"
C3_RUN_NAME="${3:?Pass a fresh run name}"
export HF_HOME="$C3_REMOTE_ROOT/hf_cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
exec "$C3_PYTHON" -u "$C3_REMOTE_ROOT/source/scripts/run_c3_teacher_labels.py" \
  --bundle "$C3_REMOTE_ROOT/teacher_bundle" --model "$C3_REMOTE_ROOT/model" \
  --output "$C3_REMOTE_ROOT/runs/$C3_RUN_NAME"
