#!/bin/bash
set -euo pipefail
C3_REMOTE_ROOT="${1:?Pass the isolated C3 remote directory}"
C3_PYTHON="${2:?Pass the verified Qwen Python executable}"
export HF_HOME="$C3_REMOTE_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
exec "$C3_PYTHON" -u "$C3_REMOTE_ROOT/source/scripts/diagnose_c3_teacher_vision.py" \
  --bundle "$C3_REMOTE_ROOT/vision_diagnostic" --model "$C3_REMOTE_ROOT/model" \
  --output "$C3_REMOTE_ROOT/runs/vision_diagnostic_v1"
