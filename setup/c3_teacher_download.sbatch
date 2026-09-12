#!/bin/bash
# Resource requests are supplied explicitly by the submitting command.
set -euo pipefail
C3_REMOTE_ROOT="${1:?Pass the isolated C3 remote directory}"
C3_PYTHON="${2:?Pass the verified Qwen Python executable}"
export HF_HOME="$C3_REMOTE_ROOT/hf_cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
exec "$C3_PYTHON" -u "$C3_REMOTE_ROOT/source/scripts/download_c3_teacher_model.py" \
  --spec "$C3_REMOTE_ROOT/MODEL_DOWNLOAD_SPEC.json" --output "$C3_REMOTE_ROOT/model"
