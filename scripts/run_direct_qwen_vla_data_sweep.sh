#!/usr/bin/env bash
set -euo pipefail

# Overnight data-size sweep for the Qwen-backed direct VLA baseline.
#
# What it does for each dataset size:
#   1. collect SafeExpert demos with that many episodes
#   2. train Qwen-VL + action head on that dataset
#   3. eval on the same 1000 seeds used by the first 1000 expert episodes
#   4. eval on a held-out 1000-seed split
#
# Override defaults like:
#   EPISODE_SIZES="1000 3000 5000" EPOCHS=2 bash run_direct_qwen_vla_data_sweep.sh

cd "$(dirname "$0")"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-VL-7B-Instruct}"
PROMPT_MODE="${PROMPT_MODE:-image_state}"
SEED="${SEED:-42}"
HELDOUT_SEED="${HELDOUT_SEED:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
MAX_STEPS="${MAX_STEPS:-300}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
RENDER_SIZE="${RENDER_SIZE:-128}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-regression}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-512}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.05}"
GENERATIVE_COND_DIM="${GENERATIVE_COND_DIM:-256}"
GENERATIVE_TIME_DIM="${GENERATIVE_TIME_DIM:-64}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
DIFFUSION_BETA_SCHEDULE="${DIFFUSION_BETA_SCHEDULE:-sigmoid}"
DIFFUSION_BETA_MIN="${DIFFUSION_BETA_MIN:-1e-4}"
DIFFUSION_BETA_MAX="${DIFFUSION_BETA_MAX:-0.02}"
FLOW_SAMPLE_STEPS="${FLOW_SAMPLE_STEPS:-16}"
ACTION_SAMPLES="${ACTION_SAMPLES:-1}"
TRAIN_VLM="${TRAIN_VLM:-0}"
TRAIN_LORA="${TRAIN_LORA:-0}"
LORA_LR="${LORA_LR:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LORA_GRADIENT_CHECKPOINTING="${LORA_GRADIENT_CHECKPOINTING:-0}"

# Keep this modest by default. Full Qwen forwards are the bottleneck, not expert collection.
EPISODE_SIZES="${EPISODE_SIZES:-1000 3000 5000}"

RUN_ROOT="${RUN_ROOT:-hazard/direct_qwen_vla_data_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

TRAIN_EXTRA_ARGS=()
if is_truthy "$TRAIN_VLM"; then
  TRAIN_EXTRA_ARGS+=(--train_vlm)
fi
if is_truthy "$TRAIN_LORA"; then
  TRAIN_EXTRA_ARGS+=(
    --train_lora
    --lora_lr "$LORA_LR"
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "$LORA_TARGET_MODULES"
  )
  if is_truthy "$LORA_GRADIENT_CHECKPOINTING"; then
    TRAIN_EXTRA_ARGS+=(--lora_gradient_checkpointing)
  else
    TRAIN_EXTRA_ARGS+=(--no-lora_gradient_checkpointing)
  fi
fi

echo "Direct Qwen VLA data-size sweep"
echo "  run_root=$RUN_ROOT"
echo "  model=$MODEL_PATH"
echo "  prompt_mode=$PROMPT_MODE"
echo "  action_head_type=$ACTION_HEAD_TYPE action_samples=$ACTION_SAMPLES"
echo "  episode_sizes=$EPISODE_SIZES"
echo "  epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR max_train_samples=$MAX_TRAIN_SAMPLES"
echo "  train_vlm=$TRAIN_VLM train_lora=$TRAIN_LORA lora_lr=$LORA_LR"
echo "  eval: same seed=$SEED, heldout seed=$HELDOUT_SEED, episodes=$EVAL_EPISODES"

for N in $EPISODE_SIZES; do
  RUN_DIR="$RUN_ROOT/episodes_${N}"
  mkdir -p "$RUN_DIR"

  DATASET="$RUN_DIR/expert_${N}eps.npz"
  CKPT="$RUN_DIR/direct_qwen_vla_${ACTION_HEAD_TYPE}_${N}eps.pt"
  SAME_SUMMARY="$RUN_DIR/eval_${ACTION_HEAD_TYPE}_same_seed_${EVAL_EPISODES}eps_summary.json"
  HELDOUT_SUMMARY="$RUN_DIR/eval_${ACTION_HEAD_TYPE}_heldout_seed_${EVAL_EPISODES}eps_summary.json"

  echo
  echo "=============================="
  echo "Dataset size: ${N} expert episodes"
  echo "=============================="

  if [[ ! -f "$DATASET" ]]; then
    python direct_vla_hazard.py --mode collect_expert \
      --episodes "$N" \
      --seed "$SEED" \
      --max_steps "$MAX_STEPS" \
      --out "$DATASET" \
      --log_every "$LOG_EVERY"
  else
    echo "Dataset exists, skipping collect: $DATASET"
  fi

  python direct_vla_hazard.py --mode train \
    --dataset "$DATASET" \
    --ckpt "$CKPT" \
    --model_path "$MODEL_PATH" \
    --prompt_mode "$PROMPT_MODE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --render_size "$RENDER_SIZE" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --action_head_type "$ACTION_HEAD_TYPE" \
    --head_hidden_dim "$HEAD_HIDDEN_DIM" \
    --head_dropout "$HEAD_DROPOUT" \
    --generative_cond_dim "$GENERATIVE_COND_DIM" \
    --generative_time_dim "$GENERATIVE_TIME_DIM" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --diffusion_beta_schedule "$DIFFUSION_BETA_SCHEDULE" \
    --diffusion_beta_min "$DIFFUSION_BETA_MIN" \
    --diffusion_beta_max "$DIFFUSION_BETA_MAX" \
    --flow_sample_steps "$FLOW_SAMPLE_STEPS" \
    --log_every "$LOG_EVERY" \
    "${TRAIN_EXTRA_ARGS[@]}"

  python direct_vla_hazard.py --mode eval \
    --ckpt "$CKPT" \
    --model_path "$MODEL_PATH" \
    --prompt_mode from_ckpt \
    --episodes "$EVAL_EPISODES" \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --summary "$SAME_SUMMARY" \
    --render_size "$RENDER_SIZE" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --action_samples "$ACTION_SAMPLES" \
    --log_every "$LOG_EVERY"

  python direct_vla_hazard.py --mode eval \
    --ckpt "$CKPT" \
    --model_path "$MODEL_PATH" \
    --prompt_mode from_ckpt \
    --episodes "$EVAL_EPISODES" \
    --seed "$HELDOUT_SEED" \
    --max_steps "$MAX_STEPS" \
    --summary "$HELDOUT_SUMMARY" \
    --render_size "$RENDER_SIZE" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --action_samples "$ACTION_SAMPLES" \
    --log_every "$LOG_EVERY"

  echo "Finished N=$N"
  echo "  same:    $SAME_SUMMARY"
  echo "  heldout: $HELDOUT_SUMMARY"
done

python - <<'PY' "$RUN_ROOT"
import glob, json, os, sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "episodes_*", "eval_*_summary.json"))):
    with open(path) as f:
        s = json.load(f)
    rows.append({
        "path": path,
        "train_vlm": s.get("train_vlm"),
        "train_lora": s.get("train_lora"),
        "action_head_type": s.get("action_head_type"),
        "action_samples": s.get("action_samples"),
        "success_rate": s.get("success_rate"),
        "hazard_hit_rate": s.get("hazard_hit_rate"),
        "timeout_rate": s.get("timeout_rate"),
        "mean_return": s.get("mean_return"),
        "mean_final_dist": s.get("mean_final_dist"),
        "mean_min_clearance": s.get("mean_min_clearance"),
        "best_loss": s.get("best_loss"),
    })

out = os.path.join(root, "sweep_summary.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=2)

print("\nSweep summary saved:", out)
for r in rows:
    print(
        os.path.relpath(r["path"], root),
        "success=", r["success_rate"],
        "hazard=", r["hazard_hit_rate"],
        "timeout=", r["timeout_rate"],
        "return=", r["mean_return"],
    )
PY

echo
echo "All done: $RUN_ROOT"
