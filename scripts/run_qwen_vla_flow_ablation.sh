#!/usr/bin/env bash
set -euo pipefail

# Flow-head ablation for the Qwen direct VLA baseline.
#
# Runs:
#   1. Flow head-only      frozen Qwen -> flow action expert
#   2. Flow + LoRA         Qwen LoRA + flow action expert
#
# Each checkpoint is evaluated on both:
#   - same seeds as the expert dataset seed
#   - held-out seeds
#
# Example:
#   bash run_qwen_vla_flow_ablation.sh
#
# Useful overrides:
#   DATASET=hazard/direct_vla_expert_demos.npz EPOCHS=3 EVAL_EPISODES=1000 \
#     bash run_qwen_vla_flow_ablation.sh

cd "$(dirname "$0")"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-VL-7B-Instruct}"
DATASET="${DATASET:-hazard/direct_vla_expert_demos.npz}"
PROMPT_MODE="${PROMPT_MODE:-image_state}"
SEED="${SEED:-42}"
HELDOUT_SEED="${HELDOUT_SEED:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
MAX_STEPS="${MAX_STEPS:-300}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
RENDER_SIZE="${RENDER_SIZE:-128}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-512}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.05}"
GENERATIVE_COND_DIM="${GENERATIVE_COND_DIM:-256}"
GENERATIVE_TIME_DIM="${GENERATIVE_TIME_DIM:-64}"
FLOW_SAMPLE_STEPS="${FLOW_SAMPLE_STEPS:-16}"
ACTION_SAMPLES="${ACTION_SAMPLES:-4}"

LORA_LR="${LORA_LR:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LORA_GRADIENT_CHECKPOINTING="${LORA_GRADIENT_CHECKPOINTING:-0}"

RUN_ROOT="${RUN_ROOT:-hazard/qwen_vla_flow_ablation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET"
  echo "Create it first, for example:"
  echo "  python direct_vla_hazard.py --mode collect_expert --episodes 3000 --seed $SEED --out $DATASET"
  exit 1
fi

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

COMMON_TRAIN_ARGS=(
  --mode train
  --dataset "$DATASET"
  --model_path "$MODEL_PATH"
  --prompt_mode "$PROMPT_MODE"
  --action_head_type flow
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --seed "$SEED"
  --max_steps "$MAX_STEPS"
  --render_size "$RENDER_SIZE"
  --torch_dtype "$TORCH_DTYPE"
  --device_map "$DEVICE_MAP"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --head_hidden_dim "$HEAD_HIDDEN_DIM"
  --head_dropout "$HEAD_DROPOUT"
  --generative_cond_dim "$GENERATIVE_COND_DIM"
  --generative_time_dim "$GENERATIVE_TIME_DIM"
  --flow_sample_steps "$FLOW_SAMPLE_STEPS"
  --log_every "$LOG_EVERY"
)

COMMON_EVAL_ARGS=(
  --mode eval
  --model_path "$MODEL_PATH"
  --prompt_mode from_ckpt
  --episodes "$EVAL_EPISODES"
  --max_steps "$MAX_STEPS"
  --render_size "$RENDER_SIZE"
  --torch_dtype "$TORCH_DTYPE"
  --device_map "$DEVICE_MAP"
  --action_samples "$ACTION_SAMPLES"
  --log_every "$LOG_EVERY"
)

LORA_ARGS=(
  --train_lora
  --lora_lr "$LORA_LR"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_target_modules "$LORA_TARGET_MODULES"
)

if is_truthy "$LORA_GRADIENT_CHECKPOINTING"; then
  LORA_ARGS+=(--lora_gradient_checkpointing)
else
  LORA_ARGS+=(--no-lora_gradient_checkpointing)
fi

run_variant() {
  local name="$1"
  local ckpt="$RUN_ROOT/${name}.pt"
  local same_summary="$RUN_ROOT/${name}_eval_same_seed_${EVAL_EPISODES}eps_summary.json"
  local heldout_summary="$RUN_ROOT/${name}_eval_heldout_seed_${EVAL_EPISODES}eps_summary.json"
  local same_log="$RUN_ROOT/${name}_eval_same_seed_${EVAL_EPISODES}eps_episodes.jsonl"
  local heldout_log="$RUN_ROOT/${name}_eval_heldout_seed_${EVAL_EPISODES}eps_episodes.jsonl"
  shift

  echo
  echo "=============================="
  echo "Training: $name"
  echo "=============================="
  python direct_vla_hazard.py "${COMMON_TRAIN_ARGS[@]}" --ckpt "$ckpt" "$@"

  echo
  echo "=============================="
  echo "Eval same-seed: $name"
  echo "=============================="
  python direct_vla_hazard.py "${COMMON_EVAL_ARGS[@]}" \
    --ckpt "$ckpt" \
    --seed "$SEED" \
    --summary "$same_summary" \
    --episode_log "$same_log"

  echo
  echo "=============================="
  echo "Eval heldout-seed: $name"
  echo "=============================="
  python direct_vla_hazard.py "${COMMON_EVAL_ARGS[@]}" \
    --ckpt "$ckpt" \
    --seed "$HELDOUT_SEED" \
    --summary "$heldout_summary" \
    --episode_log "$heldout_log"
}

echo "Qwen VLA flow ablation"
echo "  run_root=$RUN_ROOT"
echo "  dataset=$DATASET"
echo "  model=$MODEL_PATH"
echo "  prompt_mode=$PROMPT_MODE"
echo "  epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR"
echo "  flow_sample_steps=$FLOW_SAMPLE_STEPS action_samples=$ACTION_SAMPLES"
echo "  eval episodes=$EVAL_EPISODES same_seed=$SEED heldout_seed=$HELDOUT_SEED"

run_variant "flow_head_only"
run_variant "flow_lora" "${LORA_ARGS[@]}"

python - "$RUN_ROOT" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "*_summary.json"))):
    with open(path) as f:
        s = json.load(f)
    rows.append({
        "summary": os.path.relpath(path, root),
        "ckpt": os.path.relpath(s.get("ckpt", ""), root) if s.get("ckpt") else "",
        "action_head_type": s.get("action_head_type"),
        "train_lora": s.get("train_lora"),
        "action_samples": s.get("action_samples"),
        "success_rate": s.get("success_rate"),
        "hazard_hit_rate": s.get("hazard_hit_rate"),
        "timeout_rate": s.get("timeout_rate"),
        "mean_return": s.get("mean_return"),
        "mean_final_dist": s.get("mean_final_dist"),
        "mean_min_clearance": s.get("mean_min_clearance"),
        "best_loss": s.get("best_loss"),
    })

out = os.path.join(root, "flow_ablation_summary.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=2)

print()
print("Flow ablation summary saved:", out)
for r in rows:
    print(
        r["summary"],
        "lora=", r["train_lora"],
        "success=", r["success_rate"],
        "hazard=", r["hazard_hit_rate"],
        "return=", r["mean_return"],
    )
PY

echo
echo "All done: $RUN_ROOT"
