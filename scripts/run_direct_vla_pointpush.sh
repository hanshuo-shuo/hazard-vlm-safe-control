#!/usr/bin/env bash
set -euo pipefail

# Step 1: Collect expert demos
python direct_vla_pointpush.py --mode collect_expert \
    --episodes 1000 \
    --seed 42 \
    --out hazard/direct_vla_pointpush_expert_demos.npz

# Step 2: Train action head (with LoRA)
python direct_vla_pointpush.py --mode train \
    --dataset hazard/direct_vla_pointpush_expert_demos.npz \
    --ckpt hazard/direct_qwen_vla_pointpush_head.pt \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --train_lora \
    --epochs 3 \
    --batch_size 1 \
    --seed 42

# Step 3: Evaluate
python direct_vla_pointpush.py --mode eval \
    --ckpt hazard/direct_qwen_vla_pointpush_head.pt \
    --episodes 100 \
    --seed 1000 \
    --summary hazard/direct_qwen_vla_pointpush_eval_summary.json
