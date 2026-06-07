#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$ROOT_DIR/exp/grpo/example"

cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export WANDB_PROJECT="${WANDB_PROJECT:-ms-swift-grpo-example}"
export TOKENIZERS_PARALLELISM=false

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-3B-Instruct \
    --external_plugins "$EXP_DIR/countdown_plugin.py" \
    --reward_funcs external_countdown format \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset 'zouxuhong/Countdown-Tasks-3to4#64' \
    --load_from_cache_file true \
    --max_length 512 \
    --max_completion_length 128 \
    --max_steps 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --save_strategy no \
    --output_dir "$EXP_DIR/output" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2 \
    --num_generations 4 \
    --temperature 1.0 \
    --system 'You are a helpful assistant. Think through the reasoning, then provide the final equation in <answer> </answer> tags.' \
    --log_completions true \
    --report_to wandb \
    --run_name grpo-countdown-example-3steps \
    --beta 0.001
