#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/exp/sft-full/qwen2.5-3b-s1k"
MODEL="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B}"
RECIPE="${1:?r0-original or r1-conservative required}"
MODE="${2:-full}"

case "$RECIPE" in
  r0-original) LR=1e-5; EPOCHS=5; PORT=29610 ;;
  r1-conservative) LR=5e-6; EPOCHS=3; PORT=29611 ;;
  *) echo "Unknown recipe: $RECIPE" >&2; exit 2 ;;
esac

[[ -f "$RUN/data/s1k-ms-swift.jsonl" ]] || { echo "Run prepare_data.py first" >&2; exit 1; }
OUT="$RUN/output/$RECIPE"
EXTRA=()
if [[ "$MODE" == smoke ]]; then
  OUT="$RUN/output/smoke/$RECIPE"
  EXTRA=(--max_steps 2)
elif [[ "$MODE" != full ]]; then
  echo "Mode must be smoke or full" >&2; exit 2
fi
mkdir -p "$OUT" "$RUN/logs"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,4,5}"
export NPROC_PER_NODE=4
export MASTER_PORT="${MASTER_PORT:-$PORT}"
export PYTHONPATH="$ROOT/ms-swift:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

swift sft \
  --model "$MODEL" \
  --template qwen2_5 \
  --system 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' \
  --tuner_type full \
  --dataset "$RUN/data/s1k-ms-swift.jsonl" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.0001 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --loss_scale default \
  --packing false \
  --max_length 32768 \
  --truncation_strategy delete \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_only_model true \
  --save_total_limit 1 \
  --report_to none \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --load_from_cache_file false \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --deepspeed zero3 \
  --output_dir "$OUT" \
  "${EXTRA[@]}" 2>&1 | tee "$RUN/logs/train-${RECIPE}-${MODE}.log"
