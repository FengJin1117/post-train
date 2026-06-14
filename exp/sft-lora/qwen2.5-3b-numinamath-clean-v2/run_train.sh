#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-numinamath-clean-v2"
MODEL_PATH="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B}"
RECIPE="${1:-}"
MODE="${2:-full}"

[[ "${CONDA_DEFAULT_ENV:-}" == "ms-swift" ]] || { echo "Please activate ms-swift" >&2; exit 1; }
[[ -f "$RUN_DIR/data/clean-10k.jsonl" && -f "$RUN_DIR/data/val.jsonl" ]] || { echo "Broken data links" >&2; exit 1; }

case "$RECIPE" in
  b1)
    GPUS=0,1,4; EPOCHS=1; LR=1e-5; RANK=8; ALPHA=16
    MASTER_PORT=29511
    TARGETS=(q_proj k_proj v_proj o_proj)
    ;;
  b2)
    GPUS=5,6,7; EPOCHS=2; LR=5e-5; RANK=16; ALPHA=32
    MASTER_PORT=29512
    TARGETS=(all-linear)
    ;;
  *) echo "Usage: $0 b1|b2 [smoke|full]" >&2; exit 2 ;;
esac

EXTRA=()
OUT="$RUN_DIR/output/$RECIPE"
if [[ "$MODE" == smoke ]]; then
  OUT="$RUN_DIR/output/smoke/$RECIPE"
  EXTRA=(--max_steps 2 --save_steps 1 --eval_strategy no)
elif [[ "$MODE" != full ]]; then
  echo "Mode must be smoke or full" >&2; exit 2
fi
mkdir -p "$OUT" "$RUN_DIR/logs"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NPROC_PER_NODE=3
export MASTER_PORT
export PYTHONPATH="$ROOT_DIR/ms-swift:${PYTHONPATH:-}"

swift sft \
  --model "$MODEL_PATH" \
  --template qwen2_5 \
  --system 'You are a helpful assistant.' \
  --tuner_type lora \
  --dataset "$RUN_DIR/data/clean-10k.jsonl" \
  --val_dataset "$RUN_DIR/data/val.jsonl" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lora_rank "$RANK" \
  --lora_alpha "$ALPHA" \
  --lora_dropout 0.05 \
  --target_modules "${TARGETS[@]}" \
  --max_length 2048 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 20 \
  --logging_steps 5 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --load_from_cache_file false \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed zero2 \
  --output_dir "$OUT" \
  "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/logs/train-${RECIPE}-${MODE}.log"
