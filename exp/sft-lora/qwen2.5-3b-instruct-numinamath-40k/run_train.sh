#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k"
DATA_DIR="$RUN_DIR/data"
OUTPUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
MODEL_PATH="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct}"
MODE="${1:-full}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ms-swift" ]]; then
  echo "Please run: conda activate ms-swift" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model config is missing: $MODEL_PATH/config.json" >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/val.jsonl" ]]; then
  echo "Dataset is missing. Run: python $RUN_DIR/prepare_dataset.py --output-dir $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

GPU_INFO="$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits |
  awk -F ', ' '$1 == 4 || $1 == 5 { print $0 }')"
if [[ $(wc -l <<<"$GPU_INFO") -ne 2 ]] || ! awk -F ', ' '$2 ~ /A6000/ && $3 >= 40960 { ok += 1 } END { exit ok != 2 }' <<<"$GPU_INFO"; then
  echo "GPU 4,5 must be RTX A6000 with at least 40 GiB free memory each." >&2
  echo "$GPU_INFO" >&2
  exit 1
fi

EXTRA_ARGS=()
case "$MODE" in
  smoke)
    EXTRA_ARGS+=(--max_steps 2 --save_steps 1 --eval_strategy no)
    ;;
  full)
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$ROOT_DIR/ms-swift:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4

python - <<'PY'
import deepspeed
print("deepspeed", deepspeed.__version__)
PY

swift sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$DATA_DIR/train.jsonl" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --eval_steps 250 \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --max_length 2048 \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --load_from_cache_file true \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed zero2 \
  --system 'You are a helpful assistant.' \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train-${MODE}.log"
