#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2"
DATA_DIR="$RUN_DIR/data"
OUTPUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
MODEL_PATH="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct}"
EXPERIMENT="${1:-}"
MODE="${2:-full}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ms-swift" ]]; then
  echo "Please run: conda activate ms-swift" >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/clean-10k.jsonl" ]]; then
  echo "Prepared datasets are missing. Run prepare_dataset.py first." >&2
  exit 1
fi

case "$EXPERIMENT" in
  e1)
    DATASET="$DATA_DIR/clean-10k.jsonl"; LR="1e-5"; SAVE_STEPS=50 ;;
  e2)
    DATASET="$DATA_DIR/clean-10k.jsonl"; LR="2e-5"; SAVE_STEPS=50 ;;
  e3)
    DECISION="$RUN_DIR/gate/decision.json"
    [[ -f "$DECISION" ]] || { echo "Missing E3 decision: $DECISION" >&2; exit 1; }
    read -r DATASET LR < <(python - "$DECISION" "$RUN_DIR" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]}/{x['e3_dataset']} {x['e3_learning_rate']}")
PY
)
    SAVE_STEPS=100
    ;;
  *)
    echo "Usage: $0 {e1|e2|e3} [full|smoke]" >&2
    exit 2
    ;;
esac

[[ -f "$DATASET" ]] || { echo "Dataset is missing: $DATASET" >&2; exit 1; }
EXTRA_ARGS=()
TARGET_OUTPUT_DIR="$OUTPUT_DIR/$EXPERIMENT"
if [[ "$MODE" == "smoke" ]]; then
  TARGET_OUTPUT_DIR="$OUTPUT_DIR/smoke/$EXPERIMENT"
  EXTRA_ARGS+=(--max_steps 2 --save_steps 1 --eval_strategy no)
elif [[ "$MODE" != "full" ]]; then
  echo "Mode must be full or smoke." >&2
  exit 2
fi
mkdir -p "$TARGET_OUTPUT_DIR" "$LOG_DIR"

export PYTHONPATH="$ROOT_DIR/ms-swift:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,6,7
export NPROC_PER_NODE=3

swift sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$DATASET" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate "$LR" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj \
  --max_length 2048 \
  --eval_steps "$SAVE_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 10 \
  --logging_steps 5 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --load_from_cache_file false \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed zero2 \
  --output_dir "$TARGET_OUTPUT_DIR" \
  --system 'You are a helpful assistant.' \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train-${EXPERIMENT}-${MODE}.log"
