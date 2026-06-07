#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k"
OUTPUT_DIR="$RUN_DIR/output"
EVAL_DIR="$RUN_DIR/eval"
ADAPTER_PATH="${ADAPTER_PATH:-}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-}"
GPU_ID="${GPU_ID:-5}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
GENERATION_CONFIG='{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
EXTRA_EVAL_ARGS='{"judge_strategy":"rule"}'

if (($#)); then
  BENCHMARKS=("$@")
else
  BENCHMARKS=(gsm8k math_500)
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]]; then
  echo "Please run: conda activate vllm" >&2
  exit 1
fi

if [[ -z "$ADAPTER_PATH" ]]; then
  ADAPTER_PATH="$(find "$OUTPUT_DIR" -type f -name adapter_config.json -printf '%h\n' | sort -V | tail -1)"
fi
if [[ -z "$ADAPTER_PATH" || ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
  echo "No checkpoint found under $OUTPUT_DIR" >&2
  exit 1
fi
if [[ -z "$MERGED_MODEL_PATH" ]]; then
  MERGED_MODEL_PATH="${ADAPTER_PATH}-merged"
fi
if [[ ! -f "$MERGED_MODEL_PATH/config.json" ]]; then
  echo "Merged model config is missing: $MERGED_MODEL_PATH/config.json" >&2
  echo "Merge it first with: swift export --adapters $ADAPTER_PATH --merge_lora true --output_dir $MERGED_MODEL_PATH" >&2
  exit 1
fi

python - "$MERGED_MODEL_PATH/tokenizer_config.json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    config = json.load(f)
extra = config.get('extra_special_tokens')
if isinstance(extra, list):
    config.setdefault('additional_special_tokens', extra)
    del config['extra_special_tokens']
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.replace(tmp_path, path)
    print(f'Normalized tokenizer config: {path}')
PY

EXTRA_ARGS=()
if [[ -n "$EVAL_LIMIT" ]]; then
  EXTRA_ARGS+=(--eval_limit "$EVAL_LIMIT")
fi

mkdir -p "$EVAL_DIR"
echo "Using merged model: $MERGED_MODEL_PATH"

for benchmark in "${BENCHMARKS[@]}"; do
  out_dir="$EVAL_DIR/$benchmark"
  log_file="$EVAL_DIR/${benchmark}.log"
  mkdir -p "$out_dir"
  echo "Running $benchmark; log: $log_file"
  (
    cd "$out_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" swift eval \
      --model "$MERGED_MODEL_PATH" \
      --enable_thinking false \
      --eval_dataset "$benchmark" \
      --eval_backend Native \
      --infer_backend vllm \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.9 \
      --vllm_max_model_len 10000 \
      --eval_generation_config "$GENERATION_CONFIG" \
      --extra_eval_args "$EXTRA_EVAL_ARGS" \
      --eval_num_proc 8 \
      "${EXTRA_ARGS[@]}"
  ) >"$log_file" 2>&1
done
