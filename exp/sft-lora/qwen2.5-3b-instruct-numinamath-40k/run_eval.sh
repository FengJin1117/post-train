#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k"
OUTPUT_DIR="$RUN_DIR/output"
EVAL_DIR="$RUN_DIR/eval"
MODEL_PATH="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct}"
GPU_ID="${GPU_ID:-5}"
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

LATEST_CKPT="$(find "$OUTPUT_DIR" -type d -name 'checkpoint-*' | sort -V | tail -1)"
if [[ -z "$LATEST_CKPT" ]]; then
  echo "No checkpoint found under $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"
echo "Using adapter: $LATEST_CKPT"

for benchmark in "${BENCHMARKS[@]}"; do
  out_dir="$EVAL_DIR/$benchmark"
  mkdir -p "$out_dir"
  (
    cd "$out_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" swift eval \
      --model "$MODEL_PATH" \
      --adapters "$LATEST_CKPT" \
      --enable_thinking false \
      --eval_dataset "$benchmark" \
      --eval_backend Native \
      --infer_backend vllm \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.9 \
      --vllm_max_model_len 10000 \
      --eval_generation_config "$GENERATION_CONFIG" \
      --extra_eval_args "$EXTRA_EVAL_ARGS" \
      --eval_num_proc 8
  ) 2>&1 | tee "$EVAL_DIR/${benchmark}.log"
done
