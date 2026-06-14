#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-numinamath-clean-v2"
CHECKPOINT="$(realpath "${1:?checkpoint required}")"
GPU_ID="${GPU_ID:-0}"; EVAL_PORT="${EVAL_PORT:-8300}"
NAME="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")-$(basename "$CHECKPOINT")"
MERGED="/root/.cache/qwen25-3b-base-clean-v2-final/$NAME"
mkdir -p "$(dirname "$MERGED")" "$RUN_DIR/eval/$NAME"
rm -rf "$MERGED"
trap 'rm -rf "$MERGED"' EXIT
conda run -n ms-swift --no-capture-output env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/ms-swift" \
  swift export --adapters "$CHECKPOINT" --merge_lora true --output_dir "$MERGED"
python "$RUN_DIR/fix_merged_tokenizer.py" "$MERGED"
for benchmark in gsm8k math_500; do
  out="$RUN_DIR/eval/$NAME/$benchmark"; mkdir -p "$out"
  (
    cd "$out"
    setsid conda run -n vllm --no-capture-output env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift eval --model "$MERGED" --template qwen2_5 --system 'You are a helpful assistant.' \
      --enable_thinking false --eval_dataset "$benchmark" --eval_backend Native --infer_backend vllm \
      --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.85 --vllm_max_num_seqs 16 \
      --vllm_max_model_len 10000 --port "$EVAL_PORT" \
      --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
      --extra_eval_args '{"judge_strategy":"rule"}' --eval_num_proc 8 &
    pid=$!; set +e; wait "$pid"; status=$?; set -e
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$pid" 2>/dev/null || true
    exit "$status"
  ) >"$RUN_DIR/eval/$NAME/$benchmark.log" 2>&1
done
