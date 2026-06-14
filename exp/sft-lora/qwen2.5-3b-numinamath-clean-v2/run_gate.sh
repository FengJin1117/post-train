#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-numinamath-clean-v2"
BASELINE="$ROOT_DIR/exp/baseline/qwen2.5-3b/gate-results.json"
CHECKPOINT="$(realpath "${1:?checkpoint required}")"
GPU_ID="${GPU_ID:-0}"
EVAL_PORT="${EVAL_PORT:-8200}"
NAME="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")-$(basename "$CHECKPOINT")"
MERGED="/root/.cache/qwen25-3b-base-clean-v2-merged/$NAME"
GATE="$RUN_DIR/gate/runs/$NAME"
RESULT="$RUN_DIR/gate/results/$NAME.json"
mkdir -p "$(dirname "$MERGED")" "$GATE" "$(dirname "$RESULT")"
rm -rf "$MERGED"
trap 'rm -rf "$MERGED"' EXIT

conda run -n ms-swift --no-capture-output env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/ms-swift" \
  swift export --adapters "$CHECKPOINT" --merge_lora true --output_dir "$MERGED"
python "$RUN_DIR/fix_merged_tokenizer.py" "$MERGED"

run_one() {
  local benchmark="$1" limit="$2" max_tokens="$3"
  local out="$GATE/$benchmark"
  mkdir -p "$out"
  (
    cd "$out"
    setsid conda run -n vllm --no-capture-output env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift eval --model "$MERGED" --template qwen2_5 --system 'You are a helpful assistant.' \
      --enable_thinking false --eval_dataset "$benchmark" --eval_backend Native --infer_backend vllm \
      --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.75 --vllm_max_num_seqs 16 \
      --vllm_max_model_len 4096 --port "$EVAL_PORT" --eval_limit "$limit" \
      --eval_generation_config "{\"max_tokens\":$max_tokens,\"temperature\":0.0,\"do_sample\":false}" \
      --extra_eval_args '{"judge_strategy":"rule"}' --eval_num_proc 8 &
    pid=$!; set +e; wait "$pid"; status=$?; set -e
    kill -TERM -- "-$pid" 2>/dev/null || true; sleep 2; kill -KILL -- "-$pid" 2>/dev/null || true
    exit "$status"
  ) >"$GATE/$benchmark.log" 2>&1
}
run_one gsm8k 200 1024
run_one math_500 20 2048
python "$RUN_DIR/analyze_gate.py" --root "$GATE" --name "$NAME" --baseline "$BASELINE" --output "$RESULT"
