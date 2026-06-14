#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/baseline/qwen2.5-math-1.5b"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
ENV_NAME="${ENV_NAME:-vllm}"
SMOKE_GPU="${SMOKE_GPU:-1}"
GSM_GPU="${GSM_GPU:-1}"
MATH_GPU="${MATH_GPU:-4}"
MODE="${1:-all}"

mkdir -p "$RUN_DIR/raw/smoke" "$RUN_DIR/raw/full" "$RUN_DIR/logs"

run_one() {
  local stage="$1" benchmark="$2" gpu="$3" port="$4" limit="${5:-}"
  local out="$RUN_DIR/raw/$stage/$benchmark"
  local limit_args=()
  [[ -n "$limit" ]] && limit_args=(--eval_limit "$limit")
  mkdir -p "$out"

  (
    cd "$out"
    setsid conda run -n "$ENV_NAME" --no-capture-output env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift eval \
      --model "$MODEL_PATH" \
      --template qwen2_5_math \
      --system 'Please reason step by step, and put your final answer within \boxed{}.' \
      --enable_thinking false \
      --eval_dataset "$benchmark" \
      --eval_backend Native \
      --infer_backend vllm \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.85 \
      --vllm_max_num_seqs 16 \
      --vllm_max_model_len 4096 \
      --port "$port" \
      --eval_generation_config '{"max_tokens":3072,"temperature":0.0,"do_sample":false}' \
      --extra_eval_args '{"judge_strategy":"rule"}' \
      --eval_num_proc 8 \
      "${limit_args[@]}" &
    pid=$!
    set +e
    wait "$pid"; status=$?
    set -e
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$pid" 2>/dev/null || true
    exit "$status"
  ) >"$RUN_DIR/logs/${stage}-${benchmark}.log" 2>&1
}

run_smoke() {
  run_one smoke gsm8k "$SMOKE_GPU" 8401 2
  run_one smoke math_500 "$SMOKE_GPU" 8402 2
  python "$RUN_DIR/analyze.py" \
    --root "$RUN_DIR/raw/smoke" \
    --output "$RUN_DIR/smoke-results.json" \
    --expected-gsm8k 2 \
    --expected-math-500 10
}

run_full() {
  run_one full gsm8k "$GSM_GPU" 8411 &
  gsm_pid=$!
  run_one full math_500 "$MATH_GPU" 8412 &
  math_pid=$!
  set +e
  wait "$gsm_pid"; gsm_status=$?
  wait "$math_pid"; math_status=$?
  set -e
  ((gsm_status == 0 && math_status == 0)) || return 1
  python "$RUN_DIR/analyze.py" \
    --root "$RUN_DIR/raw/full" \
    --output "$RUN_DIR/full-results.json" \
    --expected-gsm8k 1319 \
    --expected-math-500 500 \
    --markdown "$RUN_DIR/RESULTS.md"
}

case "$MODE" in
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_smoke; run_full ;;
  *) echo "Usage: $0 [smoke|full|all]" >&2; exit 2 ;;
esac
