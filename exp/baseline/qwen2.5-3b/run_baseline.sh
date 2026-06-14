#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/baseline/qwen2.5-3b"
MODEL_PATH="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B}"
GAOKAO_PLUGIN="$ROOT_DIR/benchmarks/gaokao_math/plugin.py"
MODE="${1:-all}"
mkdir -p "$RUN_DIR/raw/full" "$RUN_DIR/raw/gate" "$RUN_DIR/raw/gaokao" "$RUN_DIR/logs"

run_one() {
  local stage="$1" benchmark="$2" gpu="$3" port="$4" limit="$5" max_tokens="$6"
  local out="$RUN_DIR/raw/$stage/$benchmark"
  local limit_args=()
  [[ -n "$limit" ]] && limit_args=(--eval_limit "$limit")
  mkdir -p "$out"
  (
    cd "$out"
    setsid conda run -n vllm --no-capture-output env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift eval \
      --model "$MODEL_PATH" \
      --template qwen2_5 \
      --system 'You are a helpful assistant.' \
      --enable_thinking false \
      --external_plugins "$GAOKAO_PLUGIN" \
      --eval_dataset "$benchmark" \
      --eval_backend Native \
      --infer_backend vllm \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.85 \
      --vllm_max_num_seqs 16 \
      --vllm_max_model_len 10000 \
      --port "$port" \
      --eval_generation_config "{\"max_tokens\":$max_tokens,\"temperature\":0.0,\"do_sample\":false}" \
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

run_pair() {
  local stage="$1" gsm_limit="$2" math_limit="$3" gsm_tokens="$4" math_tokens="$5"
  run_one "$stage" gsm8k 0 8101 "$gsm_limit" "$gsm_tokens" &
  p1=$!
  run_one "$stage" math_500 1 8102 "$math_limit" "$math_tokens" &
  p2=$!
  set +e
  wait "$p1"; s1=$?
  wait "$p2"; s2=$?
  set -e
  ((s1 == 0 && s2 == 0)) || return 1
  python "$RUN_DIR/analyze.py" --root "$RUN_DIR/raw/$stage" --output "$RUN_DIR/${stage}-results.json" \
    --benchmarks gsm8k math_500
}

run_gaokao() {
  run_one gaokao gaokao_math_cloze 0 8111 "" 3072 &
  p1=$!
  run_one gaokao gaokao_math_qa 1 8112 "" 3072 &
  p2=$!
  set +e
  wait "$p1"; s1=$?
  wait "$p2"; s2=$?
  set -e
  ((s1 == 0 && s2 == 0)) || return 1
  python "$RUN_DIR/analyze.py" --root "$RUN_DIR/raw/gaokao" --output "$RUN_DIR/gaokao-results.json" \
    --benchmarks gaokao_math_cloze gaokao_math_qa
}

case "$MODE" in
  full) run_pair full "" "" 8192 8192 ;;
  gate) run_pair gate 200 100 1024 2048 ;;
  gaokao) run_gaokao ;;
  all) run_pair gate 200 100 1024 2048; run_pair full "" "" 8192 8192; run_gaokao ;;
  *) echo "Usage: $0 [gate|full|gaokao|all]" >&2; exit 2 ;;
esac
