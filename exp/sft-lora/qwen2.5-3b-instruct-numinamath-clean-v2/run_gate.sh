#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2"
MODEL_PATH="/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct"
TARGET="${1:-}"
GPU_ID="${GPU_ID:-7}"

if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 baseline|<checkpoint-path>" >&2
  exit 2
fi
if [[ "$TARGET" == "baseline" ]]; then
  NAME="baseline"
  MODEL="$MODEL_PATH"
  BASELINE_ARGS=()
else
  CHECKPOINT="$(realpath "$TARGET")"
  [[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "Invalid checkpoint: $CHECKPOINT" >&2; exit 1; }
  EXPERIMENT="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")"
  NAME="$EXPERIMENT-$(basename "$CHECKPOINT")"
  MODEL="$RUN_DIR/gate/merged/$NAME"
  mkdir -p "$(dirname "$MODEL")"
  if [[ ! -f "$MODEL/config.json" ]]; then
    conda run -n ms-swift --no-capture-output env PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift export --adapters "$CHECKPOINT" --merge_lora true --output_dir "$MODEL"
  fi
  BASELINE_ARGS=(--baseline "$RUN_DIR/gate/results/baseline.json")
fi

GATE_DIR="$RUN_DIR/gate/runs/$NAME"
RESULT="$RUN_DIR/gate/results/$NAME.json"
mkdir -p "$GATE_DIR" "$RUN_DIR/gate/results"

run_eval() {
  local benchmark="$1" limit="$2" max_tokens="$3"
  local out_dir="$GATE_DIR/$benchmark"
  local status
  mkdir -p "$out_dir"
  set +e
  (
    cd "$out_dir"
    setsid conda run -n vllm --no-capture-output env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift eval \
      --model "$MODEL" \
      --enable_thinking false \
      --eval_dataset "$benchmark" \
      --eval_backend Native \
      --infer_backend vllm \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.75 \
      --vllm_max_num_seqs 16 \
      --vllm_max_model_len 4096 \
      --eval_limit "$limit" \
      --eval_generation_config "{\"max_tokens\":$max_tokens,\"temperature\":0.0,\"do_sample\":false}" \
      --extra_eval_args '{"judge_strategy":"rule"}' \
      --eval_num_proc 8 &
    eval_pid=$!
    wait "$eval_pid"
    eval_status=$?
    kill -TERM -- "-$eval_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$eval_pid" 2>/dev/null || true
    exit "$eval_status"
  ) >"$GATE_DIR/$benchmark.log" 2>&1
  status=$?
  set -e
  sleep 3
  return "$status"
}

run_eval gsm8k 200 1024
# EvalScope applies eval_limit per MATH-500 difficulty subset: 20 x 5 = 100.
run_eval math_500 20 2048

python "$RUN_DIR/analyze_gate.py" \
  --gate-dir "$GATE_DIR" \
  --name "$NAME" \
  --output "$RESULT" \
  "${BASELINE_ARGS[@]}"

if [[ "$TARGET" != "baseline" ]]; then
  rm -rf "$MODEL"
fi
