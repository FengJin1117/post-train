#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/baseline/qwen2.5-3b-instruct"
MODEL_PATH="${MODEL_PATH:-/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct}"
ENV_NAME="${ENV_NAME:-vllm}"
MODE="${1:-all}"
BENCHMARKS=(gsm8k)
GENERATION_CONFIG='{"max_tokens":8192,"temperature":0.0,"do_sample":false}'
EXTRA_EVAL_ARGS='{"judge_strategy":"rule"}'

GPU_ID="${GPU_ID:-0}"
GPU_INFO="$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits |
  awk -F ', ' -v gpu_id="$GPU_ID" '$1 == gpu_id { print $0 }')"
if [[ -z "$GPU_INFO" ]] || ! awk -F ', ' '$2 ~ /A6000/ && $3 >= 40960 { found = 1 } END { exit !found }' <<<"$GPU_INFO"; then
  echo "GPU $GPU_ID must be an RTX A6000 with at least 40 GiB free memory." >&2
  exit 1
fi
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
  awk -F ', ' -v gpu_id="$GPU_ID" '$1 == gpu_id { print $2 }')"
INITIAL_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader |
  awk -F ', ' -v gpu_uuid="$GPU_UUID" '$2 == gpu_uuid { print $1 }')"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model config is missing: $MODEL_PATH/config.json" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/raw"

echo "Using GPU $GPU_ID"
echo "Using model $MODEL_PATH"

cleanup_gpu_processes() {
  local pid
  local pids=()

  while IFS= read -r pid; do
    if [[ -n "$pid" ]] && ! grep -qxF "$pid" <<<"$INITIAL_GPU_PIDS"; then
      pids+=("$pid")
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader |
    awk -F ', ' -v gpu_uuid="$GPU_UUID" '$2 == gpu_uuid { print $1 }')

  if ((${#pids[@]})); then
    echo "Stopping evaluation GPU processes: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_gpu_processes EXIT
trap 'cleanup_gpu_processes; exit 130' INT
trap 'cleanup_gpu_processes; exit 143' TERM

conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
from evalscope.api.registry import BENCHMARK_REGISTRY

required = {"gsm8k"}
available = set(BENCHMARK_REGISTRY)
missing = required - available
assert not missing, f"Missing Native benchmarks: {sorted(missing)}"
print("Native benchmarks verified:", ", ".join(sorted(required)))
PY

run_eval() {
  local stage="$1"
  local benchmark="$2"
  local limit="${3:-}"
  local output_dir="$RUN_DIR/raw/$stage"
  local log_file="$output_dir/run.log"
  local args=()

  mkdir -p "$output_dir"
  if [[ -n "$limit" ]]; then
    args+=(--eval_limit "$limit")
  fi

  (
    cd "$output_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n "$ENV_NAME" swift eval \
      --model "$MODEL_PATH" \
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
      "${args[@]}"
  ) > >(tee "$log_file") 2>&1
  cleanup_gpu_processes
}

run_smoke() {
  local benchmark
  for benchmark in "${BENCHMARKS[@]}"; do
    echo "Running smoke test: $benchmark"
    run_eval "smoke/$benchmark" "$benchmark" 2
  done
}

run_full() {
  local benchmark
  for benchmark in "${BENCHMARKS[@]}"; do
    echo "Running full baseline: $benchmark"
    run_eval "full/$benchmark" "$benchmark"
  done
}

case "$MODE" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  all)
    run_smoke
    run_full
    ;;
  *)
    echo "Usage: $0 [smoke|full|all]" >&2
    exit 2
    ;;
esac
