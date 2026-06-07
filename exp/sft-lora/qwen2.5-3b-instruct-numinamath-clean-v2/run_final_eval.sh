#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2"
CHECKPOINT="${1:-}"
GPU_ID="${GPU_ID:-7}"
EVAL_PORT="${EVAL_PORT:-8000}"

[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "Usage: $0 <checkpoint-path>" >&2; exit 2; }
CHECKPOINT="$(realpath "$CHECKPOINT")"
NAME="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")-$(basename "$CHECKPOINT")"
MERGED="$RUN_DIR/eval/merged/$NAME"
mkdir -p "$(dirname "$MERGED")" "$RUN_DIR/eval/$NAME"

if [[ ! -f "$MERGED/config.json" ]]; then
  conda run -n ms-swift --no-capture-output env PYTHONPATH="$ROOT_DIR/ms-swift" \
    swift export --adapters "$CHECKPOINT" --merge_lora true --output_dir "$MERGED"
fi

python - "$MERGED/tokenizer_config.json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    config = json.load(f)
extra = config.get("extra_special_tokens")
if isinstance(extra, list):
    config.setdefault("additional_special_tokens", extra)
    del config["extra_special_tokens"]
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)
    print(f"Normalized tokenizer config: {path}")
PY

if [[ "${MERGE_ONLY:-0}" == "1" ]]; then
  echo "Merged model ready: $MERGED"
  exit 0
fi

if (($# > 1)); then
  BENCHMARKS=("${@:2}")
else
  BENCHMARKS=(gsm8k math_500)
fi

for benchmark in "${BENCHMARKS[@]}"; do
  out_dir="$RUN_DIR/eval/$NAME/$benchmark"
  mkdir -p "$out_dir"
  set +e
  (
    cd "$out_dir"
    setsid conda run -n vllm --no-capture-output env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT_DIR/ms-swift" \
      swift eval \
      --model "$MERGED" \
      --enable_thinking false \
      --eval_dataset "$benchmark" \
      --eval_backend Native \
      --infer_backend vllm \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.85 \
      --vllm_max_num_seqs 16 \
      --vllm_max_model_len 10000 \
      --port "$EVAL_PORT" \
      --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
      --extra_eval_args '{"judge_strategy":"rule"}' \
      --eval_num_proc 8 &
    eval_pid=$!
    wait "$eval_pid"
    eval_status=$?
    kill -TERM -- "-$eval_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$eval_pid" 2>/dev/null || true
    exit "$eval_status"
  ) >"$RUN_DIR/eval/$NAME/$benchmark.log" 2>&1
  status=$?
  set -e
  sleep 3
  ((status == 0)) || exit "$status"
done

echo "Full evaluation completed: $RUN_DIR/eval/$NAME"
