#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/exp/sft-full/qwen2.5-3b-s1k"
NAME="${1:?name required}"
MODEL="$(realpath "${2:?model path required}")"
STAGE="${3:?gate or full required}"
python "$RUN/fix_tokenizer.py" "$MODEL"

if [[ "$STAGE" == gate ]]; then
  BENCH=(gsm8k math_500); LIMITS=(200 20); TOKENS=(1024 2048)
elif [[ "$STAGE" == full ]]; then
  BENCH=(gsm8k math_500 aime24 aime25); LIMITS=("" "" "" ""); TOKENS=(8192 8192 8192 8192)
else
  echo "Stage must be gate or full" >&2; exit 2
fi

GPUS=(${EVAL_GPUS:-0 1 4 5})
mkdir -p "$RUN/eval/$STAGE/$NAME" "$RUN/logs"

run_one() {
  local benchmark="$1" limit="$2" tokens="$3" gpu="$4" port="$5"
  local out="$RUN/eval/$STAGE/$NAME/$benchmark"
  local limit_args=()
  [[ -n "$limit" ]] && limit_args=(--eval_limit "$limit")
  rm -rf "$out"
  mkdir -p "$out"
  (
    cd "$out"
    setsid conda run -n vllm --no-capture-output env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/ms-swift" \
      swift eval --model "$MODEL" --template qwen2_5 --system 'You are a helpful assistant.' \
      --enable_thinking false --eval_dataset "$benchmark" --eval_backend Native --infer_backend vllm \
      --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.85 --vllm_max_num_seqs 16 \
      --vllm_max_model_len 10000 --port "$port" \
      --eval_generation_config "{\"max_tokens\":$tokens,\"temperature\":0.0,\"do_sample\":false}" \
      --extra_eval_args '{"judge_strategy":"rule"}' --eval_num_proc 8 "${limit_args[@]}" &
    pid=$!; set +e; wait "$pid"; status=$?; set -e
    kill -TERM -- "-$pid" 2>/dev/null || true; sleep 2; kill -KILL -- "-$pid" 2>/dev/null || true
    exit "$status"
  ) >"$RUN/logs/eval-${STAGE}-${NAME}-${benchmark}.log" 2>&1
}

pids=()
for i in "${!BENCH[@]}"; do
  run_one "${BENCH[$i]}" "${LIMITS[$i]}" "${TOKENS[$i]}" "${GPUS[$i]}" "$((8500+i))" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status == 0)) || exit 1
python "$RUN/analyze_eval.py" --root "$RUN/eval/$STAGE/$NAME" --output "$RUN/${STAGE}-${NAME}.json" --stage "$STAGE"
