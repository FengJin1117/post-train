#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2"
OUTPUT_DIR="$RUN_DIR/output/e1/v4-20260606-161704"
EVAL_SCRIPT="$RUN_DIR/run_final_eval.sh"

CHECKPOINTS=(50 100 150)
for step in "${CHECKPOINTS[@]}"; do
  MERGE_ONLY=1 bash "$EVAL_SCRIPT" "$OUTPUT_DIR/checkpoint-$step"
done

STEPS=(50 50 100 100 150 150)
BENCHMARKS=(gsm8k math_500 gsm8k math_500 gsm8k math_500)
GPUS=(0 1 4 5 6 7)
PORTS=(8001 8002 8003 8004 8005 8006)

for i in "${!STEPS[@]}"; do
  step="${STEPS[$i]}"
  benchmark="${BENCHMARKS[$i]}"
  gpu="${GPUS[$i]}"
  port="${PORTS[$i]}"
  session="clean-v2-e1-cp${step}-${benchmark}"
  log="$RUN_DIR/logs/e1-checkpoint-${step}-${benchmark}.log"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "Session already exists: $session" >&2
    exit 1
  }
  tmux new-session -d -s "$session" \
    "cd '$ROOT_DIR' && GPU_ID='$gpu' EVAL_PORT='$port' bash '$EVAL_SCRIPT' '$OUTPUT_DIR/checkpoint-$step' '$benchmark' > '$log' 2>&1"
  echo "Started $session on GPU $gpu, port $port"
done
