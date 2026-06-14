#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-numinamath-clean-v2"
ANALYZE="$ROOT_DIR/exp/baseline/qwen2.5-3b/analyze.py"

while tmux has-session -t qwen25-base-final-b1 2>/dev/null ||
      tmux has-session -t qwen25-base-final-b2 2>/dev/null; do
  sleep 60
done

python "$ANALYZE" \
  --root "$RUN_DIR/eval/b1-checkpoint-209" \
  --output "$RUN_DIR/full-results-b1.json"
python "$ANALYZE" \
  --root "$RUN_DIR/eval/b2-checkpoint-200" \
  --output "$RUN_DIR/full-results-b2.json"
