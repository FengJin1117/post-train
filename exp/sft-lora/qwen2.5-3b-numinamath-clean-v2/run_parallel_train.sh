#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-numinamath-clean-v2"
for recipe in b1 b2; do
  session="qwen25-base-$recipe"
  tmux has-session -t "$session" 2>/dev/null && { echo "Session exists: $session" >&2; exit 1; }
  tmux new-session -d -s "$session" \
    "cd '$ROOT_DIR' && eval \"\$(conda shell.bash hook)\" && conda activate ms-swift && bash '$RUN_DIR/run_train.sh' '$recipe' full"
  echo "Started $session"
done

