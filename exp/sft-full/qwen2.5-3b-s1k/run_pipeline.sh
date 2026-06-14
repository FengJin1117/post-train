#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/exp/sft-full/qwen2.5-3b-s1k"
BASE="/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B"

latest_model() {
  find "$RUN/output/$1" -type f -name config.json -printf '%h\n' | sort | tail -1
}

for recipe in r0-original r1-conservative; do
  if [[ -z "$(latest_model "$recipe")" ]]; then
    bash "$RUN/run_train.sh" "$recipe" full
  fi
  model="$(latest_model "$recipe")"
  [[ -n "$model" ]] || { echo "No model for $recipe" >&2; exit 1; }
  EVAL_GPUS="6 7" bash "$RUN/run_eval.sh" "$recipe" "$model" gate
done
python "$RUN/select_model.py"

EVAL_GPUS="0 1 4 5" bash "$RUN/run_eval.sh" base "$BASE" full
for recipe in r0-original r1-conservative; do
  EVAL_GPUS="0 1 4 5" bash "$RUN/run_eval.sh" "$recipe" "$(latest_model "$recipe")" full
done
python "$RUN/contamination.py" \
  --train "$RUN/data/s1k-ms-swift.jsonl" \
  --eval-root "$RUN/eval/full" \
  --output "$RUN/contamination-report.json"
