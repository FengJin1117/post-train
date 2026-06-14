#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$ROOT_DIR/exp/sft-lora/qwen2.5-3b-numinamath-clean-v2"
RECIPE="${1:?b1 or b2}"
[[ -f "$ROOT_DIR/exp/baseline/qwen2.5-3b/gate-results.json" ]] || { echo "Missing baseline gate"; exit 1; }
mapfile -t RUNS < <(find "$RUN_DIR/output/$RECIPE" -maxdepth 2 -name args.json -printf '%h\n' | sort -V)
(( ${#RUNS[@]} > 0 )) || { echo "No run found for $RECIPE" >&2; exit 1; }
LATEST="${RUNS[-1]}"
mapfile -t ALL_CPS < <(find "$LATEST" -maxdepth 2 -name adapter_config.json -printf '%h\n' | sort -V)
CPS=()
for cp in "${ALL_CPS[@]}"; do
  result="$RUN_DIR/gate/results/$RECIPE-$(basename "$cp").json"
  [[ -f "$result" ]] || CPS+=("$cp")
done
IFS=',' read -r -a GPUS <<< "${GATE_GPUS:-0,1,4,5,6,7}"
WAVE_SIZE="${#GPUS[@]}"
failures=0
for ((offset=0; offset<${#CPS[@]}; offset+=WAVE_SIZE)); do
  pids=()
  names=()
  for ((i=0; i<WAVE_SIZE && offset+i<${#CPS[@]}; i++)); do
    cp="${CPS[offset+i]}"
    GPU_ID="${GPUS[i]}" EVAL_PORT="$((8201 + i))" bash "$RUN_DIR/run_gate.sh" "$cp" \
      >"$RUN_DIR/logs/gate-${RECIPE}-$(basename "$cp").log" 2>&1 &
    pids+=("$!")
    names+=("$(basename "$cp")")
  done
  for ((i=0; i<${#pids[@]}; i++)); do
    if ! wait "${pids[i]}"; then
      echo "Gate failed: $RECIPE ${names[i]}" >&2
      failures=$((failures + 1))
    fi
  done
done
python "$RUN_DIR/summarize_gates.py"
(( failures == 0 )) || { echo "$failures gate task(s) failed" >&2; exit 1; }
