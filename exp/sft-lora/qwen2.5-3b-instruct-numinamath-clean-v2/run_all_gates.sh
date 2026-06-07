#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-}"
[[ "$EXPERIMENT" == "e1" || "$EXPERIMENT" == "e2" ]] || { echo "Usage: $0 e1|e2" >&2; exit 2; }

[[ -f "$RUN_DIR/gate/results/baseline.json" ]] || bash "$RUN_DIR/run_gate.sh" baseline

mapfile -t CHECKPOINTS < <(python - "$RUN_DIR/output/$EXPERIMENT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
full_runs = []
for args_path in root.glob("v*/args.json"):
    args = json.loads(args_path.read_text())
    if args.get("max_steps", -1) == -1:
        full_runs.append(args_path.parent)
if full_runs:
    latest = sorted(full_runs)[-1]
    for checkpoint in sorted(latest.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        if (checkpoint / "adapter_config.json").is_file():
            print(checkpoint)
PY
)
((${#CHECKPOINTS[@]})) || { echo "No checkpoints found for $EXPERIMENT" >&2; exit 1; }
for checkpoint in "${CHECKPOINTS[@]}"; do
  bash "$RUN_DIR/run_gate.sh" "$checkpoint"
done

python "$RUN_DIR/summarize_gates.py"
