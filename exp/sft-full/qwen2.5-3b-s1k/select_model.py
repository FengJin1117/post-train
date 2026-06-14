#!/usr/bin/env python3
"""Select the winning recipe using only predeclared gate metrics."""

import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
names = ("r0-original", "r1-conservative")
rows = {}
for name in names:
    data = json.loads((RUN / f"gate-{name}.json").read_text())
    rows[name] = {
        "gate_mean": round((data["gsm8k"]["accuracy"] + data["math_500"]["accuracy"]) / 2, 6),
        "max_token_mean": round((data["gsm8k"]["max_token_rate"] + data["math_500"]["max_token_rate"]) / 2, 6),
        "metrics": data,
    }
winner = sorted(names, key=lambda x: (-rows[x]["gate_mean"], rows[x]["max_token_mean"], names.index(x)))[0]
result = {"winner": winner, "selection_rule": "gate mean desc, max-token mean asc, r0 tie-break", "runs": rows}
(RUN / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
