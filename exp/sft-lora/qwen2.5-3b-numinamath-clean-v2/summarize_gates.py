#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parent
results = [json.loads(p.read_text()) | {"result_path": str(p)} for p in (root / "gate/results").glob("b*-checkpoint-*.json")]
results.sort(key=lambda x: (x["math_500"]["accuracy"], x["gsm8k"]["accuracy"],
                            -x["math_500"]["max_token_rate"] - x["gsm8k"]["max_token_rate"],
                            x["math_500"]["boxed_rate"] + x["gsm8k"]["boxed_rate"]), reverse=True)
best = {}
for recipe in ("b1", "b2"):
    candidates = [x for x in results if x["name"].startswith(recipe + "-")]
    best[recipe] = candidates[0]["name"] if candidates else None
summary = {"ranking": results, "best": best}
(root / "gate/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

