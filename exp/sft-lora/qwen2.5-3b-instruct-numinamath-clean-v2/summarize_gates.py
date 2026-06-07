#!/usr/bin/env python3
"""Rank checkpoint gate results and write the conditional E3 decision."""

from __future__ import annotations

import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
RESULT_DIR = RUN_DIR / "gate" / "results"


def rank_key(result: dict) -> tuple:
    return (
        result["math_500"]["accuracy"],
        result["gsm8k"]["accuracy"],
        -result["math_500"]["max_token_rate"] - result["gsm8k"]["max_token_rate"],
        -result["math_500"]["output_tokens_median"] - result["gsm8k"]["output_tokens_median"],
    )


def main() -> None:
    results = []
    for path in RESULT_DIR.glob("e[12]-checkpoint-*.json"):
        result = json.loads(path.read_text())
        result["path"] = str(path)
        results.append(result)
    if not results:
        raise RuntimeError("No E1/E2 checkpoint gate results found.")
    results.sort(key=rank_key, reverse=True)
    passed = [result for result in results if result.get("passed")]
    if passed:
        winner = passed[0]
        experiment = winner["name"].split("-")[0]
        decision = {
            "reason": "at_least_one_checkpoint_passed",
            "winner": winner["name"],
            "e3_dataset": "data/clean-20k.jsonl",
            "e3_learning_rate": "1e-5" if experiment == "e1" else "2e-5",
        }
    else:
        decision = {
            "reason": "no_checkpoint_passed",
            "winner": None,
            "e3_dataset": "data/non-synthetic-10k.jsonl",
            "e3_learning_rate": "5e-6",
        }
    summary = {"ranking": results, "decision": decision}
    (RUN_DIR / "gate" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RUN_DIR / "gate" / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
