#!/usr/bin/env python3
"""Summarize EvalScope artifacts for gates and full evaluations."""

import argparse
import json
import statistics
from pathlib import Path


def latest(root: Path, kind: str) -> list[dict]:
    files = list(root.glob(f"eval_output/native/*/{kind}/**/*.jsonl"))
    if not files:
        raise RuntimeError(f"No {kind} artifacts under {root}")
    stamp = max(x.parts[x.parts.index("native") + 1] for x in files)
    return [json.loads(line) for x in files if stamp in x.parts for line in x.open() if line.strip()]


def summarize(root: Path) -> dict:
    pred, review = latest(root, "predictions"), latest(root, "reviews")
    if len(pred) != len(review):
        raise RuntimeError(f"Prediction/review mismatch: {len(pred)}/{len(review)}")
    text = [x["model_output"]["choices"][0]["message"]["content"] or "" for x in pred]
    stop = [x["model_output"]["choices"][0].get("stop_reason") for x in pred]
    toks = [x["model_output"]["usage"]["output_tokens"] for x in pred]
    score = [x["sample_score"]["score"]["value"]["acc"] for x in review]
    return {
        "samples": len(score),
        "correct": int(sum(score)),
        "accuracy": round(sum(score) / len(score), 6),
        "non_empty_rate": round(sum(bool(x.strip()) for x in text) / len(text), 6),
        "boxed_rate": round(sum(r"\boxed{" in x for x in text) / len(text), 6),
        "max_token_rate": round(sum(x == "max_tokens" for x in stop) / len(stop), 6),
        "output_tokens_mean": round(statistics.mean(toks), 2),
        "output_tokens_median": statistics.median(toks),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--stage", choices=["gate", "full"], required=True)
    a = p.parse_args()
    result = {}
    for benchmark in ("gsm8k", "math_500", "aime24", "aime25"):
        path = a.root / benchmark
        if path.exists():
            result[benchmark] = summarize(path)
    expected = {
        "gate": {"gsm8k": 200, "math_500": 100},
        "full": {"gsm8k": 1319, "math_500": 500, "aime24": 30, "aime25": 30},
    }[a.stage]
    for benchmark, count in expected.items():
        actual = result.get(benchmark, {}).get("samples")
        if actual != count:
            raise RuntimeError(f"{benchmark}: expected {count} samples, got {actual}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
