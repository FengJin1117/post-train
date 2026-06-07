#!/usr/bin/env python3
"""Validate and summarize a full GSM8K/MATH-500 evaluation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


EXPECTED = {"gsm8k": 1319, "math_500": 500}
BASELINE = {"gsm8k": 0.8469, "math_500": 0.6660}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def repeated(text: str) -> bool:
    chunks = [text[i:i + 80] for i in range(0, len(text) - 79, 80)]
    return len(chunks) >= 6 and 1 - len(set(chunks)) / len(chunks) > 0.40


def latest_files(root: Path, benchmark: str, kind: str) -> list[Path]:
    files = list(root.glob(f"{benchmark}/eval_output/native/*/{kind}/**/*.jsonl"))
    if not files:
        raise RuntimeError(f"No {kind} files found for {benchmark} under {root}")
    stamps = [path.parts[path.parts.index("native") + 1] for path in files]
    latest_stamp = max(stamps)
    return [path for path in files if latest_stamp in path.parts]


def load_jsonl(paths: list[Path]) -> list[dict]:
    return [
        json.loads(line)
        for path in paths
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def benchmark_metrics(root: Path, benchmark: str) -> dict:
    predictions = load_jsonl(latest_files(root, benchmark, "predictions"))
    reviews = load_jsonl(latest_files(root, benchmark, "reviews"))
    if len(predictions) != EXPECTED[benchmark] or len(reviews) != EXPECTED[benchmark]:
        raise RuntimeError(
            f"{benchmark}: expected {EXPECTED[benchmark]} predictions/reviews, "
            f"found {len(predictions)}/{len(reviews)}"
        )

    responses, output_tokens, stop_reasons = [], [], []
    for record in predictions:
        choice = record["model_output"]["choices"][0]
        responses.append(choice["message"]["content"] or "")
        stop_reasons.append(choice["stop_reason"])
        output_tokens.append(record["model_output"]["usage"]["output_tokens"])

    scores = [record["sample_score"]["score"]["value"]["acc"] for record in reviews]
    invalid = sum(
        not text.strip() or not math.isfinite(float(score))
        for text, score in zip(responses, scores)
    )
    correct = int(sum(scores))
    metrics = {
        "samples": len(scores),
        "correct": correct,
        "accuracy": round(correct / len(scores), 6),
        "baseline_accuracy": BASELINE[benchmark],
        "delta": round(correct / len(scores) - BASELINE[benchmark], 6),
        "non_empty_rate": round(sum(bool(text.strip()) for text in responses) / len(responses), 6),
        "boxed_rate": round(sum(r"\boxed{" in text for text in responses) / len(responses), 6),
        "max_token_rate": round(sum(reason == "max_tokens" for reason in stop_reasons) / len(stop_reasons), 6),
        "repetition_rate": round(sum(repeated(text) for text in responses) / len(responses), 6),
        "empty_or_invalid": invalid,
        "output_tokens_mean": round(statistics.mean(output_tokens), 2),
        "output_tokens_median": statistics.median(output_tokens),
    }
    if benchmark == "math_500":
        levels = {}
        for path in latest_files(root, benchmark, "reviews"):
            level_scores = [
                record["sample_score"]["score"]["value"]["acc"]
                for record in load_jsonl([path])
            ]
            level = path.stem.removeprefix("math_500_")
            levels[level] = {
                "samples": len(level_scores),
                "correct": int(sum(level_scores)),
                "accuracy": round(sum(level_scores) / len(level_scores), 6),
            }
        metrics["levels"] = dict(sorted(levels.items()))
    return metrics


def main() -> None:
    args = parse_args()
    root = Path(args.eval_dir)
    summary = {
        "name": root.name,
        "gsm8k": benchmark_metrics(root, "gsm8k"),
        "math_500": benchmark_metrics(root, "math_500"),
    }
    summary["success"] = {
        "math_500_improved": summary["math_500"]["accuracy"] > BASELINE["math_500"],
        "math_500_target_met": summary["math_500"]["accuracy"] >= 0.6760,
        "gsm8k_preserved": summary["gsm8k"]["accuracy"] >= 0.8269,
        "math_500_max_token_target_met": summary["math_500"]["max_token_rate"] < 0.02,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
