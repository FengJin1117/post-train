#!/usr/bin/env python3
"""Summarize one fixed-subset checkpoint gate evaluation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def repeated(text: str) -> bool:
    chunks = [text[i:i + 80] for i in range(0, len(text) - 79, 80)]
    return len(chunks) >= 6 and 1 - len(set(chunks)) / len(chunks) > 0.40


def latest_files(root: Path, kind: str, benchmark: str) -> list[Path]:
    files = list(root.glob(f"{benchmark}/eval_output/native/*/{kind}/**/*.jsonl"))
    if not files:
        raise RuntimeError(f"No {kind} files found for {benchmark} under {root}")
    latest_stamp = max(path.parts[path.parts.index("native") + 1] for path in files)
    return [path for path in files if latest_stamp in path.parts]


def benchmark_metrics(root: Path, benchmark: str) -> dict:
    predictions = []
    reviews = []
    for path in latest_files(root, "predictions", benchmark):
        predictions.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    for path in latest_files(root, "reviews", benchmark):
        reviews.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    output_tokens, stop_reasons, responses = [], [], []
    for record in predictions:
        output_tokens.append(record["model_output"]["usage"]["output_tokens"])
        choice = record["model_output"]["choices"][0]
        stop_reasons.append(choice["stop_reason"])
        responses.append(choice["message"]["content"] or "")
    scores = [record["sample_score"]["score"]["value"]["acc"] for record in reviews]
    invalid = sum(not text.strip() or not math.isfinite(float(score)) for text, score in zip(responses, scores))
    return {
        "samples": len(scores),
        "accuracy": round(sum(scores) / len(scores), 6),
        "boxed_rate": round(sum(r"\boxed{" in text for text in responses) / len(responses), 6),
        "max_token_rate": round(sum(reason == "max_tokens" for reason in stop_reasons) / len(stop_reasons), 6),
        "repetition_rate": round(sum(repeated(text) for text in responses) / len(responses), 6),
        "empty_or_invalid": invalid,
        "output_tokens_mean": round(statistics.mean(output_tokens), 2),
        "output_tokens_median": statistics.median(output_tokens),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.gate_dir)
    metrics = {
        "name": args.name,
        "gsm8k": benchmark_metrics(root, "gsm8k"),
        "math_500": benchmark_metrics(root, "math_500"),
    }
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        metrics["thresholds"] = {
            "gsm8k_min": round(baseline["gsm8k"]["accuracy"] - 0.03, 6),
            "math_500_min": round(baseline["math_500"]["accuracy"] - 0.02, 6),
        }
        metrics["passed"] = (
            metrics["gsm8k"]["accuracy"] >= metrics["thresholds"]["gsm8k_min"]
            and metrics["math_500"]["accuracy"] >= metrics["thresholds"]["math_500_min"]
            and metrics["gsm8k"]["max_token_rate"] < 0.01
            and metrics["math_500"]["max_token_rate"] < 0.01
            and metrics["gsm8k"]["boxed_rate"] >= 0.98
            and metrics["math_500"]["boxed_rate"] >= 0.98
            and metrics["gsm8k"]["empty_or_invalid"] == 0
            and metrics["math_500"]["empty_or_invalid"] == 0
        )
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
