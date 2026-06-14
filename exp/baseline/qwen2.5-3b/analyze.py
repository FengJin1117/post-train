#!/usr/bin/env python3
"""Summarize Qwen2.5-3B baseline evaluation artifacts."""

import argparse
import json
import statistics
from pathlib import Path


def load_latest(root: Path, benchmark: str, kind: str) -> list[dict]:
    files = list(root.glob(f"{benchmark}/eval_output/native/*/{kind}/**/*.jsonl"))
    if not files:
        raise RuntimeError(f"No {kind} files for {benchmark}")
    stamp = max(p.parts[p.parts.index("native") + 1] for p in files)
    return [json.loads(line) for p in files if stamp in p.parts for line in p.open() if line.strip()]


def metrics(root: Path, benchmark: str) -> dict:
    predictions = load_latest(root, benchmark, "predictions")
    reviews = load_latest(root, benchmark, "reviews")
    responses = [x["model_output"]["choices"][0]["message"]["content"] or "" for x in predictions]
    stops = [x["model_output"]["choices"][0]["stop_reason"] for x in predictions]
    tokens = [x["model_output"]["usage"]["output_tokens"] for x in predictions]
    scores = [x["sample_score"]["score"]["value"]["acc"] for x in reviews]
    extracted = [x["sample_score"]["score"].get("extracted_prediction") or "" for x in reviews]
    result = {
        "samples": len(scores),
        "correct": int(sum(scores)),
        "accuracy": round(sum(scores) / len(scores), 6),
        "extraction_failure_count": sum(not x.strip() for x in extracted),
        "boxed_rate": round(sum(r"\boxed{" in x for x in responses) / len(responses), 6),
        "max_token_rate": round(sum(x == "max_tokens" for x in stops) / len(stops), 6),
        "non_empty_rate": round(sum(bool(x.strip()) for x in responses) / len(responses), 6),
        "output_tokens_mean": round(statistics.mean(tokens), 2),
        "output_tokens_median": statistics.median(tokens),
    }
    if benchmark == "gaokao_math_cloze":
        special = [x for x in reviews if ";" in x["target"]]
        label = "multi_blank"
    elif benchmark == "gaokao_math_qa":
        special = [x for x in reviews if x["sample_score"]["sample_metadata"].get("is_multi_select")]
        label = "multi_select"
    else:
        special = []
        label = ""
    if special:
        special_scores = [x["sample_score"]["score"]["value"]["acc"] for x in special]
        result[f"{label}_samples"] = len(special_scores)
        result[f"{label}_correct"] = int(sum(special_scores))
        result[f"{label}_accuracy"] = round(sum(special_scores) / len(special_scores), 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    result = {name: metrics(root, name) for name in args.benchmarks}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
