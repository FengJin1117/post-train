#!/usr/bin/env python3
"""Summarize and judge one base-model checkpoint gate."""

import argparse
import json
import statistics
from pathlib import Path


def load_latest(root: Path, benchmark: str, kind: str) -> list[dict]:
    files = list(root.glob(f"{benchmark}/eval_output/native/*/{kind}/**/*.jsonl"))
    stamp = max(p.parts[p.parts.index("native") + 1] for p in files)
    return [json.loads(line) for p in files if stamp in p.parts for line in p.open() if line.strip()]


def repeated(text: str) -> bool:
    chunks = [text[i:i + 80] for i in range(0, len(text) - 79, 80)]
    return len(chunks) >= 6 and 1 - len(set(chunks)) / len(chunks) > 0.40


def metrics(root: Path, benchmark: str) -> dict:
    pred = load_latest(root, benchmark, "predictions")
    rev = load_latest(root, benchmark, "reviews")
    text = [x["model_output"]["choices"][0]["message"]["content"] or "" for x in pred]
    stop = [x["model_output"]["choices"][0]["stop_reason"] for x in pred]
    toks = [x["model_output"]["usage"]["output_tokens"] for x in pred]
    score = [x["sample_score"]["score"]["value"]["acc"] for x in rev]
    return {
        "samples": len(score),
        "accuracy": round(sum(score) / len(score), 6),
        "boxed_rate": round(sum(r"\boxed{" in x for x in text) / len(text), 6),
        "max_token_rate": round(sum(x == "max_tokens" for x in stop) / len(stop), 6),
        "repetition_rate": round(sum(repeated(x) for x in text) / len(text), 6),
        "empty": sum(not x.strip() for x in text),
        "output_tokens_median": statistics.median(toks),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True); p.add_argument("--name", required=True)
    p.add_argument("--baseline", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    base = json.loads(Path(a.baseline).read_text())
    result = {"name": a.name, "gsm8k": metrics(Path(a.root), "gsm8k"), "math_500": metrics(Path(a.root), "math_500")}
    result["thresholds"] = {x: round(base[x]["accuracy"] + 0.05, 6) for x in ("gsm8k", "math_500")}
    result["passed"] = all([
        result["gsm8k"]["accuracy"] >= result["thresholds"]["gsm8k"],
        result["math_500"]["accuracy"] >= result["thresholds"]["math_500"],
        result["gsm8k"]["boxed_rate"] >= .90, result["math_500"]["boxed_rate"] >= .90,
        result["gsm8k"]["max_token_rate"] < .05, result["math_500"]["max_token_rate"] < .05,
        result["gsm8k"]["empty"] == 0, result["math_500"]["empty"] == 0,
        result["gsm8k"]["repetition_rate"] == 0, result["math_500"]["repetition_rate"] == 0,
    ])
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

