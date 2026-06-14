#!/usr/bin/env python3
"""Scan s1K questions against questions recorded in EvalScope predictions."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def ngrams(text: str, n: int = 13) -> set[str]:
    words = re.sub(r"\s+", " ", text.lower().strip()).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def question(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return message.get("content") or ""
    raise RuntimeError("No user question in EvalScope artifact")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    train = [json.loads(x) for x in a.train.read_text().splitlines() if x.strip()]
    lookup = defaultdict(set)
    for idx, row in enumerate(train):
        for gram in ngrams(row["messages"][0]["content"]):
            lookup[gram].add(idx)

    result = {}
    for benchmark in ("gsm8k", "math_500", "aime24", "aime25"):
        files = list(a.eval_root.glob(f"base/{benchmark}/eval_output/native/*/predictions/**/*.jsonl"))
        if not files:
            raise RuntimeError(f"Missing base predictions for {benchmark}")
        stamp = max(x.parts[x.parts.index("native") + 1] for x in files)
        rows = [json.loads(line) for x in files if stamp in x.parts for line in x.open() if line.strip()]
        matches = []
        for idx, row in enumerate(rows):
            overlap = sorted(ngrams(question(row)) & lookup.keys())
            if overlap:
                matches.append({"eval_index": idx, "matching_ngram": overlap[0]})
        result[benchmark] = {"samples": len(rows), "matched": len(matches), "examples": matches}

    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
