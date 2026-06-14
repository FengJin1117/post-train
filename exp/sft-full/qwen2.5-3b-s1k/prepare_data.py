#!/usr/bin/env python3
"""Prepare and validate the fixed s1K dataset for MS-SWIFT."""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
DATA = RUN / "data"
MODEL = "/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B"
REVISION = "278d72baaa2b887a7e76a70a0ae254a5a45536e4"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def assistant(row: dict) -> str:
    attempt = row["attempt"].strip()
    if "Answer:" not in attempt:
        attempt = "Answer: " + attempt
    thinking = "\n".join(row["thinking_trajectories"]).strip()
    return f"<|im_start|>think\n{thinking}\n<|im_start|>answer\n{attempt}"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    raw = load_dataset("simplescaling/s1K", revision=REVISION, split="train")
    official = load_dataset("simplescaling/s1K_tokenized", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    records = []
    lengths = []
    exact_matches = 0
    for index, row in enumerate(raw):
        response = assistant(row)
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": response},
        ]
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, *messages],
            tokenize=False,
        )
        length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        lengths.append(length)
        exact_matches += rendered == official[index]["text"]
        records.append({
            "messages": messages,
            "cot_type": row["cot_type"],
            "source_type": row["source_type"],
            "s1k_index": index,
        })

    if len(records) != 1000:
        raise RuntimeError(f"Expected 1000 rows, got {len(records)}")
    if any(not x["messages"][0]["content"] or not x["messages"][1]["content"] for x in records):
        raise RuntimeError("Empty question or assistant response")
    if max(lengths) > 32768:
        raise RuntimeError(f"Found sequence longer than 32768: {max(lengths)}")
    if exact_matches != 1000:
        raise RuntimeError(f"Only {exact_matches}/1000 rendered rows match official s1K_tokenized")

    with (DATA / "s1k-ms-swift.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "dataset": "simplescaling/s1K",
        "revision": REVISION,
        "rows": len(records),
        "official_render_exact_matches": exact_matches,
        "cot_type": Counter(x["cot_type"] for x in records),
        "source_type": Counter(x["source_type"] for x in records),
        "lengths": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": round(statistics.mean(lengths), 2),
            "max": max(lengths),
        },
        "known_contamination_risk": {
            "aime24": "s1K contains 287 rows from qq8933/AIME_1983_2024",
            "scan": "Run contamination.py after EvalScope has cached all benchmark datasets.",
        },
    }
    (RUN / "data-report.json").write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
