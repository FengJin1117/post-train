#!/usr/bin/env python3
"""Prepare a filtered NuminaMath-1.5 SFT-LoRA dataset.

The output uses ms-swift's native messages JSONL shape:
{"messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": solution}], ...}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET_ID = "AI-MO/NuminaMath-1.5"
REVISION = "1b05109f9e5c1ad06c0663519502416c30b300f8"
PARQUET_FILES = [f"data/train-{idx:05d}-of-00003.parquet" for idx in range(3)]
IMAGE_PATTERNS = (
    re.compile(r"!\[[^\]]*\]\(", re.IGNORECASE),
    re.compile(r"<img\b", re.IGNORECASE),
)
HIGH_SOURCES = {"olympiads", "aops_forum", "cn_contest", "amc_aime"}
MID_SOURCES = {"orca_math", "synthetic_math"}
CN_TYPES = {"Algebra", "Geometry"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data", help="Directory for train/val JSONL and manifest.")
    parser.add_argument("--train-size", type=int, default=40000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-max", type=int, default=12000)
    return parser.parse_args()


def has_image(text: str) -> bool:
    return any(pattern.search(text) for pattern in IMAGE_PATTERNS)


def valid_row(row: Dict) -> bool:
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()
    answer = (row.get("answer") or "").strip()
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return False
    if not problem or not solution or not answer or answer == "notfound":
        return False
    if has_image(problem) or has_image(solution):
        return False
    return True


def iter_rows() -> Iterable[Dict]:
    columns = [
        "problem",
        "solution",
        "answer",
        "problem_type",
        "question_type",
        "problem_is_valid",
        "solution_is_valid",
        "source",
        "synthetic",
    ]
    for filename in PARQUET_FILES:
        path = hf_hub_download(DATASET_ID, filename, repo_type="dataset", revision=REVISION)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=20000, columns=columns):
            yield from batch.to_pylist()


def sample_without_replacement(items: List[Dict], size: int, rng: random.Random, label: str) -> List[Dict]:
    if len(items) < size:
        raise RuntimeError(f"Not enough rows for {label}: need {size}, have {len(items)}")
    return rng.sample(items, size)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "messages": [
                    {"role": "user", "content": row["problem"].strip()},
                    {"role": "assistant", "content": row["solution"].strip()},
                ],
                "answer": row["answer"],
                "source": row["source"],
                "problem_type": row["problem_type"],
                "question_type": row["question_type"],
                "synthetic": row["synthetic"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(rows: List[Dict]) -> Dict:
    return {
        "size": len(rows),
        "source": dict(Counter(row["source"] for row in rows).most_common()),
        "problem_type": dict(Counter(row["problem_type"] for row in rows).most_common()),
        "question_type": dict(Counter(row["question_type"] for row in rows).most_common()),
        "synthetic": dict(Counter(str(row["synthetic"]) for row in rows).most_common()),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    high, mid_synthetic, mid_fallback, cn = [], [], [], []
    all_valid = []
    seen_ids = set()

    for idx, row in enumerate(iter_rows()):
        if not valid_row(row):
            continue
        row["_idx"] = idx
        all_valid.append(row)
        source = row["source"]
        qtype = row["question_type"]
        ptype = row["problem_type"]
        synthetic = bool(row["synthetic"])
        if source in HIGH_SOURCES:
            high.append(row)
        if source in MID_SOURCES and qtype == "math-word-problem" and synthetic:
            mid_synthetic.append(row)
        if qtype == "math-word-problem" and not synthetic and source not in HIGH_SOURCES:
            mid_fallback.append(row)
        if source == "cn_k12" and ptype in CN_TYPES:
            cn.append(row)

    selected = []

    high_rows = sample_without_replacement(high, 18000, rng, "high competition bucket")
    selected.extend(high_rows)
    seen_ids.update(row["_idx"] for row in high_rows)

    mid_syn_rows = sample_without_replacement(mid_synthetic, min(12000, args.synthetic_max), rng, "synthetic mid bucket")
    selected.extend(mid_syn_rows)
    seen_ids.update(row["_idx"] for row in mid_syn_rows)

    mid_need = 14000 - len(mid_syn_rows)
    mid_pool = [row for row in mid_fallback if row["_idx"] not in seen_ids]
    mid_fb_rows = sample_without_replacement(mid_pool, mid_need, rng, "non-synthetic mid fallback bucket")
    selected.extend(mid_fb_rows)
    seen_ids.update(row["_idx"] for row in mid_fb_rows)

    cn_pool = [row for row in cn if row["_idx"] not in seen_ids]
    cn_rows = sample_without_replacement(cn_pool, 8000, rng, "cn_k12 algebra/geometry bucket")
    selected.extend(cn_rows)
    seen_ids.update(row["_idx"] for row in cn_rows)

    if len(selected) != args.train_size:
        raise RuntimeError(f"Expected train size {args.train_size}, got {len(selected)}")
    if sum(1 for row in selected if row["synthetic"]) > args.synthetic_max:
        raise RuntimeError("Synthetic cap exceeded")

    val_pool = [row for row in all_valid if row["_idx"] not in seen_ids and not row["synthetic"]]
    val_rows = sample_without_replacement(val_pool, args.val_size, rng, "validation bucket")

    rng.shuffle(selected)
    rng.shuffle(val_rows)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train_path, selected)
    write_jsonl(val_path, val_rows)

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "seed": args.seed,
        "filters": [
            'problem_is_valid == "Yes"',
            'solution_is_valid == "Yes"',
            "problem/solution/answer non-empty",
            'answer != "notfound"',
            "no Markdown/HTML image links",
            f"synthetic train rows <= {args.synthetic_max}",
        ],
        "paths": {"train": str(train_path), "val": str(val_path)},
        "train": summarize(selected),
        "val": summarize(val_rows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
