#!/usr/bin/env python3
"""Prepare clean, benchmark-deduplicated NuminaMath SFT datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


DATASET_ID = "AI-MO/NuminaMath-1.5"
REVISION = "1b05109f9e5c1ad06c0663519502416c30b300f8"
PARQUET_FILES = [f"data/train-{idx:05d}-of-00003.parquet" for idx in range(3)]
MODEL_PATH = "/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct"
COMPETITION_SOURCES = {"aops_forum", "metamath", "amc_aime"}
SYNTHETIC_SOURCES = {"orca_math", "synthetic_math"}
TYPE_TARGETS = {
    "Algebra": 0.35,
    "Geometry": 0.20,
    "Number Theory": 0.20,
    "Combinatorics": 0.15,
    "Other": 0.10,
}
LETTER_ANSWER = re.compile(r"^\s*[A-E]\s*$", re.IGNORECASE)
URL_OR_IMAGE = re.compile(r"https?://|!\[[^\]]*\]\(|<img\b", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-root", default="exp/baseline/qwen2.5-3b-instruct")
    return parser.parse_args()


def normalize_problem(text: str) -> str:
    return "".join(WORD.findall(text.lower()))


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def extract_boxed(text: str) -> list[str]:
    results = []
    cursor = 0
    marker = r"\boxed{"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        depth = 1
        idx = start + len(marker)
        begin = idx
        while idx < len(text) and depth:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            results.append(text[begin:idx - 1])
        cursor = max(idx, start + len(marker))
    return results


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^\$+|\$+$", "", text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", "").replace(r"\!", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def high_repetition(text: str) -> bool:
    chunks = [text[i:i + 80] for i in range(0, len(text) - 79, 80)]
    return len(chunks) >= 6 and 1 - len(set(chunks)) / len(chunks) > 0.40


def map_type(problem_type: str) -> str:
    return problem_type if problem_type in TYPE_TARGETS and problem_type != "Other" else "Other"


def bucket(row: dict) -> str | None:
    source = row["source"]
    if not row["synthetic"] and source in COMPETITION_SOURCES:
        return "competition"
    if not row["synthetic"] and source == "cn_k12":
        return "cn_k12"
    if row["synthetic"] and source in SYNTHETIC_SOURCES:
        return "synthetic"
    return None


def benchmark_prompts(root: Path) -> list[str]:
    patterns = [
        "raw/full/gsm8k/**/reviews/**/*.jsonl",
        "raw/math_500/full/**/reviews/**/*.jsonl",
    ]
    prompts = []
    for pattern in patterns:
        for path in root.glob(pattern):
            for line in path.open(encoding="utf-8"):
                record = json.loads(line)
                prompt = record["messages"][0]["content"].split("Please reason step by step")[0]
                prompts.append(prompt)
    if len(prompts) < 1800:
        raise RuntimeError(f"Expected local GSM8K/MATH-500 prompts, found only {len(prompts)}")
    return prompts


class ContaminationIndex:
    def __init__(self, prompts: list[str]):
        self.compact = [normalize_problem(prompt) for prompt in prompts]
        self.exact = set(self.compact)
        self.grams = [ngrams(words(prompt)) for prompt in prompts]
        self.inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for idx, grams in enumerate(self.grams):
            for gram in grams:
                self.inverted[gram].add(idx)

    def contaminated(self, problem: str) -> tuple[bool, str | None]:
        compact = normalize_problem(problem)
        if compact in self.exact:
            return True, "exact"
        grams = ngrams(words(problem))
        candidates = set()
        for gram in grams:
            candidates.update(self.inverted.get(gram, ()))
        for idx in candidates:
            other = self.compact[idx]
            short, long = sorted((compact, other), key=len)
            if short and len(short) / len(long) >= 0.50 and short in long and len(short) / len(long) >= 0.90:
                return True, "containment"
            union = grams | self.grams[idx]
            if union and len(grams & self.grams[idx]) / len(union) >= 0.80:
                return True, "jaccard"
        return False, None


def iter_rows() -> Iterable[dict]:
    columns = [
        "problem", "solution", "answer", "problem_type", "question_type",
        "problem_is_valid", "solution_is_valid", "source", "synthetic",
    ]
    for filename in PARQUET_FILES:
        path = hf_hub_download(DATASET_ID, filename, repo_type="dataset", revision=REVISION)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=20000, columns=columns):
            yield from batch.to_pylist()


def basic_filter(row: dict) -> str | None:
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()
    answer = (row.get("answer") or "").strip()
    if row.get("problem_is_valid") != "Yes" or row.get("solution_is_valid") != "Yes":
        return "invalid_label"
    if row.get("question_type") != "math-word-problem":
        return "question_type"
    if not problem or not solution or not answer:
        return "empty"
    if answer.lower() in {"proof", "notfound"} or LETTER_ANSWER.fullmatch(answer):
        return "answer_type"
    if URL_OR_IMAGE.search(problem + solution):
        return "url_or_image"
    if solution.count("{") != solution.count("}"):
        return "unbalanced_braces"
    if high_repetition(solution):
        return "repetition"
    boxes = extract_boxed(solution)
    if len(boxes) != 1:
        return "boxed_count"
    if normalize_answer(boxes[0]) != normalize_answer(answer):
        return "boxed_answer_mismatch"
    if bucket(row) is None:
        return "source_bucket"
    return None


def quota(total: int, proportions: dict[str, float]) -> dict[str, int]:
    result = {key: int(total * value) for key, value in proportions.items()}
    result[next(iter(result))] += total - sum(result.values())
    return result


def select_stratified(
    pools: dict[tuple[str, str], list[dict]],
    bucket_targets: dict[str, int],
    type_targets: dict[str, int],
    used: set[str],
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    selected: list[dict] = []
    notes: list[str] = []
    remaining_bucket = dict(bucket_targets)
    remaining_type = dict(type_targets)

    for bucket_name, bucket_total in bucket_targets.items():
        desired = quota(bucket_total, TYPE_TARGETS)
        for type_name, count in desired.items():
            available = [row for row in pools[(bucket_name, type_name)] if row["_id"] not in used]
            take = min(count, len(available))
            chosen = rng.sample(available, take)
            selected.extend(chosen)
            used.update(row["_id"] for row in chosen)
            remaining_bucket[bucket_name] -= take
            remaining_type[type_name] -= take
            if take < count:
                notes.append(f"{bucket_name}/{type_name}: requested {count}, selected {take}")

    while sum(remaining_bucket.values()) > 0:
        candidates = []
        for (bucket_name, type_name), rows in pools.items():
            if remaining_bucket.get(bucket_name, 0) <= 0:
                continue
            for row in rows:
                if row["_id"] not in used:
                    priority = (remaining_type.get(type_name, 0) > 0, not row["synthetic"])
                    candidates.append((priority, bucket_name, type_name, row))
        if not candidates:
            raise RuntimeError(f"Unable to fill quotas; remaining buckets: {remaining_bucket}")
        best_priority = max(item[0] for item in candidates)
        eligible = [item for item in candidates if item[0] == best_priority]
        _, bucket_name, type_name, row = rng.choice(eligible)
        selected.append(row)
        used.add(row["_id"])
        remaining_bucket[bucket_name] -= 1
        remaining_type[type_name] = remaining_type.get(type_name, 0) - 1
    return selected, notes


def write_jsonl(path: Path, rows: list[dict]) -> None:
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
                "solution_tokens": row["_solution_tokens"],
                "message_tokens": row["_message_tokens"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    return {
        "size": len(rows),
        "source": dict(Counter(row["source"] for row in rows).most_common()),
        "bucket": dict(Counter(bucket(row) for row in rows).most_common()),
        "problem_type": dict(Counter(map_type(row["problem_type"]) for row in rows).most_common()),
        "synthetic": dict(Counter(str(row["synthetic"]) for row in rows).most_common()),
        "short_stable": sum(
            row["_solution_tokens"] <= 512
            and row["source"] in {"cn_k12", "orca_math", "synthetic_math"}
            and map_type(row["problem_type"]) == "Algebra"
            for row in rows
        ),
        "solution_tokens": {
            "min": min(row["_solution_tokens"] for row in rows),
            "max": max(row["_solution_tokens"] for row in rows),
            "mean": round(sum(row["_solution_tokens"] for row in rows) / len(rows), 2),
        },
    }


def chat_template_token_count(tokenizer, messages: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    contamination = ContaminationIndex(benchmark_prompts(Path(args.benchmark_root)))
    rejected = Counter()
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    unique: set[str] = set()

    for row in iter_rows():
        reason = basic_filter(row)
        if reason:
            rejected[reason] += 1
            continue
        problem_id = normalize_problem(row["problem"])
        if problem_id in unique:
            rejected["duplicate"] += 1
            continue
        is_contaminated, contamination_type = contamination.contaminated(row["problem"])
        if is_contaminated:
            rejected[f"benchmark_{contamination_type}"] += 1
            continue
        solution_tokens = len(tokenizer.encode(row["solution"], add_special_tokens=False))
        if not 128 <= solution_tokens <= 1536:
            rejected["solution_tokens"] += 1
            continue
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": row["problem"].strip()},
            {"role": "assistant", "content": row["solution"].strip()},
        ]
        message_tokens = chat_template_token_count(tokenizer, messages)
        # Keep a safety margin for ms-swift's template-side tokens so training
        # does not silently drop examples at max_length=2048.
        if message_tokens > 1950:
            rejected["message_tokens"] += 1
            continue
        unique.add(problem_id)
        row["_id"] = problem_id
        row["_solution_tokens"] = solution_tokens
        row["_message_tokens"] = message_tokens
        pools[(bucket(row), map_type(row["problem_type"]))].append(row)

    for rows in pools.values():
        rng.shuffle(rows)

    type_10k = quota(10000, TYPE_TARGETS)
    clean_10k, notes_10k = select_stratified(
        pools, {"competition": 3500, "cn_k12": 3500, "synthetic": 3000}, type_10k, set(), rng)
    used_20k = {row["_id"] for row in clean_10k}
    extra_10k, notes_20k = select_stratified(
        pools, {"competition": 3500, "cn_k12": 3500, "synthetic": 3000}, type_10k, used_20k, rng)
    clean_20k = clean_10k + extra_10k
    non_synthetic_10k, notes_non_syn = select_stratified(
        pools, {"competition": 5000, "cn_k12": 5000}, type_10k, set(), rng)

    selected_ids = {row["_id"] for row in clean_20k + non_synthetic_10k}
    val_pool = [
        row for (bucket_name, _), rows in pools.items() if bucket_name != "synthetic"
        for row in rows if row["_id"] not in selected_ids
    ]
    if len(val_pool) < 1000:
        raise RuntimeError(f"Not enough non-synthetic validation rows: {len(val_pool)}")
    val = rng.sample(val_pool, 1000)

    for rows in (clean_10k, clean_20k, non_synthetic_10k, val):
        rng.shuffle(rows)
    outputs = {
        "clean-10k": clean_10k,
        "clean-20k": clean_20k,
        "non-synthetic-10k": non_synthetic_10k,
        "val": val,
    }
    for name, rows in outputs.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "seed": args.seed,
        "model_tokenizer": args.model,
        "filters": {
            "question_type": "math-word-problem",
            "solution_tokens": [128, 1536],
            "message_tokens_max": 1950,
            "exactly_one_boxed_answer": True,
            "boxed_must_match_answer": True,
            "benchmark_jaccard_threshold": 0.80,
            "benchmark_containment_threshold": 0.90,
        },
        "candidate_pool": {
            f"{bucket_name}/{type_name}": len(rows)
            for (bucket_name, type_name), rows in sorted(pools.items())
        },
        "rejected": dict(rejected.most_common()),
        "datasets": {name: summarize(rows) for name, rows in outputs.items()},
        "quota_notes": {
            "clean-10k": notes_10k,
            "clean-20k-extra": notes_20k,
            "non-synthetic-10k": notes_non_syn,
        },
        "strict_superset_check": all(row["_id"] in {r["_id"] for r in clean_20k} for row in clean_10k),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
