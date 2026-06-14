#!/usr/bin/env python3
"""Validate and summarize Qwen2.5-Math-1.5B EvalScope artifacts."""

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MODEL = "Qwen/Qwen2.5-Math-1.5B"
SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."


def load_latest(root: Path, benchmark: str, kind: str) -> tuple[list[dict], str]:
    files = list(root.glob(f"{benchmark}/eval_output/native/*/{kind}/**/*.jsonl"))
    if not files:
        raise RuntimeError(f"No {kind} files for {benchmark}")
    stamps = {p.parts[p.parts.index("native") + 1] for p in files}
    stamp = max(stamps)
    rows = []
    for path in files:
        if stamp not in path.parts:
            continue
        subset = path.stem.removeprefix(f"{benchmark}_")
        for line in path.open():
            if line.strip():
                row = json.loads(line)
                row["_artifact_subset"] = subset
                rows.append(row)
    return rows, stamp


def get_score(row: dict) -> float:
    return float(row["sample_score"]["score"]["value"]["acc"])


def summarize(root: Path, benchmark: str, expected: int) -> dict:
    predictions, stamp = load_latest(root, benchmark, "predictions")
    reviews, review_stamp = load_latest(root, benchmark, "reviews")
    if stamp != review_stamp:
        raise RuntimeError(f"{benchmark}: prediction/review timestamps differ")
    if len(predictions) != expected or len(reviews) != expected:
        raise RuntimeError(
            f"{benchmark}: expected {expected} predictions/reviews, "
            f"got {len(predictions)}/{len(reviews)}"
        )

    responses = [row["model_output"]["choices"][0]["message"]["content"] or "" for row in predictions]
    stops = [row["model_output"]["choices"][0].get("stop_reason") for row in predictions]
    tokens = [int(row["model_output"]["usage"]["output_tokens"]) for row in predictions]
    scores = [get_score(row) for row in reviews]
    result = {
        "timestamp": stamp,
        "samples": len(scores),
        "correct": int(sum(scores)),
        "accuracy": round(sum(scores) / len(scores), 6),
        "non_empty_rate": round(sum(bool(x.strip()) for x in responses) / len(responses), 6),
        "boxed_rate": round(sum(r"\boxed{" in x for x in responses) / len(responses), 6),
        "max_token_rate": round(sum(x == "max_tokens" for x in stops) / len(stops), 6),
        "output_tokens_mean": round(statistics.mean(tokens), 2),
        "output_tokens_median": statistics.median(tokens),
    }
    if benchmark == "math_500":
        levels = {}
        for row in reviews:
            subset = row["_artifact_subset"]
            levels.setdefault(subset, []).append(get_score(row))
        result["levels"] = {
            level: {
                "samples": len(values),
                "correct": int(sum(values)),
                "accuracy": round(sum(values) / len(values), 6),
            }
            for level, values in sorted(levels.items())
        }
    return result


def package_version(env: str, package: str) -> str:
    code = (
        "import importlib.metadata as m; "
        f"print(m.version({package!r}))"
    )
    return subprocess.check_output(
        ["conda", "run", "-n", env, "--no-capture-output", "python", "-c", code],
        text=True,
    ).strip()


def markdown(result: dict, root: Path, env: str) -> str:
    gsm = result["gsm8k"]
    math = result["math_500"]
    repo_root = root.parents[4]
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root / "ms-swift"), "rev-parse", "HEAD"], text=True
    ).strip()
    versions = {
        name: package_version(env, package)
        for name, package in (
            ("MS-SWIFT", "ms-swift"),
            ("EvalScope", "evalscope"),
            ("vLLM", "vllm"),
            ("Torch", "torch"),
            ("Transformers", "transformers"),
        )
    }
    level_lines = "\n".join(
        f"| {level} | {data['samples']} | {data['correct']} | {data['accuracy'] * 100:.2f} |"
        for level, data in math["levels"].items()
    )
    diag_lines = "\n".join(
        f"| {name} | {data['non_empty_rate'] * 100:.2f} | {data['boxed_rate'] * 100:.2f} | "
        f"{data['max_token_rate'] * 100:.2f} | {data['output_tokens_mean']:.2f} | "
        f"{data['output_tokens_median']:.1f} |"
        for name, data in (("GSM8K", gsm), ("MATH-500", math))
    )
    version_lines = "\n".join(f"- {name}: `{value}`" for name, value in versions.items())
    return f"""# Qwen2.5-Math-1.5B Base Benchmark Results

Status: completed

## Evaluation Policy

- Model: `{MODEL}` (base, non-Instruct)
- Template: `qwen2_5_math`
- System prompt: `{SYSTEM}`
- Backend: Native EvalScope with vLLM
- Generation: deterministic, `temperature=0.0`, `do_sample=false`, `max_tokens=3072`
- Context length: native model limit, `vllm_max_model_len=4096`
- Scoring: rule judge only
- Evaluation date: `{datetime.now(timezone.utc).date().isoformat()}` UTC

## Results

| Benchmark | Samples | Correct | Accuracy (%) |
| --- | ---: | ---: | ---: |
| GSM8K | {gsm['samples']} | {gsm['correct']} | {gsm['accuracy'] * 100:.2f} |
| MATH-500 | {math['samples']} | {math['correct']} | {math['accuracy'] * 100:.2f} |

## MATH-500 Difficulty Breakdown

| Difficulty | Samples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
{level_lines}

## Output Diagnostics

| Benchmark | Non-empty (%) | Boxed (%) | Max-token (%) | Mean tokens | Median tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
{diag_lines}

## Environment

- MS-SWIFT commit: `{commit}`
{version_lines}

## Command

```bash
cd {repo_root}
tmux new-session -d -s qwen25-math-15b-baseline \\
  "bash exp/baseline/qwen2.5-math-1.5b/run_baseline.sh all"
```

The runner uses GPUs `1` and `4` by default for parallel full evaluation. Raw
EvalScope artifacts are stored under `raw/full/`; logs are stored under `logs/`.
The originally planned 10,000-token service context and 8,192-token generation
limit were not used because this base model declares a 4,096-token context and
crashed vLLM with a CUDA device-side assertion when forced beyond that limit.

## Raw Artifacts

- GSM8K: `raw/full/gsm8k/eval_output/native/{gsm['timestamp']}/`
- MATH-500: `raw/full/math_500/eval_output/native/{math['timestamp']}/`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-gsm8k", required=True, type=int)
    parser.add_argument("--expected-math-500", required=True, type=int)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--env", default="vllm")
    args = parser.parse_args()

    result = {
        "gsm8k": summarize(args.root, "gsm8k", args.expected_gsm8k),
        "math_500": summarize(args.root, "math_500", args.expected_math_500),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(result, args.root, args.env))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
