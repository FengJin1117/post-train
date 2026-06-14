#!/usr/bin/env python3
"""Validate linked data and exact ms-swift template lengths."""

import json
from pathlib import Path

from swift.arguments import SftArguments
from swift.template import MaxLengthError


ROOT = Path(__file__).resolve().parent
MODEL = "/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B"


def check(path: Path, expected: int) -> dict:
    args = SftArguments(
        model=MODEL,
        dataset=[str(path)],
        template="qwen2_5",
        system="You are a helpful assistant.",
        max_length=2048,
        output_dir="/tmp/qwen25-base-clean-v2-length-check",
    )
    processor = args.get_model_processor(load_model=False)[1]
    template = args.get_template(processor)
    template.set_mode("train")
    lengths, bad = [], []
    for index, line in enumerate(path.open()):
        try:
            value = template.encode(json.loads(line), return_length=True)["lengths"]
            lengths.append(max(value) if isinstance(value, list) else value)
        except MaxLengthError:
            bad.append(index)
    assert len(lengths) + len(bad) == expected
    assert not bad, f"{path}: over-length rows {bad[:10]}"
    return {"rows": expected, "min": min(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 2)}


def main() -> None:
    for name in ("clean-10k.jsonl", "val.jsonl", "manifest.json"):
        path = ROOT / "data" / name
        assert path.is_symlink() and path.exists(), f"Invalid symlink: {path}"
    result = {
        "clean-10k": check(ROOT / "data/clean-10k.jsonl", 10000),
        "val": check(ROOT / "data/val.jsonl", 1000),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

