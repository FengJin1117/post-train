#!/usr/bin/env python3
"""Restore the base tokenizer config for Transformers 4.x/vLLM compatibility."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--base",
        type=Path,
        default=Path("/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2___5-3B"),
    )
    a = p.parse_args()
    checkpoint_config = json.loads((a.checkpoint / "tokenizer_config.json").read_text())
    extra = checkpoint_config.get("extra_special_tokens")
    if isinstance(extra, list):
        shutil.copy2(a.checkpoint / "tokenizer_config.json", a.checkpoint / "tokenizer_config.transformers5.json")
        shutil.copy2(a.base / "tokenizer_config.json", a.checkpoint / "tokenizer_config.json")
        print(f"Restored tokenizer_config.json in {a.checkpoint}")
    else:
        print(f"Tokenizer config already compatible: {a.checkpoint}")


if __name__ == "__main__":
    main()
