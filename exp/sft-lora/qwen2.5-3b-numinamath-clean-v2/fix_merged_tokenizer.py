#!/usr/bin/env python3
"""Make ms-swift exported tokenizer metadata readable by the vLLM environment."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    path = Path(args.model_dir) / "tokenizer_config.json"
    config = json.loads(path.read_text())
    extra = config.pop("extra_special_tokens", None)
    if extra and "additional_special_tokens" not in config:
        config["additional_special_tokens"] = extra
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(f"Patched tokenizer metadata: {path}")


if __name__ == "__main__":
    main()
