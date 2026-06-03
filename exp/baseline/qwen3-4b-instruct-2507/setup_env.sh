#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-vllm}"
CONSTRAINTS="$ROOT_DIR/exp/baseline/qwen3-4b-instruct-2507/constraints.txt"

cd "$ROOT_DIR"

conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  --constraint "$CONSTRAINTS" \
  evalscope==1.8.0

conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  --no-deps \
  --editable ./ms-swift

conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  --no-deps \
  attrdict \
  binpacking \
  cpm_kernels \
  dacite \
  gradio==5.50.0 \
  json_repair \
  matplotlib \
  peft==0.19.1 \
  rouge \
  tensorboard \
  transformers_stream_generator \
  trl==0.19.1 \
  zstandard

conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import pathlib

import evalscope
import modelscope
import swift
import torch
import transformers
import vllm

expected_swift = pathlib.Path("ms-swift/swift").resolve()
actual_swift = pathlib.Path(swift.__file__).resolve().parent
assert actual_swift == expected_swift, (actual_swift, expected_swift)
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert transformers.__version__ == "4.52.3", transformers.__version__

print("swift", swift.__version__, swift.__file__)
print("evalscope", evalscope.__version__)
print("modelscope", modelscope.__version__)
print("vllm", vllm.__version__, vllm.__file__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
