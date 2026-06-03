# Qwen3-4B-Instruct-2507 Baseline

Status: completed

## Evaluation Policy

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Local model path: `/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507`
- Benchmark: `gsm8k`
- Backend: Native EvalScope with vLLM
- Prompting: EvalScope default GSM8K 4-shot chain-of-thought prompt
- Generation: deterministic decoding, `max_tokens=8192`, thinking disabled (`qwen3_nothinking`)
- Scoring: rule-based scoring only; no LLM judge
- GPU: `0` (`NVIDIA RTX A6000`)
- Evaluation date: `2026-06-02` UTC

## Results

| Benchmark | Samples | Accuracy | Runtime |
| --- | ---: | ---: | ---: |
| GSM8K | 1319 | 0.9409 (1241/1319) | 1140.44s |

The EvalScope benchmark phase took `1140.44s` (`19m 00s`). The full `swift eval` process, including vLLM startup and
shutdown, ran from `2026-06-02T12:08:38Z` to `2026-06-02T12:29:12Z`.

## Environment

- MS-SWIFT commit: `eb35f330e5754d64fe34358716d80804a280a647`
- MS-SWIFT version: `4.4.0.dev0`
- EvalScope: `1.8.0`
- vLLM: `0.1.dev5699+gde8f43fbe`
- Torch: `2.6.0+cu124`
- Transformers: `4.52.3`
- Datasets: `4.8.4`
- ModelScope: `1.37.0`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 swift eval \
  --model /data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
  --enable_thinking false \
  --eval_dataset gsm8k \
  --eval_backend Native \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 10000 \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --extra_eval_args '{"judge_strategy":"rule"}' \
  --eval_num_proc 8
```

## Verification

- Smoke test: `2/2` correct.
- Full prediction and review files: `1319` records each.
- Output sanity check: `1319/1319` outputs are non-empty and include a `\boxed{}` answer.
- One output reached the `8192` token generation limit; the rule scorer still completed all samples.

## Raw Artifacts

Raw logs, predictions, and EvalScope intermediate files are stored locally under `raw/` and intentionally excluded
from Git. The full EvalScope output is under:

```text
raw/full/gsm8k/eval_output/native/20260602_121011/
```

At report time, `/data2` had no generally available disk space remaining. This baseline added approximately `20 MiB`
under `raw/`; the full-disk host state was not caused by the experiment artifacts.
