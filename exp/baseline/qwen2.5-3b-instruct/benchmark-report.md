# Qwen2.5-3B-Instruct Baseline：GSM8K

Status: completed

## Evaluation Policy

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Local model path: `/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct`
- Benchmark: `gsm8k`
- Backend: Native EvalScope with vLLM
- Prompting: EvalScope default GSM8K 4-shot chain-of-thought prompt
- Generation: deterministic decoding, `max_tokens=8192`, thinking disabled
- Scoring: rule-based scoring only; no LLM judge
- GPU: `0` (`NVIDIA RTX A6000`)
- Evaluation date: `2026-06-02` UTC

## Results

| Benchmark | Samples | Accuracy | Runtime |
| --- | ---: | ---: | ---: |
| GSM8K | 1319 | 0.8469 (1117/1319) | 824.08s |

The EvalScope benchmark phase took `824.08s` (`13m 44s`). The full `swift eval` process, including vLLM startup and
shutdown, ran from `2026-06-02T17:15:13Z` to `2026-06-02T17:30:27Z`.

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
  --model /data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct \
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
- Output sanity check: `1319/1319` outputs are non-empty.
- `1318/1319` outputs include a `\boxed{}` answer.
- Sample `index=153` entered a repetitive generation loop, reached the `8192` token limit, and was scored incorrect.

## Raw Artifacts

Raw logs, predictions, and EvalScope intermediate files are stored locally under `raw/` and intentionally excluded
from Git. The full EvalScope output is under:

```text
raw/full/gsm8k/eval_output/native/20260602_171641/
```


## AIME 25 Baseline

Status: completed

### Evaluation Policy

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Local model path: `/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct`
- Benchmark: `aime25` (`AIME-2025`)
- Backend: Native EvalScope with vLLM
- Prompting: EvalScope default AIME25 0-shot math prompt
- Generation: deterministic decoding, `max_tokens=8192`, thinking disabled
- Scoring: rule-based scoring only; no LLM judge
- GPU: `0` (`NVIDIA RTX A6000`)
- Evaluation date: `2026-06-03` UTC

### Results

| Benchmark | Samples | Accuracy | Runtime |
| --- | ---: | ---: | ---: |
| AIME 25 | 30 | 0.0333 (1/30) | 150.40s |

EvalScope reported average latency `20.508278s`, average output throughput `70.06 tok/s`, and average output length
`1436.866667` tokens.

### Environment

- MS-SWIFT commit: `eb35f330e5754d64fe34358716d80804a280a647`
- MS-SWIFT version: `4.4.0.dev0`
- EvalScope: `1.8.0`
- vLLM: `0.1.dev5699+gde8f43fbe.cu118`
- Torch: `2.6.0+cu124`
- Transformers: `4.52.3`
- Datasets: `4.8.4`
- ModelScope: `1.37.0`

### Command

The evaluation command is also recorded in `math-reasoning/README.md`.

```bash
CUDA_VISIBLE_DEVICES=0 swift eval \
  --model /data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct \
  --enable_thinking false \
  --eval_dataset aime25 \
  --eval_backend Native \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 10000 \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --extra_eval_args '{"judge_strategy":"rule"}' \
  --eval_num_proc 8
```

### Verification

- Full prediction and review files: `30` records each.
- Rule-scored correct samples: `1/30`; the correct sample index is `5`.
- Output sanity check: `30/30` predictions are non-empty and have extractable answers; `28/30` include a `\boxed{}` answer.
- Two outputs reached the `8192` token generation limit at sample indices `15` and `18`; EvalScope still completed rule scoring.

### Raw Artifacts

Raw logs, predictions, reviews, and EvalScope reports are stored locally under `raw/` and intentionally excluded from
Git. The full EvalScope output is under:

```text
raw/aime25-qwen2.5-3b-instruct/full/eval_output/native/20260603_041248/
```

The top-level EvalScope result file is under:

```text
raw/aime25-qwen2.5-3b-instruct/full/result/Qwen2.5-3B-Instruct/eval_result.jsonl
```

