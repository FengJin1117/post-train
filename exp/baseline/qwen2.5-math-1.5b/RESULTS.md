# Qwen2.5-Math-1.5B Base Benchmark Results

Status: completed

## Evaluation Policy

- Model: `Qwen/Qwen2.5-Math-1.5B` (base, non-Instruct)
- Template: `qwen2_5_math`
- System prompt: `Please reason step by step, and put your final answer within \boxed{}.`
- Backend: Native EvalScope with vLLM
- Generation: deterministic, `temperature=0.0`, `do_sample=false`, `max_tokens=3072`
- Context length: native model limit, `vllm_max_model_len=4096`
- Scoring: rule judge only
- Evaluation date: `2026-06-13` UTC

## Results

| Benchmark | Samples | Correct | Accuracy (%) |
| --- | ---: | ---: | ---: |
| GSM8K | 1319 | 728 | 55.19 |
| MATH-500 | 500 | 115 | 23.00 |

## MATH-500 Difficulty Breakdown

| Difficulty | Samples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Level 1 | 43 | 6 | 13.95 |
| Level 2 | 90 | 21 | 23.33 |
| Level 3 | 105 | 33 | 31.43 |
| Level 4 | 128 | 28 | 21.88 |
| Level 5 | 134 | 27 | 20.15 |

## Output Diagnostics

| Benchmark | Non-empty (%) | Boxed (%) | Max-token (%) | Mean tokens | Median tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 100.00 | 97.95 | 28.05 | 1076.36 | 330.0 |
| MATH-500 | 100.00 | 95.60 | 53.20 | 1977.87 | 3072.0 |

## Environment

- MS-SWIFT commit: `eb35f330e5754d64fe34358716d80804a280a647`
- MS-SWIFT: `4.4.0.dev0`
- EvalScope: `1.8.0`
- vLLM: `0.1.dev5699+gde8f43fbe.cu118`
- Torch: `2.6.0`
- Transformers: `4.52.3`

## Command

```bash
cd /data2/fwh/post-train
tmux new-session -d -s qwen25-math-15b-baseline \
  "bash exp/baseline/qwen2.5-math-1.5b/run_baseline.sh all"
```

The runner uses GPUs `1` and `4` by default for parallel full evaluation. Raw
EvalScope artifacts are stored under `raw/full/`; logs are stored under `logs/`.
The originally planned 10,000-token service context and 8,192-token generation
limit were not used because this base model declares a 4,096-token context and
crashed vLLM with a CUDA device-side assertion when forced beyond that limit.

## Raw Artifacts

- GSM8K: `raw/full/gsm8k/eval_output/native/20260613_091753/`
- MATH-500: `raw/full/math_500/eval_output/native/20260613_091753/`
