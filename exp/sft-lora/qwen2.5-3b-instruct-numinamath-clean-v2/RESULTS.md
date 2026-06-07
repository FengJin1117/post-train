# Clean V2 Results

Last updated: 2026-06-07

## Fixed-Subset Baseline Gate

| Model | GSM8K 200 | MATH-500 100 | Boxed rate | Max-token rate |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 84.00 | 68.00 | 100% | 0% |

## Training Runs

| Run | Dataset | LR | Steps | Train loss | Eval loss | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E1 | clean-10K | 1e-5 | 209 | 0.38255 | 0.45749 | Training and selected full-checkpoint evaluation complete |
| E2 | clean-10K | 2e-5 | TBD | TBD | TBD | Not started |
| E3 | Conditional | Conditional | TBD | TBD | TBD | Not started |

E1 formal checkpoints: `50`, `100`, `150`, `200`, `209`.

# Benchmark Results

Evaluation used Native EvalScope with vLLM, deterministic decoding,
`max_tokens=8192`, thinking disabled, and rule-based scoring. Scores are full
benchmark accuracy percentages.

| Model | GSM8K | Delta vs base | MATH-500 | Delta vs base |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 84.69 | - | 66.60 | - |
| checkpoint-50 | 70.74 | -13.95 | 43.00 | -23.60 |
| checkpoint-100 | 69.75 | -14.94 | **44.40** | -22.20 |
| checkpoint-150 | 69.75 | -14.94 | 42.80 | -23.80 |
| checkpoint-209 | 70.51 | -14.18 | 43.60 | -23.00 |

All evaluated checkpoints substantially degrade both benchmarks. The
degradation is already present at checkpoint-50 and does not recover with
additional training. Among the trained checkpoints, checkpoint-50 has the
highest GSM8K score and checkpoint-100 has the highest MATH-500 score.

## Generation Quality

| Model | Benchmark | Correct / Samples | Non-empty | Boxed | Max-token | Repetition |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint-50 | GSM8K | 933 / 1,319 | 100.00% | 96.66% | 2.81% | 2.65% |
| checkpoint-50 | MATH-500 | 215 / 500 | 100.00% | 91.40% | 8.20% | 8.60% |
| checkpoint-100 | GSM8K | 920 / 1,319 | 100.00% | 96.89% | 2.43% | 2.96% |
| checkpoint-100 | MATH-500 | 222 / 500 | 100.00% | 88.80% | 9.80% | 11.80% |
| checkpoint-150 | GSM8K | 920 / 1,319 | 100.00% | 96.66% | 2.65% | 2.43% |
| checkpoint-150 | MATH-500 | 214 / 500 | 100.00% | 89.60% | 9.60% | 9.40% |
| checkpoint-209 | GSM8K | 930 / 1,319 | 100.00% | 96.97% | 2.65% | 2.58% |
| checkpoint-209 | MATH-500 | 218 / 500 | 100.00% | 88.00% | 10.80% | 11.40% |

MATH-500 responses show a persistent formatting and generation-stability
regression: boxed-answer extraction falls below 92%, while 8.2%-10.8% of
responses reach the 8,192-token limit.

## MATH-500 Difficulty Breakdown

| Model | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 90.70 | 85.56 | 80.95 | 63.28 | 38.06 |
| checkpoint-50 | 74.42 | 66.67 | 47.62 | 33.59 | **22.39** |
| checkpoint-100 | 79.07 | **71.11** | 53.33 | 35.94 | 16.42 |
| checkpoint-150 | **81.40** | 65.56 | 50.48 | 35.16 | 16.42 |
| checkpoint-209 | 76.74 | 65.56 | **57.14** | **36.72** | 14.18 |

## Verification And Artifacts

- Prediction and review counts were verified independently: GSM8K `1,319`
  each and MATH-500 `500` each for every checkpoint.
- Accuracy was recomputed from review records and matches EvalScope reports.
- All responses are non-empty and no CUDA OOM occurred.
- Machine-readable summaries and raw EvalScope outputs are under
  `eval/e1-checkpoint-<step>/`.
- Parallel evaluation used GPUs `0,1,4,5,6,7`. An initial shared-port
  conflict was corrected; failed checkpoint-50/100 attempts were rerun with
  isolated service ports.
