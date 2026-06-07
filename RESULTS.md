# Experiment Results

Last updated: 2026-06-06

## Main Results

All scores are accuracy percentages. `TBD` means that no valid full-benchmark result is available yet.

| Method     | GSM8K | MATH-500 | GAOKAO | AIME25 |
| ---------- | ----: | -------: | -----: | -----: |
| Base Model | 84.69 |    66.60 |   TBD  |   3.33 |
| LoRA-SFT   | 61.41 |    34.80 |   TBD  |    TBD |
| Full SFT   |   TBD |      TBD |   TBD  |    TBD |
| DPO        |   TBD |      TBD |   TBD  |    TBD |
| PPO        |   TBD |      TBD |   TBD  |    TBD |
| GRPO       |   TBD |      TBD |   TBD  |    TBD |
| DAPO       |   TBD |      TBD |   TBD  |    TBD |
| Agent      |   TBD |      TBD |   TBD  |    TBD |

## LoRA-SFT: NuminaMath-1.5 40K

### Training

- Base model: `Qwen2.5-3B-Instruct`
- Dataset: filtered and sampled `AI-MO/NuminaMath-1.5`
- Samples: 40,000 train + 1,000 validation before max-length filtering
- Effective samples: 39,910 train + 996 validation
- Method: LoRA-SFT, rank 16, alpha 32, all-linear
- Training: 2 epochs, 1,248 steps, max length 2,048
- Hardware: 4 x NVIDIA RTX A6000
- Final train loss: 0.5109
- Final validation loss: 0.5212
- Final validation token accuracy: 0.8460
- Runtime: 2h 53m 37s
- Checkpoint: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k/output/v2-20260605-140515/checkpoint-1248`
- Merged model: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k/output/v2-20260605-140515/checkpoint-1248-merged`

### Evaluation

Evaluation used the Native backend, vLLM inference, deterministic generation, and rule judge.

| Benchmark | Samples | Correct | Score | Base | Delta |
| --------- | ------: | ------: | ----: | ---: | ----: |
| GSM8K     |   1,319 |     810 | 61.41 | 84.69 | -23.28 |
| MATH-500  |     500 |     174 | 34.80 | 66.60 | -31.80 |
| AIME25    |      30 |     TBD |   TBD |  3.33 |    TBD |

MATH-500 breakdown:

| Difficulty | Samples | Score |
| ---------- | ------: | ----: |
| Level 1    |      43 | 65.12 |
| Level 2    |      90 | 57.78 |
| Level 3    |     105 | 43.81 |
| Level 4    |     128 | 28.12 |
| Level 5    |     134 |  8.96 |

### Findings

- The LoRA-SFT pipeline completed successfully, but this training recipe substantially reduced benchmark accuracy.
- Generated responses sometimes repeat until reaching the 8,192-token limit, which likely hurts answer extraction and accuracy.
- The current NuminaMath sampling mix and two-epoch SFT setup should not be used as the final recipe without further diagnosis.
- AIME25 does not yet have a valid result because the current evaluation attempt hit a CUDA out-of-memory error.
- A Clean V2 experiment is prepared with strict answer-format filtering, lower learning rates, attention-only LoRA, and checkpoint benchmark gates. Results are pending.

### Result Sources

- GSM8K: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k/eval/gsm8k.log`
- MATH-500: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k/eval/math_500.log`
- AIME25: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k/eval/aime25.log`
- Training metrics: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-40k/output/v2-20260605-140515/logging.jsonl`

## GRPO Countdown Example

- Status: completed the requested 3 training steps successfully
- Environment: `ms-swift-grpo`
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: GRPO with LoRA, Countdown accuracy reward, and format reward
- Hardware: GPU 0 and 1, NVIDIA RTX A6000
- Rollout backend: Transformers
- Runtime: 46.1685 seconds
- Final global step: `3/3`
- Final reward: 0.0
- W&B run: https://wandb.ai/fengwenhao-renmin-university-of-china/ms-swift-grpo-example/runs/ytfmbfag
- Training log: `exp/grpo/example/train.log`
- Environment record: `exp/grpo/example/environment.txt`

The CUDA 12.4 custom vLLM build can be imported, but its colocate APIs are incompatible with the current ms-swift main branch. The acceptance run therefore used Transformers rollout.

## LoRA-SFT Clean V2 Progress

- E1 clean-10K, LR `1e-5`: training complete, checkpoint benchmark gate pending
- E1 final training metrics: train loss `0.38255`, validation loss `0.45749`
- Fixed-subset base gate: GSM8K `84.00`, MATH-500 `68.00`
- Detailed state: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/HANDOFF.md`
