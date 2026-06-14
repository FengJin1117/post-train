# Current Status

Date: 2026-06-06

## Goal

Compare the effects of the following post-training methods on mathematical reasoning:

- Full SFT
- LoRA-SFT
- DPO
- PPO
- GRPO
- DAPO
- Agent

## Current Baseline

Model: `Qwen2.5-3B-Instruct`

Benchmarks:

- GSM8K: 84.69
- MATH-500: 66.60
- GAOKAO: TBD
- AIME25: 3.33

## Finished

- Base model evaluation on GSM8K, MATH-500, and AIME25
- NuminaMath-1.5 dataset analysis
- Filtered NuminaMath-1.5 dataset preparation: 40,000 train + 1,000 validation
- LoRA-SFT training pipeline
- LoRA checkpoint merge
- LoRA-SFT evaluation on GSM8K: 61.41
- LoRA-SFT evaluation on MATH-500: 34.80
- Central result tracking in `RESULTS.md`

## In Progress

- LoRA-SFT AIME25 evaluation
- Analysis of LoRA-SFT benchmark regression and repetitive long generations
- Clean V2 LoRA-SFT pipeline: strict data filtering, three controlled runs, and checkpoint benchmark gates
- Clean V2 E1 checkpoint benchmark gate
- Qwen2.5-3B base Clean-10K baseline and dual-recipe LoRA-SFT experiment

## Blockers

- The latest AIME25 evaluation attempt hit CUDA OOM because the selected evaluation GPU was already heavily occupied.
- GAOKAO evaluation has not been configured or run.
- Full SFT, DPO, PPO, GRPO, DAPO, and Agent experiments have not started.

## Current Finding

The first LoRA-SFT recipe completed successfully but reduced accuracy compared with the base model:

| Benchmark | Base Model | LoRA-SFT | Delta |
| --------- | ---------: | -------: | ----: |
| GSM8K     |      84.69 |    61.41 | -23.28 |
| MATH-500  |      66.60 |    34.80 | -31.80 |

The next experiment should diagnose response repetition and data/format mismatch before scaling this recipe or using it as the basis for preference/RL training.

## Next Step

1. Gate the five Clean V2 E1 checkpoints.
2. Complete Qwen2.5-3B base baseline, then run B1/B2 dual-recipe LoRA-SFT.
3. Train and gate Instruct Clean V2 E2, then use the recorded decision for E3.
4. Run full GSM8K/MATH-500/AIME25 evaluation for winning checkpoints.

## Clean V2 Checkpoint Gate

- Clean datasets generated: 10K, 20K strict superset, non-synthetic 10K, and non-synthetic validation 1K
- ms-swift template length check: no rejected clean-10K records; max length 1,850
- Three-GPU smoke test: passed, 2/2 steps, loss finite, checkpoints saved
- Fixed-subset baseline: GSM8K 84.00, MATH-500 68.00
- Baseline stability: boxed extraction 100%, max-token rate 0%, repetition rate 0%
- E1 training complete: 209/209 steps, train loss 0.38255, eval loss 0.45749
- Detailed continuation state: `exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/HANDOFF.md`
