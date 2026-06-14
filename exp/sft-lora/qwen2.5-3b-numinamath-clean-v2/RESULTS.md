# Qwen2.5-3B Base Clean V2 Results

| Run | Dataset | LR | LoRA | Epochs | GSM8K | MATH-500 | Status |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Raw base | - | - | - | - | 50.72 | 57.60 | Baseline complete |
| B1 best gate: checkpoint-209 | clean-10K | 1e-5 | rank 8, attention | 1 | TBD | TBD | Full eval running |
| B2 best gate: checkpoint-200 | clean-10K | 5e-5 | rank 16, all-linear | 2 | 35.48 | 27.2 | Full eval running |

Fixed checkpoint-gate baseline: GSM8K `48.50` (200 examples), MATH-500
`71.00` (100 stratified examples).

## Checkpoint gate

No checkpoint passed the gate.

| Run | Best checkpoint | GSM8K-200 | MATH-500-100 | GSM max-token | MATH max-token |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | checkpoint-209 | 14.00 | 39.00 | 89.50 | 32.00 |
| B2 | checkpoint-200 | 32.50 | 43.00 | 98.50 | 99.00 |

B2 learned more task behavior than B1, but neither recipe beat the raw base
on the fixed subsets. Both recipes have severe output termination problems.
See `gate/summary.json` for all 14 checkpoint results.
