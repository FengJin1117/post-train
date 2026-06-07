# Clean V2 Handoff

Last updated: 2026-06-07

This file is the canonical handoff for continuing the Clean V2 SFT-LoRA experiment.
Run commands from `/data2/fwh/post-train`.

## Objective

Improve MATH-500 over the Qwen2.5-3B-Instruct baseline without materially
degrading GSM8K. All E1/E2/E3 runs start from the original base model.

## Current State

- Data preparation: complete and validated
- Base-model fixed-subset gate: complete
- Three-GPU smoke training: complete
- E1 clean-10K, LR `1e-5`: complete
- E1 checkpoint gate: not run
- E2, E3, final full evaluation: not run
- Active tmux session: none

## Environments And Hardware

- Training/data/LoRA merge: `conda activate ms-swift`
- Benchmark evaluation: `conda activate vllm`
- Training GPUs: `CUDA_VISIBLE_DEVICES=4,6,7`, `NPROC_PER_NODE=3`
- Gate/final evaluation GPU: GPU `7`
- Base model:
  `/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct`

## Data

Source: `AI-MO/NuminaMath-1.5`, revision
`1b05109f9e5c1ad06c0663519502416c30b300f8`.

Generated files are under `data/` and are gitignored:

| Dataset | Size | Synthetic | Notes |
| --- | ---: | ---: | --- |
| `clean-10k.jsonl` | 10,000 | 30% | E1/E2 dataset |
| `clean-20k.jsonl` | 20,000 | 30% | Strict superset of clean-10K |
| `non-synthetic-10k.jsonl` | 10,000 | 0% | E3 rescue branch |
| `val.jsonl` | 1,000 | 0% | Independent validation |

The final `clean-10k` bucket/type quotas are exact:

- Buckets: competition 3,500; cn_k12 3,500; synthetic 3,000
- Types: Algebra 3,500; Geometry 2,000; Number Theory 2,000;
  Combinatorics 1,500; Other 1,000

Important validation:

- Full ms-swift template encode check: `10,000/10,000` usable
- clean-10K template token lengths: min `166`, max `1,850`, mean `602.78`
- clean-20K is a strict superset of clean-10K
- 25 source candidates were rejected by the final message-token filter
- Full details: `data/manifest.json`

Important implementation fix:

- `transformers.apply_chat_template()` returns a `BatchEncoding`, not always a
  plain `dict`. `prepare_dataset.py` must read `input_ids` through `Mapping`.
  Before this fix, all message lengths were incorrectly recorded as `2`.

## Training Configuration

Common parameters:

```text
LoRA rank=8, alpha=16, dropout=0.05
target_modules=q_proj k_proj v_proj o_proj
max_length=2048
epochs=1
per_device_train_batch_size=2
gradient_accumulation_steps=8
effective_batch_size=48
cosine scheduler, warmup_ratio=0.05
weight_decay=0.01, max_grad_norm=1.0
bf16, DeepSpeed ZeRO-2, gradient checkpointing
```

## Base Gate

Result file: `gate/results/baseline.json`

| Benchmark | Samples | Accuracy | Boxed | Max-token | Repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 200 | 84.00% | 100% | 0% | 0% |
| MATH-500 | 100 | 68.00% | 100% | 0% | 0% |

Checkpoint pass thresholds:

- GSM8K accuracy >= `81.00%`
- MATH-500 accuracy >= `66.00%`
- Both max-token rates < `1%`
- Both boxed extraction rates >= `98%`
- No empty/invalid responses

Evaluation implementation note:

- This vLLM build can leave an orphan engine process between sequential
  benchmarks. `run_gate.sh` and `run_final_eval.sh` launch each eval in its own
  process group and terminate that group after completion.
- Stable gate settings: `vllm_gpu_memory_utilization=0.75`,
  `vllm_max_num_seqs=16`, `vllm_max_model_len=4096`.

## Smoke Result

- Output: `output/smoke/e1/v0-20260606-161359/`
- World size: 3, GPUs 4/6/7
- Dataset retained: train 10,000; val 1,000
- Steps: 2/2
- Final train loss: `0.4207`
- LoRA trainable parameters: `3.6864M` (`0.1193%`)

## E1 Result

- Dataset: clean-10K
- Learning rate: `1e-5`
- Output: `output/e1/v4-20260606-161704/`
- Log: `logs/train-e1-full.log`
- Status: completed `209/209` steps, 1 epoch
- Runtime: `26m 11s`
- Final train loss: `0.38254666`
- Final validation loss: `0.45748961`
- Final validation token accuracy: `0.86361521`
- Best/last checkpoint according to validation loss: `checkpoint-209`
- No NaN or training instability observed

Formal checkpoints to gate:

```text
checkpoint-50
checkpoint-100
checkpoint-150
checkpoint-200
checkpoint-209
```

Validation loss by checkpoint:

| Step | Eval loss | Eval token accuracy |
| ---: | ---: | ---: |
| 50 | 0.478746 | 0.861386 |
| 100 | 0.463695 | 0.862907 |
| 150 | 0.458537 | 0.863549 |
| 200 | 0.457558 | 0.863705 |
| 209 | 0.457490 | 0.863615 |

## Exact Continuation

First, gate all E1 checkpoints. This is the immediate next action:

```bash
tmux new-session -d -s clean-v2-e1-gates \
  "cd /data2/fwh/post-train && \
   bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_all_gates.sh e1 \
   > exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/logs/e1-gates.log 2>&1"
```

Then run E2 from the original base model:

```bash
tmux new-session -d -s clean-v2-e2 \
  "cd /data2/fwh/post-train && \
   eval \"\$(conda shell.bash hook)\" && conda activate ms-swift && \
   bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_train.sh e2 full"
```

After E2 finishes:

```bash
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_all_gates.sh e2
```

`summarize_gates.py` writes `gate/decision.json`. Use it unchanged for E3:

```bash
tmux new-session -d -s clean-v2-e3 \
  "cd /data2/fwh/post-train && \
   eval \"\$(conda shell.bash hook)\" && conda activate ms-swift && \
   bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_train.sh e3 full"
```

Do not select checkpoints using validation loss alone. The benchmark gate is
the decision criterion.

