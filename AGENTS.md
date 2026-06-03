# MS-SWIFT Math Post-training Notes

## Scope

This repository exposes the `swift` CLI for training, inference, sampling, deployment, and evaluation. For the math
reasoning study, use Qwen2.5/Qwen2.5-Math models and keep experiment outputs under `output/` or `eval_output/`
(both are gitignored).

## conda environment

vllm 评测场景（benchmark），使用环境：
`conda activate vllm`


## Relevant Paths

- `examples/train/lora_sft.sh`, `examples/train/full/train.sh`: LoRA-SFT and full SFT starting points.
- `examples/train/rlhf/dpo/`, `examples/train/rlhf/ppo/`: DPO and PPO scripts.
- `examples/train/grpo/`: GRPO scripts; start with `internal/transformers.sh` or `plugin/gsm8k/gsm8k.sh`.
- `docs/source/Instruction/GRPO/AdvancedResearch/DAPO.md`: DAPO flags implemented on top of GRPO.
- `examples/train/multi-gpu/deepspeed/`: DeepSpeed ZeRO-2 and ZeRO-3 launch examples.
- `examples/eval/`, `docs/source/Instruction/Evaluation.md`: EvalScope-backed benchmark evaluation.
- `swift/dataset/dataset/llm.py`, `swift/dataset/data/dataset_info.json`: built-in dataset registrations.
- `swift/model/models/qwen.py`: Qwen and Qwen2.5-Math model registrations.
- `swift/rewards/orm.py`: built-in GRPO math rewards.

## Core Commands

```bash
# SFT: choose --tuner_type full or lora
swift sft --model Qwen/Qwen2.5-Math-7B-Instruct --dataset <dataset> --tuner_type lora --output_dir output

# DPO or PPO
swift rlhf --rlhf_type dpo --model <model> --dataset <preference-dataset> --tuner_type lora --output_dir output

# GRPO; DAPO adds the flags documented in docs/source/Instruction/GRPO/AdvancedResearch/DAPO.md
swift rlhf --rlhf_type grpo --model <model> --dataset <prompt-dataset> \
  --reward_funcs accuracy format --tuner_type lora --output_dir output

# Benchmark evaluation
swift eval --model <model-or-checkpoint> --eval_backend Native --eval_dataset gsm8k
```

Use `CUDA_VISIBLE_DEVICES=...` to select GPUs. Set `NPROC_PER_NODE=<gpu-count>` for distributed `sft` and `rlhf`.
For LoRA checkpoints, pass `--adapters <checkpoint>` during inference/evaluation when appropriate.

## Dataset Shapes

- SFT: `messages` ending with an assistant answer.
- DPO: SFT-shaped `messages` plus `rejected_response`.
- PPO/GRPO: prompt-only `messages`; GRPO math rewards commonly also need a `solution` column.
- Custom local JSON/JSONL/CSV files can be passed directly with `--dataset <path>`; use `--columns` for field mapping.

