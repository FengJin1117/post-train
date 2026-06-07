# NuminaMath Clean V2 SFT-LoRA

目标：优先提升 MATH-500，同时通过固定 GSM8K/MATH-500 checkpoint 门禁避免能力退化。

## Current Status

- Clean datasets: generated and validated
- ms-swift exact length check: clean-10K has 10,000/10,000 usable records, max length 1,850
- Three-GPU smoke: passed
- Fixed-subset base gate: GSM8K 84.00, MATH-500 68.00
- E1: completed 209/209 steps; checkpoint gate pending
- Canonical continuation notes: `HANDOFF.md`
- Experiment-local result summary: `RESULTS.md`

## 环境与 GPU

- 数据准备、训练、LoRA 合并：`conda activate ms-swift`
- Benchmark 评测：`conda activate vllm`
- 训练 GPU：`4,6,7`
- 门禁与完整评测 GPU：默认 `7`

## 1. 准备数据

```bash
cd /data2/fwh/post-train
conda activate ms-swift
python exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/prepare_dataset.py \
  --output-dir exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/data
```

生成：

- `data/clean-10k.jsonl`
- `data/clean-20k.jsonl`
- `data/non-synthetic-10k.jsonl`
- `data/val.jsonl`
- `data/manifest.json`

## 2. Smoke 训练

```bash
conda activate ms-swift
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_train.sh e1 smoke
```

## 3. 基模门禁

门禁必须在训练停止后运行，避免 vLLM 与训练争抢 GPU。

```bash
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_gate.sh baseline
```

## 4. E1 / E2

```bash
conda activate ms-swift
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_train.sh e1 full

# 训练完成后
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_all_gates.sh e1

conda activate ms-swift
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_train.sh e2 full

# 训练完成后
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_all_gates.sh e2
```

`run_all_gates.sh` 会更新：

- `gate/summary.json`
- `gate/decision.json`

## 5. E3 条件训练

E3 严格读取 `gate/decision.json`：

- E1/E2 有 checkpoint 通过门禁：使用胜出 LR 训练 clean 20K。
- 均未通过：使用 LR `5e-6` 训练 non-synthetic 10K。

```bash
conda activate ms-swift
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_train.sh e3 full
```

## 6. 最终完整评测

```bash
bash exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/run_final_eval.sh \
  exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/output/e3/<run>/checkpoint-<step>
```

成功标准：

- MATH-500 >= 67.60
- GSM8K >= 82.69
- MATH-500 达到生成上限的比例 < 2%
