# Qwen2.5-3B Base + s1K Full SFT

## 结论

本实验使用完整 s1K 对 `Qwen/Qwen2.5-3B` base 进行 Full SFT，并按预先固定的
GSM8K/MATH-500 gate 选择模型。最终选中 `r0-original`（`1e-5`、5 epochs）。

在标准 deterministic pass@1 评测下，`r0-original` 将 GSM8K 从 **52.01%**
提升至 **56.63%**（+4.62 pp），但 MATH-500 从 **59.40%** 降至
**29.20%**（-30.20 pp），AIME24/25 均为 0。模型在较难问题上经常持续生成直到
8192-token 上限，因此本轮 Full SFT 没有复刻出稳定的整体数学能力提升。

## 全量评测

| Model | GSM8K (1319) | MATH-500 (500) | AIME24 (30) | AIME25 (30) |
| --- | ---: | ---: | ---: | ---: |
| Base | 686 / 1319 (**52.01%**) | 297 / 500 (**59.40%**) | 0 / 30 (**0.00%**) | 1 / 30 (**3.33%**) |
| `r0-original`（最终选中） | 747 / 1319 (**56.63%**) | 146 / 500 (**29.20%**) | 0 / 30 (**0.00%**) | 0 / 30 (**0.00%**) |
| `r1-conservative` | 737 / 1319 (**55.88%**) | 130 / 500 (**26.00%**) | 0 / 30 (**0.00%**) | 1 / 30 (**3.33%**) |

相对 Base，`r0-original` 的四项变化分别为 **+4.62、-30.20、+0.00、-3.33 pp**。
所有全量评测的 prediction/review 数量均与预期一致，所有输出均非空。

### MATH-500 难度分层

| Model | Level 1 (43) | Level 2 (90) | Level 3 (105) | Level 4 (128) | Level 5 (134) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 37 (**86.05%**) | 75 (**83.33%**) | 70 (**66.67%**) | 67 (**52.34%**) | 48 (**35.82%**) |
| `r0-original` | 25 (**58.14%**) | 41 (**45.56%**) | 41 (**39.05%**) | 27 (**21.09%**) | 12 (**8.96%**) |
| `r1-conservative` | 27 (**62.79%**) | 38 (**42.22%**) | 34 (**32.38%**) | 23 (**17.97%**) | 8 (**5.97%**) |

## Gate 与模型选择

Gate 固定使用 GSM8K 前 200 条（`max_tokens=1024`）和 MATH-500 每级 20 条、
共 100 条（`max_tokens=2048`）。选择规则为两项准确率算术平均最高；平分时选择
平均 max-token rate 更低者，再平分选择 `r0-original`。

| Run | GSM8K | MATH-500 | Gate mean | Mean max-token rate | 选择 |
| --- | ---: | ---: | ---: | ---: | --- |
| `r0-original` | 109 / 200 (54.50%) | 47 / 100 (47.00%) | **50.75%** | 36.50% | **最终模型** |
| `r1-conservative` | 118 / 200 (59.00%) | 36 / 100 (36.00%) | 47.50% | 45.00% | 未选中 |

完整 gate 指标见
[`gate-r0-original.json`](gate-r0-original.json)、
[`gate-r1-conservative.json`](gate-r1-conservative.json) 和
[`selection.json`](selection.json)。

## 输出质量诊断

表中为 `boxed rate / max-token rate / 平均输出 tokens / 中位输出 tokens`。

| Model | GSM8K | MATH-500 | AIME24 | AIME25 |
| --- | --- | --- | --- | --- |
| Base | 96.82% / 26.54% / 2705 / 408 | 92.60% / 2.00% / 683 / 439 | 83.33% / 16.67% / 2032 / 812 | 83.33% / 10.00% / 1660 / 882 |
| `r0-original` | 82.49% / 19.03% / 2286 / 703 | 46.80% / 48.40% / 4939 / 7617 | 6.67% / 86.67% / 8044 / 8192 | 6.67% / 80.00% / 7949 / 8192 |
| `r1-conservative` | 76.80% / 27.45% / 2812 / 550 | 37.80% / 66.60% / 6115 / 8192 | 0.00% / 93.33% / 8140 / 8192 | 0.00% / 93.33% / 8114 / 8192 |

主要失败模式不是空输出，而是 SFT 后模型在较难题上无法停止或无法形成最终
`\boxed{}` 答案。该问题在 MATH-500 Level 4/5 与 AIME 上最明显。

## 数据与训练

- 数据：`simplescaling/s1K`，固定 revision
  `278d72baaa2b887a7e76a70a0ae254a5a45536e4`。
- 1,000 条全部成功转换；与官方 `s1K_tokenized` 渲染结果逐条精确匹配
  `1000/1000`；math/science/crossword 分别为 `877/108/15`。
- token 长度：min `712`，median `5898`，mean `5658.48`，max `9109`；
  无样本超过训练 `max_length=32768`。
- 模型：`Qwen/Qwen2.5-3B` base；模板 `qwen2_5`；Full SFT、bf16、
  assistant-only loss、无 packing、无验证集。
- 训练：4×RTX A6000、DeepSpeed ZeRO-3、SDPA、gradient checkpointing；
  每卡 batch 1、gradient accumulation 2、有效 batch 8。

| Run | LR | Epochs / steps | Train loss | Runtime | Peak memory/GPU | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `r0-original` | `1e-5` | 5 / 625 | 0.2238 | 1h 20m 44s | 35.48 GiB | 完成，无 OOM/NaN |
| `r1-conservative` | `5e-6` | 3 / 375 | 0.3288 | 48m 38s | 35.13 GiB | 完成，无 OOM/NaN |

数据统计见 [`data-report.json`](data-report.json)，复刻来源与格式细节见
[`references/s1K/REPRODUCTION_NOTES.md`](../../../references/s1K/REPRODUCTION_NOTES.md)。

## 评测口径

- 执行日期：2026-06-13 至 2026-06-14。
- Native EvalScope + vLLM + rule judge；评测在 conda `vllm` 环境执行。
- 模板 `qwen2_5`；system prompt：`You are a helpful assistant.`
- deterministic decoding：`temperature=0.0`、`do_sample=false`。
- 全量评测：`max_tokens=8192`、`vllm_max_model_len=10000`、
  tensor parallel size 1。
- GSM8K 使用 EvalScope 默认 4-shot；MATH-500、AIME24、AIME25 为 0-shot。
- 环境：评测 `swift 4.4.0.dev0`、`vllm 0.1.dev5699+gde8f43fbe`、
  `evalscope 1.8.0`、`transformers 4.52.3`、`torch 2.6.0+cu124`；
  训练 `swift 4.1.0.dev0`、`transformers 5.3.0`、`deepspeed 0.18.9`。

评测命令入口为 [`run_eval.sh`](run_eval.sh)，训练命令入口为
[`run_train.sh`](run_train.sh)。汇总原始 JSON 为
[`full-base.json`](full-base.json)、
[`full-r0-original.json`](full-r0-original.json) 和
[`full-r1-conservative.json`](full-r1-conservative.json)；EvalScope prediction、
review、report 与 task config 保存在 `eval/full/<model>/<benchmark>/`，运行日志保存在
`logs/eval-full-<model>-<benchmark>.log`。

## 污染说明

s1K 中有 287 条来自 `qq8933/AIME_1983_2024`，因此 AIME24 必须视为训练污染结果，
不能作为无污染泛化证据。本次 13-gram 扫描还命中 MATH-500 `4/500`、AIME24
`1/30`、AIME25 `5/30`、GSM8K `0/1319`；AIME25 命中多为通用答案措辞，可能存在
误报。完整命中片段见 [`contamination-report.json`](contamination-report.json)。

本轮不实现 budget forcing，结论仅针对标准 deterministic pass@1 Full SFT。
