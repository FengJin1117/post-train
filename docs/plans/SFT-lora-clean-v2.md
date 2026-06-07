# Qwen2.5-3B-Instruct 干净数据 SFT-LoRA 三轮训练计划

## Summary

- 目标优先提升 **MATH-500**，同时将 GSM8K 能力保持作为硬门禁。
- 训练使用 GPU `4,6,7`，环境 `conda activate ms-swift`；checkpoint 评测使用 `conda activate vllm`。
- 开展三轮受控实验：
  1. 干净 10K，LR `1e-5`
  2. 相同 10K，LR `2e-5`
  3. 根据前两轮门禁结果，扩展到干净 20K，或执行更保守的非合成数据救援实验
- 所有训练从原始基模重新开始，不在上一轮 LoRA checkpoint 上继续训练。

## 数据方案

新建独立实验目录：

`exp/sft-lora/qwen2.5-3b-instruct-numinamath-clean-v2/`

从 NuminaMath-1.5 全量池重新采样，不复用旧 40K 数据。

### 强制过滤

- `problem_is_valid == "Yes"`
- `solution_is_valid == "Yes"`
- `question_type == "math-word-problem"`
- problem、solution、answer 非空
- 排除 `answer=proof/notfound` 和单字母 MCQ 答案
- solution 必须包含且仅包含一个 `\boxed{}`
- boxed 内容规范化后必须与 `answer` 一致
- 排除 URL、图片、花括号不平衡、异常重复文本
- assistant solution 长度限制为 `128–1536 tokens`
- 完整 messages token 长度不超过 `2048`
- 数据集内部按规范化 problem 去重
- 排除与 GSM8K、MATH-500 的精确重合及高相似样本：
  - 规范化文本完全相同
  - word 5-gram Jaccard 相似度 `>= 0.80`
  - 长文本包含比例 `>= 0.90`

### 10K 数据配比

| 数据桶 | 比例 | 数量 |
| --- | ---: | ---: |
| 非合成竞赛数学：aops_forum / metamath / amc_aime | 35% | 3,500 |
| 非合成 cn_k12 | 35% | 3,500 |
| orca_math / synthetic_math | 30% | 3,000 |

全局题型目标：

| 类型 | 比例 |
| --- | ---: |
| Algebra | 35% |
| Geometry | 20% |
| Number Theory | 20% |
| Combinatorics | 15% |
| Inequalities / Calculus / Logic / Other | 10% |

- synthetic 总比例不得超过 30%。
- 其中至少 1,500 条选择较短、稳定的应用题，用于保持 GSM8K 能力。
- 生成 1,000 条独立、非合成验证集。
- 20K 数据集必须是 10K 数据集的严格超集，并保持相同比例。
- 若某个交叉配额不足，依次从同题型非合成桶、同来源相邻题型、synthetic 桶补齐，并在 manifest 中记录实际偏差。

## 三轮训练方案

公共训练参数：

```text
model: Qwen2.5-3B-Instruct
CUDA_VISIBLE_DEVICES: 4,6,7
NPROC_PER_NODE: 3
tuner_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: q_proj k_proj v_proj o_proj
max_length: 2048
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
effective_batch_size: 48
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_grad_norm: 1.0
torch_dtype: bfloat16
deepspeed: zero2
save_total_limit: 10
```

### 实验 E1：保守基准

- 数据：clean 10K
- Learning rate：`1e-5`
- 保存 checkpoint：每 50 steps
- 目标：验证干净数据和低更新强度能否保持基模能力

### 实验 E2：学习率消融

- 数据：与 E1 完全相同的 clean 10K
- Learning rate：`2e-5`
- 其他参数与 E1 完全一致
- 目标：判断更高学习率能否提升 MATH-500，同时不造成退化

### 实验 E3：条件分支

若 E1 或 E2 至少一个通过门禁：

- 使用通过方案中 MATH 门禁分数最高的学习率
- 数据扩展到 clean 20K
- 从原始基模重新训练 1 epoch
- 每 100 steps 保存 checkpoint

若 E1、E2 均未通过门禁：

- 使用 10K 全非合成超干净数据
- Learning rate：`5e-6`
- 其余参数保持不变
- 用于判断 synthetic 数据或更新强度是否仍导致退化

选择并列方案时，依次比较：

1. MATH-500 门禁准确率
2. GSM8K 门禁准确率
3. 达到生成上限的比例
4. 输出 token 中位数

## Checkpoint Benchmark 门禁

训练时不并行启动 vLLM，避免与三卡训练争抢显存。每轮训练完成后，在 GPU `7` 上依次评测该轮全部 checkpoint。

### 基线门禁

正式训练前，使用相同门禁参数评测原始基模并保存基线：

- GSM8K：固定前 200 条，`max_tokens=1024`
- MATH-500：固定 100 条，覆盖五个难度层，`max_tokens=2048`
- temperature `0`
- Native backend + vLLM + rule judge

### Checkpoint 通过条件

- MATH-500 门禁分数不低于对应基线 `-2` 个百分点
- GSM8K 门禁分数不低于对应基线 `-3` 个百分点
- `stop_reason=max_tokens` 比例 `< 1%`
- boxed 最终答案可抽取率 `>= 98%`
- 无 NaN、空响应或明显重复循环

每轮保留门禁最优 checkpoint。中间 checkpoint 的 merged 模型在结果持久化后删除，LoRA adapter 保留。

### 最终完整评测

对三轮中门禁最优的 checkpoint 执行完整评测：

- GSM8K
- MATH-500
- AIME25
- GAOKAO 配置完成后补充

正式成功标准：

- MATH-500 `>= 67.60`，至少超过基线 1 个百分点
- GSM8K `>= 82.69`，相对基线下降不超过 2 个百分点
- MATH-500 达到 8192 token 上限的比例 `< 2%`
- 若没有方案达到标准，明确记录为“未观察到正向 SFT-LoRA 效果”，不继续盲目扩数据

## 实现与记录

新增以下能力：

- 干净数据准备脚本：生成 clean-10k、clean-20k、non-synthetic-10k 与详细 manifest
- 三轮训练启动脚本：固定 GPU `4,6,7` 和各轮参数
- checkpoint 门禁脚本：合并 checkpoint、运行固定子集评测、统计准确率和生成稳定性
- 汇总脚本：输出每轮 checkpoint 排名、门禁判定和最终推荐 checkpoint
- 更新实验 README、`RESULTS.md` 与 `PROJECT_STATE.md`

## Test Plan

1. 数据准备检查：
   - 数量、来源比例、题型比例、synthetic 比例正确
   - 无 proof、MCQ、URL、图片、异常括号、重复样本
   - boxed 终答与 answer 一致
   - 与 GSM8K/MATH-500 无精确或高相似重合
2. 三卡 smoke 训练：
   - GPU 为 `4,6,7`
   - world size 为 3
   - `max_steps=2`
   - LoRA 仅挂载 attention projection
   - loss 非 NaN，并成功保存 checkpoint
3. 基模门禁评测：
   - 固定子集可重复运行
   - 两次结果一致
   - 能统计准确率、输出长度、max-token 比例和 boxed 提取率
4. 每轮训练后：
   - 全部 checkpoint 完成门禁
   - 自动选出最优 checkpoint
   - 严格按门禁结果决定 E3 分支
5. 最终验证：
   - 完整 GSM8K/MATH-500 结果写入 `RESULTS.md`
   - 数据、训练参数、checkpoint 排名和失败原因可追溯

## Assumptions

- 优先目标为 MATH-500，GSM8K 用于检测能力遗忘。
- 训练期间三张 GPU 全部用于训练；checkpoint 评测在每轮训练结束后顺序执行。
- NuminaMath-1.5 仍作为唯一主要训练数据池，不额外引入未经分析的新数据集。
- 第一目标是获得不退化且可复现的正向趋势，而不是最大化单轮数据量。
