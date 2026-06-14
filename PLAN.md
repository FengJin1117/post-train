以下结论基于截至 **2026 年 6 月 13 日**公开的论文、GitHub 仓库、训练脚本和数据集。

## 核心结论

在你的硬件与时间约束下，我建议采用：

| 实验角色 | 推荐方案 |
|---|---|
| Base Model | `Qwen2.5-Math-1.5B` |
| Full SFT | `s1K` 或 LIMO 数据，Transformers/TRL 全参训练 |
| LoRA-SFT | 与 Full SFT 使用完全相同数据和超参数，仅切换 LoRA |
| DPO | Step-DPO-10K |
| GRPO | SimpleRL-Zoo / Open-R1 recipe |
| 主要评测 | GSM8K、MATH-500、AIME24、AIME25 |
| 推荐硬件 | 4×A6000 48GB |
| 推荐统一数据预算 | 8K–10K prompts/examples |
| 推荐统一 token 预算 | 每种方法限制相同训练 token 数 |

不建议把 Full DeepScaleR 或完整 TinyZero 作为主实验：它们的原始配置明显超出 4×A6000、24 小时预算。

---

# 推荐项目

## 1. s1：最适合 SFT 基线

### 基本信息

- **论文标题**：[s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)
- **年份**：2025
- **机构**：Stanford、University of Washington、Allen Institute for AI 等
- **任务**：使用少量高质量长推理数据训练数学推理模型
- **GitHub**：[simplescaling/s1](https://github.com/simplescaling/s1)

### 模型

- 原项目主要模型：Qwen2.5-32B-Instruct
- 数据和训练方法可迁移至：Qwen2.5-1.5B、3B、7B
- **是否适合 7B 以下**：是，尤其适合作为小数据 SFT 消融实验

### 数据

- 训练数据：[s1K](https://huggingface.co/datasets/simplescaling/s1K)
- 样本量：约 1,000
- 是否公开：是
- 特点：经过难度、质量与多样性筛选的推理轨迹

### Benchmark

| Benchmark | 覆盖情况 |
|---|---|
| GSM8K | 可使用公开评测框架补充 |
| MATH-500 | 是 |
| AIME24 | 是 |
| AIME25 | 原论文未覆盖，可补充 |

### 代码

- 训练框架：Transformers、DeepSpeed
- 提供训练与评测脚本
- 可迁移至：TRL、ms-swift
- **是否可直接复现**：较高，但原始 32B 配置不适合你的预算

### 成本评估

| 配置 | 估计成本 |
|---|---|
| 1.5B Full SFT | 1×A6000，约 1–3 小时 |
| 3B Full SFT | 1–2×A6000，约 2–6 小时 |
| 7B Full SFT | 2–4×A6000，约 6–15 小时 |
| LoRA-SFT | 1×A6000，通常低于 4 小时 |

- **适合作为个人项目**：非常适合
- **复现难度**：★★
- 原因：数据极小、代码公开、训练逻辑简单；主要风险是小模型不一定复现论文中 32B 模型的提升幅度。

---

## 2. LIMO：最适合小数据 SFT 对照

### 基本信息

- **论文标题**：[LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)
- **年份**：2025
- **机构**：GAIR-NLP 团队
- **任务**：使用极少量复杂数学推理样本进行监督微调
- **GitHub**：[GAIR-NLP/LIMO](https://github.com/GAIR-NLP/LIMO)

### 模型

- 原始模型：Qwen2.5-Math-7B
- 参数量：7B
- **是否适合 7B 以下**：是，但 1.5B/3B 的能力增益可能弱于 7B

### 数据

- 数据集：[GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO)
- 样本量：817
- 是否公开：是
- 特点：高难度、高质量长推理轨迹

### Benchmark

| Benchmark | 覆盖情况 |
|---|---|
| GSM8K | 可补充 |
| MATH-500 | 是 |
| AIME24 | 是 |
| AIME25 | 可补充 |

### 代码与成本

- 训练框架：Transformers、DeepSpeed
- 是否可直接复现：是
- 7B Full SFT：建议 4×A6000，约 6–15 小时
- 1.5B/3B LoRA：1×A6000，约 1–4 小时
- **适合作为个人项目**：非常适合
- **复现难度**：★★

LIMO 和 s1 非常适合共同构成“小数据质量是否比数量重要”的消融实验。

---

## 3. Step-DPO：最适合 DPO 基线

### 基本信息

- **论文标题**：[Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs](https://arxiv.org/abs/2406.18629)
- **年份**：2024
- **机构**：CUHK、Shanghai AI Laboratory 等
- **任务**：针对数学推理中的错误步骤构造偏好对并进行 DPO
- **GitHub**：[dvlab-research/Step-DPO](https://github.com/dvlab-research/Step-DPO)

### 模型

- 原项目使用较大 Qwen 模型
- 方法可迁移至 Qwen2.5-Math-1.5B/3B/7B
- **是否适合 7B 以下**：是

### 数据

- 数据集：[Step-DPO-10K](https://huggingface.co/datasets/Riema/step-dpo)
- 样本量：约 10K preference pairs
- 是否公开：是
- 数据结构：问题、前置正确步骤、chosen step、rejected step

### Benchmark

| Benchmark | 覆盖情况 |
|---|---|
| GSM8K | 是 |
| MATH-500 | 是 |
| AIME24 | 需要自行补充 |
| AIME25 | 需要自行补充 |

### 代码与成本

- 训练框架：Transformers、TRL 风格 DPO
- 可迁移至：ms-swift DPO
- 是否可直接复现：基本可以，但需要适配 Qwen2.5 chat template

| 配置 | 估计成本 |
|---|---|
| 1.5B DPO | 1–2×A6000，约 3–8 小时 |
| 3B DPO | 2×A6000，约 5–12 小时 |
| 7B DPO | 4×A6000，约 12–24 小时 |

- **适合作为个人项目**：适合
- **复现难度**：★★★
- 原因：DPO 同时涉及 reference model、偏好数据格式与序列长度，显存占用高于 SFT。

---

## 4. SimpleRL-Zoo：最适合主 GRPO 基线

### 基本信息

- **项目名称**：SimpleRL-Zoo
- **年份**：2025
- **机构**：HKUST-NLP 团队
- **任务**：使用规则奖励进行数学 RLVR/GRPO 训练
- **GitHub**：[hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)

### 模型

- 支持 Qwen2.5-Math-1.5B、7B 等
- **是否适合 7B 以下**：非常适合
- 1.5B 是最推荐配置

### 数据

- 公开数学 prompt 数据
- 推荐使用项目中的 8K 级训练集配置
- 奖励：基于数学答案正确性与格式的 rule-based reward
- 是否公开：是

### Benchmark

| Benchmark | 覆盖情况 |
|---|---|
| GSM8K | 是 |
| MATH-500 | 是 |
| AIME24 | 是 |
| AIME25 | 可补充 |

### 代码与成本

- 训练框架：veRL
- 算法：PPO/GRPO 风格 RLVR
- 是否可直接复现：较高

| 配置 | 估计成本 |
|---|---|
| 1.5B、2K–4K rollout | 4×A6000，约 8–24 小时 |
| 3B | 4×A6000，可能超过 24 小时 |
| 7B | 不适合作为你的主配置 |

- **适合作为个人项目**：适合，但需要熟悉 Ray、vLLM、veRL
- **复现难度**：★★★★

---

## 5. Open-R1：最适合工程实现与社区复现

### 基本信息

- **项目名称**：Open-R1
- **年份**：2025
- **机构**：Hugging Face 社区
- **任务**：开放复现 DeepSeek-R1 风格 SFT、GRPO 和蒸馏流程
- **GitHub**：[huggingface/open-r1](https://github.com/huggingface/open-r1)

### 模型与数据

- 支持 Qwen2.5、Qwen2.5-Math 等模型
- 提供数学 reasoning 数据处理、SFT、GRPO、评测流程
- 可直接使用 TRL GRPO
- 支持公开数据集与自定义数学 prompt 数据

### Benchmark

- 与 LightEval 集成
- 可评测 MATH-500、AIME24、AIME25 等
- GSM8K 可以补充配置

### 成本与难度

- 1.5B LoRA-SFT：1×A6000
- 1.5B GRPO：建议 4×A6000
- 是否适合作为个人项目：非常适合
- **复现难度**：★★★
- 原因：文档与社区支持好，但 Open-R1 更像完整训练工具链，不是一篇单一算法论文。

---

## 6. DeepScaleR：适合作为参考上限，不适合作为主复现

### 基本信息

- **项目名称**：DeepScaleR
- **年份**：2025
- **机构**：Agentica、Berkeley Sky Computing Lab
- **任务**：通过分阶段长上下文 RL 提升数学推理能力
- **GitHub**：[agentica-project/deepscaler](https://github.com/agentica-project/deepscaler)

### 模型与数据

- 模型：DeepSeek-R1-Distill-Qwen-1.5B
- 训练数据：约 40K 数学问题
- 数据、代码、模型公开
- 使用 veRL

### Benchmark

| Benchmark | 覆盖情况 |
|---|---|
| GSM8K | 非核心 |
| MATH-500 | 是 |
| AIME24 | 是 |
| AIME25 | 是 |

### 成本与难度

- 原始 16K/24K 上下文阶段使用约 32×A100-80GB
- 无法在你的统一预算下完整复现
- 可以缩减为 2K–4K rollout 的 mini-DeepScaleR
- **复现难度**：★★★★★
- **个人项目建议**：只作为参考与扩展实验，不作为主表格基线

---

# LoRA-SFT 的正确处理方式

目前没有一个同时满足以下条件、并被广泛认可为“数学 LoRA-SFT 标准项目”的独立项目：

- 近两年代表性论文
- 小模型
- 小数据
- LoRA 专属方法创新
- 完整公开脚本
- 社区广泛复现

这是因为 **LoRA-SFT 是参数更新方式，而不是独立的数学推理训练目标**。

最公平的实验是：

```text
Full SFT:
Qwen2.5-Math-1.5B + s1K/LIMO + full parameters

LoRA-SFT:
Qwen2.5-Math-1.5B + 完全相同数据 + 完全相同 token budget
LoRA rank = 32 或 64
target_modules = q/k/v/o/up/down/gate_proj
```

这样 Full SFT 与 LoRA-SFT 的差异才能归因于参数高效微调，而不是数据差异。

---

# 最终排序

| Rank | Project | Algorithm | Cost | Reproducibility | Recommendation |
|---:|---|---|---|---|---|
| 1 | s1 / s1K | Full SFT、LoRA-SFT | 低 | 很高 | 最适合一周内完成 |
| 2 | LIMO | Full SFT、LoRA-SFT | 低 | 很高 | 最适合小数据消融 |
| 3 | Step-DPO | DPO | 中 | 高 | 最适合 DPO 基线 |
| 4 | SimpleRL-Zoo | GRPO/RLVR | 中高 | 高 | 最适合主 GRPO 基线 |
| 5 | Open-R1 | SFT、GRPO | 中高 | 很高 | 最适合工程实现与简历展示 |
| 6 | DeepScaleR | GRPO/RLVR | 极高 | 中 | 仅建议缩小复现 |

---

# 五项重点推荐

1. **最适合作为 SFT 基线**：`s1K + Qwen2.5-3B Full SFT`
2. **最适合作为 DPO 基线**：`Step-DPO-10K`
3. **最适合作为 GRPO 基线**：`SimpleRL-Zoo 1.5B recipe`
4. **最适合一周内完成复现**：`s1K Full SFT + LoRA-SFT`
5. **最适合作为简历项目**：`Full SFT vs LoRA-SFT vs Step-DPO vs GRPO` 完整统一预算对比

## 推荐最终实验配置

| Method | Model | Data Budget | 预计硬件 |
|---|---|---:|---|
| Base | Qwen2.5-3B | 0 | 1×A6000 评测 |
| Full SFT | s1K/LIMO 或统一 8K trajectories | 统一 token 数 | 1×A6000 |
| LoRA-SFT | 与 Full SFT 完全相同 | 统一 token 数 | 1×A6000 |
| DPO | Step-DPO-10K 或统一构造 preference pairs | 8K–10K | 1–2×A6000 |
| GRPO | SimpleRL-Zoo/Open-R1 prompts | 8K–10K | 4×A6000 |

为了保证实验可信度，还应报告：

- 训练总 tokens
- 总 GPU-hours
- 最大上下文长度
- 平均生成长度
- AIME 的 `pass@1` 与多次采样平均值
- 训练数据与 benchmark 的污染检查
- 至少 3 个随机种子或置信区间

这套设计能够在控制成本的同时，形成相当完整且有说服力的数学推理后训练对比项目。