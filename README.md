# 基于后训练的 LLM 数学推理能力增强

## 项目简介

针对大语言模型在数学推理任务中存在多步推理不稳定、答案生成易偏差等问题，本项目基于 MS-Swift 框架，以 Qwen 系列模型为基座，系统复现并对比 SFT、LoRA-SFT、DPO 与 GRPO 等主流后训练方法，分析不同算法对数学推理能力的增强效果。

## 实验结果

在统一训练预算下，基于 GSM8K、MATH 等 Benchmark 对不同后训练方法进行评测，对比其在推理准确率、训练稳定性、参数效率和资源开销等方面的表现，分析不同策略之间的性能权衡关系。


| Method     | GSM8K | MATH-500 | GAOKAO | AIME25 |
| ---------- | ----: | -------: | -----: | -----: |
| Base Model | 84.69 |    66.60 |   TBD  |   3.33 |
| Full SFT   |  TBD  |     TBD  |   TBD  |   TBD  |
| LoRA-SFT   |  TBD  |     TBD  |   TBD  |   TBD  |
| DPO        |  TBD  |     TBD  |   TBD  |   TBD  |
| PPO        |  TBD  |     TBD  |   TBD  |   TBD  |
| GRPO       |  TBD  |     TBD  |   TBD  |   TBD  |
| DAPO       |  TBD  |     TBD  |   TBD  |   TBD  |
| Agent   | TBD   | TBD   | TBD      | TBD    |
- Base Model： [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

- 添加benchmark：Gaokao

## 技术方案

### 模型

* Qwen2.5-7B
* Qwen2.5-Math-7B

### 后训练方法

* Full SFT
* LoRA-SFT
* DPO
* GRPO

### 训练框架

* MS-Swift
* Transformers
* DeepSpeed

### 评测数据集

* GSM8K
* MATH

## 项目收获

* 理解并复现主流 LLM 后训练流程；
* 掌握 SFT、DPO、GRPO 等算法的训练与评测方法；
* 熟悉 MS-Swift、DeepSpeed 等大模型训练框架；
* 具备从数据构建、模型训练到 Benchmark 评测的完整实验经验。
