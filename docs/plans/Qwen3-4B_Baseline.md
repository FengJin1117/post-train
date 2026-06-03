# Qwen3-4B Baseline 评测计划

## Summary
在 `/data2/fwh/post-train` 中跑通 `Qwen/Qwen3-4B-Instruct-2507` 的 GSM8K baseline，并输出可复现报告。

使用现有 `vllm` conda 环境，因为宿主机 `glibc 2.27` 无法直接安装当前官方 vLLM wheel。模型使用本地缓存：

```text
/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507
```

## Environment And Artifacts
- 修复缺失的 `.gitmodules`，将现有 `ms-swift` gitlink 映射到官方仓库。
- 使用仓库内 `./ms-swift` submodule，固定提交 `eb35f330e`。
- 在 `vllm` 环境安装 editable `./ms-swift` 与 `evalscope==1.8.0`。
- 使用约束文件锁定 `torch==2.6.0`、`transformers==4.52.3`、`datasets==4.8.4`、`modelscope==1.37.0`，避免破坏已可导入的 vLLM。
- 新增：
  - `exp/baseline/qwen3-4b-instruct-2507/setup_env.sh`
  - `exp/baseline/qwen3-4b-instruct-2507/run_baseline.sh`
  - `exp/baseline/qwen3-4b-instruct-2507/constraints.txt`
  - `exp/baseline/qwen3-4b-instruct-2507/environment.txt`
  - `exp/baseline/qwen3-4b-instruct-2507/benchmark-report.md`
- 将 `raw/` 下的日志、预测和 EvalScope 中间文件保留在本地并加入忽略规则；报告、脚本和环境快照纳入 Git。

## Evaluation
运行前验证 GPU `0` 是空闲显存不少于 `40 GiB` 的 A6000。先验证 Native 注册表包含：

```text
gsm8k
```

随后先做每项 2 条样本的 smoke test，再跑完整评测。完整命令固定为：

```bash
CUDA_VISIBLE_DEVICES=0 swift eval \
  --model /data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
  --enable_thinking false \
  --eval_dataset gsm8k \
  --eval_backend Native \
  --infer_backend vllm \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 10000 \
  --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}' \
  --extra_eval_args '{"judge_strategy":"rule"}' \
  --eval_num_proc 8
```

从 `raw/full/` 目录执行命令，使自动生成的 `result/` 与 `eval_output/` 也落入 baseline 目录。

## Report
`benchmark-report.md` 记录：
- 模型 ID、本地路径、MS-SWIFT 提交和关键依赖版本。
- GPU 型号、实际 GPU 编号、评测日期、完整命令。
- GSM8K 的样本数、准确率与运行耗时。
- 原始结果目录位置。
- 口径说明：使用 `8192` tokens，关闭 thinking，使用规则判分，未启用 LLM Judge。

## Verification
1. 安装后验证 `swift` 指向 `./ms-swift`，并确认 `import swift, evalscope, vllm, torch` 成功。
2. 确认 vLLM 版本保持 `0.1.dev5699+gde8f43fbe.cu118`，Torch 保持 `2.6.0+cu124`。
3. 运行 GSM8K benchmark 的 smoke test；失败时停止完整任务并保留日志。
4. 运行完整评测，检查 GSM8K 指标存在。
5. 抽查预测，确认模型输出正常且规则判分未出现明显解析异常。
6. 检查 Git 状态，确保仅报告、脚本、约束和环境快照进入版本控制。

## Assumptions
- 使用已有 `vllm` 环境是经过确认的兼容方案，不再修改原 `ms-swift` conda 环境。
- 规则判分作为本项目后续所有方法的统一 baseline 口径。
- EvalScope 对复杂数学表达式可能存在少量规则解析误差，报告中明确保留该限制。
