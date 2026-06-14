# Qwen2.5-3B Base NuminaMath Clean V2

使用相同 clean-10K 数据并行比较：

- B1：Instruct Clean V2 同配方，LR `1e-5`、rank `8`、1 epoch。
- B2：Base 加强配方，LR `5e-5`、rank `16`、all-linear、2 epochs。

数据通过 `data/` 下的相对软链接复用。

## 执行顺序

```bash
# baseline
bash exp/baseline/qwen2.5-3b/run_baseline.sh all

# smoke
conda activate ms-swift
bash exp/sft-lora/qwen2.5-3b-numinamath-clean-v2/run_train.sh b1 smoke
bash exp/sft-lora/qwen2.5-3b-numinamath-clean-v2/run_train.sh b2 smoke

# 正式并行训练
tmux new-session -d -s qwen25-base-b1 "cd /data2/fwh/post-train && eval \"\$(conda shell.bash hook)\" && conda activate ms-swift && bash exp/sft-lora/qwen2.5-3b-numinamath-clean-v2/run_train.sh b1 full"
tmux new-session -d -s qwen25-base-b2 "cd /data2/fwh/post-train && eval \"\$(conda shell.bash hook)\" && conda activate ms-swift && bash exp/sft-lora/qwen2.5-3b-numinamath-clean-v2/run_train.sh b2 full"
```

