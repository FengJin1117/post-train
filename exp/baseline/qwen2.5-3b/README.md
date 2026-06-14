# Qwen2.5-3B Base Baseline

统一使用 `qwen2_5` ChatML 模板和系统提示 `You are a helpful assistant.`。

```bash
cd /data2/fwh/post-train
tmux new-session -d -s qwen25-3b-base-baseline \
  "bash exp/baseline/qwen2.5-3b/run_baseline.sh all"
```

输出：

- `gate-results.json`: 固定 GSM8K-200 / MATH-500-100 门禁基线
- `full-results.json`: 完整 GSM8K / MATH-500 基线
- `gaokao-results.json`: 完整 GaoKao Math Cloze / GaoKao Math QA 严格全对基线
- `raw/`: EvalScope 原始结果

只运行两个 GaoKao benchmark：

```bash
bash exp/baseline/qwen2.5-3b/run_baseline.sh gaokao
```
