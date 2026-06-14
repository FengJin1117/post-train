# Qwen2.5-Math-1.5B Base Baseline

This experiment evaluates `Qwen/Qwen2.5-Math-1.5B` on full GSM8K and
MATH-500 using Native EvalScope, vLLM, deterministic decoding, and rule
scoring.

The model has a native 4,096-token context. Evaluation therefore uses
`vllm_max_model_len=4096` and `max_tokens=3072`; forcing the initially planned
10,000/8,192 limits crashes vLLM after generation exceeds the model's rotary
position cache.

```bash
cd /data2/fwh/post-train
tmux new-session -d -s qwen25-math-15b-baseline \
  "bash exp/baseline/qwen2.5-math-1.5b/run_baseline.sh all"
```

The runner performs a two-sample GSM8K smoke test and a two-per-level
MATH-500 smoke test before launching the full benchmarks in parallel. Final
metrics are written to `RESULTS.md`.
