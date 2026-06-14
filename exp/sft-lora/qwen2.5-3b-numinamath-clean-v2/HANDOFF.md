# Qwen2.5-3B Base Clean V2 Handoff

目标：比较 B1 同配方与 B2 Base 加强配方，判断 clean-10K 能否为 raw base 建立数学指令能力。

模型：`/data2/fwh/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B`

## Current status

- Raw-base baseline complete:
  - Full GSM8K: `50.72`
  - Full MATH-500: `57.60`
  - Fixed gate GSM8K-200: `48.50`
  - Fixed gate MATH-500-100: `71.00`
- B1 and B2 two-step smoke runs passed. Both saved LoRA checkpoints with finite loss.
- Parallel formal training completed on 2026-06-08:
  - B1 completed `209/209` steps on GPUs `0,1,4`.
  - B2 completed `418/418` steps on GPUs `5,6,7`.
- `run_train.sh` assigns independent torchrun ports (`29511` and `29512`) so B1/B2 can run concurrently.
- All 14 checkpoint gates completed; no checkpoint passed:
  - B1 best by gate ranking: `checkpoint-209`, GSM8K-200 `14%`, MATH-500-100 `39%`.
  - B2 best by gate ranking: `checkpoint-200`, GSM8K-200 `32.5%`, MATH-500-100 `43%`.
- Full evaluation is running in tmux:
  - `qwen25-base-final-b1`, GPU 0, B1 checkpoint-209.
  - `qwen25-base-final-b2`, GPU 1, B2 checkpoint-200.
  - `qwen25-base-finalize` waits for both and writes `full-results-b1.json` and
    `full-results-b2.json`.
- Full evaluation is slow because most outputs reach the 8192-token limit.
- Gate infrastructure fixes made during execution:
  - Repair exported tokenizer metadata for the older vLLM environment.
  - Bind LoRA merge to each gate task's assigned GPU.
  - Continue a gate wave after individual task failures and skip completed results.
  - Support `GATE_GPUS=0,1,4` for evaluation when other GPUs are unavailable.

## Continue from here

1. Monitor `qwen25-base-final-b1` and `qwen25-base-final-b2` until both finish.
2. Summarize each full eval with `exp/baseline/qwen2.5-3b/analyze.py`, using
   `eval/b1-checkpoint-209` and `eval/b2-checkpoint-200` as roots.
3. Record full results in this directory. Only update top-level `RESULTS.md` after explicit permission.

所有训练从原始 base 模型开始。数据为 Instruct Clean V2 的相对软链接，不得复制或修改。
