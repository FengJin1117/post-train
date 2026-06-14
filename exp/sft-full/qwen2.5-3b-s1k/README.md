# Qwen2.5-3B Base + s1K Full SFT

This experiment transfers the original s1 full-SFT recipe to
`Qwen/Qwen2.5-3B` base using MS-SWIFT.

```bash
conda run -n ms-swift --no-capture-output python prepare_data.py
bash run_train.sh r0-original smoke
bash run_train.sh r1-conservative smoke
```

Long-running training and evaluation are launched through `run_pipeline.sh`
inside tmux. Detailed source notes are in
`references/s1K/REPRODUCTION_NOTES.md`.
