# s1 / s1K Reproduction Notes

## Fixed Sources

- Repository: `https://github.com/simplescaling/s1`
- Repository commit: `77272c6e925d610257a50b520bad15330b513389`
- Local clone: `references/s1-repo/` (gitignored)
- Dataset: `simplescaling/s1K`
- Dataset revision: `278d72baaa2b887a7e76a70a0ae254a5a45536e4`

## Original Data Construction

The public s1K dataset contains 1,000 examples: 877 math, 108 science, and 15
crossword examples. The original `data/tokenization.py` creates one Qwen ChatML
conversation per row:

```text
user: question
assistant:
<|im_start|>think
{thinking_trajectories joined with newlines}
<|im_start|>answer
{attempt, prefixed with "Answer: " when absent}
```

The original training code uses `DataCollatorForCompletionOnlyLM`, so loss is
computed only over the assistant response. MS-SWIFT SFT's default
`loss_scale=default` provides the same assistant-only behavior.

## Original Training Recipe

- Base model: `Qwen/Qwen2.5-32B-Instruct`
- Full-parameter SFT, bf16, no validation
- 5 epochs, learning rate `1e-5`, cosine scheduler, warmup ratio `0.05`
- Adam beta1 `0.9`, beta2 `0.95`, weight decay `1e-4`
- Per-device batch size 1 and reported effective batch size 16
- Maximum sequence length 32,768
- FSDP full shard; gradient checkpointing is recommended when needed

This reproduction transfers the recipe to the non-Instruct `Qwen2.5-3B` base
model and uses DeepSpeed ZeRO-3 through MS-SWIFT.

MS-SWIFT's training environment uses Transformers 5.x while the vLLM
evaluation environment uses Transformers 4.52. The saved Transformers 5.x
`extra_special_tokens` list is incompatible with 4.52, so evaluation restores
the unchanged base model `tokenizer_config.json`; model weights and chat
template are not modified.

## Evaluation And Caveats

The original s1 paper evaluates MATH-500, AIME24, and GPQA and combines SFT
with test-time budget forcing. This reproduction evaluates standard
deterministic pass@1 without budget forcing on GSM8K, MATH-500, AIME24, and
AIME25.

s1K includes 287 examples sourced from `qq8933/AIME_1983_2024`; therefore
AIME24 must be reported as training-contaminated and is not clean evidence of
generalization. The experiment creates a 13-word n-gram contamination report
against all four evaluation benchmarks.
