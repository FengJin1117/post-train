# GaoKao Math EvalScope Benchmarks

This local EvalScope plugin registers two independent AGIEval v1.1 benchmarks:

- `gaokao_math_cloze`: 118 fill-in-the-blank questions.
- `gaokao_math_qa`: 351 choice questions, including 7 multi-select questions.

Both use strict all-or-nothing accuracy. Multi-blank Cloze answers must have every
ordered part correct. QA predictions must exactly match the gold option set.

```bash
conda run -n vllm env PYTHONPATH=ms-swift swift eval \
  --model <model-or-checkpoint> \
  --eval_backend Native \
  --external_plugins benchmarks/gaokao_math/plugin.py \
  --eval_dataset gaokao_math_cloze gaokao_math_qa
```

The bundled JSONL files preserve the standard rows, including duplicates, for
comparability. Their integrity and scoring rules are covered by:

```bash
conda run -n vllm python -m unittest discover -s benchmarks/gaokao_math/tests -v
```
