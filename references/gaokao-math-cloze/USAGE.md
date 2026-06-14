# GaoKao Math Cloze Evaluation Usage

## Scope

This is an independent benchmark for Chinese GaoKao mathematics fill-in-the-blank
questions. Do not merge its score with GaoKao Math QA.

- Standard subset name: `gaokao_math_cloze` / AGIEval `gaokao-mathcloze`
- Standard size: 118 questions
- Deduplicated audit size: 116 questions
- Input: `question`
- Gold answer: `answer`
- Metric: per-question mathematical-equivalence accuracy

The standard file is:

```text
exp/dataset-analysis/gaokao-math-cloze/AGIEval/data/v1_1/gaokao-mathcloze.jsonl
```

Its SHA-256 is:

```text
088675c147794970a3ed25c7147a3bbc59715d6813837d5f483f46dcb1b5008d
```

## Prompt

Use zero-shot reasoning and require a machine-readable final answer:

```text
请解答下面的中国高考数学填空题。请给出推理过程，并将最终答案放在
\boxed{} 中；多空答案请按题目顺序用分号分隔。

{question}
```

Use deterministic decoding for the primary result:

```text
temperature=0
top_p=1
max_tokens=3072
```

## Answer Extraction

1. Extract the last complete `\boxed{...}` expression.
2. If no box exists, fall back to the final answer line only and mark the fallback
   rate in the evaluation log.
3. Strip outer `$...$`, whitespace, `\left`, and `\right`.
4. For answers containing semicolons, split both prediction and reference in order
   and score every part independently.
5. A multi-blank question is correct only when every ordered part is correct.

Use mathematical equivalence, not BLEU/ROUGE or raw string equality. The Qwen2.5-Math
evaluation `grader.py::math_equal` is a suitable starting point for numeric,
symbolic, equation, interval, and coordinate answers.

One two-blank question has Chinese free-text conditions. Keep it in the standard
118-question score, but audit it separately with normalized aliases or manual review.
Two questions use circled proposition numbers and should be scored as exact sets.

## Reporting

Report this benchmark separately:

```text
GaoKao Math Cloze accuracy: correct / 118
```

Also report:

- Deduplicated accuracy over 116 questions.
- Number of answer-extraction failures.
- Accuracy on 16 two-blank questions.
- The free-text question result.

Never combine this score with GaoKao Math QA into a single `GAOKAO` score.

## Caveats

- Standard rows contain two duplicates: index pairs `12/47` and `13/48`.
- The benchmark is short-input and fully compatible with a 4096-token context.
- Original GAOKAO-Bench categorizes fill-in-the-blank questions under subjective
  questions. Here, "objective" means rule-scored, not multiple-choice.
- Use the standard 118 rows for external comparability; use 116 rows only as an
  additional internal audit.

