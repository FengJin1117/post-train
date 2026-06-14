# GaoKao Math QA Evaluation Usage

## Scope

This is an independent benchmark for Chinese GaoKao mathematics choice questions.
Do not merge its score with GaoKao Math Cloze.

- Standard subset name: `gaokao_math_qa` / AGIEval `gaokao-mathqa`
- Standard size: 351 questions
- Deduplicated audit size: 348 questions
- Input: `question` plus four options
- Gold answer: `label`
- Metric: exact option-set accuracy

The standard file is:

```text
exp/dataset-analysis/gaokao-math-cloze/AGIEval/data/v1_1/gaokao-mathqa.jsonl
```

Its SHA-256 is:

```text
d246f12752d121289ef55cbf1bcf954243cefb65d110f71d09358e24065c808f
```

## Important: Mixed Single-Select And Multi-Select

The benchmark contains:

- 344 single-select questions.
- 7 multi-select questions.

Do not use a scorer that always extracts only the last option letter. Normalize each
gold label and prediction to an unordered set of `A`, `B`, `C`, and `D`, then require
exact set equality. For multi-select questions, missing or adding any option is wrong.

## Prompt

```text
请解答下面的中国高考数学选择题。请给出推理过程。
最后一行仅输出最终选项字母；若有多个正确选项，请连续输出并按字母排序，
例如：AC。

{question}
{options}
```

Use deterministic decoding for the primary result:

```text
temperature=0
top_p=1
max_tokens=3072
```

## Answer Extraction And Scoring

1. Read the final non-empty output line.
2. Extract only standalone option letters `A` through `D`.
3. Deduplicate and sort the letters.
4. Normalize the gold label the same way; spaces in labels such as `A B D` are
   insignificant.
5. Score `set(prediction) == set(label)`.

Examples:

```text
prediction "答案：A C" -> AC
gold "AC"              -> AC  -> correct

prediction "A"         -> A
gold "AC"              -> AC  -> incorrect
```

MS-SWIFT Native `general_mcq` is appropriate only after verifying or extending it
to support exact-set multi-select scoring. A single-choice-only scorer silently
mis-scores seven rows.

## Reporting

Report this benchmark separately:

```text
GaoKao Math QA exact-set accuracy: correct / 351
```

Also report:

- Single-select accuracy over 344 questions.
- Multi-select exact-set accuracy over 7 questions.
- Deduplicated accuracy over 348 questions.
- Answer-extraction failure count.

Never combine this score with GaoKao Math Cloze into a single `GAOKAO` score.

## Caveats

- Standard rows contain three duplicates: index pairs `45/166`, `48/171`, and
  `50/173`.
- All rows have four options and fit comfortably within a 4096-token context.
- Keep all 351 rows for external comparability; use 348 rows only as an additional
  internal audit.
- AGIEval's generic Chinese QA path historically assumes single-choice answers.
  Validate the multi-select behavior before trusting an off-the-shelf result.

