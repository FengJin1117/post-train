# Qwen2.5-3B Base Baseline Results

| Benchmark | Fixed gate | Full benchmark |
| --- | ---: | ---: |
| GSM8K | 48.50 | 50.72 |
| MATH-500 | 71.00 | 57.60 |

## GaoKao Mathematics

| Benchmark | Correct / total | Strict accuracy | Special subset |
| --- | ---: | ---: | ---: |
| GaoKao Math Cloze | 48 / 118 | 40.68 | Multi-blank: 1 / 16 (6.25) |
| GaoKao Math QA | 158 / 351 | 45.01 | Multi-select: 2 / 7 (28.57) |

Both GaoKao benchmarks use deterministic decoding with `max_tokens=3072` and
strict all-or-nothing scoring. Cloze multi-blank answers require every ordered
part to be correct. QA answers require exact option-set equality, so partial
multi-select answers receive zero.

The Cloze and QA extraction-failure counts are 20 and 25. Their max-token rates
are `11.86%` and `10.26%`, respectively.

The raw base model is evaluated with the same `qwen2_5` ChatML template and
system prompt used by the SFT models.

The fixed gate uses 200 GSM8K examples and 100 MATH-500 examples. The high
GSM8K max-token rate (`36.5%`) is a known raw-base failure mode and is tracked
as an additional checkpoint-selection signal.
