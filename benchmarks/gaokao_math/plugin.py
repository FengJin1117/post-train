"""EvalScope registrations for strict GaoKao mathematics benchmarks."""

import re
from pathlib import Path
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.math_parser import extract_answer, math_equal
from evalscope.utils.multi_choices import MultipleChoiceTemplate


ROOT = Path(__file__).resolve().parent
CLOZE_DATA = ROOT / 'data' / 'cloze'
QA_DATA = ROOT / 'data' / 'qa'

CLOZE_PROMPT = r"""请解答下面的中国高考数学填空题。请给出推理过程，并将最终答案放在 \boxed{{}} 中；多空答案请按题目顺序用分号分隔。

{question}"""

QA_PROMPT = MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT


def normalize_cloze_part(value: str) -> str:
    value = str(value).strip()
    while len(value) >= 2 and value[0] == '$' and value[-1] == '$':
        value = value[1:-1].strip()
    value = value.replace(r'\left', '').replace(r'\right', '')
    value = value.replace('，', ',').replace('。', '').replace('；', ';')
    return re.sub(r'\s+', '', value)


def split_cloze_answer(value: str) -> list[str]:
    return [normalize_cloze_part(part) for part in re.split(r'[;；]', str(value))]


def cloze_answers_equal(prediction: str, reference: str) -> bool:
    pred_parts = split_cloze_answer(prediction)
    ref_parts = split_cloze_answer(reference)
    if len(pred_parts) != len(ref_parts) or any(not part for part in pred_parts):
        return False
    return all(
        pred == ref or math_equal(pred, ref)
        for pred, ref in zip(pred_parts, ref_parts)
    )


def normalize_choice_answer(value: str) -> str:
    return ''.join(sorted(set(re.findall(r'[A-D]', str(value).upper()))))


def extract_choice_answer(prediction: str) -> str:
    lines = [line.strip() for line in str(prediction).splitlines() if line.strip()]
    if not lines:
        return ''
    final_line = lines[-1]
    marker = re.search(r'(?:答案|ANSWER)\s*[:：]\s*([A-D\s,，、]+)', final_line, re.IGNORECASE)
    return normalize_choice_answer(marker.group(1) if marker else final_line)


class LocalJSONLMixin:

    def load_from_disk(self, use_local_loader: bool = False):
        return super().load_from_disk(use_local_loader=True)


@register_benchmark(
    BenchmarkMeta(
        name='gaokao_math_cloze',
        pretty_name='GaoKao Math Cloze',
        dataset_id=str(CLOZE_DATA),
        tags=[Tags.MATH, Tags.REASONING, Tags.QA],
        description='AGIEval v1.1 GaoKao mathematics fill-in-the-blank questions with strict all-correct scoring.',
        subset_list=['default'],
        eval_split='test',
        few_shot_num=0,
        metric_list=['acc'],
        prompt_template=CLOZE_PROMPT,
    )
)
class GaoKaoMathClozeAdapter(LocalJSONLMixin, DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            target=record['answer'],
            metadata={'source': record.get('other', {}).get('source', '')},
        )

    def extract_answer(self, prediction: str, task_state) -> str:
        return extract_answer(prediction, use_last_number=False)

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state) -> Score:
        return Score(
            value={'acc': int(cloze_answers_equal(filtered_prediction, reference))},
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )


@register_benchmark(
    BenchmarkMeta(
        name='gaokao_math_qa',
        pretty_name='GaoKao Math QA',
        dataset_id=str(QA_DATA),
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description='AGIEval v1.1 GaoKao mathematics choice questions with strict exact-set scoring.',
        subset_list=['default'],
        eval_split='test',
        few_shot_num=0,
        metric_list=['acc'],
        prompt_template=QA_PROMPT,
    )
)
class GaoKaoMathQAAdapter(LocalJSONLMixin, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiple_correct = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [re.sub(r'^\s*\([A-D]\)\s*', '', choice) for choice in record['options']]
        return Sample(
            input=record['question'],
            choices=choices,
            target=normalize_choice_answer(record['label']),
            metadata={
                'source': record.get('other', {}).get('source', ''),
                'is_multi_select': len(normalize_choice_answer(record['label'])) > 1,
            },
        )

    def extract_answer(self, prediction: str, task_state) -> str:
        return extract_choice_answer(prediction)

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state) -> Score:
        pred = normalize_choice_answer(filtered_prediction)
        gold = normalize_choice_answer(reference)
        return Score(
            value={'acc': int(bool(pred) and pred == gold)},
            extracted_prediction=pred,
            prediction=original_prediction,
        )
