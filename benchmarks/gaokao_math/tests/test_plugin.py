import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gaokao_math_plugin', ROOT / 'plugin.py')
PLUGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLUGIN)


class GaoKaoMathPluginTest(unittest.TestCase):

    def test_dataset_integrity(self):
        cases = [
            ('cloze/default_test.jsonl', 118, '088675c147794970a3ed25c7147a3bbc59715d6813837d5f483f46dcb1b5008d'),
            ('qa/default_test.jsonl', 351, 'd246f12752d121289ef55cbf1bcf954243cefb65d110f71d09358e24065c808f'),
        ]
        for relative_path, expected_rows, expected_hash in cases:
            path = ROOT / 'data' / relative_path
            self.assertEqual(len(path.read_text().splitlines()), expected_rows)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)

    def test_qa_dataset_has_seven_multi_select_questions(self):
        path = ROOT / 'data' / 'qa' / 'default_test.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        multi = [row for row in rows if len(PLUGIN.normalize_choice_answer(row['label'])) > 1]
        self.assertEqual(len(multi), 7)

    def test_choice_strict_exact_set(self):
        self.assertEqual(PLUGIN.extract_choice_answer('推理\n答案：C A'), 'AC')
        self.assertEqual(PLUGIN.extract_choice_answer('推理\nAC'), 'AC')
        self.assertNotEqual(PLUGIN.extract_choice_answer('答案：A'), 'AC')
        self.assertNotEqual(PLUGIN.extract_choice_answer('答案：ABC'), 'AC')
        self.assertEqual(PLUGIN.normalize_choice_answer('A A, C'), 'AC')

    def test_cloze_single_math_equivalence(self):
        self.assertTrue(PLUGIN.cloze_answers_equal(r'\frac{1}{2}', '0.5'))
        self.assertFalse(PLUGIN.cloze_answers_equal('2', '3'))

    def test_cloze_multi_part_is_ordered_and_all_correct(self):
        self.assertTrue(PLUGIN.cloze_answers_equal(r'$5$ ; $\frac{10}{2}$', '$5$;$5$'))
        self.assertFalse(PLUGIN.cloze_answers_equal('5', '5;10'))
        self.assertFalse(PLUGIN.cloze_answers_equal('10;5', '5;10'))
        self.assertFalse(PLUGIN.cloze_answers_equal('5;9', '5;10'))

    def test_benchmarks_are_registered(self):
        from evalscope.api.registry import BENCHMARK_REGISTRY

        self.assertIn('gaokao_math_cloze', BENCHMARK_REGISTRY)
        self.assertIn('gaokao_math_qa', BENCHMARK_REGISTRY)


if __name__ == '__main__':
    unittest.main()
