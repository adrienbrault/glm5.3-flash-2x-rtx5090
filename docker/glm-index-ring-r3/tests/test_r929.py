#!/usr/bin/env python3
"""Regression gates for effective launcher env, matched arms and ABA reporting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

PACKET = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r929_helpers', PACKET / 'r929_helpers.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
spec = importlib.util.spec_from_file_location('workload', PACKET / 'tests/serve_workload.py')
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class R929Tests(unittest.TestCase):
    def test_launcher_args_override_image_defaults_last_occurrence_wins(self):
        inspect = [{'Config': {'Env': ['EXL3_DSA_INDEX_RING=0', 'EXL3_DSA_INDEX_RING_SHADOW=0',
                                     'EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104']},
                    'Args': ['-u', 'EXL3_DSA_INDEX_RING', 'EXL3_DSA_INDEX_RING=0',
                             'EXL3_DSA_INDEX_RING=1', 'EXL3_DSA_INDEX_RING_SHADOW=1',
                             'EXL3_MOE_CPU_SPLIT_BY_DEVICE=98,102', 'python3', 'main.py']}]
        self.assertTrue(helpers.check_env(inspect, '98,102', 1, 1)['passed'])
        self.assertFalse(helpers.check_env(inspect, '100,104', 1, 1)['passed'])

    def test_later_unset_removes_value_and_equals_inside_value_is_preserved(self):
        obj = {'Config': {'Env': ['FLAG=old', 'EXL3_DSA_INDEX_RING=1']},
               'Args': ['FLAG=a=b', '--unset=EXL3_DSA_INDEX_RING', 'python3', 'main.py']}
        self.assertEqual(helpers.effective_env(obj), {'FLAG': 'a=b'})
        self.assertFalse(helpers.check_env(obj, '100,104', 1, 0)['passed'])

    def test_empty_launcher_environment_clears_config_env(self):
        obj = {'Config': {'Env': ['OLD=1']}, 'Args': ['-i', 'NEW=2', 'NEW=3', 'python3']}
        self.assertEqual(helpers.effective_env(obj), {'NEW': '3'})

    def test_daily_extra_preserves_all_other_flags_and_overrides_pair(self):
        daily = (PACKET / 'tests/daily.env').read_text()
        original = helpers.daily_values(daily)['EXL3_EXTRA'].split(';')
        for pair in helpers.PAIRS:
            actual = helpers.extra(daily, pair, 1, 0).split(';')
            for flag in original:
                if not flag.startswith('EXL3_MOE_CPU_SPLIT_BY_DEVICE='):
                    self.assertIn(flag, actual)
            self.assertEqual([f for f in actual if f.startswith('EXL3_MOE_CPU_SPLIT_BY_DEVICE=')],
                             ['EXL3_MOE_CPU_SPLIT_BY_DEVICE=' + pair])
            self.assertIn('EXL3_DSA_INDEX_RING=1', actual)
            self.assertIn('EXL3_DSA_INDEX_RING_SHADOW=0', actual)
        self.assertEqual(helpers.PAIRS, ('96,100', '97,101', '98,102', '99,103'))

    def test_wrong_daily_and_invalid_shadow_are_rejected(self):
        daily = (PACKET / 'tests/daily.env').read_text()
        for text in (daily.replace('_splitdev2', '_other'), daily.replace('100,104', '104,104')):
            with self.assertRaises(ValueError):
                helpers.extra(text, '96,100', 1, 0)
        with self.assertRaises(ValueError):
            helpers.extra(daily, '100,104', 0, 1)

    def test_aba_ratios_compare_b_with_both_daily_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for tag, value in (('A1', 80), ('B-96-100', 100), ('A2', 120)):
                (root / tag).mkdir()
                row = {'phase': 'decode-summary', 'summaries':
                       [{'c': c, 'ss_agg_tps_median': value * c} for c in (1, 2, 4)]}
                (root / tag / 'dec.jsonl').write_text(json.dumps(row) + '\n')
            report = helpers.report(root, 'B-96-100')
            self.assertEqual(report['B_vs_A_mean'], {'1': 1., '2': 1., '4': 1.})
            self.assertEqual(report['A2_vs_A1'], {'1': 1.5, '2': 1.5, '4': 1.5})
            (root / 'A2/dec.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                helpers.report(root, 'B-96-100')

    def test_workload_chat_max_tokens_16384_and_length_is_failure(self):
        runner = object.__new__(workload.Runner)
        runner.a = SimpleNamespace(model='fixture')
        bodies = []
        runner.record_request = lambda tag, route, body: bodies.append(body)
        runner.save = lambda row: None
        runner.json_post = lambda route, body: {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
        runner.chat('agentic_resume', [{'role': 'user', 'content': 'test'}])
        self.assertEqual(bodies[-1]['max_tokens'], 16384)
        runner.json_post = lambda route, body: {'choices': [{'message': {'content': 'partial'}, 'finish_reason': 'length'}]}
        with self.assertRaises(ValueError):
            runner.chat('agentic_resume', [{'role': 'user', 'content': 'test'}])

    def test_wrong_image_and_error_probe_rows_are_rejected(self):
        obj = {'Config': {'Image': 'wrong', 'Env': ['EXL3_DSA_INDEX_RING=0',
               'EXL3_DSA_INDEX_RING_SHADOW=0', 'EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104']}, 'Args': []}
        self.assertFalse(helpers.check_env(obj, '100,104', 0, 0, helpers.DAILY_IMAGE)['passed'])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'probe.jsonl'
            for text in ('', '{"error":"failed"}\n', '{"ok":false}\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    helpers.probe_check([path])


if __name__ == '__main__':
    unittest.main()
