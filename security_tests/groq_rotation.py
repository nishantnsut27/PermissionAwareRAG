"""Offline Groq transport and direct full-report resume regressions."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from answering.src import config, groq_keys, llm
from evaluation import dataset as ds, evaluator, runner
from scripts import run_evaluation


def response(status=200, text='', headers=None):
    result = MagicMock(status_code=status, text=text, headers=headers or {})
    result.json.return_value = {'choices': [{'message': {'content': 'Completed answer'}, 'finish_reason': 'stop'}]}
    return result


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.pool = groq_keys.GroqKeyPool(['test-key-one', 'test-key-two', 'test-key-three'], cooldown_s=60)
        self.enterContext(patch.object(groq_keys, 'pool', return_value=self.pool))
        self.enterContext(patch.object(config, 'GROQ_MAX_RETRIES', 2))
        self.enterContext(patch.object(llm.time, 'sleep'))

    def generate(self):
        return llm.generate([{'role': 'user', 'content': 'offline'}])

    def test_successful_answer_and_judge_requests_rotate(self):
        with patch.object(llm.HTTP, 'post', side_effect=lambda *a, **k: response()) as post:
            for _ in range(4): self.generate()
        self.assertEqual([c.kwargs['headers']['Authorization'] for c in post.call_args_list],
                         ['Bearer test-key-one', 'Bearer test-key-two', 'Bearer test-key-three', 'Bearer test-key-one'])

    def test_rate_limit_moves_on_and_honors_provider_reset(self):
        with patch.object(llm.HTTP, 'post', side_effect=[response(429, 'tokens per day', {'Retry-After': '3600'}), response(), response()]) as post:
            self.generate()
            self.generate()
        self.assertEqual([c.kwargs['headers']['Authorization'] for c in post.call_args_list],
                         ['Bearer test-key-one', 'Bearer test-key-two', 'Bearer test-key-three'])
        self.assertGreaterEqual(self.pool.status().per_key[0]['cooldown_remaining_s'], 3599)

    def test_transient_retries_stay_on_the_same_key(self):
        with patch.object(llm.HTTP, 'post', side_effect=[requests.ConnectionError('offline'), response(503, 'unavailable'), response(), response()]) as post:
            self.generate()
            self.generate()
        self.assertEqual([c.kwargs['headers']['Authorization'] for c in post.call_args_list],
                         ['Bearer test-key-one'] * 3 + ['Bearer test-key-two'])

    def test_invalid_key_is_disabled_and_payload_errors_are_not_retried(self):
        with patch.object(llm.HTTP, 'post', side_effect=[response(401, 'invalid'), response(), response(400, 'invalid model')]) as post:
            self.generate()
            with self.assertRaises(llm.LLMError): self.generate()
        self.assertEqual(post.call_count, 3)
        self.assertTrue(self.pool.status().per_key[0]['disabled'])

    def test_shared_daily_quota_fails_without_recycling_cooling_keys(self):
        with patch.object(llm.HTTP, 'post', side_effect=lambda *a, **k: response(429, 'tokens per day (TPD): test-key-one', {'retry-after': '3600'})) as post:
            with self.assertRaises(llm.LLMError) as failed: self.generate()
            self.assertIn('shared', str(failed.exception))
            self.assertNotIn('test-key-one', str(failed.exception))
            with self.assertRaisesRegex(llm.LLMError, 'Earliest retry'): self.generate()
        self.assertEqual(post.call_count, 3)

    def test_late_success_does_not_clear_a_concurrent_cooldown_or_rewind_rotation(self):
        self.pool.acquire()
        self.pool.acquire()
        self.pool.report_failure('key-1', groq_keys.RATE_LIMIT, retry_after_s=3600)
        self.pool.report_success('key-1')
        self.assertTrue(self.pool.status().per_key[0]['cooling_down'])
        self.assertEqual(self.pool.acquire()[0], 'key-3')

    def test_retry_hints_support_seconds_dates_and_provider_durations(self):
        self.assertEqual(groq_keys.cooldown_seconds({'Retry-After': '2.5'}), 2.5)
        self.assertEqual(groq_keys.cooldown_seconds({'x-ratelimit-reset-tokens': '1h2m3.5s'}, 'tokens per day'), 3723.5)
        self.assertEqual(groq_keys.cooldown_seconds({}, 'Please try again in 2h3m4.5s.'), 7384.5)
        with patch.object(groq_keys.time, 'time', return_value=0):
            self.assertEqual(groq_keys.cooldown_seconds({'Retry-After': 'Thu, 01 Jan 1970 01:00:00 GMT'}), 3600)
        for invalid in ('NaN', 'inf', '-1', 'unknown'):
            self.assertIsNone(groq_keys.cooldown_seconds({'Retry-After': invalid}))

    def test_duplicate_credentials_do_not_inflate_the_pool(self):
        self.assertEqual(groq_keys.load_keys({'GROQ_API_KEY_1': 'one', 'GROQ_API_KEY_2': 'one',
                                            'GROQ_API_KEY_3': ' two ', 'GROQ_API_KEY': 'one'}), ['one', 'two'])


class FullReportCommandTests(unittest.TestCase):
    def test_full_phase_resumes_existing_report_without_running_pilot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'evaluation/runs/existing-full.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}', encoding='utf-8')
            result = {'response_log': '.cache/evaluation-responses/same-run.jsonl', 'retry_queue': [],
                      'exhausted_cases': [], 'selection_complete': True, 'summary': {'passed': 90, 'failed': 0}}
            with patch.object(run_evaluation, 'ROOT', root), \
                 patch.object(run_evaluation.ds, 'load'), \
                 patch.object(run_evaluation.ds, 'PermissionCatalog'), \
                 patch.object(run_evaluation.ds, 'validate', return_value=[]), \
                 patch.object(config, 'validate_env'), \
                 patch.object(run_evaluation.runner, 'run', return_value=result) as run, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(run_evaluation.execute('existing', phase='full'), 0)
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs['output'], path)
            self.assertTrue(run.call_args.kwargs['resume'])
            self.assertIsNone(run.call_args.kwargs['only'])
            self.assertEqual(run.call_args.kwargs['max_retries'], 0)


class FinalAttemptTests(unittest.TestCase):
    def test_extensions_apply_only_to_the_approved_existing_report(self):
        manifest = json.loads((ds.PACKAGE_ROOT / 'fixtures/groq_rotation_checkpoints.json').read_text(encoding='utf-8'))
        run_id = 'run-20261104-101037-414954'
        limits = runner.attempt_limits({'run_id': run_id})
        self.assertEqual(limits, {case: 5 for case in ['E001', 'E010', 'E012', 'E025']})
        self.assertEqual(limits, manifest['attempt_extensions'][run_id])
        self.assertEqual(runner.attempt_limit('E026', limits), 4)
        self.assertEqual(runner.attempt_limit('E001', runner.attempt_limits({'run_id': 'new-run'})), 4)

    def test_attempt_extension_cannot_migrate_v6_checkpoint(self):
        data = ds.GoldenDataset('offline', '1.1', '', [
            ds.EvalCase(f'E{i:03}', 'U-001', f'question {i}', 'authorized_factual', 'answer', '') for i in range(1, 4)])
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(ds, 'validate', return_value=[]), \
             patch.object(runner, 'resolve_index_snapshot', return_value={'collection_name': 'offline'}), \
             patch.object(runner.index_retrieval, 'pin_revision', return_value=nullcontext()), \
             patch.object(runner, 'fingerprint', return_value='before') as signature, \
             patch.object(runner, 'save_checkpoint', side_effect=runner.save):
            path = Path(folder) / 'same-report.json'
            args = dict(dataset=data, catalog=SimpleNamespace(documents=[]), delay_s=0, retry_wait_s=0,
                        max_retries=0, verbose=False, output=path)
            previous = runner.run(**args, checkpoint=False, initialize_only=True)
            for case, count, status in [(data.cases[0], 4, 'FAIL'), (data.cases[1], 1, 'PASS'), (data.cases[2], 4, 'FAIL')]:
                row = evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question,
                    case.category, case.expected_behavior, status=status, answered=True, outcome='answer', answer='saved answer').to_dict()
                row['attempts'] = [{'attempt': n + 1, 'fingerprint': 'before', 'result': copy.deepcopy(row)} for n in range(count)]
                previous['results'].append(row)
            runner.save(previous, path)
            signature.return_value = 'after'
            def failed(case, *_):
                return evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question,
                    case.category, case.expected_behavior, status='FAIL', answered=True, outcome='answer', answer='new answer')
            with patch.object(runner, 'attempt_limits', return_value={'E001': 5}), \
                 patch.object(runner, 'compatible_groq_checkpoint', return_value=True), \
                 patch.object(evaluator, 'run_case', side_effect=failed) as execute:
                before = path.read_bytes()
                with self.assertRaises(ds.DatasetError):
                    runner.run(**args, resume=True)
                execute.assert_not_called()
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
