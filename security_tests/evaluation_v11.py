"""Offline dataset, checkpoint, denominators and safe-report contracts."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext, redirect_stdout
from unittest.mock import patch

from evaluation import dataset as ds, evaluator, judge, metrics, reports, runner
from answering.src import config
from scripts import run_evaluation


class DatasetContracts(unittest.TestCase):
    def test_invalid_judge_response_is_retained_without_reclassifying_as_pass(self):
        raw = '{"leaked":true,"refused":false,"evidence":["invented quote"],"reason":"bad"}'
        with patch.object(judge.llm, 'generate', return_value=(raw, {})), judge.record_calls() as calls:
            with self.assertRaises(judge.JudgeError): judge.leakage('q', 'actual answer', {'identity_scope': {}, 'denied_documents': [{'document_id': 'D'}], 'restricted_facts': []}, 'authorized evidence')
        self.assertEqual(calls[0]['response'], raw)
        self.assertIn('authorized evidence', calls[0]['messages'][1]['content'])

    @classmethod
    def setUpClass(cls):
        cls.dataset = ds.load()
        cls.catalog = ds.PermissionCatalog()

    def test_source_provenance_and_permissions_for_all_90(self):
        self.assertEqual(len(self.dataset), 90)
        self.assertEqual(ds.validate(self.dataset, self.catalog), [])
        self.assertEqual(ds.load(ds.V11_DATASET_FILE).summary()['expected_source_documents'], 47)
        broken = copy.deepcopy(self.dataset)
        broken.cases[60].source_provenance[0]['quote'] = 'not in source'
        self.assertTrue(any('evidence' in p for p in ds.validate(broken, self.catalog)))

    def test_original_cases_unchanged_except_documented_source_correction(self):
        raw = json.loads(ds.V11_DATASET_FILE.read_text(encoding='utf-8'))['cases'][:60]
        fixes = json.loads((ds.PACKAGE_ROOT / 'fixtures/expectation_corrections.json').read_text(encoding='utf-8'))
        for fix in fixes:
            row = next(r for r in raw if r['eval_id'] == fix['eval_id'])
            self.assertEqual(row[fix['field']], fix['new'])
            row[fix['field']] = fix['old']
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(digest, (ds.PACKAGE_ROOT / 'fixtures/original_60.sha256').read_text().strip())

    def test_filter_is_passed_through_without_widening_identity(self):
        case = self.dataset.by_id('E081')
        with patch.object(evaluator.service, 'answer_query', side_effect=RuntimeError('offline')) as answer:
            result = evaluator.run_case(case, self.catalog)
        self.assertEqual(result.status, 'ERROR')
        self.assertEqual(answer.call_args.kwargs['seller_filter'], ['S001', 'S002'])
        self.assertEqual(answer.call_args.kwargs['principal'].user.user_id, 'U-001')

    def test_history_uses_actual_answers_in_order_and_bounds_them(self):
        case = self.dataset.by_id('E086')
        rows = {i: {'eval_id': i, 'question': 'q-' + i, 'answer': 'actual-' + i, 'answered': True, 'outcome': 'answer'}
                for i in ['E061', 'E083', 'E084', 'E085']}
        history = runner.conversation_history(case, self.dataset, rows)
        self.assertEqual(len(history), config.DEFAULT_ANSWERING.max_history_turns)
        self.assertEqual(history[-1]['content'], 'actual-E085')
        self.assertEqual([r['content'] for r in history if r['role'] == 'assistant'], ['actual-E083', 'actual-E084', 'actual-E085'])
        rows['E084']['outcome'] = 'partial'
        with self.assertRaises(ds.DatasetError): runner.conversation_history(case, self.dataset, rows)

    def test_selection_adds_entire_dependency_chain(self):
        selected = runner.select_cases(self.dataset, only=['E086'])
        self.assertEqual([c.eval_id for c in selected], ['E061', 'E083', 'E084', 'E085', 'E086'])


class ReportContracts(unittest.TestCase):
    def report(self, date='2026-01-01T00:00:00Z', complete=True):
        return {'schema_version': 5, 'run_id': 'safe-run', 'started_at': date,
                'generated_at': date, 'selection_complete': complete, 'partial': not complete,
                'dataset': {'version': '1.1'}, 'selected_ids': ['E001', 'E002'],
                'case_specs': [{'eval_id': 'E001', 'category': 'authorized_factual', 'expected_behavior': 'answer'},
                               {'eval_id': 'E002', 'category': 'authorized_factual', 'expected_behavior': 'answer'}],
                'summary': {'private': 'DO NOT COPY'},
                'results': [{'eval_id': 'E001', 'status': 'FAIL', 'outcome': 'answer', 'expected_behavior': 'answer',
                             'question': 'PRIVATE QUESTION', 'answer': 'PRIVATE ANSWER',
                             'category': 'authorized_factual', 'latency_ms': 100,
                             'security_trace': {'secret': 'PRIVATE TRACE'},
                             'retrieval': {'recall_at_k': .25, 'precision_at_k': .2, 'mrr': 1},
                             'generation': {'groundedness': .8, 'answer_correctness': None, 'applicable': True},
                             'security': {'passed': None, 'raw_candidate_exposure': False, 'leakage_checked': False,
                                          'answer_leakage': False}}]}

    def test_measured_denominators_missing_scores_and_no_latency_fabrication(self):
        doc = self.report()
        data = reports.public_summary(doc)['measurements']
        self.assertEqual(data['retrieval']['recall_at_k'], {'value': .25, 'measured': 1, 'applicable': 2, 'total': 2})
        self.assertEqual(data['quality']['answer_correctness']['measured'], 0)
        self.assertIsNone(data['security']['answer_leakage']['value'])
        self.assertEqual(data['latency']['measured'], 1)
        self.assertEqual(data['reliability']['pending'], 1)
        denied = metrics.measurements([{'eval_id': 'x', 'expected_behavior': 'refuse'}])
        self.assertEqual(denied['retrieval']['recall_at_k']['applicable'], 0)

    def test_summary_drops_nested_private_data_and_preserves_cohorts(self):
        result = reports.public_summary(self.report())
        text = json.dumps(result)
        self.assertNotIn('PRIVATE', text)
        self.assertNotIn('DO NOT COPY', text)
        self.assertIn('original_60', result['cohorts'])

    def test_latest_incomplete_run_wins_started_time_history_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'runs').mkdir()
            old = self.report()
            old['generated_at'] = '2027-01-01T00:00:00Z'
            new = self.report('2026-11-04T00:00:00Z', False)
            (root / 'results.json').write_text(json.dumps(old))
            (root / 'runs/new.json').write_text(json.dumps(new))
            (root / 'runs/corrupt.json').write_text('{bad')
            (root / 'runs/wrong.json').write_text('{"results":[null]}')
            with patch.object(ds, 'PACKAGE_ROOT', root), patch.object(ds, 'RESULTS_FILE', root / 'results.json'):
                self.assertEqual(reports.public_payload()['report_id'], 'runs/new.json')
                self.assertEqual(reports.public_payload()['status'], 'Incomplete')
                self.assertEqual(reports.public_payload('results.json')['started_at'], old['started_at'])
                self.assertEqual(len(reports.list_reports()), 2)
                with self.assertRaises(reports.ReportNotFound): reports.load('../secret.json')
                newer = self.report('2026-11-05T00:00:00Z', False)
                (root / 'runs/newest.json').write_text(json.dumps(newer))
                self.assertEqual(reports.public_payload()['report_id'], 'runs/newest.json')
                self.assertEqual(reports.public_payload('runs/new.json')['report_id'], 'runs/new.json')

    def test_legacy_missing_fields_is_labeled_and_not_invented(self):
        legacy = {'results': [{'eval_id': 'E001'}]}
        result = reports.public_summary(legacy)
        self.assertEqual(result['status'], 'Legacy')
        self.assertIsNone(result['measurements']['quality']['groundedness']['value'])
        self.assertEqual(result['measurements']['latency']['measured'], 0)


class ResumeContracts(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        # Keep response logs with the temporary checkpoints, outside the live
        # workspace and its OneDrive/antivirus file watchers.
        self.enterContext(patch.object(ds, 'PROJECT_ROOT', Path(self.folder)))
        self.path = Path(self.folder) / 'checkpoint.json'
        self.dataset = ds.GoldenDataset('offline', '1.1', '', [
            ds.EvalCase(f'E{i:03}', 'U-001', f'question {i}', 'authorized_factual', 'answer', '') for i in range(1, 4)])
        self.catalog = SimpleNamespace(documents=[])
        self.enterContext(patch.object(ds, 'validate', return_value=[]))
        self.enterContext(patch.object(runner, 'resolve_index_snapshot', return_value={'collection_name': 'offline'}))
        self.enterContext(patch.object(runner, 'fingerprint', return_value='same'))
        self.enterContext(patch.object(runner.index_retrieval, 'pin_revision', return_value=nullcontext()))

    def result(self, case, status='PASS', error=None):
        return evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question, case.category,
                                    case.expected_behavior, status=status, answer='actual ' + case.eval_id,
                                    outcome='answer', answered=True, error=error)

    def run_batch(self, resume=False):
        return runner.run(self.dataset, self.catalog, delay_s=0, retry_wait_s=0, max_retries=0,
                          output=self.path, resume=resume, verbose=False)

    def test_api_error_and_quality_failure_resume_after_fresh_cases_with_history(self):
        def first(case, *_):
            return self.result(case, status='FAIL', error='grading judge failed: API unavailable') if case.eval_id == 'E002' else self.result(case, status='FAIL')
        with patch.object(evaluator, 'run_case', side_effect=first): first_run = self.run_batch()
        self.assertEqual(first_run['stop_reason'], 'execution_error')
        old = copy.deepcopy(first_run['results'][1]['attempts'][0])
        called = []
        def resumed(case, *_):
            called.append(case.eval_id)
            return self.result(case)
        with patch.object(evaluator, 'run_case', side_effect=resumed): final = self.run_batch(True)
        self.assertEqual(called, ['E003', 'E001', 'E002'])
        self.assertTrue(final['selection_complete'])
        self.assertEqual(final['results'][0]['status'], 'PASS')
        self.assertEqual(final['results'][0]['attempts'][0]['result']['status'], 'FAIL')
        self.assertEqual(final['results'][1]['attempts'][0], old)
        self.assertEqual(len(final['results'][1]['attempts']), 2)
        self.assertTrue((ds.PROJECT_ROOT / final['response_log']).is_file())

    def test_ctrl_c_resumes_from_the_interrupted_case(self):
        def first(case, *_):
            if case.eval_id == 'E002': raise KeyboardInterrupt()
            return self.result(case)
        with patch.object(evaluator, 'run_case', side_effect=first): paused = self.run_batch()
        self.assertEqual(paused['stop_reason'], 'interrupted')
        self.assertEqual(len(paused['results']), 2)
        reservation = paused['results'][1]['attempts'][0]
        self.assertIsNone(reservation['finished_at'])
        self.assertTrue(reservation['result']['interrupted'])
        called = []
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: (called.append(c.eval_id), self.result(c))[1]):
            final = self.run_batch(True)
        self.assertEqual(called, ['E003', 'E002'])
        self.assertTrue(final['selection_complete'])
        self.assertEqual(final['results'][1]['attempts'][0], reservation)
        self.assertEqual(len(final['results'][1]['attempts']), 2)

    def test_quality_failures_wait_for_restart_and_stop_at_four_total_attempts(self):
        calls = []
        def execute(case, *_):
            calls.append(case.eval_id)
            return self.result(case, status='PASS' if case.eval_id == 'E002' else 'FAIL')
        with patch.object(evaluator, 'run_case', side_effect=execute):
            first = self.run_batch()
            self.assertEqual(calls, ['E001', 'E002', 'E003'])
            self.assertEqual(first['retry_queue'], ['E001', 'E003'])
            self.assertTrue(first['selection_complete'])  # A complete baseline can contain failures.
            original = copy.deepcopy(first['results'][0]['attempts'][0])
            for attempt in (2, 3, 4):
                calls.clear()
                result = self.run_batch(True)
                self.assertEqual(calls, ['E001', 'E003'])
                self.assertEqual(len(result['results'][0]['attempts']), attempt)
                self.assertEqual(result['results'][0]['attempts'][0], original)
            calls.clear()
            final = self.run_batch(True)
        self.assertEqual(calls, [])
        self.assertEqual(final['exhausted_cases'], ['E001', 'E003'])
        self.assertEqual(final['retry_queue'], [])
        self.assertEqual(len(final['results'][1]['attempts']), 1)

    def test_recovered_pass_keeps_first_attempt_scores_and_is_not_retried_again(self):
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c, status='FAIL')):
            self.run_batch()
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c)):
            result = self.run_batch(True)
        self.assertEqual(result['summary']['passed'], 3)
        self.assertEqual(result['first_attempt_summary']['failed'], 3)
        with patch.object(evaluator, 'run_case') as execute:
            self.run_batch(True)
        execute.assert_not_called()

    def test_repeated_interruptions_count_toward_cap_and_preserve_each_reservation(self):
        self.dataset.cases = self.dataset.cases[:1]
        with patch.object(evaluator, 'run_case', side_effect=KeyboardInterrupt()) as execute:
            for attempt in range(1, 5):
                result = self.run_batch(resume=attempt > 1)
                self.assertEqual(len(result['results'][0]['attempts']), attempt)
                self.assertEqual(result['stop_reason'], 'interrupted')
            final = self.run_batch(True)
        self.assertEqual(execute.call_count, 4)
        self.assertFalse(final['selection_complete'])
        self.assertEqual(final['exhausted_cases'], ['E001'])
        self.assertEqual(final['first_attempt_summary']['generation']['scored_cases'], 0)

    def test_pause_during_pacing_does_not_consume_an_attempt(self):
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c)), \
             patch.object(runner.time, 'sleep', side_effect=KeyboardInterrupt()):
            result = runner.run(self.dataset, self.catalog, delay_s=90, retry_wait_s=0,
                                max_retries=0, output=self.path, verbose=False)
        self.assertEqual(result['stop_reason'], 'interrupted')
        self.assertEqual([r['eval_id'] for r in result['results']], ['E001'])
        self.assertEqual(result['retry_queue'], [])

    def test_interrupted_retry_queue_continues_fresh_cases_and_retains_passes(self):
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c, status='FAIL')):
            self.run_batch()
        def interrupt_second(case, *_):
            if case.eval_id == 'E002': raise KeyboardInterrupt()
            return self.result(case)
        with patch.object(evaluator, 'run_case', side_effect=interrupt_second): self.run_batch(True)
        called = []
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: (called.append(c.eval_id), self.result(c))[1]):
            final = self.run_batch(True)
        self.assertEqual(called, ['E002', 'E003'])
        self.assertEqual([len(r['attempts']) for r in final['results']], [2, 3, 2])

    def test_dependency_retries_follow_changed_parent_after_fresh_cases_and_obey_cap(self):
        self.dataset.cases[1] = replace(self.dataset.cases[1], conversation_from='E001')
        calls = []
        generation = 0
        def execute(case, _, history):
            calls.append(case.eval_id)
            result = self.result(case, status='FAIL' if case.eval_id == 'E001' else 'PASS')
            if case.eval_id == 'E001': result.answer = f'parent {generation}'
            if case.eval_id == 'E002': self.assertEqual(history[-1]['content'], f'parent {generation}')
            return result
        with patch.object(evaluator, 'run_case', side_effect=execute):
            for generation in range(1, 5):
                calls.clear()
                result = self.run_batch(generation > 1)
                self.assertEqual(calls, ['E001', 'E002', 'E003'] if generation == 1 else ['E001', 'E002'])
            calls.clear()
            self.run_batch(True)
        self.assertEqual(calls, [])
        self.assertEqual([len(r['attempts']) for r in result['results']], [4, 4, 1])
        rows = {r['eval_id']: r for r in result['results']}
        # With a fresh independent case, a failed parent and its child move together to the tail.
        rows.pop('E003')
        rows['E001']['attempts'].pop()
        self.assertEqual([c.eval_id for c in runner.execution_order(self.dataset.cases, rows)], ['E003', 'E001', 'E002'])

    def test_exhausted_child_is_invalidated_without_fifth_attempt(self):
        self.dataset.cases[1] = replace(self.dataset.cases[1], conversation_from='E001')
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c, status='FAIL' if c.eval_id == 'E001' else 'PASS')):
            first = self.run_batch()
        child = first['results'][1]
        child['attempts'] = [copy.deepcopy(child['attempts'][0]) for _ in range(4)]
        runner.save(first, self.path)
        def changed(case, *_):
            result = self.result(case)
            result.answer = 'changed parent'
            return result
        with patch.object(evaluator, 'run_case', side_effect=changed) as execute:
            final = self.run_batch(True)
        self.assertEqual([c.args[0].eval_id for c in execute.call_args_list], ['E001'])
        self.assertTrue(final['results'][1]['dependency_blocked'])
        self.assertEqual(len(final['results'][1]['attempts']), 4)
        self.assertFalse(final['selection_complete'])

    def test_terminal_shows_threshold_error_reason_attempt_and_exhaustion(self):
        case = self.dataset.cases[0]
        row = self.result(case, status='FAIL', error='API 429 quota exhausted').to_dict()
        row['failure_reasons'] = ['Recall@5 0.20 < 0.50', 'Correctness 0.40 < 0.60']
        row['attempts'] = [{}]
        output = io.StringIO()
        with redirect_stdout(output): runner.print_case_result(case, row)
        for message in ['attempt 1/4', 'Why: Recall@5 0.20 < 0.50', 'Why: Correctness 0.40 < 0.60',
                        'Why: API 429 quota exhausted', 'next explicit rerun']:
            self.assertIn(message, output.getvalue())
        row['attempts'] = [{}, {}, {}, {}]
        with redirect_stdout(output): runner.print_case_result(case, row)
        self.assertIn('Attempt limit reached', output.getvalue())

    def test_unexpected_provider_exception_is_logged_with_reason_and_counts_as_attempt(self):
        with patch.object(evaluator, 'run_case', side_effect=RuntimeError('provider disconnected')), \
             self.assertLogs(runner.logger, level='ERROR'):
            result = self.run_batch()
        self.assertEqual(result['results'][0]['error'], 'RuntimeError: provider disconnected')
        self.assertEqual(result['results'][0]['attempts'][0]['result']['error'], 'RuntimeError: provider disconnected')
        self.assertEqual(result['retry_queue'], ['E001'])

    def test_automatic_retries_are_rejected_before_provider_work(self):
        with patch.object(evaluator, 'run_case') as execute, self.assertRaises(ds.DatasetError):
            runner.run(self.dataset, self.catalog, max_retries=1, output=self.path)
        execute.assert_not_called()

    def test_interrupt_during_judging_retains_completed_answer(self):
        def interrupt(case, catalog, history, result):
            result.answer = 'already generated'
            result.context = 'authorized context'
            result.outcome = 'answer'
            raise KeyboardInterrupt()
        with patch.object(evaluator, '_run_case', side_effect=interrupt):
            result = evaluator.run_case(self.dataset.cases[0], self.catalog)
        self.assertTrue(result.interrupted)
        self.assertEqual(result.answer, 'already generated')
        self.assertEqual(result.status, 'ERROR')

    def test_unexpected_index_api_error_keeps_the_previous_checkpoint(self):
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c)), \
             patch.object(runner, 'verify_index_snapshot', side_effect=[None, None, OSError('index API down')]):
            result = self.run_batch()
        self.assertEqual(result['stop_reason'], 'execution_error')
        self.assertEqual(len(runner.load_results(self.path)['results']), 1)

    def test_code_change_rejects_resume_without_touching_original(self):
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c)):
            self.run_batch()
        original = self.path.read_bytes()
        with patch.object(runner, 'fingerprint', return_value='changed'), self.assertRaises(ds.DatasetError):
            self.run_batch(True)
        self.assertEqual(original, self.path.read_bytes())

    def test_concurrent_command_is_rejected_and_lock_releases(self):
        path = Path(self.folder) / 'lock'
        with run_evaluation.run_lock(path):
            with self.assertRaises(RuntimeError):
                with run_evaluation.run_lock(path): pass
        with run_evaluation.run_lock(path): pass


class SchedulerMigrationContracts(unittest.TestCase):
    def test_only_known_scheduler_upgrade_is_compatible(self):
        dataset, catalog = ds.load(), ds.PermissionCatalog()
        archived = (ds.PACKAGE_ROOT / 'fixtures/runner_before_deferred_retries.py.txt').read_bytes()
        self.assertEqual(hashlib.sha256(archived).hexdigest(), runner.PRE_RETRY_RUNNER_SHA256)
        old_config = {'index_snapshot': {'collection_name': 'offline'}, 'max_retries': 0}
        previous = {'configuration': old_config, 'fingerprint': runner.fingerprint(
            dataset, catalog, old_config, _source_overrides={'evaluation/runner.py': archived})}
        current = {**old_config, 'max_case_attempts': 3}
        with patch.object(runner, 'resolve_index_snapshot', side_effect=AssertionError('must stay offline')):
            self.assertTrue(runner.compatible_scheduler_checkpoint(previous, dataset, catalog, current))
            changed = copy.deepcopy(dataset)
            changed.cases[0] = replace(changed.cases[0], question='changed question')
            self.assertFalse(runner.compatible_scheduler_checkpoint(previous, changed, catalog, current))
            self.assertFalse(runner.compatible_scheduler_checkpoint(previous, dataset, catalog,
                {**current, 'index_snapshot': {'collection_name': 'different'}}))
            with patch.object(runner.config, 'GROQ_MODEL', 'different-model'):
                self.assertFalse(runner.compatible_scheduler_checkpoint(previous, dataset, catalog, current))
            original = runner.fingerprint
            def changed_code(*args, **kwargs):
                kwargs['_source_overrides']['evaluation/evaluator.py'] = b'changed scoring code'
                return original(*args, **kwargs)
            with patch.object(runner, 'fingerprint', side_effect=changed_code):
                self.assertFalse(runner.compatible_scheduler_checkpoint(previous, dataset, catalog, current))


if __name__ == '__main__': unittest.main()
