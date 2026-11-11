"""Offline contracts for Jina rotation, failure handling and checkpoint migration."""
import base64
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
from contextvars import copy_context
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from answering.src import config, jina_keys, reranker, service
from evaluation import dataset as ds, evaluator, reports, runner
from indexing.src import config as index_config, embeddings


KEYS = [('JINA_API_KEY_1', 'offline-secret-jina-first'),
        ('JINA_API_KEY_2', 'offline-secret-jina-second')]


def response(body=None, status=200):
    result = Mock(status_code=status, text='provider unavailable' if status != 200 else '')
    result.json.return_value = body
    return result


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.keys = jina_keys.JinaKeyPool(KEYS)
        self.enterContext(patch.object(jina_keys, 'pool', return_value=self.keys))
        self.enterContext(patch.object(config, 'RERANK_ENABLED', True))
        self.chunk = SimpleNamespace(text='authorized passage')

    def invoke(self, operation):
        if operation == 'embedding':
            return embeddings.embed_query('query')
        return reranker.rerank('query', [self.chunk])

    def test_key_loading_numbered_precedence_blank_duplicate_and_legacy(self):
        self.assertEqual(jina_keys.load_keys({'JINA_API_KEY_1': ' A ', 'JINA_API_KEY_2': 'B', 'JINA_API_KEY': 'C'}),
                         [('JINA_API_KEY_1', 'A'), ('JINA_API_KEY_2', 'B')])
        self.assertEqual(jina_keys.load_keys({'JINA_API_KEY_1': 'A', 'JINA_API_KEY_2': ' A '}), [('JINA_API_KEY_1', 'A')])
        self.assertEqual(jina_keys.load_keys({'JINA_API_KEY_1': ' ', 'JINA_API_KEY_2': 'B'}), [('JINA_API_KEY_2', 'B')])
        self.assertEqual(jina_keys.load_keys({'JINA_API_KEY_1': '', 'JINA_API_KEY': 'C'}), [('JINA_API_KEY', 'C')])
        self.assertEqual(jina_keys.load_keys({}), [])
        with self.assertRaisesRegex(RuntimeError, 'Configure JINA_API_KEY_1'):
            jina_keys.JinaKeyPool([]).acquire()

    def test_startup_accepts_numbered_keys_without_legacy(self):
        env = dict(KEYS, QDRANT_URL='https://offline.invalid', QDRANT_API_KEY='offline')
        with patch.dict(os.environ, env, clear=True), \
             patch.object(config.groq_keys, 'pool', return_value=SimpleNamespace(size=1)), \
             patch.object(Path, 'is_file', return_value=True):
            config.validate_env()
            index_config.validate_env()
        with patch.dict(os.environ, {'QDRANT_URL': 'offline', 'QDRANT_API_KEY': 'offline'}, clear=True), \
             patch.object(config.groq_keys, 'pool', return_value=SimpleNamespace(size=1)):
            for validate in (config.validate_env, index_config.validate_env):
                with self.assertRaisesRegex(RuntimeError, 'JINA_API_KEY_1'):
                    validate()

    def test_embedding_and_reranking_share_one_alternating_sequence(self):
        used = []
        def post(url, **kwargs):
            used.append(kwargs['headers']['Authorization'])
            if url == index_config.JINA_API_URL:
                return response({'data': [{'index': 0, 'embedding': [0.1, 0.2]}]})
            return response({'results': [{'index': 0, 'relevance_score': 0.9}]})
        with patch.object(embeddings.HTTP, 'post', side_effect=post), \
             patch.object(reranker.HTTP, 'post', side_effect=post), \
             jina_keys.evaluation_requests() as state:
            for operation in ('embedding', 'reranking', 'embedding', 'reranking'):
                self.invoke(operation)
        self.assertEqual(used, [f'Bearer {KEYS[i % 2][1]}' for i in range(4)])
        self.assertEqual([c['sequence'] for c in state.calls], [1, 2, 3, 4])
        self.assertTrue(all(c['outcome'] == 'success' for c in state.calls))
        encoded = json.dumps(state.calls)
        for _, value in KEYS: self.assertNotIn(value, encoded)
        self.assertNotIn('Authorization', encoded)

    def test_concurrent_allocation_is_alternating_in_sequence_order(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            allocated = list(workers.map(lambda _: self.keys.acquire(), range(40)))
        allocated.sort(key=lambda item: item[2])
        self.assertEqual([i[2] for i in allocated], list(range(1, 41)))
        self.assertEqual([i[0] for i in allocated], [KEYS[i % 2][0] for i in range(40)])

    def test_http_failures_stop_without_retry_or_failover_in_both_stages(self):
        for stage in ('embedding', 'reranking'):
            for status in (400, 401, 402, 403, 429, 503):
                with self.subTest(stage=stage, status=status):
                    self.keys = jina_keys.JinaKeyPool(KEYS)
                    with patch.object(jina_keys, 'pool', return_value=self.keys), \
                         patch.object(embeddings.HTTP if stage == 'embedding' else reranker.HTTP,
                                      'post', return_value=response(status=status)) as post, \
                         patch.object(embeddings.time, 'sleep') as sleep, \
                         jina_keys.evaluation_requests() as state:
                        with self.assertRaisesRegex(jina_keys.JinaRequestError, 'Check/change JINA_API_KEY_1'):
                            self.invoke(stage)
                        with self.assertRaises(jina_keys.JinaRequestError):
                            self.invoke(stage)
                    self.assertEqual(post.call_count, 1)
                    sleep.assert_not_called()
                    self.assertEqual(len(state.calls), 1)
                    self.assertEqual(state.calls[0]['http_status'], status)

    def test_transport_failure_is_identified_and_secret_free(self):
        for stage in ('embedding', 'reranking'):
            with self.subTest(stage=stage), jina_keys.evaluation_requests() as state, \
                 patch.object(embeddings.HTTP if stage == 'embedding' else reranker.HTTP, 'post',
                              side_effect=requests.ConnectTimeout('handshake timed out ' + KEYS[1][1])) as post:
                with self.assertRaisesRegex(jina_keys.JinaRequestError, 'ConnectTimeout') as error:
                    self.invoke(stage)
                self.assertNotIn(KEYS[1][1], str(error.exception))
            self.assertEqual(post.call_count, 1)
            self.assertIsNone(state.calls[0]['http_status'])

    def test_malformed_responses_are_logged_and_never_retried(self):
        invalid = [
            ('embedding', {'data': []}),
            ('embedding', {'data': [{'index': 1, 'embedding': [0.1]}]}),
            ('embedding', {'data': [{'index': 0, 'embedding': [float('nan')]}]}),
            ('embedding', {'data': [{'index': 0, 'embedding': [True]}]}),
            ('reranking', {'results': []}),
            ('reranking', {'results': [{'index': 0, 'relevance_score': float('nan')}]}),
        ]
        for stage, body in invalid:
            with self.subTest(stage=stage, body=body), jina_keys.evaluation_requests() as state, \
                 patch.object(embeddings.HTTP if stage == 'embedding' else reranker.HTTP,
                              'post', return_value=response(body)) as post:
                with self.assertRaises(jina_keys.JinaRequestError): self.invoke(stage)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(state.calls[0]['outcome'], 'error')
            self.assertEqual(state.calls[0]['http_status'], 200)

    def test_failure_blocks_later_worker_calls_in_copied_context(self):
        with jina_keys.evaluation_requests() as state, ThreadPoolExecutor(max_workers=1) as workers:
            def fail():
                with jina_keys.request('embedding'):
                    raise requests.Timeout('offline')
            failed = workers.submit(copy_context().run, fail)
            with self.assertRaises(jina_keys.JinaRequestError): failed.result()
            post = Mock()
            def later():
                with jina_keys.request('reranking'): post()
            pending = workers.submit(copy_context().run, later)
            with self.assertRaises(jina_keys.JinaRequestError): pending.result()
            post.assert_not_called()
            self.assertEqual(len(state.calls), 1)
        with jina_keys.evaluation_requests() as fresh, jina_keys.request('embedding'):
            pass
        self.assertEqual(fresh.calls[0]['key_label'], 'JINA_API_KEY_2')

    def test_ordinary_app_fallback_does_not_disable_or_skip_failed_key(self):
        used = []
        def post(*_, **kwargs):
            used.append(kwargs['headers']['Authorization'])
            return response(status=401)
        with patch.object(reranker.HTTP, 'post', side_effect=post), self.assertLogs(reranker.logger):
            for _ in range(3):
                result = self.invoke('reranking')
                self.assertFalse(result.applied)
                self.assertEqual(result.items, [(self.chunk, None)])
        self.assertEqual(used, [f'Bearer {KEYS[i % 2][1]}' for i in range(3)])

    def test_failed_case_stops_runner_saves_diagnostics_and_never_generates_or_judges(self):
        case = ds.EvalCase('E001', 'U-001', 'query', 'authorized_factual', 'answer', '')
        dataset = ds.GoldenDataset('offline', '1.1', '', [case, replace(case, eval_id='E002')])
        def prepare(*args, **kwargs):
            embeddings.embed_query('query')  # key 1 succeeds
            reranker.rerank('query', [self.chunk])  # key 2 fails
            raise AssertionError('must stop before generation')
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(ds, 'PROJECT_ROOT', Path(folder)), \
             patch.object(ds, 'validate', return_value=[]), \
             patch.object(runner, 'resolve_index_snapshot', return_value={'collection_name': 'offline'}), \
             patch.object(runner, 'fingerprint', return_value='same'), \
             patch.object(runner.index_retrieval, 'pin_revision', return_value=nullcontext()), \
             patch.object(service, '_prepare', side_effect=prepare), \
             patch.object(embeddings.HTTP, 'post', return_value=response({'data': [{'index': 0, 'embedding': [0.1]}]})), \
             patch.object(reranker.HTTP, 'post', return_value=response(status=402)) as rerank_post, \
             patch.object(service.llm, 'generate') as generate, \
             patch.object(evaluator.judge, 'grade') as grade, \
             patch.object(evaluator.judge, 'leakage') as leakage, \
             self.assertLogs(evaluator.logger, level='ERROR'):
            path = Path(folder) / 'checkpoint.json'
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                result = runner.run(dataset, SimpleNamespace(documents=[]), delay_s=0,
                                    retry_wait_s=0, max_retries=0, output=path)
            self.assertEqual(len(result['results']), 1)
            self.assertEqual(result['results'][0]['status'], 'ERROR')
            self.assertFalse(result['selection_complete'])
            self.assertEqual(len(result['results'][0]['attempts']), 1)
            self.assertIn('Check/change JINA_API_KEY_2', terminal.getvalue())
            self.assertIn('attempt 1/4', terminal.getvalue())
            saved = runner.load_results(path)
            calls = saved['results'][0]['attempts'][0]['result']['jina_calls']
            self.assertEqual([c['key_label'] for c in calls], ['JINA_API_KEY_1', 'JINA_API_KEY_2'])
            logs = (Path(folder) / result['response_log']).read_text(encoding='utf-8')
            self.assertIn('JINA_API_KEY_2', logs)
            for _, value in KEYS: self.assertNotIn(value, logs)
            self.assertNotIn('jina_calls', json.dumps(reports.public_summary(result)))
        generate.assert_not_called()
        grade.assert_not_called()
        leakage.assert_not_called()
        self.assertEqual(rerank_post.call_count, 1)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.catalog = ds.load(), ds.PermissionCatalog()
        self.raw_archive = (ds.PACKAGE_ROOT / 'fixtures/before_jina_rotation.json').read_bytes()
        self.assertEqual(hashlib.sha256(self.raw_archive).hexdigest(), runner.PRE_JINA_ROTATION_SHA256)
        archive = json.loads(self.raw_archive)
        self.overrides = {k: base64.b64decode(v) for k, v in archive['sources'].items()}
        self.added = archive['added']
        self.old_config = {'index_snapshot': {'collection_name': 'offline'}, 'max_case_attempts': 3}
        self.previous = {'configuration': self.old_config, 'fingerprint': runner.fingerprint(
            self.dataset, self.catalog, self.old_config, _source_overrides=self.overrides, _source_excludes=self.added)}
        self.current = {**self.old_config, 'max_case_attempts': 4, 'jina_request_policy': runner.JINA_REQUEST_POLICY}

    def compatible(self, dataset=None, current=None):
        return runner.compatible_jina_checkpoint(self.previous, dataset or self.dataset,
                                                self.catalog, current or self.current)

    def test_historical_upgrade_is_incompatible_with_v6(self):
        with patch.object(runner, 'resolve_index_snapshot', side_effect=AssertionError('offline only')):
            self.assertFalse(self.compatible())
            changed = copy.deepcopy(self.dataset)
            changed.cases[0] = replace(changed.cases[0], question='changed')
            self.assertFalse(self.compatible(changed))
            self.assertFalse(self.compatible(current={**self.current, 'index_snapshot': {'collection_name': 'changed'}}))
            with patch.object(index_config, 'JINA_EMBEDDING_MODEL', 'different'):
                self.assertFalse(self.compatible())
            with patch.object(evaluator, 'MIN_RECALL', .9):
                self.assertFalse(self.compatible())

    def test_new_scoring_code_or_tampered_archive_is_rejected(self):
        original = Path.read_bytes
        def read(path):
            value = original(path)
            return value + b'\n# unrelated change' if path == ds.PROJECT_ROOT / 'evaluation/evaluator.py' else value
        with patch.object(Path, 'read_bytes', read): self.assertFalse(self.compatible())
        def tamper(path):
            value = original(path)
            return value + b' ' if path.name == 'before_jina_rotation.json' else value
        with patch.object(Path, 'read_bytes', tamper): self.assertFalse(self.compatible())

    def test_credential_value_changes_do_not_change_signature(self):
        first = runner.fingerprint(self.dataset, self.catalog, self.current)
        with patch.dict(os.environ, {'JINA_API_KEY_1': 'replacement-1', 'JINA_API_KEY_2': 'replacement-2'}):
            self.assertEqual(first, runner.fingerprint(self.dataset, self.catalog, self.current))

    def test_resume_rejects_historical_migration_without_writing(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runner, 'resolve_index_snapshot', return_value={'collection_name': 'offline'}), \
             patch.object(runner.index_retrieval, 'pin_revision', return_value=nullcontext()), \
             patch.object(runner, 'save_checkpoint', side_effect=runner.save):
            path = Path(folder) / 'checkpoint.json'
            controls = dict(dataset=self.dataset, catalog=self.catalog, only=['E001', 'E002'],
                            delay_s=0, retry_wait_s=0, max_retries=0, verbose=False, output=path)
            previous = runner.run(**controls, checkpoint=False, initialize_only=True)
            previous['configuration'].pop('jina_request_policy')
            previous['configuration']['max_case_attempts'] = 3
            previous['fingerprint'] = runner.fingerprint(self.dataset, self.catalog,
                previous['configuration'], _source_overrides=self.overrides, _source_excludes=self.added)
            previous['results'] = []
            for case_id, count, status in [('E001', 3, 'ERROR'), ('E002', 1, 'PASS')]:
                case = self.dataset.by_id(case_id)
                row = evaluator.CaseResult(case.eval_id, case.user_id, case.user_name,
                    case.question, case.category, case.expected_behavior, status=status,
                    error='previous embedding failure' if status == 'ERROR' else None).to_dict()
                attempts = [{'attempt': i + 1, 'fingerprint': previous['fingerprint'],
                             'configuration': copy.deepcopy(previous['configuration']),
                             'started_at': '2026-11-04T00:00:00Z', 'finished_at': '2026-11-04T00:00:01Z',
                             'result': copy.deepcopy(row)} for i in range(count)]
                row['attempts'] = attempts
                previous['results'].append(row)
            runner.save(previous, path)
            def success(case, *_):
                return evaluator.CaseResult(case.eval_id, case.user_id, case.user_name,
                    case.question, case.category, case.expected_behavior, status='PASS',
                    outcome='answer', answered=True, answer='offline answer')
            before = path.read_bytes()
            with patch.object(evaluator, 'run_case', side_effect=success) as execute:
                with self.assertRaises(ds.DatasetError):
                    runner.run(**controls, resume=True)
            execute.assert_not_called()
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
