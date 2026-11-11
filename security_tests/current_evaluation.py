"""Offline contracts for publishing and displaying latest case outcomes."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import dataset as ds, evaluator, reports
from evaluation.artifacts import matches_sha256
from scripts.publish_evaluation_snapshot import build_snapshot


class CurrentEvaluationContracts(unittest.TestCase):
    def test_frozen_hash_allows_git_line_endings_but_rejects_content_edits(self):
        lf = b'{"status": "FAIL"}\n'
        crlf = lf.replace(b'\n', b'\r\n')
        for original in (lf, crlf):
            digest = hashlib.sha256(original).hexdigest()
            self.assertTrue(matches_sha256(lf, digest))
            self.assertTrue(matches_sha256(crlf, digest))
            self.assertFalse(matches_sha256(lf.replace(b'FAIL', b'PASS'), digest))

    def test_default_snapshot_matches_original_source_rows_and_current_totals(self):
        payload = reports.public_payload()
        self.assertEqual(payload['report_id'], 'runs/latest-cases-20261111.json')
        self.assertEqual(payload['report_kind'], 'latest_by_case')
        self.assertEqual(payload['source_run_count'], 3)
        counts = payload['measurements']['reliability']
        self.assertEqual([counts[k] for k in ('selected', 'recorded', 'passed', 'failed', 'errors', 'pending')],
                         [23, 23, 14, 9, 0, 0])
        snapshot = reports.load(payload['report_id'])
        sources = {}
        for source in snapshot['snapshot_sources']:
            path = reports.resolve(source['report_id'])
            self.assertTrue(matches_sha256(path.read_bytes(), source['sha256']))
            sources[source['report_id']] = {r['eval_id']: r for r in reports.load(source['report_id'])['results']}
        for row in snapshot['results']:
            origin = snapshot['case_provenance'][row['eval_id']]
            self.assertEqual(row, sources[origin['report_id']][row['eval_id']])
        public = json.dumps(payload)
        for private in ('case_provenance', 'snapshot_sources', 'judge_calls', 'conversation_history', 'expected_facts'):
            self.assertNotIn(private, public)
        self.assertEqual(reports.public_payload('runs/baseline-v11-20261104-full.json')['selected_cases'], 90)

    def source(self, finished, status='PASS'):
        row = evaluator.CaseResult('E001', 'U001', 'User', 'PRIVATE QUESTION', 'authorized_factual',
                                   'answer', status=status, outcome='answer', answered=True,
                                   error='provider failed' if status == 'ERROR' else None).to_dict()
        row['attempts'] = [{'finished_at': finished, 'result': copy.deepcopy(row)}]
        return {'schema_version': 6, 'scoring_policy': 'answer_security_v6_retrieval_diagnostic',
                'run_id': finished, 'generated_at': finished, 'dataset': {'version': '1.2'},
                'model': 'model', 'selected_ids': ['E001'], 'progress': {'phase': 'finished'},
                'case_specs': [{'eval_id': 'E001', 'category': 'authorized_factual', 'expected_behavior': 'answer'}],
                'results': [row]}

    def test_newest_outcome_wins_even_if_failure_or_error_and_older_report_saved_later(self):
        for status in ('FAIL', 'ERROR'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'runs').mkdir()
                old = self.source('2026-11-06T00:00:00Z')
                old['generated_at'] = '2026-11-12T00:00:00Z'
                new = self.source('2026-11-11T00:00:00Z', status)
                for name, document in [('old', old), ('new', new)]:
                    (root / f'runs/{name}.json').write_text(json.dumps(document), encoding='utf-8')
                with patch.object(ds, 'PACKAGE_ROOT', root):
                    snapshot = build_snapshot(['runs/new.json', 'runs/old.json'], 'snapshot')
                self.assertEqual(snapshot['results'][0], new['results'][0])
                self.assertEqual(snapshot['selection_complete'], status != 'ERROR')
                self.assertEqual(snapshot['case_provenance']['E001']['report_id'], 'runs/new.json')
                self.assertNotIn('fingerprint', snapshot)  # Combined snapshots cannot resume.

    def test_rejects_mixed_policy_active_runs_and_unfinished_attempts(self):
        base = self.source('2026-11-06T00:00:00Z')
        invalid = [dict(base, schema_version=5), dict(base, progress={'phase': 'evaluating'}),
                   dict(base, dataset={'version': '1.1'})]
        interrupted = copy.deepcopy(base)
        interrupted['results'][0]['attempts'][0]['finished_at'] = None
        invalid.append(interrupted)
        for bad in invalid:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'runs').mkdir()
                for name, document in [('base', base), ('bad', bad)]:
                    (root / f'runs/{name}.json').write_text(json.dumps(document), encoding='utf-8')
                with patch.object(ds, 'PACKAGE_ROOT', root), self.assertRaises(ValueError):
                    build_snapshot(['runs/base.json', 'runs/bad.json'], 'snapshot')


if __name__ == '__main__':
    unittest.main()
