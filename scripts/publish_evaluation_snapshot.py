"""Save the latest recorded outcome per case, with its original run provenance.

This reads existing reports only. It neither executes nor re-scores cases, and
the resulting combined snapshot cannot be resumed as an evaluation run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import metrics, reports, runner


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Source timestamps must include a timezone')
    return parsed


def build_snapshot(source_ids, name, generated_at=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', name):
        raise ValueError('Use a simple snapshot name')
    if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
        raise ValueError('Choose at least two distinct source reports')
    generated_at = generated_at or datetime.now(timezone.utc).isoformat(timespec='seconds')
    timestamp(generated_at)
    sources, selected, specs, rows, provenance, dates = [], [], {}, {}, {}, {}
    dataset = policy = None
    models = set()
    for source_id in source_ids:
        document = reports.load(source_id)
        if (document.get('schema_version') != 6 or document.get('report_kind') == 'latest_by_case'
                or document.get('scoring_policy') != 'answer_security_v6_retrieval_diagnostic'):
            raise ValueError('Snapshots require original v6 answer/security reports')
        if dataset is None:
            dataset, policy = document['dataset'], document['scoring_policy']
        if document['dataset'] != dataset:
            raise ValueError('Source datasets differ')
        if (document.get('progress') or {}).get('phase') in ('starting', 'waiting', 'evaluating'):
            raise ValueError('Wait for active source reports to stop before publishing')
        sources.append({'report_id': source_id, 'run_id': document['run_id'],
                        'fingerprint': document.get('fingerprint'),
                        'sha256': hashlib.sha256(reports.resolve(source_id).read_bytes()).hexdigest()})
        models.add(document.get('model'))
        for spec in document['case_specs']:
            cid = spec['eval_id']
            if cid in specs and specs[cid] != spec:
                raise ValueError(f'Case definition changed across reports: {cid}')
            if cid not in specs:
                selected.append(cid)
                specs[cid] = copy.deepcopy(spec)
        for row in document['results']:
            cid = row['eval_id']
            if cid not in specs:
                raise ValueError(f'Result lacks a case definition: {cid}')
            attempts = row.get('attempts') or []
            if attempts and not attempts[-1].get('finished_at'):
                raise ValueError(f'Unfinished attempt: {cid}')
            recorded_at = (attempts[-1]['finished_at'] if attempts else document['generated_at'])
            recorded = timestamp(recorded_at)
            if cid in dates and recorded == dates[cid] and rows[cid] != row:
                raise ValueError(f'Conflicting outcomes at the same time: {cid}')
            if cid not in dates or recorded > dates[cid]:
                # Select by time, never by score. FAIL and ERROR replace older PASS.
                rows[cid], dates[cid] = copy.deepcopy(row), recorded
                provenance[cid] = {'report_id': source_id, 'run_id': document['run_id'],
                                   'recorded_at': recorded_at, 'attempt': len(attempts)}
    selected.sort()
    results = [rows[cid] for cid in selected if cid in rows]
    complete = len(results) == len(selected) and all(
        row.get('status') in ('PASS', 'FAIL') and not row.get('error') for row in results)
    return {'schema_version': 6, 'report_kind': 'latest_by_case', 'run_id': name,
            'started_at': generated_at, 'generated_at': generated_at,
            'scoring_policy': policy, 'status_semantics': 'answer_security',
            'selection_complete': complete, 'partial': True, 'stop_reason': None,
            'progress': {'phase': 'finished', 'case_id': None, 'updated_at': generated_at},
            'dataset': copy.deepcopy(dataset), 'model': next(iter(models)) if len(models) == 1 else 'Multiple models',
            'selected_ids': selected, 'case_specs': [specs[cid] for cid in selected],
            'snapshot_sources': sources, 'case_provenance': provenance,
            'snapshot_note': 'Latest recorded outcome per case across saved runs; original code, evidence, and conversation contexts are retained. Not a new full-suite evaluation.',
            'summary': metrics.aggregate(results), 'measurements': metrics.measurements(results, list(specs.values())),
            'matrix': [runner._matrix_row(row) for row in results], 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', action='append', required=True, help='Report ID, e.g. runs/example.json')
    args = parser.parse_args()
    output = reports.resolve(f'runs/{args.name}.json')
    if output.exists():
        parser.error('Snapshot exists; use a new name to preserve published history')
    document = build_snapshot(args.source, args.name)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = reports.public_summary(document)['measurements']['reliability']
    print(f"Saved {output}: {summary['passed']} PASS, {summary['failed']} FAIL, {summary['errors']} ERROR")


if __name__ == '__main__':
    main()
