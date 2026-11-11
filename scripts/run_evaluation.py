"""Run the pilot then the full baseline. Repeat the same command to resume.

Ctrl+C pauses safely. Fresh cases run before earlier failures on each explicit
restart, up to four total attempts per case. All attempts remain in the logs.
--plan is offline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import dataset as ds, runner
from answering.src import config

PILOT_IDS = ['E001', 'E002', 'E013', 'E022', 'E027', 'E033', 'E035', 'E043',
             'E049', 'E053', 'E056', 'E061', 'E073', 'E079', 'E083', 'E087']


@contextmanager
def run_lock(path):
    """An OS lock is released even if the terminal or process is killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This evaluation is already running in another terminal.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def execute_targeted(name, only, plan=False):
    dataset, catalog = ds.load(), ds.PermissionCatalog()
    problems = ds.validate(dataset, catalog)
    if problems:
        raise ds.DatasetError('\n'.join(problems))
    fixture = json.loads((ds.PACKAGE_ROOT / 'fixtures/remediation_v12_saved_contexts.json').read_text(encoding='utf-8'))
    requested = only.split(',') if only else fixture['targeted_ids']
    cases = runner.select_cases(dataset, only=requested)
    ids = [case.eval_id for case in cases]
    if len(cases) == len(dataset):
        raise ds.DatasetError('The targeted phase must select fewer than all cases.')
    path = ds.PACKAGE_ROOT / 'runs' / f'{name}-targeted.json'
    print(f'Targeted selection with prerequisite closure ({len(ids)}): {", ".join(ids)}', flush=True)
    print(f'Output: {path}\n90-second pacing; scored quality failures are preserved without retries.', flush=True)
    if plan:
        if path.exists():
            previous = runner.load_results(path)
            controls = previous['configuration']
            manifest = runner.input_manifest(dataset, catalog, controls)
            if previous.get('selected_ids') != ids or previous.get('fingerprint') != runner.fingerprint(dataset, catalog, controls):
                raise ds.DatasetError('Checkpoint mismatch: ' + ', '.join(runner.changed_components(previous, manifest)))
            print('Resume pending/error cases only; retain completed PASS and FAIL rows.')
        print('Offline preview complete. No provider calls made.')
        return 0
    config.validate_env()
    with run_lock(ROOT / '.cache/evaluation.lock'):
        result = runner.run(dataset, catalog, only=ids, delay_s=90, retry_wait_s=90,
                            max_retries=0, resume=path.exists(), output=path, retry_quality=False)
    print(f"Targeted result: {result['summary']['passed']} PASS, {result['summary']['failed']} FAIL, {result['summary']['errors']} ERROR", flush=True)
    return 2 if result.get('stop_reason') or result['summary']['errors'] else 1 if result['summary']['failed'] else 0


def execute(name: str, *, plan: bool = False, phase: str = 'all', targeted_retry: bool = False, only=None) -> int:
    if phase == 'targeted':
        if targeted_retry:
            raise ds.DatasetError('Quality retries are disabled for targeted validation.')
        return execute_targeted(name, only, plan)
    if only:
        raise ds.DatasetError('--only requires --phase targeted')
    if targeted_retry and phase != 'full':
        raise ds.DatasetError('--retry-errors-and-low-attempt-failures requires --phase full')
    dataset = ds.load()
    catalog = ds.PermissionCatalog()
    problems = ds.validate(dataset, catalog)
    if problems:
        raise ds.DatasetError('\n'.join(problems))
    paths = [(stage, ROOT / 'evaluation' / 'runs' / f'{name}-{stage}.json', only)
             for stage, only in [('pilot', PILOT_IDS), ('full', None)]
             if phase == 'all' or stage == phase]
    if plan:
        for phase, path, only in paths:
            cases = runner.select_cases(dataset, only=only)
            previous = runner.load_results(path) if path.exists() else None
            if targeted_retry and not previous:
                raise ds.DatasetError('Targeted retries require an existing full report')
            if previous:
                if previous.get('selected_ids') != [case.eval_id for case in cases]:
                    raise ds.DatasetError('The saved report has a different case selection.')
                limits = runner.attempt_limits(previous)
                controls = {**previous['configuration'], 'groq_request_policy': runner.GROQ_REQUEST_POLICY,
                            'case_attempt_limits': limits}
                signature = runner.fingerprint(dataset, catalog, controls)
                if previous.get('schema_version') != runner.SCHEMA_VERSION or previous['fingerprint'] != signature:
                    raise ds.DatasetError('The saved report does not match the current evaluation inputs.')
                rows = {row['eval_id']: row for row in previous['results']}
                batch = runner.targeted_retry_plan(previous) if targeted_retry else None
                if batch:
                    limits = {**limits, **batch['attempt_caps']}
                pending = [case.eval_id for case in runner.execution_order(cases, rows, limits)
                           if (not batch or case.eval_id in batch['attempt_caps'])
                           and runner.attempt_count(rows.get(case.eval_id, {})) < runner.attempt_limit(case.eval_id, limits)
                           and (not runner.case_finished(rows.get(case.eval_id, {}))
                                or runner.retry_candidate(rows.get(case.eval_id, {})))]
                print(f'{phase}: keep all {len(cases)} cases in {path}')
                print(f'Retry once this invocation ({len(pending)}): {", ".join(pending) or "none"}')
                print(f'Preserve {sum(row.get("status") == "PASS" for row in rows.values())} passed cases.')
                if batch:
                    print('Error cases: ' + (', '.join(batch['error_ids']) or 'none'))
                    print('FAIL cases with <=2 attempts: ' + (', '.join(batch['failure_ids']) or 'none'))
                    print('One additional attempt per selected case, including exhausted errors; '
                          'repeating this command resumes the same frozen batch.')
                elif limits:
                    print('One final fifth attempt only for: ' + ', '.join(limits) + '. All other cases stay capped at four.')
                print(f'Existing response log: {previous["response_log"]}')
            else:
                print(f'{phase}: {len(cases)} cases; {max(0, len(cases)-1)*1.5:.1f} minutes pacing; {path}')
        if not targeted_retry:
            print(f'Each restart: one attempt per queued case; default cap {runner.MAX_CASE_ATTEMPTS}, with only the listed report-specific exceptions.')
        print('Offline validation passed. No provider or index calls made.')
        return 0
    config.validate_env()
    with run_lock(ROOT / '.cache' / 'evaluation.lock'):
        for phase, path, only in paths:
            exists = path.exists()
            if targeted_retry and not exists:
                raise ds.DatasetError('Targeted retries require an existing full report')
            print(f'{"Resuming" if exists else "Starting"} {phase}: {path}', flush=True)
            if targeted_retry:
                batch = runner.targeted_retry_plan(runner.load_results(path))
                print('Targeted cases: ' + (', '.join(batch['attempt_caps']) or 'none'), flush=True)
                print('One additional attempt per selected case. Ctrl+C pauses; repeat this command '
                      'to continue the remaining cases in this batch. All history stays in the full report.', flush=True)
            else:
                print('Ctrl+C once to pause. Repeat this command to resume: fresh cases first, '
                      f'previous failures at the end; default maximum {runner.MAX_CASE_ATTEMPTS} total attempts per case.', flush=True)
            if exists and not targeted_retry:
                limits = runner.attempt_limits(runner.load_results(path))
                if limits:
                    print('One final fifth attempt is permitted only for: ' + ', '.join(limits) + '.', flush=True)
            print('Jina requests alternate between configured keys. A failed Jina request pauses '
                  'evaluation; check/change the key named in the error, then rerun.', flush=True)
            print(f'Groq requests rotate across {config.groq_keys.pool().size} unique configured keys; '
                  'rate-limited keys wait for provider cooldowns. Keys in the same organization may share daily quota.', flush=True)
            result = runner.run(dataset, catalog, only=only, delay_s=90,
                                retry_wait_s=90, max_retries=0, resume=exists, output=path,
                                targeted_retry=targeted_retry)
            print(f'Report: {path}\nResponse log: {result["response_log"]}', flush=True)
            print(f'Queued for next restart: {len(result["retry_queue"])}; '
                  f'attempt limit reached: {len(result["exhausted_cases"])}.', flush=True)
            if not result['selection_complete'] or result.get('stop_reason'):
                print(f'Paused: {result.get("stop_reason") or "incomplete dependencies"}. '
                      'Inspect the failure reasons and logs before restarting.', flush=True)
                if result.get('stop_detail'):
                    print(f'Why: {result["stop_detail"]}', flush=True)
                if result['exhausted_cases']:
                    print('Cases at their attempt limit remain unresolved and cannot be '
                          'retried in this report.', flush=True)
                return 130 if result.get('stop_reason') == 'interrupted' else 2
            print(f'{phase.title()} complete: {result["summary"]["passed"]} passed, '
                  f'{result["summary"]["failed"]} quality/security failures. Saved scores are retained.', flush=True)
        print('Full baseline complete. Review the saved failures and response logs.', flush=True)
        return 1 if result['summary']['failed'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='remediation-v12-20261105',
                        help='Stable checkpoint name; keep it unchanged to resume, use a new name for a new baseline')
    parser.add_argument('--plan', action='store_true', help='offline validation and preview only')
    parser.add_argument('--only', help='comma-separated case IDs for targeted validation; includes prerequisite closure')
    parser.add_argument('--phase', choices=('all', 'pilot', 'full', 'targeted'), default='targeted',
                        help='Use full to resume the existing 90-case report directly, without visiting the pilot')
    parser.add_argument('--retry-errors-and-low-attempt-failures', action='store_true',
                        help='With --phase full: one frozen extra attempt for every error and FAIL with <=2 total attempts; keep the full report and logs')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', args.name):
        parser.error('--name must be a simple filename, without a path')
    logging.basicConfig(level=logging.ERROR)
    try:
        return execute(args.name, plan=args.plan, phase=args.phase,
                       targeted_retry=args.retry_errors_and_low_attempt_failures, only=args.only)
    except KeyboardInterrupt:
        print('Paused. Run the same command to resume.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Evaluation could not continue: {exc}', file=sys.stderr)
        print('Existing checkpoints are preserved. Run the same command after resolving the issue.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
