"""Compare frozen historical results with a new v6 run without rewriting either."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'evaluation/runs/remediation-v12-20261105-targeted.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/remediation_v12_comparison.md')
    args = parser.parse_args()
    frozen = read(ROOT / 'evaluation/fixtures/remediation_v12_saved_contexts.json')
    current = read(args.report)
    if current.get('schema_version') != 6:
        raise ValueError('Comparison requires a fresh v6 report')
    before = {r['eval_id']: r for r in frozen['cases']}
    after = {r['eval_id']: r for r in current['results']}
    changed_expectations = {r['eval_id'] for r in read(ROOT / 'evaluation/fixtures/remediation_v12_review.json')['changes']}
    lines = ['# Remediation v1.2: before / after', '',
             f"Report: `{args.report.name}` · fingerprint `{current['fingerprint']}`", '',
             'Before statuses use the stored v1.1 combined policy. After statuses measure answer/security; '
             'retrieval is a separate diagnostic. Differences are not a controlled causal experiment. '
             'Corrected expectations and scoring changes must not be counted solely as product improvements.', '',
             '| Case | Before (combined) | After (answer/security) | Retrieval | Change context |',
             '|---|---|---|---|---|']
    details = []
    for cid in current['selected_ids']:
        old, new = before[cid], after.get(cid, {})
        context = []
        if cid in changed_expectations:
            context.append('Reviewed expectation changed')
        if (old.get('security') or {}).get('answer_leakage'):
            context.append('Leakage judge corrected')
        if any('recall@' in text for text in old.get('failure_reasons', [])):
            context.append('Retrieval removed from answer gate')
        old_chunks = {c for s in old.get('service_response', {}).get('sources', []) for c in s.get('chunk_ids', [])}
        new_chunks = {c for s in new.get('service_response', {}).get('sources', []) for c in s.get('chunk_ids', [])}
        if new_chunks != old_chunks and new:
            context.append('Selected evidence changed')
        corrections = new.get('security_trace', {}).get('verification', {}).get('corrections', [])
        if corrections:
            context.append(f'{len(corrections)} verifier corrections')
        lines.append(f"| {cid} | {old['status']} | {new.get('status', 'PENDING')} | {new.get('retrieval_status', 'NOT_MEASURED')} | {'; '.join(context) or 'Control / unchanged expectation'} |")
        details.append({'eval_id': cid, 'before_status': old['status'], 'after_status': new.get('status'),
                        'change_context': context, 'before_failure_reasons': old.get('failure_reasons'),
                        'after_failure_reasons': new.get('failure_reasons'), 'error': new.get('error'),
                        'retrieval_failure_reasons': new.get('retrieval_failure_reasons'),
                        'before_quality': old.get('generation'), 'after_quality': new.get('generation'),
                        'added_chunks': sorted(new_chunks - old_chunks), 'dropped_chunks': sorted(old_chunks - new_chunks),
                        'verification_corrections': corrections})
    lines += ['', '## Remaining failures and errors', '']
    failures = [r for r in details if r['after_status'] != 'PASS']
    for row in failures:
        reasons = list(row['after_failure_reasons'] or [])
        if row['error']:
            reasons.append(row['error'])
        lines.append(f"- **{row['eval_id']}**: {'; '.join(reasons) or 'Not yet recorded.'}")
    if not failures:
        lines.append('None in this selected run. This is not a full 90-case evaluation.')
    lines += ['', f"Recorded {len(after)}/{len(current['selected_ids'])} cases. "
              f"Stop reason: {current.get('stop_reason') or 'none'}. No quality failures were retried by this comparison.", '']
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps({
        'baseline_sha256': frozen['baseline_sha256'],
        'report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
        'scoring_policy': current['scoring_policy'], 'cases': details}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
