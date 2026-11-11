import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { evaluation } from './api';
import type { Counts, EvaluationSummary, Measure, Measurements } from './evaluationData';
import { evaluationState } from './evaluationData';
import Icon from './Icons';
import './evaluation.css';

const labels: Record<string, string> = {
  recall_at_k: 'Recall@5', precision_at_k: 'Precision@5', mrr: 'MRR', coverage: 'Context coverage', precision: 'Context precision',
  groundedness: 'Groundedness', answer_correctness: 'Correctness', answer_relevancy: 'Relevancy', fact_coverage: 'Expected-fact coverage',
  passed: 'Security pass rate', raw_candidate_exposure: 'Raw candidate exposure', postfilter_exposure: 'Post-filter exposure', context_exposure: 'Context exposure',
  citation_exposure: 'Citation exposure', restricted_fact_exposure: 'Restricted-fact exposure', answer_leakage: 'Answer leakage', behavior_match: 'Behavior match',
  refusal_match: 'Refusal match', prompt_injection_blocked: 'Prompt injection blocked',
};
const title = (value: string) => labels[value] || value.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
const savedTime = (value: string) => Number.isNaN(Date.parse(value)) ? value : new Date(value).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
export function metricText(value?: Measure) {
  if (!value) return 'Not measured';
  if (value.applicable === 0) return 'Not applicable';
  return value.value === null ? 'Not measured' : `${(value.value * 100).toFixed(1)}%`;
}
function MetricList({ values }: { values: Record<string, Measure> }) {
  return <dl className="metric-list">{Object.entries(values).map(([key, value]) => <div key={key}>
    <dt>{title(key)}<small>{value.measured}/{value.applicable} applicable cases measured · {value.total} selected</small></dt><dd><strong>{metricText(value)}</strong></dd>
  </div>)}</dl>;
}
function Meter({ value }: { value: number | null }) {
  return <span className={`eval-meter ${value === null ? 'unmeasured' : ''}`} aria-hidden="true"><span style={{ width: `${Math.min(100, Math.max(0, (value ?? 0) * 100))}%` }} /></span>;
}
function Table({ name, rows, total }: { name: string; rows: Record<string, Counts>; total?: Counts }) {
  const entries = Object.entries(rows);
  return <section className="panel eval-table">
    <div className="eval-table-heading"><div className="eval-heading-group"><span className="eval-section-icon"><Icon name="grid" /></span><div><h2>{name}</h2>
      <p className="caption">{name === 'Capabilities' ? 'How each capability performs across the selected cases.' : 'Outcomes across authorized requests, refusals, and safety checks.'}</p></div></div><span className="eval-group-count">{entries.length} groups</span></div>
    {entries.length ? <><div className="table-scroll" tabIndex={0} role="region" aria-label={name}>
      <table><caption className="sr-only">{name}. Pass rate uses recorded cases; pending cases are not yet recorded.</caption><thead><tr><th scope="col">Group</th><th scope="col">Recorded</th><th scope="col">Pass rate</th><th scope="col">Passed</th><th scope="col">Failed</th><th scope="col">Errors</th><th scope="col">Pending</th></tr></thead><tbody>
        {entries.map(([key, row], index) => <tr key={key}><th scope="row"><div className="matrix-group"><span className="matrix-index" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span><div>{title(key)}<small>{row.selected} selected cases</small></div></div></th>
          <td className="matrix-recorded"><strong>{row.recorded}</strong><span> / {row.selected}</span></td>
          <td><div className="matrix-rate"><strong>{row.recorded ? `${(row.passed / row.recorded * 100).toFixed(1)}%` : 'Not measured'}</strong><Meter value={row.recorded ? row.passed / row.recorded : null} /></div></td>
          {(['passed', 'failed', 'errors', 'pending'] as const).map(outcome => <td className="matrix-outcome" key={outcome}><span className={`matrix-count ${row[outcome] ? outcome : 'zero'}`}>{row[outcome]}</span></td>)}</tr>)}
      </tbody>{total && <tfoot><tr><th scope="row">All selected cases</th><td>{total.recorded} / {total.selected}</td>
        <td>{total.recorded ? `${(total.passed / total.recorded * 100).toFixed(1)}%` : 'Not measured'}</td>
        {(['passed', 'failed', 'errors', 'pending'] as const).map(key => <td className="matrix-outcome" key={key}>{total[key]}</td>)}
      </tr></tfoot>}</table></div><div className="eval-table-foot"><span><i className="matrix-legend" aria-hidden="true" />Pass rate is based on recorded cases.</span><span>{name === 'Capabilities' ? 'A case can belong to multiple capabilities.' : 'Pending cases have no recorded outcome yet.'}</span></div></>
      : <p className="eval-empty caption">Not measured in this report.</p>}
  </section>;
}
function Cohorts({ data }: { data: NonNullable<EvaluationSummary['cohorts']> }) {
  return <section className="panel eval-table eval-cohorts"><div className="eval-table-heading"><div className="eval-heading-group"><span className="eval-section-icon"><Icon name="chart" /></span><div><h2>Dataset cohorts</h2><p className="caption">The original dataset and its expanded selection, side by side.</p></div></div></div>
    <div className="table-scroll" tabIndex={0} role="region" aria-label="Dataset cohorts"><table><thead><tr><th scope="col">Cohort</th><th scope="col">Recorded</th><th scope="col">Passed</th><th scope="col">Recall@5</th><th scope="col">Correctness</th></tr></thead><tbody>
      {Object.entries(data).map(([key, value]) => <tr key={key}><th scope="row">{key === 'original_60' ? 'Original 60' : 'Expanded selection'}</th><td>{value.reliability.recorded}/{value.reliability.selected}</td><td>{value.reliability.passed}/{value.reliability.recorded}</td>
        {[value.retrieval.recall_at_k, value.quality.answer_correctness].map((m, index) => <td key={index}>{metricText(m)}<small>{m ? `${m.measured}/${m.applicable} measured` : '0 measured'}</small></td>)}</tr>)}
    </tbody></table></div><p className="eval-table-foot">The original cohort includes the documented E049 retention-policy correction.</p></section>;
}
function Scorecards({ m }: { m: Measurements }) {
  const cards = [
    { label: 'Groundedness', description: 'Claims supported by evidence', value: m.quality.groundedness, tone: 'quality' },
    { label: 'Answer correctness', description: 'Matches the expected facts', value: m.quality.answer_correctness, tone: 'coverage' },
    { label: 'Retrieval recall@5', description: 'Relevant documents found', value: m.retrieval.recall_at_k, tone: 'recall' },
    { label: 'Answer leakage', description: 'Unauthorized content in answers', value: m.security.answer_leakage, tone: 'security', lower: true },
  ];
  return <div className="eval-scorecards">{cards.map(({ label, description, value, tone, lower }) =>
    <section className={`panel eval-scorecard ${tone} ${lower && (value?.value ?? 0) > 0 ? 'leakage-detected' : ''}`} key={label}>
      <div className="eval-score-heading"><h2>{label}</h2></div><strong className="eval-score-value">{metricText(value)}</strong><p>{description}</p>
      <span className="caption">{value?.measured ?? 0}/{value?.applicable ?? m.reliability.selected} applicable cases measured<br />{lower ? 'Lower is better' : 'Higher is better'}</span>
      <Meter value={value?.value ?? null} />
    </section>)}</div>;
}
function Results({ m, separate }: { m: Measurements; separate: boolean }) {
  const r = m.reliability;
  return <>
    <div className="eval-section-label"><span className="eyebrow">Results by group</span><h2>Evaluation matrix</h2><p className="caption">See where the system performs well and which groups need attention.</p></div>
    <Table name="Behavioral categories" rows={m.categories} total={r} /><Table name="Capabilities" rows={m.capabilities} />
    <div className="eval-section-label"><span className="eyebrow">A closer look</span><h2>Quality and reliability</h2><p className="caption">Detailed measurements from retrieval through the final response.</p></div>
    <div className="eval-columns"><section className="panel"><h2>Retrieval and context</h2><MetricList values={{ ...m.retrieval, ...m.context }} /></section>
      <section className="panel"><h2>Answer quality</h2><MetricList values={m.quality} /><p className="caption">{separate ? 'Answer/security thresholds: groundedness ≥70%, correctness ≥60%, relevancy ≥60%. Recall@5 ≥50% is a separate retrieval diagnostic.' : 'Historical combined thresholds: recall ≥50%, groundedness ≥70%, correctness ≥60%, relevancy ≥60%.'} Security failures always fail.</p></section></div>
    {separate && <section className="panel"><h2>Retrieval diagnostics</h2><p>{m.retrieval_diagnostics?.pass ?? 0} passed · {m.retrieval_diagnostics?.fail ?? 0} failed · {m.retrieval_diagnostics?.not_applicable ?? 0} not applicable · {m.retrieval_diagnostics?.not_measured ?? 0} not measured</p><p className="caption">These results do not change the answer/security status.</p></section>}
    <section className="panel eval-security"><h2>Permissions and behavior</h2><MetricList values={m.security} /><p className="caption">Exposure rates count cases with unauthorized material at each stage. Lower exposure is better.</p></section>
    <div className="eval-columns"><section className="panel"><h2>Reliability detail</h2><dl className="metric-list">
      {([['Partial responses', r.partial], ['Failed pipelines', r.pipeline_failed], ['Unjudged cases', r.unjudged], ['Dependency blocked', r.dependency_blocked], ['Pending cases', r.pending]] as const).map(([label, n]) => <div key={label}><dt>{label}</dt><dd>{n}/{r.selected}</dd></div>)}
      <div><dt>Degraded retrieval</dt><dd>{r.degradation_measured ? `${r.degraded_retrieval}/${r.degradation_measured}` : 'Not measured'}<small>{r.degradation_measured}/{r.selected} traces measured</small></dd></div>
    </dl><p className="caption">Reliability counts can overlap: a response may complete while its judge fails.</p></section>
    <section className="panel eval-latency"><h2>Pipeline latency</h2><dl className="metric-list">
      <div><dt>p50</dt><dd>{m.latency.p50_ms === null ? 'Not measured' : `${(m.latency.p50_ms / 1000).toFixed(2)} s`}</dd></div>
      <div><dt>p95</dt><dd>{m.latency.p95_ms === null ? 'Not measured' : `${(m.latency.p95_ms / 1000).toFixed(2)} s`}</dd></div>
    </dl><p className="caption">{m.latency.measured}/{m.latency.total} cases measured. Generation pipeline only; judging and pacing are excluded.</p></section></div>
    <section className="panel"><h2>Failure categories</h2><div className="failure-counts">{Object.entries(m.failure_categories).map(([key, value]) => <p key={key}><span>{title(key)}</span><strong>{value}/{r.recorded}</strong></p>)}</div>
      <p className="caption">Counts may overlap. A completed baseline can contain quality or security failures.</p></section>
  </>;
}
function OutcomeSummary({ data }: { data: EvaluationSummary & { measurements: Measurements } }) {
  const r = data.measurements.reliability, state = evaluationState(data);
  const outcomes = [['passed', 'Passed'], ['failed', 'Failed'], ['errors', 'Errors'], ['pending', 'Pending']] as const;
  return <section className={`panel eval-outcomes ${state.tone}`} aria-label="Evaluation results summary">
    <div className="eval-outcome-heading"><div><span className={`run-status ${state.tone}`}><span aria-hidden="true" />{state.label}</span>
      <h2>{state.heading}</h2><p>{data.status_semantics === 'answer_security' ? 'Answer/security results' : 'Historical combined results'} · {r.recorded} of {r.selected} cases recorded.</p></div>
      {data.generated_at && <p className="caption">{data.report_kind === 'latest_by_case' ? 'Snapshot saved' : state.finished ? 'Completed' : 'Last saved'}<time dateTime={data.generated_at}>{savedTime(data.generated_at)}</time></p>}
    </div>
    {data.report_kind === 'latest_by_case' && <p className="caption">Latest recorded outcomes across {data.source_run_count} saved runs. Cases retain the answers and conversation context from their original runs. This is a combined snapshot, not a new full-suite evaluation.</p>}
    <dl className="eval-outcome-counts"><div><dt>Pass rate</dt><dd>{r.recorded ? `${(r.passed / r.recorded * 100).toFixed(1)}%` : 'Not measured'}</dd></div>
      {outcomes.map(([key, label]) => <div className={key} key={key}><dt>{label}</dt><dd>{r[key]}</dd></div>)}
    </dl>
    <div className="eval-outcome-bar" role="img" aria-label={outcomes.map(([key, label]) => `${r[key]} ${label.toLowerCase()}`).join(', ')}>
      {outcomes.map(([key]) => <span className={key} key={key} style={{ width: `${r.selected ? r[key] / r.selected * 100 : 0}%` }} />)}
    </div>
    <p className="caption">{state.finished && r.errors > 0 ? `${r.errors} cases ended with errors; unavailable scores remain unmeasured. ` : ''}Pass rate includes failed and error cases. Quality metrics use only measured cases.</p>
  </section>;
}
function download(data: EvaluationSummary) {
  const url = URL.createObjectURL(new Blob([JSON.stringify({ ...data, execution_status: evaluationState(data).label }, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = `${data.run_id || 'evaluation'}-summary.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function Evaluation() {
  const [params, setParams] = useSearchParams();
  const report = params.get('report') || '';
  const query = useQuery({ queryKey: ['evaluation', report], queryFn: ({ signal }) => evaluation(report || undefined, signal),
    staleTime: 0, retry: 2, refetchInterval: !report ? 30000 : false, refetchIntervalInBackground: false });
  const data = query.data;
  const state = data ? evaluationState(data) : undefined;
  return <section className="evaluation" aria-labelledby="evaluation-title">
    <header className="eval-header"><div><div className="eyebrow">Saved evaluation results</div><h1 id="evaluation-title">Evaluation overview</h1>
      <p>Coverage, answer quality and permissions, measured through the same pipeline as your chat.</p></div><span className="eval-header-icon" aria-hidden="true"><Icon name="chart" /></span></header>
    {data?.has_results && data.measurements && <><Scorecards m={data.measurements} /><OutcomeSummary data={{ ...data, measurements: data.measurements }} /></>}
    <div className="panel eval-controls"><div className="eval-toolbar"><label>Saved evaluation run<select value={report} onChange={event => setParams(previous => { const next = new URLSearchParams(previous); event.target.value ? next.set('report', event.target.value) : next.delete('report'); return next; })}>
      <option value="">Latest evaluation</option>{report && !data?.available_reports?.some(r => r.report_id === report) && <option value={report}>{report}</option>}
      {data?.available_reports?.map(row => <option key={row.report_id} value={row.report_id}>{row.report_kind === 'latest_by_case' ? 'Latest results by case · ' : ''}{savedTime(row.started_at || row.generated_at || row.report_id)} · {row.report_id === data.report_id ? state?.label : row.status || 'Legacy'} · v{row.dataset_version || '?'} · {row.selected_cases ?? row.total_cases ?? '?'} cases</option>)}
    </select></label><div className="eval-buttons"><button className="primary" onClick={() => void query.refetch()} disabled={query.isFetching}><Icon name="refresh" />Refresh saved results</button>
      {data?.has_results && <button onClick={() => download(data)}><Icon name="download" />Download summary JSON</button>}</div></div>
      <p className="caption eval-refresh-note"><span className="dot" aria-hidden="true" />{report ? 'Historical selection stays fixed until you choose Latest evaluation.' : 'Latest evaluation refreshes every 30 seconds while this page is visible.'}</p></div>
    {query.isPending && <p className="panel eval-empty" role="status">Loading saved results…</p>}
    {query.error && <div className="notice error" role="alert">{query.error.message}{data && ' Showing previously loaded data.'}</div>}
    {data?.has_results ? <>
      <section className="eval-run" aria-label="Selected evaluation run"><div className="eval-run-title"><span className={`run-status ${state?.tone}`}><span aria-hidden="true" />{state?.label}</span><strong>{data.run_id || data.report_id || '(checkpoint)'}</strong><span className="eval-version">Dataset {data.dataset_version || 'not recorded'}</span></div>
        <dl className="eval-run-meta"><div><dt>Cases recorded</dt><dd>{data.total_cases ?? 0} <span>/ {data.selected_cases ?? data.total_cases ?? '?'}</span></dd></div><div><dt>Model</dt><dd>{data.model || 'Not recorded'}</dd></div><div><dt>{data.report_kind === 'latest_by_case' ? 'Snapshot saved' : 'Started'}</dt><dd>{savedTime(data.started_at || data.generated_at || 'Not recorded')}</dd></div></dl>
        {(!data.status || data.status === 'Legacy') && <p className="notice">Legacy report: some measurements or completion metadata may be unavailable.</p>}
        <p className="caption">Scoring policy: {data.scoring_policy || 'historical_combined'}. {data.status_semantics !== 'answer_security' && 'Historical statuses are preserved as recorded; scores have not been reinterpreted.'}</p>
      </section>
      {data.measurements ? <Results m={data.measurements} separate={data.status_semantics === 'answer_security'} /> : <div className="panel"><p>Detailed aggregate metrics: Not measured in this legacy summary.</p></div>}
      {data.cohorts && <Cohorts data={data.cohorts} />}
    </> : data && <p className="panel">No evaluation results are available.</p>}
    <p className="caption eval-privacy"><Icon name="shield" /><span>Refreshing reads saved reports only. Downloads contain aggregate metrics; individual questions, answers and evidence remain private.</span></p>
  </section>;
}
