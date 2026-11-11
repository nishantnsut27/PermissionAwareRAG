import { isRecord } from './types';
export interface Measure { value: number | null; measured: number; applicable: number; total: number }
export interface Counts { selected: number; recorded: number; passed: number; failed: number; errors: number; pending: number }
export interface Measurements {
  retrieval_diagnostics?: Record<string, number>;
  retrieval: Record<string, Measure>; context: Record<string, Measure>; quality: Record<string, Measure>; security: Record<string, Measure>;
  reliability: Counts & { completed: number; partial: number; pipeline_failed: number; unjudged: number; dependency_blocked: number; degraded_retrieval: number; degradation_measured: number };
  categories: Record<string, Counts>; capabilities: Record<string, Counts>; failure_categories: Record<string, number>;
  latency: { p50_ms: number | null; p95_ms: number | null; measured: number; total: number };
}
export interface ReportOption {
  report_kind?: 'run' | 'latest_by_case'; source_run_count?: number;
  scoring_policy?: string; status_semantics?: string;
  report_id: string; run_id?: string; started_at?: string; generated_at?: string;
  status?: 'Legacy' | 'Incomplete' | 'Complete'; dataset_version?: string; selected_cases?: number; total_cases?: number;
}
export interface EvaluationSummary extends Partial<ReportOption> {
  has_results: boolean; available_reports?: ReportOption[]; model?: string; dataset_id?: string;
  duration_s?: number; selection_complete?: boolean; partial?: boolean; phase?: string; stop_reason?: string;
  measurements?: Measurements; cohorts?: { original_60: Measurements; expanded: Measurements };
}
export function evaluationState(data: EvaluationSummary) {
  const r = data.measurements?.reliability;
  const selected = data.selected_cases ?? r?.selected ?? 0;
  const recorded = data.total_cases ?? r?.recorded ?? 0;
  const active = ['starting', 'waiting', 'evaluating'].includes(data.phase || '');
  const finished = !active && !data.stop_reason && (data.selection_complete === true || data.status === 'Complete'
    || (data.phase === 'finished' && selected > 0 && recorded === selected && (!r || r.pending === 0)));
  if (finished && data.report_kind === 'latest_by_case') return { finished, label: r?.errors ? 'Saved with errors' : 'Saved', heading: 'Latest results by case', tone: r?.errors ? 'complete-with-errors' : 'complete' };
  if (finished) return { finished, label: r?.errors ? 'Complete with errors' : 'Complete',
    heading: r?.errors ? 'Evaluation complete with errors' : 'Evaluation complete', tone: r?.errors ? 'complete-with-errors' : 'complete' };
  if (active) return { finished, label: 'Running', heading: 'Evaluation results', tone: 'running' };
  if (data.phase === 'stopped' || data.stop_reason) return { finished, label: 'Paused', heading: 'Evaluation results', tone: 'incomplete' };
  if (!data.status || data.status === 'Legacy') return { finished, label: 'Legacy', heading: 'Saved evaluation results', tone: 'legacy' };
  return { finished, label: 'Incomplete', heading: 'Evaluation results', tone: 'incomplete' };
}
const invalid = (): never => { throw new Error('The saved evaluation summary is invalid. Refresh to try again.'); };
const record = (x: unknown): Record<string, unknown> => isRecord(x) ? x : invalid();
const count = (x: unknown, fallback = 0): number => x === undefined ? fallback : typeof x === 'number' && Number.isSafeInteger(x) && x >= 0 ? x : invalid();
const number = (x: unknown): number | null => x == null ? null : typeof x === 'number' && Number.isFinite(x) && x >= 0 ? x : invalid();
const string = (x: unknown): string | undefined => x == null ? undefined : typeof x === 'string' ? x : invalid();
const boolean = (x: unknown): boolean | undefined => x === undefined ? undefined : typeof x === 'boolean' ? x : invalid();
function option(raw: unknown, requireId = true): Partial<ReportOption> {
  const x = record(raw), report_id = string(x.report_id), status = string(x.status);
  const report_kind = string(x.report_kind);
  if (report_kind && !['run', 'latest_by_case'].includes(report_kind)) invalid();
  if (requireId && (!report_id || !/^(results\.json|runs\/[A-Za-z0-9][A-Za-z0-9_.-]*\.json)$/.test(report_id))) invalid();
  if (status && !['Legacy', 'Incomplete', 'Complete'].includes(status)) invalid();
  return { report_id, report_kind: report_kind as ReportOption['report_kind'], source_run_count: count(x.source_run_count), status: status as ReportOption['status'], scoring_policy: string(x.scoring_policy), status_semantics: string(x.status_semantics), run_id: string(x.run_id), started_at: string(x.started_at), generated_at: string(x.generated_at),
    dataset_version: string(x.dataset_version), selected_cases: count(x.selected_cases), total_cases: count(x.total_cases) };
}
function measure(raw: unknown): Measure {
  const x = record(raw), value = number(x.value), measured = count(x.measured), applicable = count(x.applicable), total = count(x.total);
  if (measured > applicable || applicable > total || (value !== null && (value > 1 || measured === 0)) || (value === null && measured > 0)) invalid();
  return { value, measured, applicable, total };
}
function counts(raw: unknown): Counts {
  const x = record(raw);
  return { selected: count(x.selected), recorded: count(x.recorded), passed: count(x.passed), failed: count(x.failed), errors: count(x.errors), pending: count(x.pending) };
}
function map<T>(raw: unknown, parse: (x: unknown) => T): Record<string, T> {
  if (raw === undefined) return {};
  return Object.fromEntries(Object.entries(record(raw)).map(([key, value]) => {
    if (!/^[a-z][a-z_0-9]*$/.test(key)) invalid();
    return [key, parse(value)];
  }));
}
export function measurements(raw: unknown): Measurements {
  const x = record(raw), r = record(x.reliability || {}), l = record(x.latency || {});
  return { retrieval: map(x.retrieval, measure), context: map(x.context, measure), quality: map(x.quality, measure), security: map(x.security, measure),
    retrieval_diagnostics: map(x.retrieval_diagnostics, value => count(value)),
    categories: map(x.categories, counts), capabilities: map(x.capabilities, counts), failure_categories: map(x.failure_categories, value => count(value)),
    reliability: { ...counts(r), completed: count(r.completed), partial: count(r.partial), pipeline_failed: count(r.pipeline_failed), unjudged: count(r.unjudged),
      dependency_blocked: count(r.dependency_blocked), degraded_retrieval: count(r.degraded_retrieval), degradation_measured: count(r.degradation_measured) },
    latency: { p50_ms: number(l.p50_ms), p95_ms: number(l.p95_ms), measured: count(l.measured), total: count(l.total) } };
}
export function parseEvaluation(raw: unknown): EvaluationSummary {
  const x = record(raw);
  if (typeof x.has_results !== 'boolean') invalid();
  if (x.available_reports !== undefined && !Array.isArray(x.available_reports)) invalid();
  const cohorts = x.cohorts === undefined ? undefined : record(x.cohorts);
  return { ...option(x, false), has_results: x.has_results as boolean,
    available_reports: (x.available_reports as unknown[] | undefined)?.map(r => option(r) as ReportOption),
    model: string(x.model), dataset_id: string(x.dataset_id), duration_s: number(x.duration_s) ?? undefined,
    phase: string(x.phase), stop_reason: string(x.stop_reason), selection_complete: boolean(x.selection_complete), partial: boolean(x.partial),
    measurements: x.measurements === undefined ? undefined : measurements(x.measurements),
    cohorts: cohorts ? { original_60: measurements(cohorts.original_60), expanded: measurements(cohorts.expanded) } : undefined };
}
