import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider, focusManager } from '@tanstack/react-query';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Evaluation, { metricText } from '../src/Evaluation';
import { evaluationState, parseEvaluation } from '../src/evaluationData';
import completedReport from './fixtures/evaluation-summary.json';
import currentReport from './fixtures/evaluation-current-summary.json';
import * as api from '../src/api';

const stored = { has_results: true, run_id: 'run-new', report_id: 'runs/new.json', status: 'Incomplete' as const,
  selected_cases: 90, total_cases: 3, dataset_version: '1.1', available_reports: [
    { report_id: 'runs/new.json', status: 'Incomplete' as const, selected_cases: 90, dataset_version: '1.1' },
    { report_id: 'runs/old.json', status: 'Legacy' as const, selected_cases: 60, dataset_version: '1.0' },
  ] };
function RouteLabel() { return <output data-testid="route">{useLocation().search}</output>; }
function ui(route = '/evaluation') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0, refetchOnWindowFocus: false } } });
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[route]}><Evaluation /><RouteLabel /></MemoryRouter></QueryClientProvider>);
}
afterEach(() => { focusManager.setFocused(undefined); vi.useRealTimers(); vi.restoreAllMocks(); });
describe('saved evaluation dashboard', () => {
  it('shows the current 23-case snapshot with provenance, metrics and preserved failures', async () => {
    const data = parseEvaluation(currentReport);
    vi.spyOn(api, 'evaluation').mockResolvedValue(data);
    ui();
    await screen.findByRole('heading', { name: 'Latest results by case' });
    const summary = screen.getByRole('region', { name: 'Evaluation results summary' });
    expect(summary).toHaveTextContent('23 of 23 cases recorded');
    expect(summary).toHaveTextContent('Latest recorded outcomes across 3 saved runs');
    expect(summary).toHaveTextContent('not a new full-suite evaluation');
    expect(within(summary).getByRole('img')).toHaveAccessibleName('14 passed, 9 failed, 0 errors, 0 pending');
    expect(screen.getByRole('row', { name: 'All selected cases 23 / 23 60.9% 14 9 0 0' })).toBeVisible();
    expect(screen.getAllByText('67.4%').length).toBeGreaterThan(0);
    expect(screen.getAllByText('88.0%').length).toBeGreaterThan(0);
    expect(data.measurements?.quality.groundedness.value).toBe(1);
    expect(data.report_kind).toBe('latest_by_case');
    expect(JSON.stringify(data)).not.toContain('case_provenance');
    expect(data.available_reports?.some(r => r.selected_cases === 90)).toBe(true);
  });
  it('separates v6 answer status from retrieval diagnostics and names the policy', async () => {
    const data = parseEvaluation({ ...completedReport, status_semantics: 'answer_security',
      scoring_policy: 'answer_security_v6_retrieval_diagnostic',
      measurements: { ...completedReport.measurements, retrieval_diagnostics: { pass: 24, fail: 8, not_applicable: 0, not_measured: 0, recorded: 32 } } });
    vi.spyOn(api, 'evaluation').mockResolvedValue(data);
    ui();
    await screen.findByRole('heading', { name: 'Retrieval diagnostics' });
    expect(screen.getByText(/24 passed · 8 failed/)).toBeVisible();
    expect(screen.getByText(/Scoring policy: answer_security_v6/)).toBeVisible();
    expect(screen.getByText(/Answer\/security results/)).toBeVisible();
    expect(screen.getByText(/Recall@5 ≥50% is a separate retrieval diagnostic/)).toBeVisible();
  });
  it('labels stored historical combined statuses without applying v6', async () => {
    vi.spyOn(api, 'evaluation').mockResolvedValue(parseEvaluation(completedReport));
    ui();
    await screen.findByText(/Historical statuses are preserved as recorded/);
    expect(screen.queryByRole('heading', { name: 'Retrieval diagnostics' })).not.toBeInTheDocument();
    expect(screen.getByText(/Historical combined thresholds/)).toBeVisible();
  });
  it('shows the finished full run, remaining errors, latest metrics and matrix totals', async () => {
    const data = parseEvaluation(completedReport);
    vi.spyOn(api, 'evaluation').mockResolvedValue(data);
    ui();
    await screen.findByRole('heading', { name: 'Evaluation complete with errors' });
    const summary = screen.getByRole('region', { name: 'Evaluation results summary' });
    expect(summary).toHaveTextContent('90 of 90 cases recorded');
    expect(summary).toHaveTextContent('76.7%');
    expect(within(summary).getByRole('img')).toHaveAccessibleName('69 passed, 19 failed, 2 errors, 0 pending');
    expect(summary).toHaveTextContent('2 cases ended with errors');
    expect(screen.queryByText(/This run is incomplete/)).not.toBeInTheDocument();
    const matrix = screen.getByRole('region', { name: 'Behavioral categories' });
    const total = within(matrix).getByRole('row', { name: 'All selected cases 90 / 90 76.7% 69 19 2 0' });
    expect(total).toBeVisible();
    expect(screen.getAllByText('95.4%').length).toBeGreaterThan(0);
    expect(screen.getAllByText('83.5%').length).toBeGreaterThan(0);
    expect(screen.getAllByText('89.4%').length).toBeGreaterThan(0);
    expect(screen.getByLabelText('Saved evaluation run')).toHaveTextContent('Complete with errors');
  }, 15000);
  it('keeps paused, active and unrecorded runs distinct from finished runs with errors', () => {
    const data = parseEvaluation(completedReport);
    expect(evaluationState(data).finished).toBe(true);
    expect(evaluationState({ ...data, phase: 'stopped', stop_reason: 'execution_error' }).label).toBe('Paused');
    expect(evaluationState({ ...data, phase: 'evaluating' }).label).toBe('Running');
    expect(evaluationState({ ...data, total_cases: 89 }).finished).toBe(false);
    expect(evaluationState({ ...data, phase: undefined }).finished).toBe(false);
    expect(evaluationState({ ...data, measurements: { ...data.measurements!, reliability: { ...data.measurements!.reliability, errors: 0 } } }).label).toBe('Complete');
  });
  it('keeps historical route selection until Latest evaluation is selected', async () => {
    const read = vi.spyOn(api, 'evaluation').mockImplementation(async report => ({ ...stored, run_id: report === 'runs/old.json' ? 'run-old' : 'run-new' }));
    ui('/evaluation?report=runs%2Fold.json');
    await screen.findByText('run-old');
    expect(read.mock.calls[0][0]).toBe('runs/old.json');
    fireEvent.click(screen.getByText('Refresh saved results'));
    await waitFor(() => expect(read).toHaveBeenCalledTimes(2));
    expect(read.mock.calls[1][0]).toBe('runs/old.json');
    fireEvent.change(screen.getByLabelText('Saved evaluation run'), { target: { value: '' } });
    await screen.findByText('run-new');
    expect(screen.getByTestId('route')).toHaveTextContent('');
    expect(read.mock.calls.at(-1)?.[0]).toBeUndefined();
    expect(screen.queryByText(/operator account/)).not.toBeInTheDocument();
  });
  it('polls latest every 30 seconds while visible and never polls historical runs', async () => {
    vi.useFakeTimers(); focusManager.setFocused(true);
    const read = vi.spyOn(api, 'evaluation').mockResolvedValue(stored);
    await act(async () => { ui(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(30001); });
    expect(read).toHaveBeenCalledTimes(2);
    focusManager.setFocused(false);
    await act(async () => { await vi.advanceTimersByTimeAsync(30001); });
    expect(read).toHaveBeenCalledTimes(2);
    focusManager.setFocused(true);
    await act(async () => { fireEvent.change(screen.getByLabelText('Saved evaluation run'), { target: { value: 'runs/old.json' } }); });
    const before = read.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(60001); });
    expect(read).toHaveBeenCalledTimes(before);
  });
  it('validates metrics and strips private fields before summary download', () => {
    const m = { value: .8, measured: 4, applicable: 7, total: 9 };
    const parsed = parseEvaluation({ ...stored, results: [{ answer: 'secret' }], measurements: { quality: { groundedness: m }, private: 'secret' } });
    expect(JSON.stringify(parsed)).not.toContain('secret');
    expect(parsed.measurements?.quality.groundedness).toEqual(m);
    expect(() => parseEvaluation({ has_results: 'yes' })).toThrow();
    expect(() => parseEvaluation({ has_results: true, measurements: { quality: { groundedness: { ...m, measured: 12 } } } })).toThrow();
    expect(metricText({ ...m, value: null, measured: 0 })).toBe('Not measured');
    expect(metricText({ ...m, value: null, measured: 0, applicable: 0 })).toBe('Not applicable');
    expect(parseEvaluation({ has_results: true }).measurements).toBeUndefined();
  });
});
