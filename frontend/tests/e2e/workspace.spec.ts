import { test, expect } from '@playwright/test';
import evaluationSummary from '../fixtures/evaluation-current-summary.json' with { type: 'json' };
const user = { user_id: 'U-001', name: 'Aditya Verma', role: 'Account Manager', department: 'Business', seller_scope: ['S001', 'S003'] };
const vikram = { ...user, user_id: 'U-003', name: 'Vikram Singh', role: 'Engineer', department: 'Engineering' };
test.beforeEach(async ({ page }) => {
  await page.route('http://127.0.0.1:8000/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (route.request().method() === 'OPTIONS') { await route.fulfill({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS' } }); return; }
    const headers = { 'Access-Control-Allow-Origin': '*' };
    const body = path === '/demo/users' ? { users: [user, vikram] }
      : path === '/demo/login' ? { access_token: 'session', expires_in: 3600, user: route.request().postDataJSON().user_id === vikram.user_id ? vikram : user }
      : path === '/demo/me' ? { user }
      : path === '/ready' ? { status: 'ready' }
      : { ...evaluationSummary, run_id: 'saved-report' };
    if (path === '/demo/query/stream') {
      const question = route.request().postDataJSON().query;
      const denied = question.includes('restricted');
      const terminal = { type: 'done', outcome: denied ? 'refusal' : 'answer', completed: true, answer: denied ? '' : 'The settlement adjustment was verified.\n\n| Item | Status |\n| --- | --- |\n| Payout | Corrected |',
        sources: denied ? [] : [{ label: 'Settlement call', document_id: 'CALL-018', seller_id: 'S001', pages: [1] }], stats: { authorized: 2, selected: 1, latency_ms: 100 }, effective_sellers: ['S001'] };
      await route.fulfill({ headers, contentType: 'text/event-stream', body: `data: {"type":"delta","text":"Unverified draft"}\n\ndata: ${JSON.stringify(terminal)}\n\n` });
    } else await route.fulfill({ headers, json: body });
  });
});
async function openMenu(page: import('@playwright/test').Page) {
  const button = page.getByRole('button', { name: 'Open navigation', exact: true });
  if (await button.isVisible()) await button.click();
}
async function closeMenu(page: import('@playwright/test').Page) {
  const backdrop = page.locator('.menu-backdrop');
  if (await backdrop.isVisible()) {
    const bounds = await backdrop.boundingBox();
    await backdrop.click({ position: { x: bounds!.width - 12, y: 16 } });
  }
}
async function toggleTheme(page: import('@playwright/test').Page) {
  await openMenu(page);
  await page.getByRole('switch', { name: 'Dark theme' }).click();
  await closeMenu(page);
}
test('login, scoped chat, source inspection, refresh restoration, theme and evaluation', async ({ page }, testInfo) => {
  test.setTimeout(60000); // Multiple full-page light/dark captures on desktop and mobile.
  await page.goto('/login');
  await expect(page.getByRole('heading', { name: 'Explore a sample account' })).toBeVisible();
  await expect(page.getByLabel('Demo password')).toHaveValue('aditya@login');
  await page.screenshot({ path: testInfo.outputPath('login-light.png'), fullPage: true });
  await toggleTheme(page);
  await page.screenshot({ path: testInfo.outputPath('login-dark.png'), fullPage: true });
  await toggleTheme(page);
  await page.getByRole('button', { name: 'Continue' }).click();
  await expect(page.getByRole('heading', { name: 'How can I help, Aditya?' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('chat-light.png'), fullPage: true });
  await toggleTheme(page);
  await page.screenshot({ path: testInfo.outputPath('chat-dark.png'), fullPage: true });
  await toggleTheme(page);
  await page.getByLabel('Ask a question').fill('settlement'); await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.getByText('The settlement adjustment was verified.')).toBeVisible();
  await expect(page.getByText('Unverified draft')).toHaveCount(0);
  await page.getByText('Sources (1)').click(); await expect(page.getByText('Settlement call')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('answer-sources.png'), fullPage: true });
  await page.reload(); await expect(page.getByText('The settlement adjustment was verified.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Send question' })).toBeDisabled();
  await toggleTheme(page);
  await page.getByText('Sources (1)').click();
  await page.screenshot({ path: testInfo.outputPath('answer-dark.png'), fullPage: true });
  await openMenu(page); await page.getByRole('button', { name: 'Evaluation overview' }).click();
  await expect(page.getByText('saved-report')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Latest results by case' })).toBeVisible();
  await expect(page.getByText(/Latest recorded outcomes across 3 saved runs/)).toBeVisible();
  const scorecards = await page.locator('.eval-scorecards').boundingBox();
  const completion = await page.getByRole('region', { name: 'Evaluation results summary' }).boundingBox();
  expect(completion!.y).toBeGreaterThanOrEqual(scorecards!.y + scorecards!.height);
  await expect(page.getByRole('region', { name: 'Evaluation results summary' })).toContainText('23 of 23 cases recorded');
  await expect(page.getByRole('row', { name: 'All selected cases 23 / 23 60.9% 14 9 0 0' })).toBeVisible();
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download summary JSON' }).click();
  const download = await downloaded;
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const summary = JSON.parse(Buffer.concat(chunks).toString());
  expect(summary.report_kind).toBe('latest_by_case');
  expect(summary.measurements.reliability).toMatchObject({ selected: 23, passed: 14, failed: 9, errors: 0 });
  expect(summary.results).toBeUndefined();
  await page.screenshot({ path: testInfo.outputPath('evaluation-dark.png'), fullPage: true });
  await toggleTheme(page);
  await page.screenshot({ path: testInfo.outputPath('evaluation-light.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
test('demo passwords autofill when switching accounts and can be shown or hidden', async ({ page }, testInfo) => {
  await page.goto('/login');
  const password = page.getByLabel('Demo password');
  await expect(password).toHaveValue('aditya@login');
  await expect(password).toHaveAttribute('type', 'password');
  await page.getByRole('button', { name: 'Show password' }).click();
  await expect(password).toHaveAttribute('type', 'text');
  await expect(page.getByRole('heading', { name: 'Explore a sample account' })).toBeVisible();
  await page.getByRole('button', { name: 'Hide password' }).click();
  await expect(password).toHaveAttribute('type', 'password');
  await page.getByLabel('Demo account').selectOption(vikram.user_id);
  await expect(password).toHaveValue('vikram@login');
  await page.getByRole('button', { name: 'Show password' }).click();
  await page.screenshot({ path: testInfo.outputPath('login-password-visible.png'), fullPage: true });
  const request = page.waitForRequest(request => request.method() === 'POST' && new URL(request.url()).pathname === '/demo/login');
  await page.getByRole('button', { name: 'Continue' }).click();
  expect((await request).postDataJSON()).toEqual({ user_id: vikram.user_id, password: 'vikram@login' });
  await expect(page.getByRole('heading', { name: 'How can I help, Vikram?' })).toBeVisible();
});
test('refusal and user switching clear the previous transcript', async ({ page }, testInfo) => {
  await page.goto('/login'); await page.getByLabel('Demo password').fill('aditya@login'); await page.getByRole('button', { name: 'Continue' }).click();
  await page.getByLabel('Ask a question').fill('restricted'); await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.getByText('No authorized information found')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('refusal.png'), fullPage: true });
  await openMenu(page); await page.getByRole('button', { name: 'New User' }).click();
  await expect(page.getByRole('heading', { name: 'Explore a sample account' })).toBeVisible();
  expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem('knowledge-workspace.v1')!).messages)).toEqual([]);
});
test('unavailable backend does not block rendering the login shell', async ({ page }) => {
  await page.unrouteAll(); await page.route('http://127.0.0.1:8000/**', route => route.abort());
  await page.goto('/chat'); await expect(page.getByRole('heading', { name: 'Explore a sample account' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Continue' })).toBeDisabled();
});

test('a missing lazy asset after redeployment offers reload without replaying a question', async ({ page }) => {
  let generations = 0;
  page.on('request', request => { if (request.url().includes('/demo/query/stream')) generations++; });
  await page.goto('/login');
  await page.getByLabel('Demo password').fill('aditya@login');
  await page.getByRole('button', { name: 'Continue' }).click();
  await expect(page.getByLabel('Ask a question')).toBeEnabled();
  await page.route('**/assets/Evaluation-*.js', route => route.abort());
  await openMenu(page);
  await page.getByRole('button', { name: 'Evaluation overview' }).click();
  await expect(page.getByRole('button', { name: 'Reload app' }).first()).toBeVisible();
  expect(generations).toBe(0);
  expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem('knowledge-workspace.v1')!).session.user.user_id)).toBe('U-001');
});

test('long answers and citations stay above the growing composer', async ({ page }, testInfo) => {
  await page.addInitScript(profile => {
    sessionStorage.setItem('knowledge-workspace.v1', JSON.stringify({
      session: { token: 'session', user: profile, expiresAt: Date.now() + 3600000 },
      theme: 'light', memory: 6, seller: '', messages: [
        { id: 'question', role: 'user', content: 'What are the current priorities for my seller accounts?' },
        { id: 'answer', role: 'assistant', outcome: 'answer',
          content: ['## Seller S001 — Current priorities', ...Array(10).fill('**Settlement reconciliation** — Review the corrected statement and confirm the outstanding adjustments.\n\n'), '## Seller S003 — Current priorities', 'Confirm the tier upgrade with the account team.'].join('\n\n'),
          sources: [{ label: 'Account priorities', document_id: 'REVIEW-001', seller_id: 'S001', pages: [1, 2] }] },
      ],
    }));
  }, user);
  await page.goto('/chat');
  const composer = page.getByLabel('Ask a question');
  await expect(composer).toBeEnabled();
  await page.getByText('Sources (1)').click();
  await page.getByText('Account priorities', { exact: true }).scrollIntoViewIfNeeded();
  const source = await page.getByText('Account priorities', { exact: true }).boundingBox();
  const dock = await page.locator('.composer-dock').boundingBox();
  expect(source!.y + source!.height).toBeLessThanOrEqual(dock!.y);
  const collapsed = await composer.boundingBox();
  await composer.fill('Follow up on S001\nand S003\nusing the cited sources.');
  expect((await composer.boundingBox())!.height).toBeGreaterThan(collapsed!.height);
  await expect(page.getByRole('button', { name: 'Send question' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('long-answer-composer.png'), fullPage: true });
});
