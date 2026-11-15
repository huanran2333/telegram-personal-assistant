/* Real Edge + public HTTPS against the authorized business account.
 * Read-only by design: never create or modify production business records. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const origin = process.env.G4B_PUBLIC_ORIGIN;
const password = process.env.G4B_WEB_PASSWORD;
const edge = process.env.V32_EDGE_PATH;
if (!origin?.startsWith('https://') || !password || !edge) {
  throw new Error('G4-B public browser environment incomplete');
}

(async () => {
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    const context = await browser.newContext({
      ignoreHTTPSErrors: false,
      timezoneId: 'Asia/Shanghai',
      viewport: { width: 1440, height: 900 },
    });
    const page = await context.newPage();
    const apiStatuses = [];
    page.on('response', response => {
      if (response.url().startsWith(`${origin}/api/v3/`)) {
        apiStatuses.push({ path: new URL(response.url()).pathname, status: response.status() });
      }
    });
    await page.goto(`${origin}/workbench/`);
    await page.locator('input[type=password]').fill(password);
    await page.getByRole('button', { name: '登录', exact: true }).click();
    await page.getByRole('heading', { name: '今日', exact: true }).waitFor();
    const me = await page.request.get(`${origin}/api/v3/me`);
    assert.equal(me.status(), 200);
    assert.equal((await me.json()).auth_source, 'password');
    const session = (await context.cookies(origin)).find(cookie => cookie.name === '__Host-pa_session');
    assert.ok(session, 'server must set the host-prefixed session cookie');
    assert.equal(session.secure, true);
    assert.equal(session.httpOnly, true);
    assert.equal(session.sameSite, 'Strict');
    assert.equal(session.domain, new URL(origin).hostname);
    assert.deepEqual(await page.locator('.rail button').allTextContents(),
      ['今日', '任务', '日历', '提醒', '审批', '退出']);
    for (const tab of ['任务', '日历', '提醒', '审批', '今日']) {
      const apiPath = tab === '任务' ? '/api/v3/todos' : tab === '审批' ? '/api/v3/approvals' : null;
      const loaded = apiPath ? page.waitForResponse(response =>
        new URL(response.url()).pathname === apiPath && response.status() === 200) : null;
      await page.locator('.rail button').filter({ hasText: tab }).click();
      await page.getByRole('heading', { name: tab, exact: true }).waitFor();
      if (loaded) await loaded;
    }
    await page.reload();
    await page.getByRole('heading', { name: '今日', exact: true }).waitFor();
    assert.equal(apiStatuses.some(item => item.path === '/api/v3/todos' && item.status === 200), true);
    assert.equal(apiStatuses.some(item => item.path === '/api/v3/approvals' && item.status === 200), true);
    await context.close();
    console.log('G4B_PUBLIC_EDGE_READONLY_PASS: trusted TLS, password session, five pages, reload');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
