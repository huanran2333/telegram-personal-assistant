/* Real trusted HTTPS through a foreground Funnel, synthetic PG18 only.
 * Never a Telegram-client, Bot-send, or Google-provider E5 assertion. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const base = process.env.G4_PUBLIC_ORIGIN;
const firstPassword = process.env.G4_PUBLIC_PASSWORD;
const secondPassword = process.env.G4_PUBLIC_PASSWORD_OTHER;
const edge = process.env.V32_EDGE_PATH;
if (!/^https:\/\/[^/]+\.ts\.net$/.test(base || '') ||
    !firstPassword || !secondPassword || !edge) throw new Error('G4 public synthetic environment incomplete');

async function login(browser, password) {
  const context = await browser.newContext({ ignoreHTTPSErrors: false });
  const page = await context.newPage();
  await page.goto(`${base}/workbench/`);
  await page.locator('input[type=password]').fill(password);
  await page.getByRole('button', { name: '登录', exact: true }).click();
  await page.getByRole('heading', { name: '今日', exact: true }).waitFor();
  return { context, page };
}

(async () => {
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    const anonymous = await browser.newContext({ ignoreHTTPSErrors: false });
    const guest = await anonymous.newPage();
    await guest.goto(`${base}/workbench/`);
    for (const route of ['/api/v3/me', '/api/v3/todos', '/webhook-info']) {
      const response = await guest.request.get(`${base}${route}`);
      assert.equal(response.status(), 401, `${route} must deny anonymous access`);
    }
    assert.equal((await guest.request.get(`${base}/.env`)).status(), 404);
    const bootstrap = await guest.evaluate(async () => {
      const response = await fetch('/api/v3/auth/bootstrap');
      return { status: response.status, body: await response.json() };
    });
    assert.equal(bootstrap.status, 200);
    const bogus = await guest.evaluate(async (csrf) => {
      const response = await fetch('/api/v3/auth/telegram/exchange', {
        method: 'POST', headers: { 'content-type': 'application/json', 'x-csrf-token': csrf },
        body: JSON.stringify({ init_data: 'invalid', launch_ticket: 'invalid' }),
      });
      return response.status;
    }, bootstrap.body.csrf_token);
    assert.equal(bogus, 503, 'Telegram runtime disabled; no fake identity exchange');
    assert.equal((await guest.request.get(`${base}/api/v3/me`)).status(), 401);

    const first = await login(browser, firstPassword);
    const cookies = await first.context.cookies(base);
    const session = cookies.find(cookie => cookie.name === '__Host-pa_session');
    assert(session, 'session cookie missing');
    assert.equal(session.secure, true);
    assert.equal(session.httpOnly, true);
    assert.equal(session.sameSite, 'Strict');
    assert.equal(session.path, '/');
    assert.equal(session.domain.startsWith('.'), false, 'cookie must be host-only');
    const meResponse = await first.page.request.get(`${base}/api/v3/me`);
    assert.equal(meResponse.status(), 200);
    const me = await meResponse.json();
    assert.equal(me.auth_source, 'password');
    const noCsrf = await first.page.request.post(`${base}/api/v3/auth/logout`, {
      headers: { origin: base },
    });
    assert.equal(noCsrf.status(), 403);
    const wrongOrigin = await first.page.request.post(`${base}/api/v3/auth/logout`, {
      headers: { origin: 'https://untrusted.example', 'x-csrf-token': me.csrf_token },
    });
    assert.equal(wrongOrigin.status(), 403);
    assert.equal((await first.page.request.get(`${base}/api/v3/me`)).status(), 200);

    await first.page.locator('.rail button').filter({ hasText: '任务' }).click();
    const title = `G4 public synthetic owner test ${Date.now()}`;
    await first.page.getByPlaceholder('例如：整理本周计划').fill(title);
    await first.page.getByRole('button', { name: '创建任务' }).click();
    await first.page.getByText(title, { exact: true }).waitFor();
    const firstList = await first.page.request.get(`${base}/api/v3/todos?q=${encodeURIComponent(title)}`);
    assert.equal(firstList.status(), 200);
    const items = (await firstList.json()).items;
    assert.equal(items.length, 1);
    const todoId = items[0].id;

    const second = await login(browser, secondPassword);
    const otherList = await second.page.request.get(`${base}/api/v3/todos?q=${encodeURIComponent(title)}`);
    assert.equal(otherList.status(), 200);
    assert.equal((await otherList.json()).items.length, 0);
    assert.equal((await second.page.request.get(`${base}/api/v3/todos/${todoId}`)).status(), 404);
    for (const label of ['任务', '日历', '提醒', '审批', '今日']) {
      await second.page.locator('.rail button').filter({ hasText: label }).click();
      await second.page.getByRole('heading', { name: label, exact: true }).waitFor();
    }
    await first.context.close();
    await second.context.close();
    await anonymous.close();
    console.log('G4_PUBLIC_HTTPS_AUTH_PASS: trusted Edge TLS, anonymous denial, two synthetic owners, Secure host-only cookie, CSRF/Origin, five-page reachability');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
