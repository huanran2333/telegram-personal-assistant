/* Real Edge UI acceptance against the authorized Calendar read path.
 * The caller uses an isolated synthetic Web identity mapped only in this
 * temporary process to the already-authorized Calendar account. This test
 * never submits a proposal or invokes a provider write. Event data stays in
 * browser memory and is not printed or saved to screenshots. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const base = process.env.V32_E2E_ORIGIN || 'https://127.0.0.1:18443';
const password = process.env.V32_E2E_PASSWORD;
const edge = process.env.V32_EDGE_PATH;
if (!password || !edge) throw new Error('Synthetic browser test environment incomplete');

function dayOffset(days) {
  const date = new Date();
  date.setHours(0, 0, 0, 0);
  date.setDate(date.getDate() + days);
  return date.toISOString();
}

function weekStart(value) {
  const date = new Date(value);
  date.setHours(0, 0, 0, 0);
  date.setDate(date.getDate() - date.getDay());
  return date.getTime();
}

(async () => {
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    const context = await browser.newContext({
      ignoreHTTPSErrors: true, timezoneId: 'Asia/Shanghai',
      viewport: { width: 1440, height: 900 },
    });
    const page = await context.newPage();
    let safe;
    const calendarResponses = [];
    page.on('response', async response => {
      if (!response.url().includes('/api/v3/calendar/events?')) return;
      try {
        const body = await response.json();
        calendarResponses.push({ status: response.status(), count: body.events?.length || 0,
          target: Boolean(safe && body.events?.some(item => item.event_id === safe.event_id)) });
      } catch { calendarResponses.push({ status: response.status(), unreadable: true }); }
    });
    await page.goto(`${base}/workbench/`);
    await page.locator('input[type=password]').fill(password);
    const loginResponse = page.waitForResponse(response => response.url().endsWith('/api/v3/auth/login'));
    await page.getByRole('button', { name: '登录', exact: true }).click();
    assert.equal((await loginResponse).status(), 204, 'synthetic Web login');
    await page.getByRole('heading', { name: '今日', exact: true }).waitFor();
    const me = await page.evaluate(async () => (await fetch('/api/v3/me')).json());
    assert.equal(me.capabilities.calendar, true);

    // Four bounded 30-day reads; never query the entire Calendar.
    const windows = [[-60, -30], [-30, 0], [0, 30], [30, 60]];
    for (const [from, to] of windows) {
      const result = await page.evaluate(async ({ start, end }) => {
        const query = new URLSearchParams({ from: start, to: end, tz: 'Asia/Shanghai' });
        const response = await fetch(`/api/v3/calendar/events?${query}`);
        return { status: response.status, body: await response.json() };
      }, { start: dayOffset(from), end: dayOffset(to) });
      assert.equal(result.status, 200, 'bounded real Calendar API read');
      safe = result.body.events.find(item => item.event_id &&
        /synthetic|测试|test/i.test(item.title));
      if (safe) break;
    }
    assert(safe, 'SAFE_CALENDAR_FIXTURE_NOT_FOUND');
    const eventInstant = safe.start?.dateTime || safe.start?.date;
    assert(eventInstant, 'safe event has no time identity');
    const targetWeek = weekStart(eventInstant);
    const targetResult = await page.evaluate(async ({ start, end }) => {
      const query = new URLSearchParams({ from: start, to: end, tz: 'Asia/Shanghai' });
      const response = await fetch(`/api/v3/calendar/events?${query}`);
      return { status: response.status, body: await response.json() };
    }, { start: new Date(targetWeek).toISOString(), end: new Date(targetWeek + 7 * 86400000).toISOString() });
    assert.equal(targetResult.status, 200);
    assert(targetResult.body.events.some(item => item.event_id === safe.event_id),
      'safe event absent from its bounded target-week API');

    async function openExactEvent() {
      await page.locator('.rail button').filter({ hasText: '日历' }).click();
      await page.getByRole('heading', { name: '日历', exact: true }).waitFor();
      await page.locator('.fc-timeGridWeek-button').click();
      const weeks = Math.round((weekStart(eventInstant) - weekStart(new Date())) / (7 * 86400000));
      assert(Math.abs(weeks) <= 10, 'event outside bounded UI navigation');
      for (let i = 0; i < Math.abs(weeks); i++) {
        await page.locator(weeks > 0 ? '.fc-next-button' : '.fc-prev-button').click();
      }
      let targetIndex = -1;
      for (let attempt = 0; attempt < 60; attempt++) {
        targetIndex = await page.locator('.fc-event').evaluateAll(
          (nodes, title) => nodes.findIndex(node => node.textContent?.includes(title)), safe.title);
        if (targetIndex >= 0) break;
        await page.waitForTimeout(500);
      }
      if (targetIndex < 0) {
        console.error('Calendar week rendering diagnostic', {
          relativeWeeks: weeks, visibleEventCount: await page.locator('.fc-event').count(),
          visibleDates: await page.locator('.fc-timegrid-col[data-date]').evaluateAll(
            nodes => nodes.map(node => node.getAttribute('data-date')).slice(0, 7)),
          calendarResponses: calendarResponses.slice(-10),
        });
        throw new Error('Safe test event not rendered in target week');
      }
      const responsePromise = page.waitForResponse(response =>
        response.url().includes('/api/v3/calendar/events/') && response.status() === 200);
      await page.locator('.fc-event').nth(targetIndex).click();
      const detail = await (await responsePromise).json();
      assert.ok(detail.event_id === safe.event_id, 'exact Calendar resource identity');
      assert(detail.etag, 'exact provider GET must retain the current ETag');
      assert.ok((await page.getByLabel('标题').inputValue()) === detail.title,
        'exact title is displayed without logging private text');
      if (detail.start?.dateTime) {
        const expected = await page.evaluate(instant => {
          const date = new Date(instant);
          const pad = n => String(n).padStart(2, '0');
          return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
        }, detail.start.dateTime);
        assert.ok((await page.getByLabel('开始').inputValue()) === expected,
          'Calendar start is shown in the selected account timezone');
      }
      return detail;
    }

    const first = await openExactEvent();
    await page.reload();
    const second = await openExactEvent();
    assert.ok(second.event_id === first.event_id, 'same event after reload');
    assert.ok(second.etag === first.etag, 'exact event unchanged across reload');
    console.log('REAL_CALENDAR_READONLY_UI_PASS: bounded list, week view, exact GET, timezone, reload');
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
