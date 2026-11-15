/* Browser acceptance against a real local HTTPS FastAPI + isolated PG18.
 * The invoking process supplies the temporary test password and Edge path.
 * Calendar/Telegram/scheduler are disabled by the test server, so this test
 * cannot create a real remote event or send a Telegram message. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const base = process.env.V32_E2E_ORIGIN || 'https://127.0.0.1:18443';
const password = process.env.V32_E2E_PASSWORD;
const edge = process.env.V32_EDGE_PATH;
if (!password || !edge) throw new Error('Synthetic browser test environment incomplete');

(async () => {
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    const context = await browser.newContext({ ignoreHTTPSErrors: true,
      timezoneId: 'Asia/Shanghai', viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const posts = [];
    page.on('request', request => { if (request.method() === 'POST') posts.push(new URL(request.url()).pathname); });
    await page.goto(`${base}/workbench/`, { waitUntil: 'domcontentloaded' });
    await page.locator('input[type=password]').fill(password);
    await page.getByRole('button', { name: '登录', exact: true }).click();
    await page.getByRole('heading', { name: '今日', exact: true }).waitFor();
    await page.locator('.rail button').filter({ hasText: '任务' }).click();
    await page.getByRole('heading', { name: '创建任务' }).waitFor();
    const title = `V3.2 浏览器合成任务 ${Date.now()}`;
    await page.getByPlaceholder('例如：整理本周计划').fill(title);
    const today = await page.evaluate(() => new Intl.DateTimeFormat('sv-SE', {
      timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
    }).format(new Date()));
    await page.getByLabel('截止时间（可选）').fill(`${today}T23:45`);
    await page.getByRole('button', { name: '创建任务' }).click();
    await page.getByText(title, { exact: true }).waitFor();
    await page.locator('.rail button').filter({ hasText: '今日' }).click();
    await page.getByText(title, { exact: true }).waitFor();
    await page.locator('.rail button').filter({ hasText: '任务' }).click();
    await page.locator('.list-row').filter({ hasText: title }).getByRole('button', { name: '详情/编辑' }).click();
    await page.getByRole('dialog').getByLabel('说明').fill('synthetic browser detail');
    await page.getByRole('dialog').getByRole('button', { name: '保存' }).click();
    await page.getByText('synthetic browser detail').count();
    await page.locator('.rail button').filter({ hasText: '提醒' }).click();
    await page.getByRole('heading', { name: '设置一次性提醒' }).waitFor();
    await page.getByRole('textbox', { name: '任务' }).click();
    await page.getByRole('option', { name: title }).click();
    const at = new Date(Date.now() + 86400000);
    const pad = n => String(n).padStart(2, '0');
    const local = `${at.getFullYear()}-${pad(at.getMonth()+1)}-${pad(at.getDate())}T${pad(at.getHours())}:${pad(at.getMinutes())}`;
    await page.getByLabel('提醒时间').fill(local);
    await page.getByRole('button', { name: '设置/改期' }).click();
    const reminderRow = page.locator('.list-row').filter({ hasText: title }).last();
    await reminderRow.getByRole('button', { name: '暂停' }).waitFor();
    await reminderRow.getByRole('button', { name: '暂停' }).click();
    await reminderRow.getByRole('button', { name: '恢复' }).waitFor();
    await reminderRow.getByRole('button', { name: '恢复' }).click();
    await reminderRow.getByRole('button', { name: '取消' }).waitFor();
    await reminderRow.getByRole('button', { name: '取消' }).click();
    await reminderRow.getByText('CANCELLED', { exact: true }).waitFor();
    await page.locator('.rail button').filter({ hasText: '日历' }).click();
    await page.getByText('当前账号没有授权的 Calendar，不能查看或修改。').waitFor();
    await page.locator('.rail button').filter({ hasText: '对话' }).click();
    await page.getByLabel('消息').fill('你好，这是无敏感内容的 Web 浏览器合成测试。');
    await page.getByRole('button', { name: '发送' }).click();
    await page.getByText(/运行 COMPLETED/).waitFor({ timeout: 90000 });
    await page.reload();
    await page.locator('.rail button').filter({ hasText: '对话' }).click();
    await page.getByText(/运行 COMPLETED/).waitFor();
    await page.locator('.rail button').filter({ hasText: '任务' }).click();
    await page.getByText(title, { exact: true }).waitFor();
    await page.locator('.list-row').filter({ hasText: title }).getByRole('button', { name: '详情/编辑' }).click();
    page.once('dialog', dialog => dialog.accept());
    await page.getByRole('dialog').getByRole('button', { name: '申请删除' }).click();
    await page.getByText('已生成删除提案').waitFor();
    await page.locator('.rail button').filter({ hasText: '审批' }).click();
    const approvedTarget = page.getByText(`Delete Todo: ${title}`, { exact: true });
    await approvedTarget.waitFor({ timeout: 30000 });
    await page.screenshot({ path: '.local-tools/v32-browser/pending-approval.png', fullPage: true });
    await page.getByRole('button', { name: '拒绝' }).click();
    await page.getByText(/审批运行：COMPLETED/).waitFor({ timeout: 90000 });
    await approvedTarget.waitFor({ state: 'detached', timeout: 30000 });
    await page.screenshot({ path: '.local-tools/v32-browser/desktop.png', fullPage: true });
    assert(posts.some(path => path.endsWith('/todos')));
    assert(posts.some(path => path.endsWith('/chat/runs')));
    assert(posts.some(path => path.includes('/approvals/') && path.endsWith('/decision')));

    const mobile = await browser.newContext({ ignoreHTTPSErrors: true,
      timezoneId: 'Asia/Shanghai', viewport: { width: 390, height: 844 }, isMobile: true,
      hasTouch: true, deviceScaleFactor: 2 });
    const mobilePage = await mobile.newPage();
    await mobilePage.goto(`${base}/workbench/`, { waitUntil: 'domcontentloaded' });
    await mobilePage.locator('input[type=password]').fill(password);
    await mobilePage.getByRole('button', { name: '登录', exact: true }).click();
    await mobilePage.getByRole('heading', { name: '今日', exact: true }).waitFor();
    await mobilePage.locator('.bottom-nav button').filter({ hasText: '任务' }).click();
    await mobilePage.getByText(title, { exact: true }).waitFor();
    await mobilePage.locator('.bottom-nav button').filter({ hasText: '提醒' }).click();
    await mobilePage.locator('.list-row').filter({ hasText: title }).getByText('CANCELLED', { exact: true }).waitFor();
    await mobilePage.locator('.bottom-nav button').filter({ hasText: '对话' }).click();
    await mobilePage.getByText('你好，这是无敏感内容的 Web 浏览器合成测试。', { exact: true }).first().waitFor();
    await mobilePage.locator('.bottom-nav button').filter({ hasText: '审批' }).click();
    await mobilePage.getByText('没有待审批提案').waitFor();
    await mobilePage.locator('.bottom-nav button').filter({ hasText: '日历' }).click();
    await mobilePage.getByText('当前账号没有授权的 Calendar，不能查看或修改。').waitFor();
    await mobilePage.locator('.bottom-nav button').filter({ hasText: '今日' }).click();
    await mobilePage.getByText(title, { exact: true }).waitFor();
    await mobilePage.screenshot({ path: '.local-tools/v32-browser/mobile.png', fullPage: true });
    const overflow = await mobilePage.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
    assert.equal(overflow, false, 'mobile horizontal overflow');
    await mobile.close();
    await context.close();
    console.log('BROWSER_E2E_PASS: desktop login/Today/Tasks/edit/Reminders/Chat/reload/Approvals + mobile login/Tasks');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
