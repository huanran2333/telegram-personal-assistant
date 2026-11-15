/* Real Edge + HTTPS FastAPI + isolated PG18, without external model/provider writes.
 * The fuller v32_workbench_e2e.cjs remains the Chat/approval combined test. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const base = process.env.V32_E2E_ORIGIN || 'https://127.0.0.1:18443';
const password = process.env.V32_E2E_PASSWORD;
const edge = process.env.V32_EDGE_PATH;
if (!password || !edge) throw new Error('Synthetic browser test environment incomplete');

(async () => {
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    const desktop = await browser.newContext({ ignoreHTTPSErrors: true, timezoneId: 'Asia/Shanghai', viewport: {width:1440,height:900} });
    const page = await desktop.newPage();
    await page.goto(`${base}/workbench/`);
    await page.locator('input[type=password]').fill(password);
    await page.getByRole('button',{name:'登录',exact:true}).click();
    await page.getByRole('heading',{name:'今日',exact:true}).waitFor();
    await page.locator('.rail button').filter({hasText:'任务'}).click();
    const title = `V3.3 合成长文本与手机布局 ${Date.now()} ${'任务详情'.repeat(8)}`;
    await page.getByPlaceholder('例如：整理本周计划').fill(title);
    await page.getByRole('button',{name:'创建任务'}).click();
    await page.getByText(title,{exact:true}).waitFor();
    await page.locator('.list-row').filter({hasText:title}).getByRole('button',{name:'详情/编辑'}).click();
    await page.getByRole('dialog').getByLabel('说明').fill('synthetic detail');
    await page.getByRole('dialog').getByRole('button',{name:'保存'}).click();
    await page.locator('.rail button').filter({hasText:'提醒'}).click();
    await page.getByRole('textbox',{name:'任务'}).click();
    await page.getByRole('option',{name:title}).click();
    const tomorrow = new Date(Date.now()+86400000);
    const pad = n => String(n).padStart(2,'0');
    await page.getByLabel('提醒时间').fill(`${tomorrow.getFullYear()}-${pad(tomorrow.getMonth()+1)}-${pad(tomorrow.getDate())}T${pad(tomorrow.getHours())}:${pad(tomorrow.getMinutes())}`);
    await page.getByRole('button',{name:'设置/改期'}).click();
    const row = page.locator('.list-row').filter({hasText:title}).last();
    await row.getByRole('button',{name:'暂停'}).waitFor();
    await row.getByRole('button',{name:'暂停'}).click();
    await row.getByRole('button',{name:'恢复'}).waitFor();
    await row.getByRole('button',{name:'恢复'}).click();
    await row.getByRole('button',{name:'取消'}).waitFor();
    await row.getByRole('button',{name:'取消'}).click();
    await row.getByText('CANCELLED',{exact:true}).waitFor();
    await page.locator('.rail button').filter({hasText:'日历'}).click();
    await page.getByText('当前账号没有授权的 Calendar，不能查看或修改。').waitFor();
    await page.locator('.rail button').filter({hasText:'审批'}).click();
    await page.getByText('没有待审批提案').waitFor();

    const mobile = await browser.newContext({ignoreHTTPSErrors:true,timezoneId:'Asia/Shanghai',viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    const small = await mobile.newPage();
    await small.goto(`${base}/workbench/`);
    await small.locator('input[type=password]').fill(password);
    await small.getByRole('button',{name:'登录',exact:true}).click();
    await small.getByRole('heading',{name:'今日',exact:true}).waitFor();
    for (const tab of ['任务','提醒','日历','对话','审批','今日']) {
      await small.locator('.bottom-nav button').filter({hasText:tab}).click();
      await small.getByRole('heading',{name:tab,exact:true}).waitFor();
      assert.equal(await small.evaluate(() => document.documentElement.scrollWidth > innerWidth+2),false,`${tab} horizontal overflow`);
    }
    assert.equal(await small.locator('.bottom-nav button').first().evaluate(button => button.getBoundingClientRect().height >= 44),true,'mobile tap height');
    await small.locator('.bottom-nav button').filter({hasText:'任务'}).click();
    await small.getByText(title,{exact:true}).waitFor();
    await small.screenshot({path:'.local-tools/v32-browser/v33-mobile.png',fullPage:true});
    await page.screenshot({path:'.local-tools/v32-browser/v33-desktop.png',fullPage:true});
    await page.locator('.rail button').filter({hasText:'退出'}).click();
    await page.locator('input[type=password]').waitFor();
    await page.reload();
    await page.locator('input[type=password]').waitFor();
    await mobile.close(); await desktop.close();
    console.log('V33_BROWSER_SMOKE_PASS: HTTPS login/logout, real-PG Todo edit, Reminder lifecycle, six-page desktop/mobile navigation, no mobile overflow');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
