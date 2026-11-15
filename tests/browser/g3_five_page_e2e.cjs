/* Real Edge against local HTTPS FastAPI and a dedicated synthetic PG18 fixture.
 * This is browser E3/E4 evidence, never Telegram WebView E5. No provider writes. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const base = process.env.V32_E2E_ORIGIN || 'https://127.0.0.1:18443';
const trustedPublicTls = new URL(base).hostname.endsWith('.ts.net');
const password = process.env.V32_E2E_PASSWORD;
const edge = process.env.V32_EDGE_PATH;
if (!password || !edge) throw new Error('Synthetic browser test environment incomplete');

(async () => {
  const browser = await chromium.launch({executablePath:edge,headless:true});
  try {
    const desktop = await browser.newContext({ignoreHTTPSErrors:!trustedPublicTls,timezoneId:'Asia/Shanghai',viewport:{width:1440,height:900}});
    const page = await desktop.newPage();
    await page.goto(`${base}/workbench/`);
    await page.locator('input[type=password]').fill(password);
    await page.getByRole('button',{name:'登录',exact:true}).click();
    await page.getByRole('heading',{name:'今日',exact:true}).waitFor();
    assert.deepEqual(await page.locator('.rail button').allTextContents(),['今日','任务','日历','提醒','审批','退出']);
    assert.equal(await page.getByRole('button',{name:'对话'}).count(),0);
    await page.locator('.rail button').filter({hasText:'任务'}).click();
    const title = `G3 synthetic five-page ${Date.now()}`;
    await page.getByPlaceholder('例如：整理本周计划').fill(title);
    const today = await page.evaluate(() => new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date()));
    await page.getByLabel('截止时间（可选）').fill(`${today}T23:45`);
    await page.getByRole('button',{name:'创建任务'}).click();
    await page.getByText(title,{exact:true}).waitFor();
    await page.getByLabel('搜索任务').fill(title);
    await page.getByText(title,{exact:true}).waitFor();
    await page.locator('.list-row').filter({hasText:title}).getByRole('button',{name:'详情/编辑'}).click();
    await page.getByRole('dialog').getByLabel('说明').fill('Synthetic G3 browser detail');
    await page.getByRole('dialog').getByRole('button',{name:'保存'}).click();
    await page.locator('.rail button').filter({hasText:'今日'}).click();
    await page.getByRole('button',{name:new RegExp(title)}).waitFor();
    await page.locator('.rail button').filter({hasText:'提醒'}).click();
    await page.getByLabel('查找任务').fill(title);
    await page.getByRole('textbox',{name:'任务',exact:true}).click();
    await page.getByRole('option',{name:title}).click();
    const tomorrow = new Date(Date.now()+86400000);
    const pad = n => String(n).padStart(2,'0');
    const at = `${tomorrow.getFullYear()}-${pad(tomorrow.getMonth()+1)}-${pad(tomorrow.getDate())}T${pad(tomorrow.getHours())}:${pad(tomorrow.getMinutes())}`;
    await page.getByLabel('提醒时间').fill(at);
    await page.getByRole('button',{name:'设置/改期'}).click();
    const row = page.locator('.list-row').filter({hasText:title}).last();
    await row.getByRole('button',{name:'暂停'}).waitFor();
    await row.getByRole('button',{name:'暂停'}).click();
    await row.getByRole('button',{name:'恢复'}).waitFor();
    await row.getByRole('button',{name:'恢复'}).click();
    await row.getByRole('button',{name:'取消'}).waitFor();
    page.once('dialog', dialog => dialog.accept());
    await row.getByRole('button',{name:'取消'}).click();
    await row.getByText('CANCELLED',{exact:true}).waitFor();
    let releaseTodoRefresh;
    if (process.env.G3_FORCE_STALE_CACHE === '1') {
      const gate = new Promise(resolve => { releaseTodoRefresh = resolve; });
      await page.route('**/api/v3/todos?*', async route => {
        if (route.request().method() === 'GET') await gate;
        await route.continue();
      });
    }
    await page.locator('.rail button').filter({hasText:'任务'}).click();
    await page.getByLabel('搜索任务').fill(title);
    await page.locator('.list-row').filter({hasText:title}).getByRole('button',{name:'详情/编辑'}).click();
    const deleteRequests = [];
    const deleteResponses = [];
    page.on('request', request => {
      if (request.url().endsWith('/delete-proposal')) deleteRequests.push(request.method());
    });
    page.on('response', response => {
      if (response.url().endsWith('/delete-proposal')) deleteResponses.push(response.status());
    });
    const displayedVersion = await page.getByRole('dialog').getByText(/^版本 \d+/).textContent();
    const beforeConflict = await page.request.get(`${base}/api/v3/todos?q=${encodeURIComponent(title)}`);
    assert.equal(beforeConflict.status(),200);
    const currentTodo = (await beforeConflict.json()).items[0];
    assert.equal(displayedVersion.match(/版本 (\d+)/)?.[1],String(currentTodo.version),'detail must be an exact fresh Todo GET, even while list refresh is delayed');
    const concurrentPatch = await page.evaluate(async ({id,etag}) => {
      const me = await (await fetch('/api/v3/me')).json();
      const response = await fetch(`/api/v3/todos/${id}`,{method:'PATCH',headers:{
        'Content-Type':'application/json','If-Match':etag,'X-CSRF-Token':me.csrf_token,
      },body:JSON.stringify({description:'Synthetic concurrent edit before delete'})});
      return response.status;
    },{id:currentTodo.id,etag:currentTodo.etag});
    assert.equal(concurrentPatch,200);
    page.once('dialog',dialog=>dialog.accept());
    const conflictResponse = page.waitForResponse(response=>response.url().endsWith('/delete-proposal'));
    await page.getByRole('dialog').getByRole('button',{name:'申请删除'}).click();
    assert.equal((await conflictResponse).status(),412,'concurrent edit must reject old If-Match');
    await page.getByText('内容已变化或操作冲突，请刷新后核对再提交。').waitFor();
    if (releaseTodoRefresh) releaseTodoRefresh();
    await page.getByRole('dialog').getByRole('button',{name:'关闭并刷新'}).click();
    await page.locator('.list-row').filter({hasText:title}).getByRole('button',{name:'详情/编辑'}).click();
    const refreshedVersion = await page.getByRole('dialog').getByText(/^版本 \d+/).textContent();
    assert.equal(refreshedVersion.match(/版本 (\d+)/)?.[1],String(currentTodo.version+1));
    page.once('dialog',dialog=>dialog.accept());
    const acceptedResponse = page.waitForResponse(response=>response.url().endsWith('/delete-proposal'));
    await page.getByRole('dialog').getByRole('button',{name:'申请删除'}).click();
    try {
      await page.getByText('已生成删除提案').waitFor();
    } catch (error) {
      if (releaseTodoRefresh) releaseTodoRefresh();
      const latest = await page.request.get(`${base}/api/v3/todos?q=${encodeURIComponent(title)}`);
      const items = latest.ok() ? (await latest.json()).items : [];
      const feedback = await page.getByRole('dialog').innerText().catch(()=>'dialog closed');
      throw new Error(`Delete proposal missing: requests=${JSON.stringify(deleteRequests)}, statuses=${JSON.stringify(deleteResponses)}, displayed=${displayedVersion}, server_version=${items[0]?.version}, feedback=${feedback.slice(-400)}`,{cause:error});
    }
    if (releaseTodoRefresh) releaseTodoRefresh();
    const accepted = await acceptedResponse;
    assert.equal(accepted.status(),202,'accepted form proposal must be durable');
    const admitted = await accepted.json();
    const persistedRun = await page.request.get(`${base}/api/v3/chat/runs/${admitted.run_id}`);
    assert.equal(persistedRun.status(),200,'WebRequest must exist before approval');
    await page.locator('.rail button').filter({hasText:'日历'}).click();
    await page.getByText('当前账号没有授权的 Calendar，不能查看或修改。').waitFor();
    await page.locator('.rail button').filter({hasText:'审批'}).click();
    await page.getByText(`Delete Todo: ${title}`,{exact:true}).waitFor();
    await page.getByRole('button',{name:'拒绝'}).click();
    await page.getByText('没有待审批提案').waitFor();
    await page.reload();
    await page.getByRole('heading',{name:'审批',exact:true}).waitFor();
    await page.locator('.rail button').filter({hasText:'任务'}).click();
    await page.getByLabel('搜索任务').fill(title);
    await page.getByText(title,{exact:true}).waitFor();

    const mobile = await browser.newContext({ignoreHTTPSErrors:!trustedPublicTls,timezoneId:'Asia/Shanghai',viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    const small = await mobile.newPage();
    await small.goto(`${base}/workbench/`);
    await small.locator('input[type=password]').fill(password);
    await small.getByRole('button',{name:'登录',exact:true}).click();
    await small.getByRole('heading',{name:'今日',exact:true}).waitFor();
    assert.deepEqual(await small.locator('.bottom-nav button').allTextContents(),['今日','任务','日历','提醒','审批']);
    for (const tab of ['任务','日历','提醒','审批','今日']) {
      await small.locator('.bottom-nav button').filter({hasText:tab}).click();
      await small.getByRole('heading',{name:tab,exact:true}).waitFor();
      assert.equal(await small.evaluate(()=>document.documentElement.scrollWidth > innerWidth+2),false,`${tab} horizontal overflow`);
    }
    assert.equal(await small.locator('.bottom-nav button').first().evaluate(button=>button.getBoundingClientRect().height>=44),true);
    await small.locator('.bottom-nav button').filter({hasText:'任务'}).click();
    await small.getByLabel('搜索任务').fill(title);
    await small.getByText(title,{exact:true}).waitFor();
    await small.screenshot({path:'.local-tools/v32-browser/g3-mobile.png',fullPage:true});
    await page.screenshot({path:'.local-tools/v32-browser/g3-desktop.png',fullPage:true});
    await small.getByRole('button',{name:'退出'}).click();
    await small.locator('input[type=password]').waitFor();
    await desktop.close();await mobile.close();
    console.log(`G3_EDGE_FIVE_PAGE_PASS: local HTTPS/PG18 Todo+Reminder, fresh detail despite ${process.env.G3_FORCE_STALE_CACHE==='1'?'held':'normal'} list refresh, 412 conflict/retry, durable deletion proposal/HITL reject, five-page desktop/mobile/reload`);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
