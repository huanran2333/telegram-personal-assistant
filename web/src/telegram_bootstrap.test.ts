// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';

afterEach(() => { vi.unstubAllGlobals(); history.replaceState(null, '', '/'); });

it('exchanges raw initData before rendering business data and clears launch fragment', async () => {
  vi.resetModules();
  history.replaceState(null, '', '/workbench/#launchTicket=synthetic-ticket');
  const ready = vi.fn(), expand = vi.fn();
  vi.stubGlobal('Telegram', { WebApp:{initData:'user=%7B%22id%22%3A123%7D&auth_date=1&hash=synthetic',ready,expand} });
  const requests: Array<{url:string;body?:string;csrf:string|null}> = [];
  vi.stubGlobal('fetch', vi.fn(async (url:string, init?:RequestInit) => {
    requests.push({url,body:String(init?.body || ''),csrf:new Headers(init?.headers).get('X-CSRF-Token')});
    if (url.endsWith('/me')) return new Response(JSON.stringify({detail:'NoSession'}),{status:401});
    return url.endsWith('/auth/bootstrap')
      ? new Response(JSON.stringify({csrf_token:'synthetic-csrf'}),{status:200})
      : new Response(null,{status:204});
  }));
  const { setCsrf } = await import('./api');
  setCsrf('stale-browser-session-csrf');
  const { bootstrapTelegram } = await import('./telegram_bootstrap');
  await bootstrapTelegram();
  expect(ready).toHaveBeenCalledOnce();
  expect(expand).toHaveBeenCalledOnce();
  expect(requests.map(x=>x.url)).toEqual(['/api/v3/me','/api/v3/auth/bootstrap',
    '/api/v3/auth/telegram/exchange']);
  expect(JSON.parse(requests[2].body || '{}')).toEqual({
    init_data:'user=%7B%22id%22%3A123%7D&auth_date=1&hash=synthetic',launch_ticket:'synthetic-ticket',
  });
  expect(requests[2].csrf).toBe('synthetic-csrf');
  expect(location.hash).toBe('');
});

it('does not load a private session when a launch has no signed initData', async () => {
  vi.resetModules();
  history.replaceState(null, '', '/workbench/#launchTicket=synthetic-ticket');
  vi.stubGlobal('Telegram', { WebApp:{initData:'',ready:vi.fn(),expand:vi.fn()} });
  const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
  const { bootstrapTelegram } = await import('./telegram_bootstrap');
  await expect(bootstrapTelegram()).rejects.toThrow('重新发送 /workbench');
  expect(fetch).not.toHaveBeenCalled();
});

it('reuses a live Session without rotating it, after signed owner and ticket-expiry verification', async () => {
  vi.resetModules();
  history.replaceState(null, '', '/workbench/#launchTicket=already-spent');
  vi.stubGlobal('Telegram', { WebApp:{initData:'signed-current-user',ready:vi.fn(),expand:vi.fn()} });
  const requests:string[] = [];
  vi.stubGlobal('fetch', vi.fn(async (url:string, init?:RequestInit) => {
    requests.push(url);
    if (url.endsWith('/me')) return new Response(JSON.stringify({csrf_token:'session-csrf',auth_source:'telegram'}),{status:200});
    expect(url).toBe('/api/v3/auth/telegram/revalidate');
    expect(new Headers(init?.headers).get('X-CSRF-Token')).toBe('session-csrf');
    expect(JSON.parse(String(init?.body))).toEqual({init_data:'signed-current-user',launch_ticket:'already-spent'});
    return new Response(null,{status:204});
  }));
  const { bootstrapTelegram } = await import('./telegram_bootstrap');
  await bootstrapTelegram();
  expect(requests).toEqual(['/api/v3/me','/api/v3/auth/telegram/revalidate']);
  expect(location.hash).toBe('');
});

it('does not reuse a button when both session revalidation and signed exchange fail', async () => {
  vi.resetModules();
  history.replaceState(null, '', '/workbench/#launchTicket=already-spent');
  vi.stubGlobal('Telegram', { WebApp:{initData:'signed-other-user',ready:vi.fn(),expand:vi.fn()} });
  vi.stubGlobal('fetch', vi.fn(async (url:string) => {
    if (url.endsWith('/auth/bootstrap')) return new Response(JSON.stringify({csrf_token:'nonce-csrf'}),{status:200});
    if (url.endsWith('/me')) return new Response(JSON.stringify({csrf_token:'session-csrf',auth_source:'telegram'}),{status:200});
    return new Response(JSON.stringify({detail:'MiniAppAuthenticationRequired'}),{status:401});
  }));
  const { bootstrapTelegram } = await import('./telegram_bootstrap');
  await expect(bootstrapTelegram()).rejects.toThrow('重新发送 /workbench');
  expect(location.hash).toContain('launchTicket');
});

it('retries the same unexpired button after a transient bootstrap network failure', async () => {
  vi.resetModules();
  history.replaceState(null, '', '/workbench/#launchTicket=still-valid');
  vi.stubGlobal('Telegram', { WebApp:{initData:'signed-current-user',ready:vi.fn(),expand:vi.fn()} });
  let bootstrapCalls = 0;
  vi.stubGlobal('fetch', vi.fn(async (url:string) => {
    if (url.endsWith('/me')) return new Response(JSON.stringify({detail:'NoSession'}),{status:401});
    if (url.endsWith('/auth/bootstrap')) {
      bootstrapCalls += 1;
      if (bootstrapCalls === 1) throw new Error('SyntheticNetworkFailure');
      return new Response(JSON.stringify({csrf_token:'fresh-nonce'}),{status:200});
    }
    return new Response(null,{status:204});
  }));
  const { bootstrapTelegram } = await import('./telegram_bootstrap');
  await expect(bootstrapTelegram()).rejects.toThrow('重新发送 /workbench');
  await expect(bootstrapTelegram()).resolves.toBeUndefined();
  expect(bootstrapCalls).toBe(2);
  expect(location.hash).toBe('');
});
