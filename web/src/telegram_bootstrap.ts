import { api } from './api';

export type TelegramWebApp = {
  initData: string; ready(): void; expand(): void;
  colorScheme?: 'light'|'dark';
  themeParams?: Record<string,string|undefined>;
  viewportStableHeight?: number; viewportHeight?: number;
  safeAreaInset?: {top?:number;right?:number;bottom?:number;left?:number};
  contentSafeAreaInset?: {top?:number;right?:number;bottom?:number;left?:number};
  BackButton?: {show():void;hide():void;onClick(callback:()=>void):void;offClick(callback:()=>void):void};
  onEvent?(name:string, callback:()=>void):void; offEvent?(name:string, callback:()=>void):void;
  enableClosingConfirmation?():void; disableClosingConfirmation?():void;
};
declare global { interface Window { Telegram?: { WebApp?: TelegramWebApp } } }

let attempt: Promise<void> | null = null;
let activeTicket: string | null = null;
const relaunchHint = '这个工作台入口已失效或无法验证。请返回 Bot 私聊，重新发送 /workbench 获取新按钮。';

export function inTelegram(): boolean { return Boolean(window.Telegram?.WebApp?.initData); }

export function bootstrapTelegram(): Promise<void> {
  const params = new URLSearchParams(location.hash.slice(1));
  const ticket = params.get('launchTicket');
  if (!ticket) return Promise.resolve();
  if (attempt && activeTicket === ticket) return attempt;
  activeTicket = ticket;
  const current = (async () => {
    const webApp = window.Telegram?.WebApp;
    if (!webApp?.initData) throw new Error(relaunchHint);
    webApp.ready();
    webApp.expand();
    // Prefer the existing Telegram Session: reopening must not rotate a
    // perfectly good cookie or consume another login nonce. The server checks
    // both fresh signed initData and this launch's ten-minute expiry.
    const sessionController = new AbortController();
    const sessionDeadline = setTimeout(() => sessionController.abort(), 15000);
    try {
      const session = await api<{csrf_token:string;auth_source:string}>('/me', {signal:sessionController.signal});
      if (session.auth_source !== 'telegram') throw new Error('WrongAuthSource');
      await api('/auth/telegram/revalidate', {
        method:'POST', headers:{'X-CSRF-Token':session.csrf_token}, signal:sessionController.signal,
        body:JSON.stringify({init_data:webApp.initData, launch_ticket:ticket}),
      });
    } catch {
      clearTimeout(sessionDeadline);
      // A lost/revoked cookie may be replaced within the same launch window,
      // but a bare URL is never sufficient: exchange checks Telegram's HMAC,
      // Bot ID, user binding, auth_date, Origin and one-time CSRF nonce.
      const exchangeController = new AbortController();
      const exchangeDeadline = setTimeout(() => exchangeController.abort(), 15000);
      try {
        const nonce = await api<{csrf_token:string}>('/auth/bootstrap', {signal:exchangeController.signal});
        await api('/auth/telegram/exchange', {
          method:'POST', headers:{'X-CSRF-Token':nonce.csrf_token}, signal:exchangeController.signal,
          body:JSON.stringify({init_data:webApp.initData, launch_ticket:ticket}),
        });
      } catch {
        // Do not echo initData, the launch ticket, or provider response details.
        throw new Error(relaunchHint);
      } finally {
        clearTimeout(exchangeDeadline);
      }
    } finally {
      clearTimeout(sessionDeadline);
    }
    history.replaceState(history.state, '', location.pathname + location.search);
  })();
  attempt = current.catch((error:unknown) => {
    // A transient WebView/network failure must not poison this button's
    // remaining ten-minute window in a retained JavaScript context.
    if (activeTicket === ticket) { activeTicket = null; attempt = null; }
    throw error;
  });
  return attempt;
}
