import { inTelegram } from './telegram_bootstrap';

const color = (value: string | undefined, fallback: string) =>
  value && /^#[0-9a-fA-F]{6}$/.test(value) ? value : fallback;
const inset = (value: number | undefined) =>
  `${Number.isFinite(value) ? Math.max(0, Math.min(120, value!)) : 0}px`;

export function mountTelegramLifecycle(onActivate: () => void): () => void {
  const app = inTelegram() ? window.Telegram?.WebApp : undefined;
  if (!app) return () => {};
  const root = document.documentElement;
  root.classList.add('telegram-webview');
  function sync() {
    const theme = app!.themeParams || {};
    root.dataset.telegramTheme = app!.colorScheme || 'light';
    root.style.setProperty('--tg-page-bg', color(theme.bg_color, '#f6f7f4'));
    root.style.setProperty('--tg-page-text', color(theme.text_color, '#172921'));
    root.style.setProperty('--tg-card-bg', color(theme.secondary_bg_color, '#ffffff'));
    const safe = app!.contentSafeAreaInset || app!.safeAreaInset;
    root.style.setProperty('--tg-safe-bottom', inset(safe?.bottom));
    root.style.setProperty('--tg-safe-top', inset(safe?.top));
    root.style.setProperty('--tg-safe-left', inset(safe?.left));
    root.style.setProperty('--tg-safe-right', inset(safe?.right));
    const height = app!.viewportStableHeight || app!.viewportHeight;
    if (height && Number.isFinite(height)) root.style.setProperty('--tg-view-height', `${height}px`);
  }
  function viewport() {
    sync();
    const visible = window.visualViewport?.height || window.innerHeight;
    root.dataset.keyboardOpen = visible < window.innerHeight - 140 ? 'true' : 'false';
  }
  const focus = (event: FocusEvent) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) {
      window.setTimeout(() => target.scrollIntoView({block:'nearest',behavior:'smooth'}), 180);
    }
  };
  sync();
  for (const name of ['themeChanged','viewportChanged','safeAreaChanged','contentSafeAreaChanged'])
    app.onEvent?.(name, viewport);
  app.onEvent?.('activated', onActivate);
  window.visualViewport?.addEventListener('resize', viewport);
  document.addEventListener('focusin', focus);
  return () => {
    for (const name of ['themeChanged','viewportChanged','safeAreaChanged','contentSafeAreaChanged'])
      app.offEvent?.(name, viewport);
    app.offEvent?.('activated', onActivate);
    window.visualViewport?.removeEventListener('resize', viewport);
    document.removeEventListener('focusin', focus);
    root.classList.remove('telegram-webview');
    root.removeAttribute('data-keyboard-open');
    root.removeAttribute('data-telegram-theme');
  };
}

export function bindTelegramBack(onBack: () => void, visible: boolean): () => void {
  const back = inTelegram() ? window.Telegram?.WebApp?.BackButton : undefined;
  if (!back) return () => {};
  if (visible) { back.show(); back.onClick(onBack); }
  else back.hide();
  return () => { if (visible) back.offClick(onBack); back.hide(); };
}

export function guardUnsavedChanges(dirty: boolean): () => void {
  const app = inTelegram() ? window.Telegram?.WebApp : undefined;
  if (dirty) app?.enableClosingConfirmation?.();
  else app?.disableClosingConfirmation?.();
  const beforeUnload = (event: BeforeUnloadEvent) => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = '';
  };
  window.addEventListener('beforeunload', beforeUnload);
  return () => { window.removeEventListener('beforeunload', beforeUnload); if (dirty) app?.disableClosingConfirmation?.(); };
}
