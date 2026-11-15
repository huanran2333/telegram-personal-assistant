// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { bindTelegramBack, guardUnsavedChanges, mountTelegramLifecycle } from './telegram_lifecycle';

afterEach(() => {
  vi.unstubAllGlobals();
  document.documentElement.className = '';
  document.documentElement.removeAttribute('data-telegram-theme');
  document.documentElement.removeAttribute('data-keyboard-open');
});

describe('Telegram WebView lifecycle', () => {
  it('uses bridge theme/insets, refreshes on activation, and unregisters handlers', () => {
    const listeners = new Map<string, () => void>();
    const app = {
      initData:'signed fixture',ready:vi.fn(),expand:vi.fn(),colorScheme:'dark',
      themeParams:{bg_color:'#123456',text_color:'#ffffff',secondary_bg_color:'#234567'},
      contentSafeAreaInset:{bottom:24,top:8,left:3,right:4},viewportStableHeight:740,
      onEvent:vi.fn((name:string, callback:()=>void)=>listeners.set(name,callback)),
      offEvent:vi.fn((name:string)=>listeners.delete(name)),
    };
    vi.stubGlobal('Telegram',{WebApp:app});
    const activate=vi.fn();
    const cleanup=mountTelegramLifecycle(activate);
    const root=document.documentElement;
    expect(root.classList.contains('telegram-webview')).toBe(true);
    expect(root.style.getPropertyValue('--tg-safe-bottom')).toBe('24px');
    expect(root.style.getPropertyValue('--tg-page-bg')).toBe('#123456');
    expect(root.dataset.telegramTheme).toBe('dark');
    listeners.get('activated')?.();
    expect(activate).toHaveBeenCalledOnce();
    app.colorScheme='light';
    listeners.get('themeChanged')?.();
    expect(root.dataset.telegramTheme).toBe('light');
    cleanup();
    expect(listeners.size).toBe(0);
    expect(root.classList.contains('telegram-webview')).toBe(false);
  });

  it('uses BackButton and closing confirmation only for a signed launch', () => {
    let back: (()=>void)|undefined;
    const app={initData:'signed fixture',ready:vi.fn(),expand:vi.fn(),
      BackButton:{show:vi.fn(),hide:vi.fn(),onClick:vi.fn((callback:()=>void)=>{back=callback;}),offClick:vi.fn()},
      enableClosingConfirmation:vi.fn(),disableClosingConfirmation:vi.fn()};
    vi.stubGlobal('Telegram',{WebApp:app});
    const handler=vi.fn();
    const stopBack=bindTelegramBack(handler,true);
    back?.();
    expect(handler).toHaveBeenCalledOnce();
    const stopGuard=guardUnsavedChanges(true);
    expect(app.enableClosingConfirmation).toHaveBeenCalledOnce();
    stopGuard();stopBack();
    expect(app.disableClosingConfirmation).toHaveBeenCalledOnce();
    expect(app.BackButton.offClick).toHaveBeenCalledOnce();
  });

  it('does not require the bridge in an ordinary browser', () => {
    expect(() => {mountTelegramLifecycle(()=>{})();bindTelegramBack(()=>{},true)();guardUnsavedChanges(false)();}).not.toThrow();
    expect(document.documentElement.classList.contains('telegram-webview')).toBe(false);
  });
});
