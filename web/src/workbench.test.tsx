// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MantineProvider } from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Workbench } from './workbench';

afterEach(() => { cleanup(); sessionStorage.clear(); history.replaceState(null, '', '/'); vi.unstubAllGlobals(); });

describe('real API-backed workbench slice', () => {
  it('shows only the four direct-management pages and never requests retired approval or draft APIs', async () => {
    history.replaceState(null, '', '/workbench/?page=Approvals');
    vi.stubGlobal('matchMedia', () => ({matches:false,addListener:()=>{},removeListener:()=>{},
      addEventListener:()=>{},removeEventListener:()=>{}}));
    vi.stubGlobal('ResizeObserver', class {observe(){} unobserve(){} disconnect(){}});
    const paths:string[]=[];
    vi.stubGlobal('fetch', vi.fn(async (url:string) => {
      const path=new URL(url,'http://localhost').pathname;
      paths.push(path);
      const body=path.endsWith('/me')
        ? {display_label:'Fixture',timezone:'UTC',csrf_token:'csrf',
           auth_source:'telegram',capabilities:{calendar:false,gmail:false,memory:false}}
        : {date:'2026-10-09',todos:[],events:[],reminders:[],sources:{calendar:'unavailable'}};
      return new Response(JSON.stringify(body),{status:200,headers:{'Content-Type':'application/json'}});
    }));
    const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(<MantineProvider><QueryClientProvider client={client}><Workbench/></QueryClientProvider></MantineProvider>);
    await screen.findByText('今日与逾期任务');
    expect(screen.getByRole('navigation',{name:'工作台导航'}).querySelectorAll('button')).toHaveLength(4);
    expect(screen.queryByRole('button',{name:'审批'})).toBeNull();
    expect(screen.queryByRole('button',{name:'草稿'})).toBeNull();
    expect(paths.some(path=>path.includes('/approvals')||path.includes('/meeting-drafts'))).toBe(false);
  });

  it('fails closed before private API reads when Telegram launch lacks signed initData', async () => {
    history.replaceState(null, '', '/workbench/#launchTicket=synthetic-unknown-launch');
    vi.stubGlobal('Telegram', {WebApp:{initData:'',ready:vi.fn(),expand:vi.fn()}});
    vi.stubGlobal('matchMedia', () => ({matches:false,addListener:()=>{},removeListener:()=>{},
      addEventListener:()=>{},removeEventListener:()=>{}}));
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    const client = new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(<MantineProvider><QueryClientProvider client={client}><Workbench /></QueryClientProvider></MantineProvider>);
    expect(await screen.findByText(/重新发送 \/workbench 获取新按钮/)).toBeTruthy();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('loads Today, creates a Todo, and refreshes the Tasks list', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: false, media: query,
      onchange: null, addListener: () => {}, removeListener: () => {},
      addEventListener: () => {}, removeEventListener: () => {}, dispatchEvent: () => false }));
    vi.stubGlobal('ResizeObserver', class {
      observe() {} unobserve() {} disconnect() {}
    });
    let created = false;
    const requests: { path: string; method: string; key: string | null }[] = [];
    vi.stubGlobal('crypto', { randomUUID: () => 'synthetic-key' });
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname;
      const method = init?.method || 'GET';
      const headers = new Headers(init?.headers);
      requests.push({ path, method, key: headers.get('Idempotency-Key') });
      const todo = { id: 'synthetic-todo', title: '合成任务', description: null, due_at: null,
        priority: 'medium', status: 'pending', tags: [], version: 1, etag: '"1"',
        reminder_summary: { state: 'NONE', at: null, version: 0 } };
      let body: unknown = {};
      if (path.endsWith('/me')) body = { display_label: 'Test', timezone: 'Asia/Shanghai',
        csrf_token: 'synthetic-csrf', capabilities: { calendar: false, gmail: false, memory: true } };
      if (path.endsWith('/today')) body = { date: '2026-10-09', todos: [], events: [], reminders: [], sources: { calendar: 'unavailable' } };
      if (path.endsWith('/todos') && method === 'GET') body = { items: created ? [todo] : [] };
      if (path.endsWith('/todos') && method === 'POST') { created = true; body = todo; }
      return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<MantineProvider><QueryClientProvider client={client}><Workbench /></QueryClientProvider></MantineProvider>);
    expect(await screen.findByText('今日与逾期任务')).toBeTruthy();
    fireEvent.click(screen.getAllByText('任务')[0]);
    expect(await screen.findByRole('heading', { name: '创建任务' })).toBeTruthy();
    fireEvent.change(screen.getByPlaceholderText('例如：整理本周计划'), { target: { value: '合成任务' } });
    fireEvent.click(screen.getByRole('button', { name: '创建任务' }));
    await waitFor(() => expect(screen.getByText('合成任务')).toBeTruthy());
    expect(requests.find(r => r.path.endsWith('/todos') && r.method === 'POST')?.key).toBe('synthetic-key');
  });


});
