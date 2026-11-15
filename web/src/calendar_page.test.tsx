// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MantineProvider } from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CalendarPage from './calendar_page';

vi.mock('@fullcalendar/react', async () => { const { forwardRef } = await import('react'); return { default: forwardRef<HTMLDivElement, { datesSet: (arg:{start:Date;end:Date})=>void;eventClick:(arg:{event:{id:string}})=>void;dateClick:(arg:{date:Date;dateStr:string})=>void;dayCellContent?: (arg:{date:Date})=>React.ReactNode;events:{id:string;title:string}[];initialView:string;eventDisplay:string;dayHeaderFormat?:{weekday:string} }>(({ datesSet, eventClick, dateClick, dayCellContent, events, initialView, eventDisplay, dayHeaderFormat }, ref) =>
  <div ref={ref} data-testid="calendar-config" data-initial-view={initialView} data-event-display={eventDisplay} data-weekday-format={dayHeaderFormat?.weekday}><button onClick={() => datesSet({start:new Date('2026-10-12T00:00:00Z'),end:new Date('2026-10-19T00:00:00Z')})}>load week</button><button onClick={() => datesSet({start:new Date('2026-10-25T00:00:00Z'),end:new Date('2026-12-06T00:00:00Z')})}>load month</button>
  <button aria-label="select Nov 12" onClick={() => dateClick({date:new Date(2026,10,12),dateStr:'2026-11-12'})}>{dayCellContent?.({date:new Date(2026,10,12)})}</button>
  <button disabled={!events.some(event => event.id === 'primary:event-1')}
    onClick={() => eventClick({event:{id:'primary:event-1'}})}>pick event</button>
  {events.filter(event=>event.id.startsWith('todo:')).map(event=><button key={event.id}
    onClick={()=>eventClick({event:{id:event.id}})}>{event.id} · {event.title}</button>)}</div>) }; });

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('Calendar direct action', () => {
  it('uses the exact GET fields and ETag, not stale list fields, in a typed write', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({matches:false,media:query,addListener:()=>{},removeListener:()=>{},addEventListener:()=>{},removeEventListener:()=>{}}));
    vi.stubGlobal('ResizeObserver', class {observe() {} unobserve() {} disconnect() {}});
    vi.stubGlobal('crypto', { randomUUID: () => 'synthetic-key' });
    let proposal: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname;
      let body: unknown;
      if (path.endsWith('/calendar/events')) body = {events:[{calendar_id:'primary',event_id:'event-1',etag:'"stale"',title:'Old title',start:{dateTime:'2026-10-14T14:00:00+08:00'},end:{dateTime:'2026-10-14T15:00:00+08:00'}}]};
      else if (path.endsWith('/calendar/events/primary/event-1')) body = {calendar_id:'primary',event_id:'event-1',etag:'"approved"',title:'Exact title',start:{dateTime:'2026-10-14T16:00:00+08:00'},end:{dateTime:'2026-10-14T17:00:00+08:00'}};
      else if (path.endsWith('/calendar/actions')) {proposal=JSON.parse(String(init?.body));body={status:'executed',message:'Google 日历操作已确认完成。'};}
      else throw new Error(`Unexpected API ${path}`);
      return new Response(JSON.stringify(body),{status:200,headers:{'Content-Type':'application/json'}});
    }));
    const client = new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(<MantineProvider><QueryClientProvider client={client}><CalendarPage me={{display_label:'Test',timezone:'Asia/Shanghai',csrf_token:'synthetic',auth_source:'password',capabilities:{calendar:true,gmail:false,memory:false}}}/></QueryClientProvider></MantineProvider>);
    fireEvent.click(screen.getByRole('button',{name:'load week'}));
    await waitFor(() => expect((screen.getByRole('button',{name:'pick event'}) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button',{name:'pick event'}));
    await waitFor(() => expect(screen.getByRole('button',{name:'编辑此 Google 日程'})).toBeDefined());
    fireEvent.click(screen.getByRole('button',{name:'编辑此 Google 日程'}));
    await waitFor(() => expect((screen.getByRole('textbox',{name:'标题'}) as HTMLInputElement).value).toBe('Exact title'));
    fireEvent.click(screen.getByRole('button',{name:'执行操作'}));
    await waitFor(() => expect(proposal).toBeDefined());
    expect(proposal?.operation).toBe('update');
    expect(proposal?.baseline_etag).toBe('"approved"');
    expect((proposal?.fields as {title:string;start:string}).title).toBe('Exact title');
    expect((proposal?.fields as {title:string;start:string}).start).toBe('2026-10-14T16:00:00+08:00');
  });
});

describe('one canonical scheduled Todo on the calendar', () => {
  it('opens in month view and requests the whole visible month grid', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({matches:false,media:query,addListener:()=>{},removeListener:()=>{},addEventListener:()=>{},removeEventListener:()=>{}}));
    vi.stubGlobal('ResizeObserver', class {observe() {} unobserve() {} disconnect() {}});
    const fetchMock=vi.fn(async (url: string) => {
      const path=new URL(url,'http://localhost').pathname;
      if (path.endsWith('/calendar/sync-preference')) return new Response(JSON.stringify({enabled:false,available:false}),{status:200});
      if (path.endsWith('/calendar/scheduled-todos')) return new Response(JSON.stringify({events:[]}),{status:200});
      throw new Error(`Unexpected API ${path}`);
    });
    vi.stubGlobal('fetch',fetchMock);
    const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(<MantineProvider><QueryClientProvider client={client}><CalendarPage
      me={{display_label:'Test',timezone:'Asia/Shanghai',csrf_token:'synthetic',auth_source:'password',capabilities:{calendar:false,gmail:false,memory:false}}}
    /></QueryClientProvider></MantineProvider>);
    expect(screen.getByTestId('calendar-config').getAttribute('data-initial-view')).toBe('dayGridMonth');
    expect(screen.getByTestId('calendar-config').getAttribute('data-event-display')).toBe('block');
    expect((screen.getByRole('radio',{name:'月'}) as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByRole('button',{name:'＋ 放大'})).toBeNull();
    fireEvent.click(screen.getByRole('radio',{name:'周'}));
    expect((screen.getByRole('radio',{name:'周'}) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole('radio',{name:'日'}));
    expect((screen.getByRole('radio',{name:'日'}) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByRole('button',{name:'load month'}));
    await waitFor(()=>expect(fetchMock.mock.calls.some(([url])=>String(url).includes('2026-10-25T00%3A00%3A00.000Z')&&String(url).includes('2026-12-06T00%3A00%3A00.000Z'))).toBe(true));
  });

  it('uses a compact phone month and shows full titles in the selected-day agenda', async () => {
    vi.stubGlobal('innerWidth',390);
    vi.stubGlobal('matchMedia', (query: string) => ({matches:false,media:query,addListener:()=>{},removeListener:()=>{},addEventListener:()=>{},removeEventListener:()=>{}}));
    vi.stubGlobal('ResizeObserver', class {observe() {} unobserve() {} disconnect() {}});
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const path=new URL(url,'http://localhost').pathname;
      const body=path.endsWith('/calendar/sync-preference')?{enabled:false,available:false}:
        path.endsWith('/calendar/scheduled-todos')?{events:[{todo_id:'local-1',title:'完整的手机待办标题',start:'2026-11-12T10:00:00+08:00',end:'2026-11-12T11:00:00+08:00',status:'pending',google_event_id:null}]}:
        path.endsWith('/todos/local-1')?{id:'local-1',title:'完整的手机待办标题',description:null,due_at:null,planned_start_at:'2026-11-12T10:00:00+08:00',planned_end_at:'2026-11-12T11:00:00+08:00',priority:'medium',status:'pending',tags:[],version:1,etag:'"1"',reminder_summary:{state:'OFF',at:null,version:0}}:null;
      if(!body)throw new Error(`Unexpected API ${path}`);
      return new Response(JSON.stringify(body),{status:200,headers:{'Content-Type':'application/json'}});
    }));
    const onTasks=vi.fn();const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(<MantineProvider><QueryClientProvider client={client}><CalendarPage me={{display_label:'Test',timezone:'Asia/Shanghai',csrf_token:'synthetic',auth_source:'password',capabilities:{calendar:false,gmail:false,memory:false}}} onTasks={onTasks}/></QueryClientProvider></MantineProvider>);
    expect(screen.getByTestId('calendar-config').getAttribute('data-event-display')).toBe('none');
    expect(screen.getByTestId('calendar-config').getAttribute('data-weekday-format')).toBe('narrow');
    expect(screen.getByRole('button',{name:'select Nov 12'}).textContent).toBe('12');
    fireEvent.click(screen.getByRole('button',{name:'load month'}));
    await waitFor(()=>expect(screen.getByRole('button',{name:'select Nov 12'}).querySelector('.calendar-mobile-markers')).not.toBeNull());
    fireEvent.click(screen.getByRole('button',{name:'select Nov 12'}));
    expect(screen.getByText('11月12日 · 1 项日程')).toBeDefined();
    expect(screen.getByRole('button',{name:/10:00.*完整的手机待办标题/})).toBeDefined();
    expect(onTasks).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('radio',{name:'舒适'}));
    expect(document.querySelector('.calendar-month-comfortable')).not.toBeNull();
    fireEvent.click(screen.getByRole('radio',{name:'周'}));
    expect(screen.getByTestId('calendar-config').getAttribute('data-initial-view')).toBe('listWeek');
  });

  it('shows local plans without Google authorization and opens the canonical Todo', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({matches:false,media:query,addListener:()=>{},removeListener:()=>{},addEventListener:()=>{},removeEventListener:()=>{}}));
    vi.stubGlobal('ResizeObserver', class {observe() {} unobserve() {} disconnect() {}});
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const path = new URL(url, 'http://localhost').pathname;
      const body = path.endsWith('/todos/local-1') ?
        {id:'local-1',title:'synthetic planned task',description:'Project note',due_at:null,planned_start_at:'2026-10-14T14:00:00+08:00',planned_end_at:'2026-10-14T15:00:00+08:00',priority:'medium',status:'pending',tags:[],version:1,etag:'"1"',reminder_summary:{state:'OFF',at:null,version:0}} :
        path.endsWith('/calendar/scheduled-todos') ?
        {events:[{todo_id:'local-1',title:'synthetic planned task',start:'2026-10-14T14:00:00+08:00',end:'2026-10-14T15:00:00+08:00',status:'pending',google_sync_state:'OFF',google_event_id:null,google_calendar_id:null}]} :
        path.endsWith('/calendar/sync-preference') ? {enabled:false,available:false} : null;
      if (!body) throw new Error(`Unexpected API ${path}`);
      return new Response(JSON.stringify(body),{status:200,headers:{'Content-Type':'application/json'}});
    }));
    const onTasks=vi.fn();
    const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(<MantineProvider><QueryClientProvider client={client}><CalendarPage
      me={{display_label:'Test',timezone:'Asia/Shanghai',csrf_token:'synthetic',auth_source:'password',capabilities:{calendar:false,gmail:false,memory:false}}}
      onTasks={onTasks}/></QueryClientProvider></MantineProvider>);
    fireEvent.click(screen.getByRole('button',{name:'load week'}));
    await waitFor(()=>expect(screen.getByRole('button',{name:/todo:local-1/})).toBeDefined());
    fireEvent.click(screen.getByRole('button',{name:/todo:local-1/}));
    expect(onTasks).not.toHaveBeenCalled();
    await waitFor(()=>expect(screen.getByText('Project note')).toBeDefined());
    fireEvent.click(screen.getByRole('button',{name:'去任务页编辑'}));
    expect(onTasks).toHaveBeenCalledWith('local-1');
    expect(screen.getByText(/本地已排期事项仍可查看/)).toBeDefined();
  });
});
