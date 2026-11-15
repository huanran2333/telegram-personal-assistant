import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Button, Card, Group, Loader, Modal, SegmentedControl, Select, Text, TextInput } from '@mantine/core';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import listPlugin from '@fullcalendar/list';
import interactionPlugin from '@fullcalendar/interaction';
import zhCnLocale from '@fullcalendar/core/locales/zh-cn';
import type { DatesSetArg, EventClickArg } from '@fullcalendar/core';
import { api, ApiError, SubmissionKeys } from './api';
import type { CalendarEvent, Me, ScheduledTodoEvent, Todo } from './api';
import { zonedInput, zonedISO } from './zoned_time';

function dateKey(value:string, timeZone:string):string {
  if(!value.includes('T'))return value.slice(0,10);
  const parts=new Intl.DateTimeFormat('en-US',{timeZone,year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(new Date(value));
  const part=(name:string)=>parts.find(item=>item.type===name)?.value||'';
  return `${part('year')}-${part('month')}-${part('day')}`;
}

function appearsOnDay(event:{start?:string|null;end?:string|null},day:string,timeZone:string):boolean {
  if(!event.start)return false;
  const first=dateKey(event.start,timeZone);
  if(!event.end)return first===day;
  const last=event.end.includes('T')
    ?dateKey(new Date(new Date(event.end).getTime()-1).toISOString(),timeZone)
    :new Date(Date.parse(`${event.end}T00:00:00Z`)-86400000).toISOString().slice(0,10);
  return first<=day&&day<=last;
}

function calendarCellKey(date:Date):string {
  return `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
}

export default function CalendarPage({me,focusId='',onTasks=()=>{},onDirty=()=>{}}:{me:Me;focusId?:string;onTasks?:(id:string)=>void;onDirty?:(dirty:boolean)=>void}) {
  const qc=useQueryClient();const calendarRef=useRef<FullCalendar>(null);
  const [range,setRange]=useState<{from:string;to:string}|null>(null);const [selected,setSelected]=useState<CalendarEvent|null>(null);const [op,setOp]=useState<'create'|'update'|'delete'>('create');
  const [title,setTitle]=useState('');const [start,setStart]=useState('');const [end,setEnd]=useState('');const [message,setMessage]=useState('');const [mobile,setMobile]=useState(()=>window.innerWidth<820);const [view,setView]=useState('dayGridMonth');const [dirty,setLocalDirty]=useState(false);const [keys]=useState(()=>new SubmissionKeys());
  const [todoId,setTodoId]=useState<string|null>(null);const [detailOpen,setDetailOpen]=useState(false);
  const [showForm,setShowForm]=useState(false);
  const [selectedDay,setSelectedDay]=useState(()=>dateKey(new Date().toISOString(),me.timezone));
  const [monthSize,setMonthSize]=useState<'compact'|'comfortable'>('compact');
  useEffect(()=>{const resize=()=>setMobile(window.innerWidth<820);window.addEventListener('resize',resize);return()=>window.removeEventListener('resize',resize);},[]);
  useEffect(()=>{onDirty(dirty);return()=>onDirty(false);},[dirty,onDirty]);
  useEffect(()=>{if(!focusId)return;try{const [calendar_id,event_id]=JSON.parse(focusId);if(typeof calendar_id==='string'&&typeof event_id==='string'){setSelected({calendar_id,event_id,etag:null,title:'',start:null,end:null,status:''});setOp('update');setDetailOpen(true);}}catch{/* invalid navigation cannot select a provider target */}},[focusId]);
  const q=useQuery({queryKey:['events',range],enabled:!!range&&me.capabilities.calendar,queryFn:()=>api<{events:CalendarEvent[];truncated:boolean}>(`/calendar/events?from=${encodeURIComponent(range!.from)}&to=${encodeURIComponent(range!.to)}&tz=${encodeURIComponent(me.timezone)}`)});
  const local=useQuery({queryKey:['scheduled-todos',range],enabled:!!range,queryFn:()=>api<{events:ScheduledTodoEvent[];truncated:boolean}>(`/calendar/scheduled-todos?from=${encodeURIComponent(range!.from)}&to=${encodeURIComponent(range!.to)}`)});
  const sync=useQuery({queryKey:['calendar-sync-preference'],queryFn:()=>api<{enabled:boolean;available:boolean;existing_remote_events_remain_when_disabled:boolean}>('/calendar/sync-preference')});
  const setSync=useMutation({mutationFn:(enabled:boolean)=>api<{enabled:boolean}>('/calendar/sync-preference',{method:'PATCH',body:JSON.stringify({enabled})}),onSuccess:()=>{qc.invalidateQueries({queryKey:['calendar-sync-preference']});qc.invalidateQueries({queryKey:['me']});}});
  const detail=useQuery({queryKey:['event-detail',selected?.calendar_id,selected?.event_id],enabled:!!selected,queryFn:()=>api<CalendarEvent>(`/calendar/events/${encodeURIComponent(selected!.calendar_id)}/${encodeURIComponent(selected!.event_id)}`)});
  const todoDetail=useQuery({queryKey:['todo-detail',todoId],enabled:!!todoId,queryFn:()=>api<Todo>(`/todos/${encodeURIComponent(todoId!)}`)});
  // Display the exact resource version used for the conditional write.
  useEffect(()=>{if(!selected||!detail.data||detail.isFetching||detail.data.event_id!==selected.event_id)return;
    setTitle(detail.data.title);
    setStart(detail.data.start?.dateTime?zonedInput(detail.data.start.dateTime,me.timezone):'');
    setEnd(detail.data.end?.dateTime?zonedInput(detail.data.end.dateTime,me.timezone):'');
  },[selected?.event_id,detail.data,detail.isFetching,me.timezone]);
  useEffect(()=>{const date=detail.data?.start?.dateTime||detail.data?.start?.date;if(date)calendarRef.current?.getApi?.().gotoDate(date);},[detail.data?.event_id]);
  const action=useMutation({mutationFn:()=>{const body={operation:op,calendar_id:op==='create'?'primary':detail.data?.calendar_id,event_id:op==='create'?undefined:detail.data?.event_id,baseline_etag:op==='create'?undefined:detail.data?.etag,fields:op==='delete'?undefined:{title,start:zonedISO(start,me.timezone),end:zonedISO(end,me.timezone),timeZone:me.timezone}};return api<{status:string;message:string}>(`/calendar/actions`,{method:'POST',body:JSON.stringify(body)},keys.for('calendar-action',body));},onSuccess:r=>{if(r.status==='executed')keys.clear('calendar-action');setLocalDirty(false);setMessage(r.message);qc.invalidateQueries({queryKey:['events']});qc.invalidateQueries({queryKey:['event-detail']});}});
  function dateRange(info:DatesSetArg){
    setRange({from:info.start.toISOString(),to:info.end.toISOString()});
    if(info.view?.type){
      setView(info.view.type==='listWeek'?'dayGridWeek':info.view.type);
      if(info.view.type==='dayGridMonth'&&info.view.currentStart){
        const month=calendarCellKey(info.view.currentStart).slice(0,7);
        setSelectedDay(current=>current.startsWith(month)?current:`${month}-01`);
      }
    }
  }
  function openEvent(id:string){
    if(id.startsWith('todo:')){setTodoId(id.slice(5));setDetailOpen(true);return;}
    const e=q.data?.events.find(item=>`${item.calendar_id}:${item.event_id}`===id);
    if(e){setTodoId(null);setSelected(e);setOp('update');setLocalDirty(false);setTitle(e.title);setStart(e.start?.dateTime?zonedInput(e.start.dateTime,me.timezone):'');setEnd(e.end?.dateTime?zonedInput(e.end.dateTime,me.timezone):'');setDetailOpen(true);}
  }
  function eventClick(info:EventClickArg){info.jsEvent?.preventDefault();openEvent(info.event.id);}
  function changeView(next:string,date?:Date){setView(next);calendarRef.current?.getApi?.().changeView(mobile&&next==='dayGridWeek'?'listWeek':next,date);}
  const mirrored=new Set((local.data?.events||[]).filter(e=>e.google_event_id).map(e=>e.google_event_id));
  const calendarEvents=[...(local.data?.events||[]).map(e=>({id:`todo:${e.todo_id}`,title:`${e.title}${e.status==='done'?' ✓':''}`,start:e.start,end:e.end,classNames:['calendar-todo-event']})),...(q.data?.events||[]).filter(e=>!mirrored.has(e.event_id)).map(e=>({id:`${e.calendar_id}:${e.event_id}`,title:e.title,start:e.start?.dateTime||e.start?.date,end:e.end?.dateTime||e.end?.date,classNames:['calendar-google-event']}))];
  const selectedDayEvents=calendarEvents.filter(e=>appearsOnDay(e,selectedDay,me.timezone)).sort((a,b)=>(a.start||'').localeCompare(b.start||''));
  const formatDateTime=(value:string|null|undefined)=>value?new Intl.DateTimeFormat('zh-CN',{timeZone:me.timezone,month:'long',day:'numeric',weekday:'short',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value)):'未设置';
  const agendaTime=(value:string|null|undefined)=>!value||!value.includes('T')?'全天':new Intl.DateTimeFormat('zh-CN',{timeZone:me.timezone,hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value));
  const error=q.error||local.error||sync.error||setSync.error||action.error;
  return <div className="stack"><Card className="card"><h3>事项与日程</h3><Text size="sm">这里显示已排期的待办和 Google 日程。点击日程可先查看详情；提醒时间与日程时间分别管理。</Text>{sync.data?.available&&<Group mt="sm"><Button variant={sync.data.enabled?'filled':'light'} disabled={setSync.isPending} onClick={()=>{if(!sync.data?.enabled&&window.confirm('开启后，新建或改期的有时间事项将自动写入 Google 日历；不会自动补写历史任务。继续？'))setSync.mutate(true);else if(sync.data?.enabled)setSync.mutate(false);}}>Google 同步：{sync.data.enabled?'已开启':'已关闭'}</Button><Text size="xs" c="dimmed">关闭只停止后续同步，不会删除此前已创建的 Google 事件。</Text></Group>}</Card>{!me.capabilities.calendar&&<Alert color="yellow">当前账号没有授权的 Google Calendar；本地已排期事项仍可查看。</Alert>}{me.capabilities.calendar&&<Group><Button onClick={()=>{setSelected(null);setOp('create');setTitle('');setStart('');setEnd('');setLocalDirty(false);setShowForm(true);setDetailOpen(false);}}>新建 Google 事件</Button><Button variant="light" onClick={()=>{q.refetch();local.refetch();}}>刷新日程</Button></Group>}
    {error&&<Alert color="red" title="操作未完成">{error instanceof ApiError&&[409,412].includes(error.status)?'操作未确认完成；请刷新事件后核对状态。':error instanceof Error?error.message:'请稍后重试'}</Alert>}
    {(q.isFetching||local.isFetching)&&<Loader size="sm"/>}{(q.data?.truncated||local.data?.truncated)&&<Alert color="yellow">该时间范围事件过多，已截断；缩小日期范围查看完整数据。</Alert>}
    <Group className="calendar-view-controls" justify="space-between" align="center">
      <SegmentedControl aria-label="日历视图" value={view} onChange={changeView} data={[{value:'dayGridMonth',label:'月'},{value:'dayGridWeek',label:'周'},{value:'listDay',label:'日'}]}/>
      {mobile&&view==='dayGridMonth'&&<SegmentedControl aria-label="月历大小" value={monthSize} onChange={value=>setMonthSize(value as 'compact'|'comfortable')} data={[{value:'compact',label:'紧凑'},{value:'comfortable',label:'舒适'}]}/>}
      <Group gap="sm" className="calendar-legend"><span className="calendar-legend-todo">待办</span><span className="calendar-legend-google">Google 日程</span></Group>
    </Group>
    <div className={`calendar-viewport ${mobile&&view==='dayGridMonth'?`calendar-mobile-month calendar-month-${monthSize}`:''}`}><FullCalendar ref={calendarRef} key={mobile?'mobile':'desktop'} plugins={[dayGridPlugin,listPlugin,interactionPlugin]} locale={zhCnLocale} initialView={mobile&&view==='dayGridWeek'?'listWeek':view} headerToolbar={mobile?{left:'prev,next',center:'title',right:'today'}:{left:'prev,next today',center:'title',right:''}} height="auto" fixedWeekCount={false} dayHeaderFormat={mobile?{weekday:'narrow'}:undefined} eventDisplay={mobile&&view==='dayGridMonth'?'none':'block'} eventTimeFormat={{hour:'2-digit',minute:'2-digit',hour12:false}} dayMaxEvents={4} moreLinkClick="popover" datesSet={dateRange} dateClick={info=>{if(mobile&&view==='dayGridMonth')setSelectedDay(info.dateStr);else changeView('listDay',info.date);}} dayCellContent={mobile&&view==='dayGridMonth'?info=>{const key=calendarCellKey(info.date);const events=calendarEvents.filter(e=>appearsOnDay(e,key,me.timezone));return <span className={`calendar-mobile-date ${selectedDay===key?'selected':''}`}><span>{info.date.getDate()}</span>{events.length>0&&<span className="calendar-mobile-markers" aria-label={`${events.length}项日程`}>{events.slice(0,3).map((event,index)=><i key={`${event.id}:${index}`} className={event.id.startsWith('todo:')?'todo':'google'}/>)}</span>}</span>;}:undefined} eventClick={eventClick} events={calendarEvents}/></div>
    {mobile&&view==='dayGridMonth'&&<Card className="card calendar-day-agenda"><h3>{selectedDay.slice(5).replace('-', '月')}日 · {selectedDayEvents.length} 项日程</h3>{selectedDayEvents.length?selectedDayEvents.map(e=><button className={`calendar-agenda-item ${e.id.startsWith('todo:')?'todo':'google'}`} key={e.id} onClick={()=>openEvent(e.id)}><span className="calendar-agenda-time">{agendaTime(e.start)}</span><span className="calendar-agenda-title">{e.title}</span><span className="calendar-agenda-type">{e.id.startsWith('todo:')?'待办':'Google'}</span></button>):<Text size="sm" c="dimmed">这一天没有已排期日程。点选其他日期查看。</Text>}</Card>}
    <Modal opened={detailOpen} onClose={()=>setDetailOpen(false)} title={todoId?'待办详情':'Google 日程详情'} centered size="md">
      {todoId?todoDetail.isPending?<Loader size="sm"/>:todoDetail.error?<Alert color="red">待办详情读取失败，请稍后重试。</Alert>:todoDetail.data?<div className="stack"><Text fw={700}>{todoDetail.data.title}</Text><Text size="sm">计划：{formatDateTime(todoDetail.data.planned_start_at)} — {formatDateTime(todoDetail.data.planned_end_at)}</Text><Text size="sm">状态：{{pending:'待办',in_progress:'进行中',done:'已完成'}[todoDetail.data.status]}</Text><Text size="sm">提醒：{todoDetail.data.reminder_summary.at?formatDateTime(todoDetail.data.reminder_summary.at):'未设置'}</Text>{todoDetail.data.description&&<Text className="long-text" size="sm">{todoDetail.data.description}</Text>}<Button variant="light" onClick={()=>{setDetailOpen(false);onTasks(todoId);}}>去任务页编辑</Button></div>:null
      :detail.isPending?<Loader size="sm"/>:detail.error?<Alert color="red">Google 日程详情读取失败，请稍后重试。</Alert>:detail.data?<div className="stack"><Text fw={700}>{detail.data.title}</Text><Text size="sm">开始：{detail.data.start?.dateTime?formatDateTime(detail.data.start.dateTime):detail.data.start?.date||'未知'}</Text><Text size="sm">结束：{detail.data.end?.dateTime?formatDateTime(detail.data.end.dateTime):detail.data.end?.date||'未知'}</Text>{detail.data.location&&<Text size="sm">地点：{detail.data.location}</Text>}{detail.data.description&&<Text className="long-text" size="sm">{detail.data.description}</Text>}{detail.data.start?.date&&!detail.data.start.dateTime?<Alert color="yellow">全天事件当前仅支持查看。</Alert>:<Button variant="light" onClick={()=>{setShowForm(true);setDetailOpen(false);}}>编辑此 Google 日程</Button>}</div>:null}
    </Modal>
    {me.capabilities.calendar&&showForm&&<Card className="card"><Group justify="space-between"><h3>Google 事件操作</h3><Button variant="subtle" onClick={()=>setShowForm(false)}>收起</Button></Group><Text size="sm" c="dimmed">{selected?`已选择精确事件：${detail.data?.title||selected.title||'读取中'}`:'仅用于独立 Google 事件；日常事项请在任务页创建并排期。'}</Text>
      <Group className="calendar-form"><Select label="操作" value={op} onChange={v=>{setOp(v as typeof op);setLocalDirty(true);}} data={['create','update','delete'].map(v=>({value:v,label:{create:'创建',update:'修改',delete:'删除'}[v as typeof op]}))}/><TextInput label="标题" value={title} onChange={e=>{setTitle(e.target.value);setLocalDirty(true);}} disabled={op==='delete'}/><TextInput label="开始" type="datetime-local" value={start} onChange={e=>{setStart(e.target.value);setLocalDirty(true);}} disabled={op==='delete'}/><TextInput label="结束" type="datetime-local" value={end} onChange={e=>{setEnd(e.target.value);setLocalDirty(true);}} disabled={op==='delete'}/></Group>
      <Text size="xs" c="dimmed">时间输入按账号时区 {me.timezone} 显示；修改和删除会核对事件版本。夏令时不存在或重复的时间会要求重新选择。</Text><Button mt="md" disabled={op!=='create'&&(!selected||!detail.data?.etag||detail.isFetching)||op!=='delete'&&(!title||!start||!end||Boolean(detail.data?.start?.date&&!detail.data.start.dateTime))} loading={action.isPending} onClick={()=>{if(op==='delete'&&!window.confirm('确定删除该精确 Google 事件？'))return;action.mutate();}}>执行操作</Button>{message&&<Alert mt="md" color="blue">{message}</Alert>}
    </Card>}</div>;
}
