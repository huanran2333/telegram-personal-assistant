import { lazy, Suspense, useEffect, useState } from 'react';
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Badge, Button, Card, Group, Loader, Modal, Select, Stack, Text, Textarea, TextInput } from '@mantine/core';
import { useForm } from 'react-hook-form';
import { api, ApiError, SubmissionKeys, setCsrf } from './api';
import type { CalendarEvent, Me, Reminder, Todo } from './api';
import { bootstrapTelegram, inTelegram } from './telegram_bootstrap';
import { bindTelegramBack, guardUnsavedChanges, mountTelegramLifecycle } from './telegram_lifecycle';
import { zonedInput, zonedISO } from './zoned_time';

const CalendarPage = lazy(() => import('./calendar_page'));

type Page = 'Today'|'Tasks'|'Calendar'|'Reminders';
const pages:Page[] = ['Today','Tasks','Calendar','Reminders'];
const labels:Record<Page,string> = {Today:'今日',Tasks:'任务',Calendar:'日历',Reminders:'提醒'};
function pageFromUrl():Page { const value=new URLSearchParams(location.search).get('page'); return pages.find(p=>p===value)||'Today'; }
function formatTime(value:string|null|undefined,zone:string):string { if(!value)return '未设置';return new Intl.DateTimeFormat('zh-CN',{timeZone:zone,dateStyle:'medium',timeStyle:'short'}).format(new Date(value)); }
function Err({error}:{error:unknown}) { return error ? <Alert color="red" title="操作未完成">{error instanceof ApiError && [409,412].includes(error.status) ? '内容已变化或操作冲突，请刷新后核对再提交。' : error instanceof ApiError && error.status===401 ? '登录已失效，请重新登录。' : error instanceof ApiError && error.status===403 ? '安全校验未通过，请刷新页面后重试。' : error instanceof ApiError && error.status===503 ? '服务暂不可用；已提交的操作请先查询状态，不要换一个请求重复执行。' : error instanceof TypeError ? '网络连接中断，请检查网络后刷新；已提交操作会在服务器继续。' : error instanceof Error ? error.message : '请稍后重试'}</Alert> : null; }
function Loading({loading,children}:{loading:boolean;children:React.ReactNode}) { return loading ? <Loader size="sm"/> : <>{children}</>; }

function Login({onDone}:{onDone:()=>void}) {
  const [password,setPassword]=useState(''); const [token,setToken]=useState(''); const [error,setError]=useState('');
  useEffect(()=>{ api<{csrf_token:string}>('/auth/bootstrap').then(r=>setToken(r.csrf_token)).catch(e=>setError(String(e))); },[]);
  async function submit(e:React.FormEvent) { e.preventDefault(); setError('');
    try { await api('/auth/login',{method:'POST',headers:{'X-CSRF-Token':token},body:JSON.stringify({password})}); onDone(); }
    catch(e) { setError(e instanceof Error ? e.message : '登录失败'); }
  }
  return <form className="login stack" onSubmit={submit}><h1>Personal Assistant</h1><Text>登录你的私人工作台</Text>
    {error && <Alert color="red">{error}</Alert>}<TextInput type="password" label="密码" value={password} onChange={e=>setPassword(e.target.value)} required/>
    <Button type="submit" disabled={!token}>登录</Button></form>;
}

export function Workbench() {
  const [page,setPage]=useState<Page>(pageFromUrl); const [focusId,setFocusId]=useState(()=>new URLSearchParams(location.search).get('item')||'');
  const [dirty,setDirty]=useState(false); const qc=useQueryClient();
  const [launch,setLaunch]=useState<'pending'|'ready'|'failed'>('pending');
  const [launchError,setLaunchError]=useState('');
  useEffect(()=>{bootstrapTelegram().then(()=>setLaunch('ready')).catch(e=>{setLaunchError(e instanceof Error?e.message:'工作台入口无法验证，请重新发送 /workbench。');setLaunch('failed');});},[]);
  const me=useQuery({queryKey:['me'],queryFn:()=>api<Me>('/me'),retry:false,enabled:launch==='ready'});
  useEffect(()=>{if(me.data) setCsrf(me.data.csrf_token);},[me.data]);
  useEffect(()=>mountTelegramLifecycle(()=>qc.invalidateQueries()),[qc]);
  useEffect(()=>guardUnsavedChanges(dirty),[dirty]);
  function navigate(target:Page,id=''):boolean {
    if(dirty&&!window.confirm('有未保存的输入，确定离开当前页面吗？'))return false;
    setDirty(false);setPage(target);setFocusId(id);
    const next=new URL(location.href);next.searchParams.set('page',target);
    if(id)next.searchParams.set('item',id);else next.searchParams.delete('item');
    history.pushState(null,'',next.pathname+next.search);
    if(target==='Today')qc.invalidateQueries({queryKey:['today']});
    if(target==='Tasks')qc.invalidateQueries({queryKey:['todos']});
    if(target==='Reminders')qc.invalidateQueries({queryKey:['reminders']});
    return true;
  }
  useEffect(()=>{
    const pop=()=>{if(dirty&&!window.confirm('有未保存的输入，确定离开当前页面吗？')){history.pushState(null,'',`?page=${page}`);return;}setDirty(false);setPage(pageFromUrl());setFocusId(new URLSearchParams(location.search).get('item')||'');};
    window.addEventListener('popstate',pop);return()=>window.removeEventListener('popstate',pop);
  },[dirty,page]);
  useEffect(()=>bindTelegramBack(()=>navigate('Today'),page!=='Today'),[page,dirty]);
  useEffect(()=>{const expired=()=>qc.invalidateQueries({queryKey:['me']});window.addEventListener('web-auth-expired',expired);return()=>window.removeEventListener('web-auth-expired',expired);},[qc]);
  async function logout() { await api('/auth/logout',{method:'POST'});setCsrf('');qc.clear();window.location.replace('/workbench/'); }
  if(launch==='pending') return <div className="login"><Loader/></div>;
  if(launch==='failed') return <div className="login"><Alert color="red" title="工作台入口不可用">{launchError}</Alert></div>;
  if(me.isPending) return <div className="login"><Loader/></div>;
  if(me.error instanceof ApiError && me.error.status===401) return inTelegram() ?
    <div className="login"><Alert color="red">登录已失效，请返回 Bot 私聊，重新发送 /workbench 获取新按钮。</Alert></div> :
    <Login onDone={()=>qc.invalidateQueries({queryKey:['me']})}/>;
  if(me.error || !me.data) return <div className="login"><Err error={me.error}/><Button onClick={()=>qc.invalidateQueries({queryKey:['me']})}>重试</Button></div>;
  if(inTelegram() && me.data.auth_source!=='telegram') return <div className="login"><Alert color="red">请从 Bot 私聊重新打开工作台。</Alert></div>;
  const content = page==='Today' ? <Today me={me.data} navigate={navigate}/> : page==='Tasks' ? <Tasks me={me.data} focusId={focusId} navigate={navigate} onDirty={setDirty}/> : page==='Calendar' ? <Suspense fallback={<Loader size="sm"/>}><CalendarPage me={me.data} focusId={focusId} onTasks={id=>navigate('Tasks',id)} onDirty={setDirty}/></Suspense> :
    page==='Reminders' ? <Reminders me={me.data} focusId={focusId} navigate={navigate} onDirty={setDirty}/> :
    <Reminders me={me.data} focusId={focusId} navigate={navigate} onDirty={setDirty}/>;
  return <div className="shell"><aside className="rail"><div className="brand">Personal Assistant</div>
    {pages.map(p=><button className={page===p?'active':''} key={p} onClick={()=>navigate(p)}>{labels[p]}</button>)}
    <div style={{flex:1}}/><Text size="sm">{me.data.display_label}</Text><button onClick={logout}>退出</button>
  </aside><main className="main" style={{ margin: '0 auto' }}><div className="page-head"><h1>{labels[page]}</h1><Group><Text c="dimmed">{me.data.display_label}</Text><Button className="mobile-logout" size="xs" variant="subtle" onClick={logout}>退出</Button></Group></div>{content}</main>
    <nav className="bottom-nav" aria-label="工作台导航">{pages.map(p=><button className={page===p?'active':''} aria-current={page===p?'page':undefined} key={p} onClick={()=>navigate(p)}>{labels[p]}</button>)}</nav>
  </div>;
}

type TodayData={date:string;todos:Todo[];events:CalendarEvent[];reminders:Reminder[];sources:Record<string,string>};
function Today({me,navigate}:{me:Me;navigate:(page:Page,id?:string)=>void}) {
  const source=(name:'todo'|'calendar'|'reminder')=>api<TodayData>(`/today?source=${name}&tz=${encodeURIComponent(me.timezone)}`);
  const todos=useQuery({queryKey:['today','todo',me.timezone],queryFn:()=>source('todo'),refetchInterval:60000});
  const calendar=useQuery({queryKey:['today','calendar',me.timezone],queryFn:()=>source('calendar'),refetchInterval:60000});
  const reminders=useQuery({queryKey:['today','reminder',me.timezone],queryFn:()=>source('reminder'),refetchInterval:60000});
  return <div className="grid">
    <Card className="card"><h3>今日与逾期任务</h3><Err error={todos.error}/><Loading loading={todos.isPending}>{todos.data?.todos.length ? todos.data.todos.map(t=><button className="list-row row-button" key={t.id} onClick={()=>navigate('Tasks',t.id)}><span>{t.title}</span><Text size="xs" c="dimmed">{formatTime(t.due_at,me.timezone)}</Text></button>) : <Text c="dimmed">暂无到期任务</Text>}</Loading></Card>
    <Card className="card"><h3>今日日程</h3><Err error={calendar.error}/><Loading loading={calendar.isPending}>{calendar.data?.sources.calendar==='local_only'&&<Alert color="yellow">Google 日历暂不可用；以下为本地已排期事项。</Alert>}{calendar.data?.events.length ? calendar.data.events.map(e=><button className="list-row row-button" key={`${e.calendar_id}:${e.event_id}`} onClick={()=>e.calendar_id==='local'&&e.event_id?navigate('Tasks',e.event_id):navigate('Calendar',JSON.stringify([e.calendar_id,e.event_id]))}>{e.calendar_id==='local'?'事项 · ':'Google · '}{e.title}<Text size="xs" c="dimmed">{e.start?.dateTime?formatTime(e.start.dateTime,me.timezone):e.start?.date||'时间未知'}</Text></button>) : <Text c="dimmed">暂无日程</Text>}</Loading></Card>
    <Card className="card"><h3>即将触发的提醒</h3><Err error={reminders.error}/><Loading loading={reminders.isPending}>{reminders.data?.reminders.length ? reminders.data.reminders.map(r=><button className="list-row row-button" key={r.todo_id} onClick={()=>navigate('Reminders',r.todo_id)}><span>{r.title}</span><Badge>{r.delivery.delivery_state||r.definition_state}</Badge></button>) : <Text c="dimmed">暂无计划提醒</Text>}</Loading></Card>
  </div>;
}

type TodoInput={title:string;description:string;due_at:string;planned_start_at:string;planned_end_at:string;reminder_at:string;priority:'low'|'medium'|'high'};
function Tasks({me,focusId,navigate,onDirty}:{me:Me;focusId:string;navigate:(page:Page,id?:string)=>boolean;onDirty:(dirty:boolean)=>void}) {
  const qc=useQueryClient(); const [filter,setFilter]=useState('all'); const [search,setSearch]=useState(''); const [term,setTerm]=useState('');
  const [selected,setSelected]=useState<Todo|null>(null); const [detailDirty,setDetailDirty]=useState(false); const [keys]=useState(()=>new SubmissionKeys()); const [notice,setNotice]=useState('');
  const {register,handleSubmit,reset,setValue,formState}=useForm<TodoInput>({defaultValues:{title:'',description:'',due_at:'',planned_start_at:'',planned_end_at:'',reminder_at:'',priority:'medium'}});
  useEffect(()=>{const timer=setTimeout(()=>setTerm(search.trim()),250);return()=>clearTimeout(timer);},[search]);
  useEffect(()=>{onDirty(formState.isDirty||detailDirty);return()=>onDirty(false);},[formState.isDirty,detailDirty,onDirty]);
  const focused=useQuery({queryKey:['todo-detail',focusId],enabled:!!focusId,refetchOnMount:'always',queryFn:()=>api<Todo>(`/todos/${encodeURIComponent(focusId)}`)});
  useEffect(()=>{if(focused.data&&!focused.isFetching)setSelected(focused.data);},[focused.data,focused.isFetching]);
  const q=useInfiniteQuery({queryKey:['todos',filter,term],initialPageParam:'',queryFn:({pageParam})=>api<{items:Todo[];next_cursor:string|null}>(`/todos?limit=100${filter==='all'?'':`&status=${filter}`}${term?`&q=${encodeURIComponent(term)}`:''}${pageParam?`&cursor=${encodeURIComponent(pageParam)}`:''}`),getNextPageParam:last=>last.next_cursor||undefined});
  const items=q.data?.pages.flatMap(page=>page.items)||[];
  const create=useMutation({mutationFn:(v:TodoInput)=>{const body={title:v.title,description:v.description||null,due_at:v.due_at?zonedISO(v.due_at,me.timezone):null,planned_start_at:v.planned_start_at?zonedISO(v.planned_start_at,me.timezone):null,planned_end_at:v.planned_end_at?zonedISO(v.planned_end_at,me.timezone):null,reminder_at:v.reminder_at?zonedISO(v.reminder_at,me.timezone):null,priority:v.priority};return api<Todo>('/todos',{method:'POST',body:JSON.stringify(body)},keys.for('todo-create',body));},onSuccess:()=>{keys.clear('todo-create');reset();qc.invalidateQueries({queryKey:['todos']});qc.invalidateQueries({queryKey:['today']});qc.invalidateQueries({queryKey:['scheduled-todos']});qc.invalidateQueries({queryKey:['reminders']});}});
  const complete=useMutation({mutationFn:(t:Todo)=>api<Todo>(`/todos/${t.id}/complete`,{method:'POST',headers:{'If-Match':t.etag}}),onSuccess:()=>{qc.invalidateQueries({queryKey:['todos']});qc.invalidateQueries({queryKey:['today']});qc.invalidateQueries({queryKey:['reminders']});qc.invalidateQueries({queryKey:['scheduled-todos']});}});
  const patch=useMutation({mutationFn:({todo,values}:{todo:Todo;values:Partial<Todo>})=>api<Todo>(`/todos/${todo.id}`,{method:'PATCH',headers:{'If-Match':todo.etag},body:JSON.stringify(values)}),onSuccess:()=>{setSelected(null);setDetailDirty(false);qc.invalidateQueries({queryKey:['todos']});qc.invalidateQueries({queryKey:['today']});qc.invalidateQueries({queryKey:['reminders']});qc.invalidateQueries({queryKey:['scheduled-todos']});qc.invalidateQueries({queryKey:['todo-detail']});}});
  const deletion=useMutation({mutationFn:(t:Todo)=>api<{deleted:boolean}>(`/todos/${t.id}`,{method:'DELETE',headers:{'If-Match':t.etag}},keys.for(`todo-delete:${t.id}`,{etag:t.etag})),onSuccess:(_,t)=>{keys.clear(`todo-delete:${t.id}`);setSelected(null);setDetailDirty(false);setNotice('任务已删除；若启用了 Google 时间块同步，远端清理会由后台继续处理。');qc.invalidateQueries({queryKey:['todos']});qc.invalidateQueries({queryKey:['today']});qc.invalidateQueries({queryKey:['scheduled-todos']});qc.invalidateQueries({queryKey:['reminders']});}});
  // A list row may be a cached snapshot while a reminder mutation refetches.
  // Open the authoritative detail before enabling version-guarded mutations.
  const openDetail=useMutation({mutationFn:(id:string)=>api<Todo>(`/todos/${encodeURIComponent(id)}`),onSuccess:t=>{patch.reset();deletion.reset();setSelected(t);setDetailDirty(false);}});
  return <div className="stack"><Card className="card"><h3>创建任务</h3><form className="stack" onSubmit={handleSubmit(v=>create.mutate(v))}>
    <TextInput label="任务" placeholder="例如：整理本周计划" {...register('title',{required:true})}/><Textarea label="说明" {...register('description')}/>
    <Group grow><TextInput label="计划开始（设定后显示在日历）" type="datetime-local" {...register('planned_start_at')}/><TextInput label="计划结束（留空默认 1 小时）" type="datetime-local" {...register('planned_end_at')}/></Group>
    <Group grow><TextInput label="截止时间（可选，与计划时段不同）" type="datetime-local" {...register('due_at')}/><TextInput label="提醒时间（可选，可稍后修改）" type="datetime-local" {...register('reminder_at')}/></Group><Select label="优先级" data={[{value:'low',label:'低'},{value:'medium',label:'中'},{value:'high',label:'高'}]} defaultValue="medium" onChange={v=>setValue('priority',(v||'medium') as TodoInput['priority'])}/>
    <Button type="submit" loading={create.isPending}>创建任务</Button><Err error={create.error}/></form></Card>
    <Group className="filter-row"><TextInput label="搜索任务" placeholder="标题或说明" value={search} onChange={e=>setSearch(e.target.value)}/><Select label="状态筛选" value={filter} onChange={v=>setFilter(v||'all')} data={[{value:'all',label:'全部'},{value:'pending',label:'待办'},{value:'in_progress',label:'进行中'},{value:'done',label:'已完成'}]}/><Button variant="light" onClick={()=>q.refetch()}>刷新</Button></Group>
    {notice&&<Alert color="blue">{notice}</Alert>}<Err error={q.error||complete.error||openDetail.error}/><Loading loading={q.isPending}><Card className="card">{items.length ? items.map(t=><div className="list-row" key={t.id}><div><b>{t.title}</b><Text size="sm" c="dimmed">{t.planned_start_at?`计划 ${formatTime(t.planned_start_at,me.timezone)}`:'未排期'} · {t.due_at?`截止 ${formatTime(t.due_at,me.timezone)}`:'无截止'} · {t.status}</Text></div><div className="actions"><Button size="xs" variant="light" loading={openDetail.isPending&&openDetail.variables===t.id} disabled={openDetail.isPending} onClick={()=>openDetail.mutate(t.id)}>详情/编辑</Button>{t.status!=='done'&&<Button size="xs" loading={complete.isPending} onClick={()=>complete.mutate(t)}>完成</Button>}</div></div>) : <Text c="dimmed">没有任务</Text>}</Card>{q.hasNextPage&&<Button variant="light" loading={q.isFetchingNextPage} onClick={()=>q.fetchNextPage()}>加载更多任务</Button>}</Loading>
    <Err error={focused.error}/><Modal opened={!!selected} onClose={()=>{if(detailDirty&&!window.confirm('有未保存的任务修改，确定关闭吗？'))return;setSelected(null);setDetailDirty(false);}} title="任务详情" centered size="lg" scrollAreaComponent={undefined}>{selected&&<TaskDetail key={`${selected.id}:${selected.version}`} todo={selected} zone={me.timezone} onDirty={setDetailDirty} onSave={values=>patch.mutate({todo:selected,values})} onDelete={()=>deletion.mutate(selected)} onReminders={()=>{if(navigate('Reminders',selected.id)){setSelected(null);setDetailDirty(false);}}}/>}<Err error={patch.error||deletion.error}/>{((patch.error instanceof ApiError&&patch.error.status===412)||(deletion.error instanceof ApiError&&deletion.error.status===412))&&<Button variant="light" onClick={()=>{setSelected(null);setDetailDirty(false);q.refetch();if(focusId)focused.refetch();}}>关闭并刷新</Button>}</Modal>
  </div>;
}
function TaskDetail({todo,zone,onDirty,onSave,onDelete,onReminders}:{todo:Todo;zone:string;onDirty:(dirty:boolean)=>void;onSave:(v:Partial<Todo>)=>void;onDelete:()=>void;onReminders:()=>void}) {
  const [title,setTitle]=useState(todo.title); const [description,setDescription]=useState(todo.description||''); const [due,setDue]=useState(todo.due_at?zonedInput(todo.due_at,zone):'');const [planStart,setPlanStart]=useState(todo.planned_start_at?zonedInput(todo.planned_start_at,zone):'');const [planEnd,setPlanEnd]=useState(todo.planned_end_at?zonedInput(todo.planned_end_at,zone):'');
  const [timeError,setTimeError]=useState('');
  useEffect(()=>{onDirty(title!==todo.title||description!==(todo.description||'')||due!==(todo.due_at?zonedInput(todo.due_at,zone):'')||planStart!==(todo.planned_start_at?zonedInput(todo.planned_start_at,zone):'')||planEnd!==(todo.planned_end_at?zonedInput(todo.planned_end_at,zone):''));},[title,description,due,planStart,planEnd,todo,zone,onDirty]);
  function save() { try { onSave({title,description,due_at:due?zonedISO(due,zone):null,planned_start_at:planStart?zonedISO(planStart,zone):null,planned_end_at:planStart&&planEnd?zonedISO(planEnd,zone):null});setTimeError(''); } catch { setTimeError('此时间在账号时区中不存在或有歧义，请重新选择。'); } }
  return <Stack><TextInput label="任务" value={title} onChange={e=>setTitle(e.target.value)}/><Textarea label="说明" value={description} onChange={e=>setDescription(e.target.value)}/><TextInput label="计划开始" type="datetime-local" value={planStart} onChange={e=>setPlanStart(e.target.value)}/><TextInput label="计划结束（留空默认 1 小时）" type="datetime-local" value={planEnd} onChange={e=>setPlanEnd(e.target.value)}/><TextInput label="截止时间" type="datetime-local" value={due} onChange={e=>setDue(e.target.value)}/>
    {timeError&&<Alert color="red">{timeError}</Alert>}<Text size="sm" c="dimmed">提醒：{todo.reminder_summary.state}</Text><Group><Button onClick={save}>保存</Button><Button variant="light" onClick={onReminders}>管理提醒</Button><Button color="red" variant="light" onClick={()=>{if(window.confirm(`确定删除「${todo.title}」？`))onDelete();}}>删除任务</Button></Group>
    <Text size="xs" c="dimmed">版本 {todo.version}；如被其他设备修改，保存会提示冲突并可刷新。</Text></Stack>;
}

function Reminders({me,focusId,navigate,onDirty}:{me:Me;focusId:string;navigate:(page:Page,id?:string)=>void;onDirty:(dirty:boolean)=>void}) {
  const qc=useQueryClient();const [selected,setSelected]=useState<Reminder|null>(null);const [at,setAt]=useState('');const [pickerSearch,setPickerSearch]=useState('');const [draftTouched,setDraftTouched]=useState(false);const [keys]=useState(()=>new SubmissionKeys());
  const q=useInfiniteQuery({queryKey:['reminders'],initialPageParam:'',queryFn:({pageParam})=>api<{items:Reminder[];next_cursor:string|null}>(`/reminders?limit=100${pageParam?`&cursor=${encodeURIComponent(pageParam)}`:''}`),getNextPageParam:last=>last.next_cursor||undefined,refetchInterval:60000});
  const reminderItems=q.data?.pages.flatMap(page=>page.items)||[];
  const focused=useQuery({queryKey:['reminder-detail',focusId],enabled:!!focusId,queryFn:()=>api<Reminder>(`/reminders/${encodeURIComponent(focusId)}`)});
  useEffect(()=>{if(focused.data){setSelected(focused.data);setAt(focused.data.configured_at?zonedInput(focused.data.configured_at,me.timezone):'');setDraftTouched(false);}},[focused.data,me.timezone]);
  const todo=useQuery({queryKey:['todos','reminder-picker',pickerSearch],queryFn:()=>api<{items:Todo[]}>(`/todos?limit=100&status=pending${pickerSearch.trim()?`&q=${encodeURIComponent(pickerSearch.trim())}`:''}`)});
  useEffect(()=>{onDirty(draftTouched);return()=>onDirty(false);},[draftTouched,onDirty]);
  const change=useMutation({mutationFn:({r,op}:{r:Reminder;op:'set'|'pause'|'resume'|'cancel'})=>{const body=op==='set'?{at:zonedISO(at,me.timezone),tz:me.timezone}:op==='resume'?{at:at?zonedISO(at,me.timezone):null}:null;return api<Reminder>(`/todos/${r.todo_id}/reminder${op==='set'?'':`/${op}`}`,{method:op==='set'?'PUT':'POST',headers:{'If-Match':r.etag},body:body?JSON.stringify(body):undefined},keys.for(`reminder:${r.todo_id}:${op}`,{etag:r.etag,body}));},onSuccess:async(result,v)=>{keys.clear(`reminder:${v.r.todo_id}:${v.op}`);setSelected(result);setAt(result.configured_at?zonedInput(result.configured_at,me.timezone):'');setDraftTouched(false);await Promise.all([qc.invalidateQueries({queryKey:['reminders']}),qc.invalidateQueries({queryKey:['todos']}),qc.invalidateQueries({queryKey:['today']}),qc.invalidateQueries({queryKey:['reminder-detail']}),qc.invalidateQueries({queryKey:['todo-detail']})]);}});
  return <div className="stack"><Card className="card"><h3>设置或改期一次性提醒</h3><TextInput label="查找任务" placeholder="输入标题或说明可查找更多任务" value={pickerSearch} onChange={e=>setPickerSearch(e.target.value)}/><Err error={todo.error||focused.error}/><Group className="filter-row"><Select label="任务" searchable data={(todo.data?.items||[]).map(t=>({value:t.id,label:t.title}))} value={selected?.todo_id||null} onChange={id=>{const t=todo.data?.items.find(x=>x.id===id);if(t){setSelected({todo_id:t.id,title:t.title,definition_state:t.reminder_summary.state,next_at:t.reminder_summary.at,configured_at:t.reminder_summary.at,todo_version:t.version,etag:t.etag,reminder_version:t.reminder_summary.version,delivery:{occurrence_state:null,delivery_state:null,chunk_states:[],current_version:false,may_have_sent:false}});setAt(t.reminder_summary.at?zonedInput(t.reminder_summary.at,me.timezone):'');setDraftTouched(false);}}}/><TextInput label="提醒时间" type="datetime-local" value={at} onChange={e=>{setAt(e.target.value);setDraftTouched(true);}}/><Button mt={24} disabled={!selected||!at||change.isPending} loading={change.isPending} onClick={()=>selected&&change.mutate({r:selected,op:'set'})}>设置/改期</Button></Group>{selected&&<Text size="sm">当前选择：{selected.title} · 定义 {selected.definition_state}，版本 {selected.reminder_version}</Text>}</Card>
    <Err error={q.error||change.error}/>{change.error&&<Text size="sm">若响应丢失，可用同一操作重试；服务器幂等键和版本会防止重复修改。</Text>}<Loading loading={q.isPending}><Card className="card"><h3>提醒定义与投递历史</h3>{reminderItems.length?reminderItems.map(r=><div className="list-row" key={r.todo_id}><div><b>{r.title}</b><Text size="sm" c="dimmed">计划：{formatTime(r.next_at||r.configured_at,me.timezone)} · 定义版本 {r.reminder_version}</Text><Text size="xs" c="dimmed">投递：{r.delivery.delivery_state||r.delivery.occurrence_state||'未触发'}{r.delivery.may_have_sent?'（可能已发送）':''}{!r.delivery.current_version&&r.delivery.occurrence_state?' · 历史版本':''}</Text><details><summary>查看发送历史</summary>{r.history?.length?r.history.map((h,i)=><Text size="xs" key={`${h.created_at}:${i}`}>版本 {h.reminder_version} · {h.occurrence_state} / {h.delivery_state||'未建投递'} · chunk {h.chunk_states.join(', ')||'无'}{h.may_have_sent?' · 可能已发送':''}{h.current_version?' · 当前版本':' · 历史版本'}</Text>):<Text size="xs">暂无发生记录</Text>}</details></div><div className="actions"><Badge>{r.definition_state}</Badge><Button size="sm" variant="light" onClick={()=>{setSelected(r);setAt(r.configured_at?zonedInput(r.configured_at,me.timezone):'');setDraftTouched(false);}}>管理</Button><Button size="sm" variant="subtle" onClick={()=>navigate('Tasks',r.todo_id)}>任务</Button>{r.definition_state==='ACTIVE'&&<Button size="sm" variant="light" disabled={change.isPending} onClick={()=>change.mutate({r,op:'pause'})}>暂停</Button>}{r.definition_state==='PAUSED'&&<Button size="sm" variant="light" disabled={change.isPending} onClick={()=>change.mutate({r,op:'resume'})}>恢复</Button>}{['ACTIVE','PAUSED'].includes(r.definition_state)&&<Button size="sm" color="red" variant="light" disabled={change.isPending} onClick={()=>{if(window.confirm('取消未来尚未开始发送的提醒？已开始发送的消息不能撤回。'))change.mutate({r,op:'cancel'});}}>取消</Button>}</div></div>):<Text c="dimmed">暂无已配置提醒</Text>}</Card>{q.hasNextPage&&<Button variant="light" loading={q.isFetchingNextPage} onClick={()=>q.fetchNextPage()}>加载更多提醒</Button>}</Loading>
    <Text size="sm" c="dimmed">暂停或取消只能阻止尚未开始发送的消息；SENDING/UNCERTAIN 不能被撤回。</Text></div>;
}
