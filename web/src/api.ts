export type Todo = { id: string; title: string; description: string | null; due_at: string | null;
  planned_start_at:string|null;planned_end_at:string|null;
  priority: 'low'|'medium'|'high'; status: 'pending'|'in_progress'|'done'; tags: string[];
  version: number; etag: string; reminder_summary: {state:string;at:string|null;version:number} };
export type Reminder = { todo_id:string; title:string; definition_state:string; next_at:string|null;
  configured_at:string|null; todo_version:number; etag:string; reminder_version:number;
  delivery:{occurrence_state:string|null;delivery_state:string|null;chunk_states:string[];
    current_version:boolean;may_have_sent:boolean};history?:Array<{occurrence_state:string;
      delivery_state:string|null;chunk_states:string[];current_version:boolean;may_have_sent:boolean;
      created_at:string;reminder_version:number|null}> };
export type CalendarEvent = {calendar_id:string;event_id:string;etag:string|null;title:string;
  start:{dateTime?:string;date?:string}|null;end:{dateTime?:string;date?:string}|null;status:string;updated?:string;
  description?:string|null;location?:string|null;recurrence?:string[]|null};
export type ScheduledTodoEvent = {todo_id:string;title:string;start:string;end:string;
  status:string;google_sync_state:string;google_event_id:string|null;google_calendar_id:string|null};
export type Me = {display_label:string;timezone:string;csrf_token:string;auth_source:'password'|'telegram';calendar_time_block_sync?:boolean;capabilities:{calendar:boolean;gmail:boolean;memory:boolean}};

let csrf = '';
export function setCsrf(token: string) { csrf = token; }
export const newKey = () => crypto.randomUUID();
// Keep one logical form submission stable across a lost HTTP response. A changed
// payload is a new operation; a confirmed response releases the key.
export class SubmissionKeys {
  private pending = new Map<string, { body: string; key: string }>();
  for(scope: string, body: unknown): string {
    const serialized = JSON.stringify(body);
    const current = this.pending.get(scope);
    if (current?.body === serialized) return current.key;
    const key = newKey();
    this.pending.set(scope, { body: serialized, key });
    return key;
  }
  clear(scope: string): void { this.pending.delete(scope); }
}
export class ApiError extends Error { constructor(public status:number, message:string) { super(message); } }
export async function api<T>(path:string, init:RequestInit = {}, key?:string):Promise<T> {
  const method = init.method || 'GET';
  const headers = new Headers(init.headers);
  if (init.body) headers.set('Content-Type','application/json');
  if (method !== 'GET' && method !== 'HEAD') {
    // Bootstrap/exchange carries a distinct one-use nonce CSRF token. Do not
    // overwrite an explicitly supplied token with an old session's token.
    if (csrf && !headers.has('X-CSRF-Token')) headers.set('X-CSRF-Token', csrf);
    if (key) headers.set('Idempotency-Key', key);
  }
  const response = await fetch(`/api/v3${path}`, { ...init, headers, credentials:'same-origin', cache:'no-store' });
  if (!response.ok) {
    if (response.status === 401 && !['/me','/auth/login','/auth/telegram/exchange',
      '/auth/telegram/revalidate'].includes(path))
      window.dispatchEvent(new Event('web-auth-expired'));
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, String(body.detail || response.statusText));
  }
  return response.status === 204 ? undefined as T : await response.json() as T;
}
