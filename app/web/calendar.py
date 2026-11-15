"""Owner-scoped Web Calendar reads and direct, idempotent form writes."""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import unquote
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agent.calendar import CalendarError, CalendarUnavailable
from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.workflow import bind_approved_calendar_version
from app.agent.workflow_runtime import get_action_executor, get_calendar_adapter
from app.config import get_settings
from app.db.models import Todo, TodoCalendarMirror, User, WebMutation
from app.db.session import AsyncSessionLocal
from app.web.api import _authenticated
from app.web.auth import no_store
from app.web.api import _key
from app.web.todos import _digest, mutation_claim

router = APIRouter(prefix="/api/v3", tags=["web-calendar"])


def stable_form_action_id(user_id: str, key: str) -> str:
    return uuid5(NAMESPACE_URL, f"web-form:{user_id}:{key}").hex


def _window(start_text: str | None, end_text: str | None) -> tuple[datetime, datetime]:
    if not start_text or not end_text:
        raise HTTPException(422, "BoundedCalendarWindowRequired")
    try:
        start, end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
    except ValueError:
        raise HTTPException(422, "InvalidCalendarWindow") from None
    # FullCalendar's month grid can show six complete weeks. Allow that
    # bounded window plus a daylight-saving hour without opening unbounded reads.
    if start.tzinfo is None or end.tzinfo is None or end <= start or end - start > timedelta(days=43):
        raise HTTPException(422, "InvalidCalendarWindow")
    return start, end


@router.get("/calendar/scheduled-todos")
async def scheduled_todos(request: Request, response: Response,
                          from_: str | None = None, to: str | None = None):
    """The local Todo is authority even when Google is offline or sync is off."""
    user, _ = await _authenticated(request)
    start, end = _window(request.query_params.get("from", from_), to)
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Todo, TodoCalendarMirror).outerjoin(
            TodoCalendarMirror, TodoCalendarMirror.todo_id == Todo.id).where(
            Todo.user_id == user.id, Todo.planned_start_at.is_not(None),
            Todo.planned_start_at < end, Todo.planned_end_at > start)
            .order_by(Todo.planned_start_at, Todo.id).limit(201))).all()
    no_store(response)
    return {"events": [{"todo_id": todo.id, "title": todo.title,
                        "start": todo.planned_start_at.isoformat(),
                        "end": todo.planned_end_at.isoformat(),
                        "status": todo.status.value,
                        "google_sync_state": mirror.status if mirror else "OFF",
                        "google_event_id": mirror.event_id if mirror and mirror.status == "SYNCED" else None,
                        "google_calendar_id": mirror.calendar_id if mirror and mirror.status == "SYNCED" else None}
                       for todo, mirror in rows[:200]], "truncated": len(rows) > 200}


class CalendarSyncPreference(BaseModel):
    enabled: bool


@router.get("/calendar/sync-preference")
async def get_calendar_sync_preference(request: Request, response: Response):
    user, _ = await _authenticated(request)
    no_store(response)
    return {"enabled": bool(user.calendar_time_block_sync),
            "available": bool(get_settings().google_calendar_enabled and
                              user.telegram_id == get_settings().google_calendar_owner_telegram_id),
            "existing_remote_events_remain_when_disabled": True}


@router.patch("/calendar/sync-preference")
async def set_calendar_sync_preference(request: Request, response: Response,
                                       body: CalendarSyncPreference):
    user, _ = await _authenticated(request, write=True)
    settings = get_settings()
    if (not settings.google_calendar_enabled or
            user.telegram_id != settings.google_calendar_owner_telegram_id):
        raise HTTPException(403, "CalendarOwnerRequired")
    async with AsyncSessionLocal() as db:
        target = await db.get(User, user.id, with_for_update=True)
        if target is None:
            raise HTTPException(404, "UserNotFound")
        target.calendar_time_block_sync = body.enabled
        await db.commit()
    no_store(response)
    return {"enabled": body.enabled, "available": True,
            "existing_remote_events_remain_when_disabled": True}


async def _calendar_for(request: Request):
    user, _ = await _authenticated(request)
    settings = get_settings()
    if not settings.google_calendar_enabled or user.telegram_id != settings.google_calendar_owner_telegram_id:
        raise HTTPException(404, "CalendarUnavailable")
    return get_calendar_adapter()


def _safe_event(event: dict) -> dict:
    return {"calendar_id": str(event.get("calendarId") or "primary"),
            "event_id": str(event.get("id") or ""),
            "etag": event.get("etag"), "title": event.get("summary") or "(untitled)",
            "start": event.get("start"), "end": event.get("end"),
            "status": event.get("status", "confirmed"),
            "updated": event.get("updated")}


@router.get("/calendar/events")
async def calendar_events(request: Request, response: Response,
                          from_: str | None = None, to: str | None = None,
                          tz: str | None = None):
    # The documented wire name is `from`; avoid Python's reserved word.
    start_text = request.query_params.get("from", from_)
    start, end = _window(start_text, to)
    calendar = await _calendar_for(request)
    try:
        await calendar.resolve_read_scope("primary")
        # FullCalendar supplies millisecond ISO timestamps. The pinned Calendar
        # MCP list-events schema accepts whole-second ISO 8601 only; preserve
        # the parsed instant/offset while removing unsupported fractional time.
        events = await calendar.search_events(
            "", start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"))
    except (CalendarError, CalendarUnavailable):
        raise HTTPException(503, "CalendarReadUnavailable") from None
    no_store(response)
    return {"events": [_safe_event(item) for item in events[:100] if isinstance(item, dict) and item.get("id")],
            "freshness": "current", "truncated": len(events) > 100}


@router.get("/calendar/events/{calendar_id}/{event_id}")
async def calendar_event_detail(request: Request, response: Response,
                                calendar_id: str, event_id: str):
    calendar = await _calendar_for(request)
    calendar_id, event_id = unquote(calendar_id), unquote(event_id)
    if not calendar_id or not event_id or len(calendar_id) > 256 or len(event_id) > 1024:
        raise HTTPException(422, "InvalidCalendarIdentity")
    try:
        _, concrete_id = await calendar.resolve_read_scope(calendar_id)
        event = await calendar.get_event(concrete_id, event_id)
    except (CalendarError, CalendarUnavailable):
        raise HTTPException(503, "CalendarReadUnavailable") from None
    no_store(response)
    return {"calendar_id": concrete_id, "event_id": event.id,
            "etag": event.etag, "title": event.summary, "start": event.start,
            "end": event.end, "status": event.status, "updated": event.updated,
            "description": event.description, "location": event.location,
            "recurrence": event.recurrence}


class CalendarFields(BaseModel):
    title: str | None = Field(default=None, max_length=512)
    start: datetime | None = None
    end: datetime | None = None
    timeZone: str | None = Field(default=None, max_length=80)


class CalendarForm(BaseModel):
    operation: str
    calendar_id: str = Field(max_length=256)
    event_id: str | None = Field(default=None, max_length=1024)
    baseline_etag: str | None = Field(default=None, max_length=256)
    fields: CalendarFields | None = None


@router.post("/calendar/actions")
async def calendar_action(request: Request, response: Response,
                          body: CalendarForm):
    user, _ = await _authenticated(request, write=True)
    key = _key(request)
    client = body.model_dump(mode="json")
    if body.operation not in {"create", "update", "delete"}:
        raise HTTPException(422, "UnsupportedCalendarOperation")
    if body.operation == "create" and (body.event_id or body.baseline_etag):
        raise HTTPException(422, "UnexpectedCalendarTarget")
    calendar = await _calendar_for(request)
    if body.operation != "create" and (not body.event_id or not body.baseline_etag):
        raise HTTPException(422, "ExactApprovedTargetRequired")
    if body.operation == "delete" and body.fields is not None:
        raise HTTPException(422, "DeleteCannotContainPatch")
    if body.operation != "delete":
        fields = body.fields
        if (fields is None or not fields.title or not fields.start or not fields.end or
                not fields.timeZone or fields.timeZone != get_settings().default_timezone or
                fields.start.tzinfo is None or fields.end.tzinfo is None or
                fields.end <= fields.start):
            raise HTTPException(422, "InvalidCalendarFields")
        zone = ZoneInfo(fields.timeZone)
        if (fields.start.utcoffset() != fields.start.astimezone(zone).utcoffset() or
                fields.end.utcoffset() != fields.end.astimezone(zone).utcoffset()):
            raise HTTPException(422, "CalendarTimezoneOffsetMismatch")
        if body.operation == "create" and fields.start <= datetime.now(fields.start.tzinfo):
            raise HTTPException(422, "FutureEventRequired")
    args = {"user_id": user.id, "calendarId": body.calendar_id}
    if body.operation != "delete":
        args.update({"title": body.fields.title, "start": body.fields.start.isoformat(),
                     "end": body.fields.end.isoformat(), "timeZone": body.fields.timeZone})
    if body.event_id:
        args["eventId"] = body.event_id
    op = {"create": "create_event", "update": "update_event",
          "delete": "delete_event"}[body.operation]
    async with AsyncSessionLocal() as db:
        prior = await db.scalar(select(WebMutation).where(
            WebMutation.user_id == user.id, WebMutation.operation == "calendar_action",
            WebMutation.idempotency_key == key))
        if prior is not None and prior.request_hash != _digest(client):
            raise HTTPException(409, "IdempotencyConflict")
        prior_proposal = (prior.result or {}).get("proposal") if prior else None
    if prior is not None and not prior_proposal:
        raise HTTPException(409, "CalendarActionNotReady")
    if prior_proposal:
        proposal = ActionProposal.model_validate(prior_proposal)
    else:
        proposal = ActionProposal(
                action_id=stable_form_action_id(user.id, key), domain="calendar", operation=op,
                args=args, risk_level=RiskLevel.DESTRUCTIVE if op == "delete_event" else RiskLevel.EXTERNAL_WRITE,
                reason="Owner-selected Calendar form and exact resource version",
                human_readable_preview=f"{body.operation} Calendar event: {body.fields.title if body.fields else body.event_id}")
        try:
            if body.operation == "create":
                account, concrete = await calendar.resolve_create_scope(body.calendar_id)
                proposal.args["account"], proposal.args["calendarId"] = account, concrete
            else:
                await bind_approved_calendar_version(calendar, proposal)
                if proposal.approved_calendar_baseline.etag != body.baseline_etag:
                    raise HTTPException(412, "CalendarVersionConflict")
        except CalendarError as exc:
            if str(exc) == "ManagedTodoCalendarEvent":
                raise HTTPException(409, "待办同步的 Google 时间块不能在日历页独立修改或删除；请操作对应待办。") from None
            if "Recurring or multi-resource" in str(exc):
                raise HTTPException(422, "UnsupportedRecurrence") from None
            if "target/version changed" in str(exc):
                raise HTTPException(409, "CalendarTargetChanged") from None
            raise HTTPException(503, "CalendarWritePreparationUnavailable") from None
    async with AsyncSessionLocal() as db:
        proof, fresh = await mutation_claim(
            db, user_id=user.id, operation="calendar_action", key=key, body=client)
        if fresh:
            proof.result = {"proposal": proposal.model_dump(mode="json")}
        else:
            proposal = ActionProposal.model_validate(proof.result["proposal"])
        await db.commit()
    # The action identity is durable before provider I/O. Retries reuse the
    # same evidence and the coordinator reconciles unknown remote outcomes.
    result = await get_action_executor().execute(proposal)
    no_store(response)
    if result.status == ActionStatus.EXECUTED:
        return {"status": "executed", "message": "Google 日历操作已确认完成。"}
    if result.status == ActionStatus.FAILED:
        category = result.result.get("failure_category")
        if category == "preparation_unavailable":
            code, reason = 503, "Google 日历在写入前不可用，本次写入未发送；请稍后重试。"
        elif category in {"provider_permission_denied", "oauth_unavailable", "oauth_refresh_failed"}:
            code, reason = 403, "Google 日历拒绝了本次操作；请检查账号授权。"
        elif category == "etag_conflict":
            code, reason = 412, "日历事件版本已变化，本次修改未执行；请刷新后重试。"
        else:
            code, reason = 409, "Google 日历未确认本次操作完成；请刷新并核对事件状态。"
        return JSONResponse(status_code=code, content={
            "status": "failed", "detail": reason},
            headers={"Cache-Control": "no-store"})
    return JSONResponse(status_code=202, content={
        "status": "pending_verification",
        "message": "Google 日历操作结果尚未确认，系统将继续核实；请勿重复提交新的操作。"},
        headers={"Cache-Control": "no-store"})
