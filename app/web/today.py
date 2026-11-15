"""Bounded owner-scoped Today composite; each source reports its own health."""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select

from app.db.models import Todo, TodoCalendarMirror, TodoStatus
from app.db.session import AsyncSessionLocal
from app.web.api import _authenticated
from app.web.auth import no_store
from app.web.calendar import calendar_events
from app.web.reminders import reminder_view
from app.web.todos import todo_view

router = APIRouter(prefix="/api/v3", tags=["web-today"])


@router.get("/today")
async def today(request: Request, response: Response, date: str | None = None,
                tz: str = "Asia/Shanghai", source: str = "all"):
    user, _ = await _authenticated(request)
    if source not in {"all", "todo", "reminder", "calendar"}:
        raise HTTPException(422, "InvalidTodaySource")
    try:
        zone = ZoneInfo(tz)
        local_day = datetime.strptime(date, "%Y-%m-%d").date() if date else datetime.now(zone).date()
    except (ValueError, ZoneInfoNotFoundError):
        raise HTTPException(422, "InvalidDayOrTimezone") from None
    start = datetime.combine(local_day, datetime.min.time(), zone)
    end = datetime.combine(local_day + timedelta(days=1), datetime.min.time(), zone)
    todos, reminder_items = [], []
    if source in {"all", "todo", "reminder"}:
        async with AsyncSessionLocal() as db:
            if source in {"all", "todo"}:
                todos = (await db.scalars(select(Todo).where(
                    Todo.user_id == user.id, Todo.status != TodoStatus.done,
                    Todo.due_date.is_not(None), Todo.due_date < end.astimezone(timezone.utc))
                    .order_by(Todo.due_date).limit(100))).all()
            if source in {"all", "reminder"}:
                reminders = (await db.scalars(select(Todo).where(
                    Todo.user_id == user.id, Todo.reminder_state == "ACTIVE",
                    Todo.reminder_sent.is_(False),
                    Todo.reminder_at >= max(start.astimezone(timezone.utc), datetime.now(timezone.utc)),
                    Todo.reminder_at < end.astimezone(timezone.utc))
                    .order_by(Todo.reminder_at).limit(100))).all()
                reminder_items = [await reminder_view(db, item) for item in reminders]
    sources = {"todo": "ready" if source in {"all", "todo"} else "not_requested",
               "reminder": "ready" if source in {"all", "reminder"} else "not_requested",
               "calendar": "not_requested"}
    events = []
    if source in {"all", "calendar"}:
        sources["calendar"] = "unavailable"
        async with AsyncSessionLocal() as db:
            planned = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.planned_start_at.is_not(None),
                Todo.planned_start_at < end.astimezone(timezone.utc),
                Todo.planned_end_at > start.astimezone(timezone.utc))
                .order_by(Todo.planned_start_at).limit(100))).all()
            mirrored_ids = set((await db.scalars(select(TodoCalendarMirror.event_id).where(
                TodoCalendarMirror.todo_id.in_([item.id for item in planned]),
                TodoCalendarMirror.status == "SYNCED",
                TodoCalendarMirror.event_id.is_not(None)))).all()) if planned else set()
        events = [{"source": "todo", "todo_id": item.id, "calendar_id": "local",
                   "event_id": item.id, "title": item.title,
                   "start": {"dateTime": item.planned_start_at.isoformat()},
                   "end": {"dateTime": item.planned_end_at.isoformat()},
                   "status": item.status.value} for item in planned]
        sources["calendar"] = "local_only"
        try:
            calendar = await calendar_events(request, response,
                                             from_=start.isoformat(), to=end.isoformat(), tz=tz)
            events += [item for item in calendar["events"]
                       if item.get("event_id") not in mirrored_ids]
            sources["calendar"] = "ready"
        except HTTPException:
            pass
    no_store(response)
    return {"date": local_day.isoformat(), "todos": [todo_view(row) for row in todos],
            "events": events, "reminders": reminder_items, "sources": sources,
            "generated_at": datetime.now(timezone.utc).isoformat()}
