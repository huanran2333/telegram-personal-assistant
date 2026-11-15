"""One-shot Todo reminder definition; R4 occurrence/chunks remain delivery truth."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select

from app.db.models import OutboundDelivery, OutboundDeliveryChunk, ScheduledOccurrence, Todo, TodoStatus
from app.db.session import AsyncSessionLocal
from app.web.api import _authenticated, _key
from app.web.auth import no_store
from app.web.todos import mutation_claim, owned_todo, require_version

router = APIRouter(prefix="/api/v3", tags=["web-reminder"])


class ReminderTime(BaseModel):
    at: datetime
    tz: str


class ResumeTime(BaseModel):
    at: datetime | None = None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise HTTPException(422, "TimezoneRequired")
    return value.astimezone(timezone.utc)


async def reminder_view(db, todo: Todo) -> dict:
    rows = (await db.execute(select(ScheduledOccurrence, OutboundDelivery).outerjoin(
        OutboundDelivery, OutboundDelivery.occurrence_id == ScheduledOccurrence.id).where(
        ScheduledOccurrence.kind == "reminder", ScheduledOccurrence.todo_id == todo.id,
        ScheduledOccurrence.user_id == todo.user_id).order_by(
            ScheduledOccurrence.created_at.desc(), ScheduledOccurrence.id.desc()).limit(5))).all()
    delivery_ids = [delivery.id for _, delivery in rows if delivery is not None]
    chunk_rows = ((await db.scalars(select(OutboundDeliveryChunk).where(
        OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))).all() if delivery_ids else [])
    chunks_by_delivery = {delivery_id: [] for delivery_id in delivery_ids}
    for chunk in chunk_rows:
        chunks_by_delivery[chunk.delivery_id].append(chunk.state)
    history = []
    for occurrence, delivery in rows:
        chunk_states = chunks_by_delivery.get(delivery.id, []) if delivery else []
        history.append({"occurrence_state": occurrence.state,
                        "delivery_state": delivery.state if delivery else None,
                        "chunk_states": chunk_states,
                        "current_version": (occurrence.reminder_version_snapshot or 0) == todo.reminder_version,
                        "may_have_sent": any(state in {"SENDING", "UNCERTAIN", "SENT"}
                                               for state in chunk_states),
                        "created_at": occurrence.created_at.isoformat(),
                        "reminder_version": occurrence.reminder_version_snapshot})
    latest = history[0] if history else None
    return {"todo_id": todo.id, "title": todo.title,
            "definition_state": todo.reminder_state,
            "next_at": todo.reminder_at.isoformat() if todo.reminder_state == "ACTIVE" and todo.reminder_at else None,
            "configured_at": todo.reminder_at.isoformat() if todo.reminder_at else None,
            "todo_version": todo.version, "etag": f'"{todo.version}"',
            "reminder_version": todo.reminder_version,
            "delivery": {key: latest[key] for key in ("occurrence_state", "delivery_state",
                         "chunk_states", "current_version", "may_have_sent")}
                        if latest else {"occurrence_state": None, "delivery_state": None,
                                        "chunk_states": [], "current_version": False,
                                        "may_have_sent": False},
            "history": history}


@router.get("/reminders")
async def list_reminders(request: Request, response: Response,
                         state: str | None = None, limit: int = 50,
                         cursor: str | None = None):
    user, _ = await _authenticated(request)
    if not 1 <= limit <= 100 or state not in {None, "ACTIVE", "PAUSED", "CANCELLED", "NONE"}:
        raise HTTPException(422, "InvalidReminderFilter")
    async with AsyncSessionLocal() as db:
        stmt = select(Todo).where(Todo.user_id == user.id)
        if state:
            stmt = stmt.where(Todo.reminder_state == state)
        else:
            stmt = stmt.where(Todo.reminder_state != "NONE")
        if cursor:
            stmt = stmt.where(Todo.id > cursor)
        rows = (await db.scalars(stmt.order_by(Todo.id).limit(limit + 1))).all()
        result = [await reminder_view(db, item) for item in rows[:limit]]
    no_store(response)
    return {"items": result, "next_cursor": rows[limit - 1].id if len(rows) > limit else None}


@router.get("/reminders/{todo_id}")
async def get_reminder(request: Request, response: Response, todo_id: str):
    user, _ = await _authenticated(request)
    async with AsyncSessionLocal() as db:
        todo = await owned_todo(db, user.id, todo_id)
        result = await reminder_view(db, todo)
    no_store(response)
    return result


async def _change(request: Request, response: Response, todo_id: str,
                  operation: str, at: datetime | None = None):
    user, _ = await _authenticated(request, write=True)
    key = _key(request)
    if at is not None:
        at = _utc(at)
        if at <= datetime.now(timezone.utc):
            raise HTTPException(422, "FutureReminderRequired")
    async with AsyncSessionLocal() as db:
        proof, fresh = await mutation_claim(db, user_id=user.id,
            operation=f"reminder_{operation}", key=key,
            body={"todo_id": todo_id, "operation": operation,
                  "at": at.isoformat() if at else None,
                  "if_match": request.headers.get("if-match")})
        if fresh:
            todo = await owned_todo(db, user.id, todo_id, lock=True)
            require_version(request, todo)
            if todo.status == TodoStatus.done:
                raise HTTPException(409, "CompletedTodoCannotRemind")
            if operation == "set":
                if at is None:
                    raise HTTPException(422, "ReminderTimeRequired")
                todo.reminder_at, todo.reminder_state = at, "ACTIVE"
            elif operation == "pause":
                if todo.reminder_state != "ACTIVE":
                    raise HTTPException(409, "ReminderNotActive")
                todo.reminder_state = "PAUSED"
            elif operation == "resume":
                if todo.reminder_state != "PAUSED":
                    raise HTTPException(409, "ReminderNotPaused")
                target = at or todo.reminder_at
                if target is None or _utc(target) <= datetime.now(timezone.utc):
                    raise HTTPException(422, "FutureReminderRequired")
                todo.reminder_at, todo.reminder_state = target, "ACTIVE"
            elif operation == "cancel":
                if todo.reminder_state not in {"ACTIVE", "PAUSED"}:
                    raise HTTPException(409, "ReminderNotConfigured")
                todo.reminder_at, todo.reminder_state = None, "CANCELLED"
            todo.reminder_version += 1
            todo.version += 1
            todo.reminder_sent = False
            todo.reminder_sent_at = None
            todo.reminder_claimed_at = None
            await db.flush()
            proof.result = await reminder_view(db, todo)
        result = proof.result
        await db.commit()
    no_store(response)
    return result


@router.put("/todos/{todo_id}/reminder")
async def set_reminder(request: Request, response: Response, todo_id: str, body: ReminderTime):
    try:
        ZoneInfo(body.tz)
    except ZoneInfoNotFoundError:
        raise HTTPException(422, "InvalidTimezone") from None
    return await _change(request, response, todo_id, "set", body.at)


@router.post("/todos/{todo_id}/reminder/pause")
async def pause_reminder(request: Request, response: Response, todo_id: str):
    return await _change(request, response, todo_id, "pause")


@router.post("/todos/{todo_id}/reminder/resume")
async def resume_reminder(request: Request, response: Response, todo_id: str,
                          body: ResumeTime | None = None):
    return await _change(request, response, todo_id, "resume", body.at if body else None)


@router.post("/todos/{todo_id}/reminder/cancel")
async def cancel_reminder(request: Request, response: Response, todo_id: str):
    return await _change(request, response, todo_id, "cancel")
