"""Owner-scoped Todo forms; the Todo row is authority, not a browser cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import Priority, Todo, TodoStatus, WebMutation
from app.db.session import AsyncSessionLocal
from app.web.api import _authenticated, _key
from app.web.auth import no_store

router = APIRouter(prefix="/api/v3", tags=["web-todo"])


class TodoCreate(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=4000)
    due_at: datetime | None = None
    planned_start_at: datetime | None = None
    planned_end_at: datetime | None = None
    reminder_at: datetime | None = None
    priority: str = "medium"
    tags: list[str] = Field(default_factory=list, max_length=20)


class TodoPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=4000)
    due_at: datetime | None = None
    planned_start_at: datetime | None = None
    planned_end_at: datetime | None = None
    reminder_at: datetime | None = None
    priority: str | None = None
    status: str | None = None
    tags: list[str] | None = Field(default=None, max_length=20)


def todo_view(todo: Todo) -> dict:
    return {"id": todo.id, "title": todo.title, "description": todo.description,
            "due_at": todo.due_date.isoformat() if todo.due_date else None,
            "planned_start_at": todo.planned_start_at.isoformat() if todo.planned_start_at else None,
            "planned_end_at": todo.planned_end_at.isoformat() if todo.planned_end_at else None,
            "priority": todo.priority.value, "status": todo.status.value,
            "tags": todo.tags or [], "version": todo.version,
            "etag": f'"{todo.version}"',
            "reminder_summary": {"state": todo.reminder_state,
                                 "at": todo.reminder_at.isoformat() if todo.reminder_at else None,
                                 "version": todo.reminder_version}}


def require_version(request: Request, todo: Todo) -> None:
    supplied = request.headers.get("if-match")
    if supplied is None:
        raise HTTPException(428, "IfMatchRequired")
    if supplied != f'"{todo.version}"':
        raise HTTPException(412, "VersionConflict")


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def _planned_window(start: datetime | None, end: datetime | None) -> datetime | None:
    if start is None:
        if end is not None:
            raise HTTPException(422, "PlannedStartRequired")
        return None
    if start.tzinfo is None or (end is not None and end.tzinfo is None):
        raise HTTPException(422, "TimezoneRequired")
    end = end or start.astimezone(timezone.utc) + timedelta(hours=1)
    if end <= start:
        raise HTTPException(422, "InvalidPlannedWindow")
    return end


async def mutation_claim(db, *, user_id: str, operation: str, key: str, body: dict):
    digest = _digest(body)
    inserted = await db.scalar(pg_insert(WebMutation).values(
        id=str(uuid4()), user_id=user_id, operation=operation,
        idempotency_key=key, request_hash=digest, result={}
    ).on_conflict_do_nothing(constraint="uq_web_mutation_key").returning(WebMutation.id))
    row = await db.scalar(select(WebMutation).where(
        WebMutation.user_id == user_id, WebMutation.operation == operation,
        WebMutation.idempotency_key == key))
    if row is None or row.request_hash != digest:
        raise HTTPException(409, "IdempotencyConflict")
    return row, inserted is not None


async def owned_todo(db, user_id: str, todo_id: str, *, lock: bool = False) -> Todo:
    stmt = select(Todo).where(Todo.id == todo_id, Todo.user_id == user_id)
    if lock:
        stmt = stmt.with_for_update()
    row = await db.scalar(stmt)
    if row is None:
        raise HTTPException(404, "TodoNotFound")
    return row


@router.get("/todos")
async def list_web_todos(request: Request, response: Response,
                         status: str | None = None, limit: int = 50,
                         cursor: str | None = None, q: str | None = None):
    user, _ = await _authenticated(request)
    if not 1 <= limit <= 100 or status not in {None, "pending", "in_progress", "done"}:
        raise HTTPException(422, "InvalidTodoFilter")
    stmt = select(Todo).where(Todo.user_id == user.id)
    if q is not None:
        term = q.strip()
        if not term or len(term) > 120:
            raise HTTPException(422, "InvalidTodoSearch")
        # Literal matching: '%' and '_' in user text are not SQL wildcards.
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(or_(Todo.title.ilike(f"%{escaped}%", escape="\\"),
                              Todo.description.ilike(f"%{escaped}%", escape="\\")))
    if status:
        stmt = stmt.where(Todo.status == TodoStatus(status))
    if cursor:
        stmt = stmt.where(Todo.id > cursor)
    async with AsyncSessionLocal() as db:
        rows = (await db.scalars(stmt.order_by(Todo.id).limit(limit + 1))).all()
    no_store(response)
    return {"items": [todo_view(t) for t in rows[:limit]],
            "next_cursor": rows[limit - 1].id if len(rows) > limit else None}


@router.get("/todos/{todo_id}")
async def get_web_todo(request: Request, response: Response, todo_id: str):
    user, _ = await _authenticated(request)
    async with AsyncSessionLocal() as db:
        todo = await owned_todo(db, user.id, todo_id)
        result = todo_view(todo)
    no_store(response)
    return result


@router.post("/todos", status_code=201)
async def create_web_todo(request: Request, response: Response, body: TodoCreate):
    user, _ = await _authenticated(request, write=True)
    key = _key(request)
    if any(value is not None and value.tzinfo is None for value in
           (body.due_at, body.reminder_at)):
        raise HTTPException(422, "TimezoneRequired")
    if body.reminder_at is not None and body.reminder_at <= datetime.now(timezone.utc):
        raise HTTPException(422, "FutureReminderRequired")
    # A newly requested notification supplies a time for an otherwise
    # unscheduled item. The notification itself remains an independent field.
    planned_start = body.planned_start_at
    if planned_start is None and body.planned_end_at is None:
        planned_start = body.reminder_at
    planned_end = _planned_window(planned_start, body.planned_end_at)
    if body.priority not in {"low", "medium", "high"}:
        raise HTTPException(422, "InvalidPriority")
    if not body.title.strip():
        raise HTTPException(422, "TodoTitleRequired")
    async with AsyncSessionLocal() as db:
        proof, fresh = await mutation_claim(db, user_id=user.id,
                                            operation="todo_create", key=key,
                                            body=body.model_dump(mode="json"))
        if fresh:
            todo = Todo(user_id=user.id, title=body.title.strip(),
                        description=body.description, due_date=body.due_at,
                        planned_start_at=planned_start, planned_end_at=planned_end,
                        reminder_at=body.reminder_at,
                        reminder_state="ACTIVE" if body.reminder_at else "NONE",
                        priority=Priority(body.priority), tags=body.tags)
            db.add(todo)
            await db.flush()
            proof.result = todo_view(todo)
        result = proof.result
        await db.commit()
    no_store(response)
    return result


@router.patch("/todos/{todo_id}")
async def patch_web_todo(request: Request, response: Response, todo_id: str,
                         body: TodoPatch):
    user, _ = await _authenticated(request, write=True)
    changes = body.model_dump(exclude_unset=True)
    if not changes or any(key not in {"title", "description", "due_at", "priority", "status", "tags",
                                      "planned_start_at", "planned_end_at", "reminder_at"}
                          for key in changes):
        raise HTTPException(422, "InvalidTodoPatch")
    if "due_at" in changes and changes["due_at"] is not None and changes["due_at"].tzinfo is None:
        raise HTTPException(422, "TimezoneRequired")
    if "reminder_at" in changes and changes["reminder_at"] is not None and changes["reminder_at"].tzinfo is None:
        raise HTTPException(422, "TimezoneRequired")
    if ("reminder_at" in changes and changes["reminder_at"] is not None and
            changes["reminder_at"] <= datetime.now(timezone.utc)):
        raise HTTPException(422, "FutureReminderRequired")
    if ("title" in changes and (changes["title"] is None or not changes["title"].strip())):
        raise HTTPException(422, "TodoTitleRequired")
    if ("priority" in changes and changes["priority"] not in {"low", "medium", "high"}) or (
            "status" in changes and changes["status"] not in {"pending", "in_progress", "done"}):
        raise HTTPException(422, "InvalidTodoValue")
    async with AsyncSessionLocal() as db:
        todo = await owned_todo(db, user.id, todo_id, lock=True)
        require_version(request, todo)
        if "planned_start_at" in changes and "planned_end_at" not in changes:
            changes["planned_end_at"] = None
        if "planned_start_at" in changes or "planned_end_at" in changes:
            changes["planned_end_at"] = _planned_window(
                changes.get("planned_start_at", todo.planned_start_at),
                changes.get("planned_end_at", todo.planned_end_at))
        for name, value in changes.items():
            if name == "title":
                value = value.strip()
            if name == "priority":
                value = Priority(value)
            elif name == "status":
                value = TodoStatus(value)
            elif name == "reminder_at":
                todo.reminder_state = "ACTIVE" if value is not None else "CANCELLED"
                todo.reminder_version += 1
                todo.reminder_sent = False
                todo.reminder_sent_at = None
                todo.reminder_claimed_at = None
            setattr(todo, "due_date" if name == "due_at" else name, value)
        if todo.status == TodoStatus.done and todo.reminder_state in {"ACTIVE", "PAUSED"}:
            todo.reminder_state, todo.reminder_at = "CANCELLED", None
            todo.reminder_version += 1
        todo.version += 1
        todo.updated_at = datetime.now()
        await db.flush()
        result = todo_view(todo)
        await db.commit()
    no_store(response)
    return result


@router.post("/todos/{todo_id}/complete")
async def complete_web_todo(request: Request, response: Response, todo_id: str):
    user, _ = await _authenticated(request, write=True)
    async with AsyncSessionLocal() as db:
        todo = await owned_todo(db, user.id, todo_id, lock=True)
        require_version(request, todo)
        todo.status = TodoStatus.done
        if todo.reminder_state in {"ACTIVE", "PAUSED"}:
            todo.reminder_state, todo.reminder_at = "CANCELLED", None
            todo.reminder_version += 1
        todo.version += 1
        await db.flush()
        result = todo_view(todo)
        await db.commit()
    no_store(response)
    return result


@router.delete("/todos/{todo_id}")
async def delete_web_todo(request: Request, response: Response, todo_id: str):
    user, _ = await _authenticated(request, write=True)
    key = _key(request)
    supplied = request.headers.get("if-match")
    if supplied is None:
        raise HTTPException(428, "IfMatchRequired")
    async with AsyncSessionLocal() as db:
        proof, fresh = await mutation_claim(
            db, user_id=user.id, operation="todo_delete", key=key,
            body={"todo_id": todo_id, "if_match": supplied})
        if fresh:
            todo = await owned_todo(db, user.id, todo_id, lock=True)
            require_version(request, todo)
            proof.result = {"deleted": True, "title": todo.title}
            await db.delete(todo)
        result = proof.result
        await db.commit()
    no_store(response)
    return result
