"""G3 real PG18 owner-scoped five-page data and cross-entry projections."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response
from sqlalchemy import select

from app.db import crud
from app.db.models import Todo
from app.scheduler.proactive import upsert_due_reminders
from app.web import today as today_api, todos as todo_api, reminders as reminder_api

pytestmark = pytest.mark.integration


def request(*, key=None, etag=None):
    headers = {}
    if key:
        headers["idempotency-key"] = key
    if etag:
        headers["if-match"] = etag
    return SimpleNamespace(headers=headers)


@pytest.mark.asyncio
async def test_bot_owned_todo_visible_in_workbench_and_web_edit_is_authoritative(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    owner, other = await new_user(), await new_user()
    async with factory() as db:
        row = await crud.create_todo(db, user_id=owner.id, title="synthetic Telegram-created G3 task")
        await db.commit()
        todo_id = row.id
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(today_api, "AsyncSessionLocal", factory)
    current = owner

    async def auth(_request, *, write=False):
        return current, None

    monkeypatch.setattr(todo_api, "_authenticated", auth)
    monkeypatch.setattr(today_api, "_authenticated", auth)
    listed = await todo_api.list_web_todos(request(), Response(), q="Telegram-created")
    assert [item["id"] for item in listed["items"]] == [todo_id]
    assert (await todo_api.list_web_todos(request(), Response(), q="%"))["items"] == []
    changed = await todo_api.patch_web_todo(
        request(etag=listed["items"][0]["etag"]), Response(), todo_id,
        todo_api.TodoPatch(title="synthetic updated in workbench"))
    async with factory() as db:
        assert (await db.get(Todo, todo_id)).title == changed["title"]
    current = other
    assert (await todo_api.list_web_todos(request(), Response(), q="workbench"))["items"] == []
    with pytest.raises(HTTPException) as forbidden:
        await todo_api.get_web_todo(request(), Response(), todo_id)
    assert forbidden.value.status_code == 404


@pytest.mark.asyncio
async def test_today_independent_sources_and_reminder_history_preserve_versions(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    owner, other = await new_user(), await new_user()
    for module in (todo_api, reminder_api, today_api):
        monkeypatch.setattr(module, "AsyncSessionLocal", factory)
    current = owner

    async def auth(_request, *, write=False):
        return current, None

    for module in (todo_api, reminder_api, today_api):
        monkeypatch.setattr(module, "_authenticated", auth)
    calls = []

    async def calendar(_request, _response, **kwargs):
        calls.append(kwargs)
        return {"events": [{"calendar_id": "primary", "event_id": "synthetic-event"}]}

    monkeypatch.setattr(today_api, "calendar_events", calendar)
    due = datetime.now(timezone.utc) + timedelta(seconds=30)
    todo = await todo_api.create_web_todo(request(key=uuid4().hex), Response(),
        todo_api.TodoCreate(title="synthetic due today", due_at=due))
    tasks = await today_api.today(request(), Response(), date=due.date().isoformat(), tz="UTC", source="todo")
    assert [item["id"] for item in tasks["todos"]] == [todo["id"]]
    assert tasks["sources"]["calendar"] == "not_requested" and not calls
    events = await today_api.today(request(), Response(), tz="UTC", source="calendar")
    assert events["events"][0]["event_id"] == "synthetic-event" and not events["todos"]
    assert len(calls) == 1
    configured = await reminder_api.set_reminder(request(key=uuid4().hex, etag=todo["etag"]),
        Response(), todo["id"], reminder_api.ReminderTime(at=due, tz="UTC"))
    await upsert_due_reminders(factory, bot_identity=123, timezone_name="UTC",
                               now=due + timedelta(seconds=1))
    rescheduled = await reminder_api.set_reminder(
        request(key=uuid4().hex, etag=configured["etag"]), Response(), todo["id"],
        reminder_api.ReminderTime(at=due + timedelta(minutes=10), tz="UTC"))
    detail = await reminder_api.get_reminder(request(), Response(), todo["id"])
    assert detail["reminder_version"] == rescheduled["reminder_version"]
    assert len(detail["history"]) == 1
    assert detail["history"][0]["current_version"] is False
    assert detail["history"][0]["reminder_version"] == configured["reminder_version"]
    current = other
    with pytest.raises(HTTPException) as forbidden:
        await reminder_api.get_reminder(request(), Response(), todo["id"])
    assert forbidden.value.status_code == 404
