"""Isolated PG18 only: canonical item, transactional queue and fake Google projection."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import Response

from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarEventSnapshot
from app.agent.todo_calendar_sync import TodoCalendarSync
from app.agent.tools import make_todo_tools
from app.db import crud
from app.db.models import Todo, TodoCalendarMirror
from app.web import calendar as calendar_api
from app.web import todos as todo_api
from app.web import reminders as reminder_api


# Every test in this module writes to the identity-checked PG18 fixture.
# Without this marker pytest's default offline selection tries to collect it.
pytestmark = pytest.mark.integration


class FakeCalendar:
    def __init__(self):
        self.events = {}
        self.creates = 0
        self.updates = 0
        self.deletes = 0

    def _snapshot(self, event):
        return CalendarEventSnapshot.model_validate(event)

    async def resolve_create_scope(self, calendar_id):
        assert calendar_id == "primary"
        return "synthetic-account", "synthetic-calendar"

    async def get_event(self, calendar_id, event_id):
        if event_id not in self.events:
            raise CalendarError("not_found_or_inaccessible")
        return self._snapshot(self.events[event_id])

    async def create_time_block(self, calendar_id, event_id, body):
        assert calendar_id == "synthetic-calendar"
        assert body["reminders"] == {"useDefault": False, "overrides": []}
        assert not body.get("attendees")
        if event_id in self.events:
            raise CalendarError("conflict")
        self.creates += 1
        self.events[event_id] = {**body, "etag": '"1"', "status": "confirmed"}
        return self._snapshot(self.events[event_id])

    async def update_event_if_match(self, calendar_id, event_id, etag, patch):
        if self.events[event_id]["etag"] != etag:
            raise CalendarError("etag_conflict")
        self.updates += 1
        self.events[event_id].update(patch)
        self.events[event_id]["etag"] = f'"{self.updates + 1}"'
        return self._snapshot(self.events[event_id])

    async def delete_event_if_match(self, calendar_id, event_id, etag):
        if self.events[event_id]["etag"] != etag:
            raise CalendarError("etag_conflict")
        self.deletes += 1
        del self.events[event_id]
        return {"accepted": True}


@pytest.mark.asyncio
async def test_new_user_sync_defaults_on_and_queues_new_item(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        owner = await db.get(type(user), user.id)
        assert owner.calendar_time_block_sync is True
        item = await crud.create_todo(db, user.id, "synthetic default on",
            planned_start_at=datetime.now(timezone.utc) + timedelta(days=3))
        await db.commit()
        mirror = await db.get(TodoCalendarMirror, item.id)
        assert mirror.status == "PENDING"


@pytest.mark.asyncio
async def test_opt_in_time_block_create_update_and_delete(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    start = datetime.now(timezone.utc) + timedelta(days=3)
    async with factory() as db:
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = False  # explicit opt-out for this case
        await db.commit()
        unsynced = await crud.create_todo(db, user.id, "synthetic off", planned_start_at=start)
        await db.commit()
        assert await db.get(TodoCalendarMirror, unsynced.id) is None
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = True
        await db.commit()
        assert await db.get(TodoCalendarMirror, unsynced.id) is None  # no historical backfill
        item = await crud.create_todo(db, user.id, "synthetic timed", planned_start_at=start)
        await db.commit()
        await db.refresh(item)
        assert item.planned_end_at - item.planned_start_at == timedelta(hours=1)
        mirror = await db.get(TodoCalendarMirror, item.id)
        assert mirror.status == "PENDING" and mirror.desired_revision == 1
    google = FakeCalendar()
    sync = TodoCalendarSync(factory, google, user.telegram_id)
    assert await sync.tick() == 1
    assert google.creates == 1
    assert await sync.tick() == 0
    async with factory() as db:
        mirror = await db.get(TodoCalendarMirror, item.id)
        assert mirror.status == "SYNCED" and mirror.event_id in google.events
        await crud.update_todo(db, item.id, user.id, title="synthetic revised")
        await db.commit()
        await db.refresh(mirror)  # trigger changes are outside the ORM identity map
        assert mirror.status == "PENDING"
    assert await sync.tick() == 1
    assert google.updates == 1 and google.creates == 1
    async with factory() as db:
        assert await crud.delete_todo(db, item.id, user.id)
        await db.commit()
        assert await db.get(Todo, item.id) is None
        mirror = await db.get(TodoCalendarMirror, item.id)
        await db.refresh(mirror)
        assert mirror.desired_operation == "DELETE"
    assert await sync.tick() == 1
    assert google.deletes == 1
    async with factory() as db:
        assert (await db.get(TodoCalendarMirror, item.id)).status == "REMOVED"


@pytest.mark.asyncio
async def test_remote_edit_requires_review_and_off_switch_stops_claim(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    start = datetime.now(timezone.utc) + timedelta(days=3)
    async with factory() as db:
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = True
        item = await crud.create_todo(db, user.id, "synthetic conflict", planned_start_at=start)
        await db.commit()
    google = FakeCalendar()
    sync = TodoCalendarSync(factory, google, user.telegram_id)
    assert await sync.tick() == 1
    async with factory() as db:
        await crud.update_todo(db, item.id, user.id, title="synthetic new intent")
        await db.commit()
        mirror = await db.get(TodoCalendarMirror, item.id)
        google.events[mirror.event_id]["etag"] = '"manual"'
        google.events[mirror.event_id]["summary"] = "manual remote edit"
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = False
        await db.commit()
    assert await sync.tick() == 0 and google.updates == 0
    async with factory() as db:
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = True
        await db.commit()
    assert await sync.tick() == 1
    async with factory() as db:
        mirror = await db.get(TodoCalendarMirror, item.id)
        assert mirror.status == "REVIEW" and mirror.last_error_category == "remote_event_changed"
    assert google.events[mirror.event_id]["summary"] == "manual remote edit"


@pytest.mark.asyncio
async def test_rolled_back_todo_does_not_enqueue_projection(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = True
        await db.commit()
        item = await crud.create_todo(db, user.id, "synthetic rollback",
            planned_start_at=datetime.now(timezone.utc) + timedelta(days=3))
        item_id = item.id
        await db.rollback()
    async with factory() as db:
        assert await db.get(Todo, item_id) is None
        assert await db.get(TodoCalendarMirror, item_id) is None


@pytest.mark.asyncio
async def test_calendar_local_timeline_and_owner_scoped_toggle(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    other = await new_user()
    start = datetime.now(timezone.utc) + timedelta(days=3)
    async with factory() as db:
        # Exercise the explicit opt-out -> enable transition; new users now
        # default to enabled and would otherwise enqueue a mirror immediately.
        owner_row = await db.get(type(owner), owner.id)
        owner_row.calendar_time_block_sync = False
        await db.flush()
        planned = await crud.create_todo(db, owner.id, "synthetic owner plan",
                                         planned_start_at=start)
        await crud.create_todo(db, owner.id, "synthetic unscheduled")
        await crud.create_todo(db, other.id, "synthetic other plan",
                               planned_start_at=start)
        await db.commit()
    active = owner

    async def authenticated(_request, *, write=False):
        return active, SimpleNamespace(csrf_token="synthetic")

    monkeypatch.setattr(calendar_api, "_authenticated", authenticated)
    monkeypatch.setattr(calendar_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(calendar_api, "get_settings", lambda: SimpleNamespace(
        google_calendar_enabled=True, google_calendar_owner_telegram_id=owner.telegram_id))
    request = Request({"type": "http", "method": "GET", "path": "/api/v3/calendar/scheduled-todos",
                       "query_string": b"", "headers": []})
    view = await calendar_api.scheduled_todos(request, Response(),
        from_=(start - timedelta(hours=1)).isoformat(), to=(start + timedelta(hours=3)).isoformat())
    assert [item["todo_id"] for item in view["events"]] == [planned.id]
    assert view["events"][0]["google_sync_state"] == "OFF"
    changed = await calendar_api.set_calendar_sync_preference(
        request, Response(), calendar_api.CalendarSyncPreference(enabled=True))
    assert changed["enabled"] is True
    async with factory() as db:
        assert (await db.get(type(owner), owner.id)).calendar_time_block_sync is True
        assert await db.get(TodoCalendarMirror, planned.id) is None  # enabling is not backfill
    active = other
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await calendar_api.set_calendar_sync_preference(
            request, Response(), calendar_api.CalendarSyncPreference(enabled=True))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_agent_timed_item_and_later_reminder_change(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    tools = {item.name: item for item in make_todo_tools(factory)}
    start = (datetime.now(timezone.utc) + timedelta(days=4)).replace(microsecond=0)
    remind = start - timedelta(hours=1)
    created = await tools["create_todo"].ainvoke({
        "title": "synthetic meeting", "user_id": user.id,
        "planned_start_str": start.isoformat()})
    assert "synthetic meeting" in created
    async with factory() as db:
        meeting = (await db.scalars(select(Todo).where(
            Todo.user_id == user.id, Todo.title == "synthetic meeting"))).one()
        assert meeting.planned_end_at - meeting.planned_start_at == timedelta(hours=1)
        assert meeting.reminder_at is None
        item_id = meeting.id
    assert "synthetic meeting" in await tools["set_reminder"].ainvoke({
        "user_id": user.id, "todo_id": item_id, "remind_at_str": remind.isoformat()})
    async with factory() as db:
        meeting = await db.get(Todo, item_id)
        assert meeting.reminder_at == remind and meeting.planned_start_at == start
    await tools["cancel_reminder"].ainvoke({"user_id": user.id, "todo_id": item_id})
    async with factory() as db:
        meeting = await db.get(Todo, item_id)
        assert meeting.reminder_at is None and meeting.planned_start_at == start
        assert meeting.reminder_state == "CANCELLED"
    await tools["update_todo"].ainvoke({
        "user_id": user.id, "todo_id": item_id,
        "planned_end_str": (start + timedelta(hours=2)).isoformat()})
    async with factory() as db:
        meeting = await db.get(Todo, item_id)
        assert meeting.planned_end_at - meeting.planned_start_at == timedelta(hours=2)

    notified = await tools["create_todo"].ainvoke({
        "title": "synthetic reminder only", "user_id": user.id,
        "reminder_at_str": remind.isoformat()})
    assert "synthetic reminder only" in notified
    async with factory() as db:
        row = (await db.scalars(select(Todo).where(
            Todo.user_id == user.id, Todo.title == "synthetic reminder only"))).one()
        assert row.planned_start_at == remind
        assert row.planned_end_at - row.planned_start_at == timedelta(hours=1)
        assert row.reminder_at == remind


@pytest.mark.asyncio
async def test_web_reminder_only_create_is_one_scheduled_item(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    remind = (datetime.now(timezone.utc) + timedelta(days=5)).replace(microsecond=0)

    async def authenticated(_request, *, write=False):
        return user, SimpleNamespace(csrf_token="synthetic")

    monkeypatch.setattr(todo_api, "_authenticated", authenticated)
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    request = Request({"type": "http", "method": "POST", "path": "/api/v3/todos",
                       "query_string": b"", "headers": [
                           (b"idempotency-key", b"syntheticreminder0001")]})
    result = await todo_api.create_web_todo(request, Response(), todo_api.TodoCreate(
        title="synthetic browser notification", reminder_at=remind))
    assert result["planned_start_at"] == remind.isoformat()
    assert result["reminder_summary"]["at"] == remind.isoformat()
    async with factory() as db:
        item = await db.get(Todo, result["id"])
        assert item.planned_end_at - item.planned_start_at == timedelta(hours=1)


@pytest.mark.asyncio
async def test_sync_unexpected_provider_failure_releases_lease(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        owner = await db.get(type(user), user.id)
        owner.calendar_time_block_sync = True
        item = await crud.create_todo(db, user.id, "synthetic retry",
            planned_start_at=datetime.now(timezone.utc) + timedelta(days=3))
        await db.commit()

    class BrokenCalendar(FakeCalendar):
        async def resolve_create_scope(self, _calendar_id):
            raise RuntimeError("synthetic provider failure")

    sync = TodoCalendarSync(factory, BrokenCalendar(), user.telegram_id)
    assert await sync.tick() == 1
    async with factory() as db:
        mirror = await db.get(TodoCalendarMirror, item.id)
        assert mirror.status == "RETRY"
        assert mirror.lease_token is None and mirror.lease_until is None
        assert mirror.next_attempt_at is not None


@pytest.mark.asyncio
async def test_web_reminder_on_unscheduled_todo_preserves_empty_plan(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    at = (datetime.now(timezone.utc) + timedelta(days=5)).replace(microsecond=0)
    async with factory() as db:
        item = await crud.create_todo(db, user.id, "synthetic initially unscheduled")
        await db.commit()
        item_id = item.id
        version = item.version

    async def authenticated(_request, *, write=False):
        return user, SimpleNamespace(csrf_token="synthetic")

    monkeypatch.setattr(reminder_api, "_authenticated", authenticated)
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)
    request = Request({"type": "http", "method": "PUT", "path": "/api/v3/todos/reminder",
                       "query_string": b"", "headers": [
                           (b"idempotency-key", b"syntheticreminderset01"),
                           (b"if-match", f'"{version}"'.encode())]})
    await reminder_api._change(request, Response(), item_id, "set", at)
    async with factory() as db:
        item = await db.get(Todo, item_id)
        assert item.reminder_at == at
        assert item.planned_start_at is None
        assert item.planned_end_at is None
