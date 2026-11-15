"""V3.2 real isolated PG18: form authority and reminder send boundary."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response
from sqlalchemy import select, text
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.actions import ActionProposal, RiskLevel
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.workflow import build_productivity_graph
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.calendar_modify import CalendarModifyCoordinator
from app.config import get_settings
from app.db import crud
from app.db.models import ActionExecution, CalendarActionEvidence, WebRequest
from app.web.runs import WebRunWorker, claim_one, submit_run
from tests.test_workflow import FakeCalendar

from app.db.models import OutboundDelivery, OutboundDeliveryChunk, ScheduledOccurrence, Todo
from app.scheduler.proactive import (claim_occurrence, ready_occurrence, reconcile_occurrence,
                                     reminder_identity, upsert_due_reminders)
from app.telegram.durable import claim_next_chunk, finish_chunk
from app.web import reminders as reminder_api
from app.web import todos as todo_api
from app.web import calendar as calendar_api
from tests.test_r5_2_create_reconciliation import FakeCalendar as FakeR5Calendar
from tests.test_r5_3_calendar_modify import FakeConditionalCalendar
from zoneinfo import ZoneInfo

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_v32_migration_legacy_reminder_upgrade_is_idempotent(pg_harness):
    """A minimal pre-V3.2 schema keeps its Todo and backfills ACTIVE once."""
    _, _, engine = pg_harness
    from app.db.migrations.v3_2_workbench import apply as apply_v3_2
    schema = f"v32_upgrade_{uuid4().hex}"
    async with engine.begin() as conn:
        await conn.exec_driver_sql(f"CREATE SCHEMA {schema}")
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql(f"SET LOCAL search_path TO {schema}")
            await conn.exec_driver_sql("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY)")
            await conn.exec_driver_sql("CREATE TABLE todos (id VARCHAR(36) PRIMARY KEY, "
                                       "user_id VARCHAR(36) NOT NULL REFERENCES users(id), "
                                       "reminder_at TIMESTAMPTZ)")
            await conn.exec_driver_sql("CREATE TABLE scheduled_occurrences (id VARCHAR(120) PRIMARY KEY)")
            await conn.exec_driver_sql("CREATE TABLE app_schema_migrations (revision VARCHAR(80) PRIMARY KEY)")
            await conn.execute(text("INSERT INTO users(id) VALUES (:id)"), {"id": "synthetic-owner"})
            await conn.execute(text("INSERT INTO todos(id,user_id,reminder_at) "
                                    "VALUES (:id,:user_id,now())"),
                               {"id": "synthetic-todo", "user_id": "synthetic-owner"})
            await conn.run_sync(apply_v3_2)
            await conn.run_sync(apply_v3_2)
            row = (await conn.execute(text("SELECT version,reminder_version,reminder_state "
                                           "FROM todos WHERE id='synthetic-todo'"))).one()
            assert row == (1, 0, "ACTIVE")
            assert (await conn.scalar(text("SELECT count(*) FROM app_schema_migrations "
                                           "WHERE revision='20261008_v3_2_workbench'"))) == 1
            assert (await conn.scalar(text("SELECT count(*) FROM web_mutations"))) == 0
    finally:
        async with engine.begin() as conn:
            await conn.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")


def request(*, key: str | None = None, etag: str | None = None):
    headers = {}
    if key:
        headers["idempotency-key"] = key
    if etag:
        headers["if-match"] = etag
    return SimpleNamespace(headers=headers)


@pytest.mark.asyncio
async def test_web_todo_create_idempotency_owner_and_version(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    a, b = await new_user(), await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    async def auth_a(_request, *, write=False):
        return a, None
    monkeypatch.setattr(todo_api, "_authenticated", auth_a)
    key = uuid4().hex
    body = todo_api.TodoCreate(title="synthetic V3.2 task", due_at=None)
    first = await todo_api.create_web_todo(request(key=key), Response(), body)
    second = await todo_api.create_web_todo(request(key=key), Response(), body)
    assert first == second
    async with factory() as db:
        assert len((await db.scalars(select(Todo).where(Todo.user_id == a.id))).all()) == 1
    with pytest.raises(HTTPException) as mismatch:
        await todo_api.create_web_todo(request(key=key), Response(),
                                       todo_api.TodoCreate(title="different"))
    assert mismatch.value.status_code == 409
    with pytest.raises(HTTPException) as stale:
        await todo_api.patch_web_todo(request(etag='"0"'), Response(), first["id"],
                                      todo_api.TodoPatch(title="changed"))
    assert stale.value.status_code == 412
    changed = await todo_api.patch_web_todo(request(etag=first["etag"]), Response(), first["id"],
                                            todo_api.TodoPatch(title="changed", due_at=None))
    assert changed["version"] == first["version"] + 1
    assert changed["due_at"] is None
    async def auth_b(_request, *, write=False):
        return b, None
    monkeypatch.setattr(todo_api, "_authenticated", auth_b)
    with pytest.raises(HTTPException) as hidden:
        await todo_api.get_web_todo(request(), Response(), first["id"])
    assert hidden.value.status_code == 404


@pytest.mark.asyncio
async def test_reminder_version_cancellation_fences_chunk_claim(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)
    async def auth(_request, *, write=False):
        return user, None
    monkeypatch.setattr(todo_api, "_authenticated", auth)
    monkeypatch.setattr(reminder_api, "_authenticated", auth)
    created = await todo_api.create_web_todo(request(key=uuid4().hex), Response(),
                                              todo_api.TodoCreate(title="synthetic reminder"))
    due = datetime.now(timezone.utc) + timedelta(seconds=20)
    configured = await reminder_api.set_reminder(
        request(key=uuid4().hex, etag=created["etag"]), Response(), created["id"],
        reminder_api.ReminderTime(at=due, tz="Asia/Shanghai"))
    assert configured["definition_state"] == "ACTIVE"
    await upsert_due_reminders(factory, bot_identity=123, timezone_name="Asia/Shanghai",
                               now=due + timedelta(seconds=1))
    async with factory() as db:
        occurrence = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.todo_id == created["id"]))
        assert occurrence.reminder_version_snapshot == 1
        occurrence_id = occurrence.id
    token = await claim_occurrence(factory, occurrence_id, "renderer", now=due + timedelta(seconds=1))
    assert token
    delivery_id = await ready_occurrence(factory, occurrence_id, token,
                                          text="synthetic harmless reminder", locale="zh",
                                          now=due + timedelta(seconds=1))
    assert delivery_id
    cancelled = await reminder_api.cancel_reminder(
        request(key=uuid4().hex, etag=configured["etag"]), Response(), created["id"])
    assert cancelled["definition_state"] == "CANCELLED"
    assert await claim_next_chunk(factory, delivery_id, "sender") is None
    async with factory() as db:
        delivery = await db.get(OutboundDelivery, delivery_id)
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 0})
        assert (delivery.state, chunk.state) == ("EXHAUSTED", "EXHAUSTED")
        assert chunk.telegram_message_id is None
    resumed_due = datetime.now(timezone.utc) + timedelta(minutes=5)
    reset = await reminder_api.set_reminder(
        request(key=uuid4().hex, etag=cancelled["etag"]), Response(), created["id"],
        reminder_api.ReminderTime(at=resumed_due, tz="Asia/Shanghai"))
    assert reset["reminder_version"] == 3
    assert reminder_identity(todo_id=created["id"], user_id=user.id, bot_identity=123,
                             chat_id=user.telegram_id, reminder_at=due, reminder_version=1) != \
           reminder_identity(todo_id=created["id"], user_id=user.id, bot_identity=123,
                             chat_id=user.telegram_id, reminder_at=due, reminder_version=3)


@pytest.mark.asyncio
async def test_concurrent_web_cancel_and_r4_send_claim_preserves_boundary(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    monkeypatch.setattr(todo_api, "_authenticated", auth)
    monkeypatch.setattr(reminder_api, "_authenticated", auth)
    created = await todo_api.create_web_todo(
        request(key=uuid4().hex), Response(), todo_api.TodoCreate(title="synthetic concurrent reminder"))
    due = datetime.now(timezone.utc) + timedelta(seconds=20)
    configured = await reminder_api.set_reminder(
        request(key=uuid4().hex, etag=created["etag"]), Response(), created["id"],
        reminder_api.ReminderTime(at=due, tz="Asia/Shanghai"))
    await upsert_due_reminders(factory, bot_identity=123, timezone_name="Asia/Shanghai",
                              now=due + timedelta(seconds=1))
    async with factory() as db:
        occurrence_id = (await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.todo_id == created["id"]))).id
    fence = await claim_occurrence(factory, occurrence_id, "renderer", now=due + timedelta(seconds=1))
    delivery_id = await ready_occurrence(factory, occurrence_id, fence,
                                          text="synthetic concurrent reminder", locale="zh",
                                          now=due + timedelta(seconds=1))
    claimed, cancelled = await asyncio.gather(
        claim_next_chunk(factory, delivery_id, "sender"),
        reminder_api.cancel_reminder(
            request(key=uuid4().hex, etag=configured["etag"]), Response(), created["id"]))
    assert cancelled["definition_state"] == "CANCELLED"
    async with factory() as db:
        delivery = await db.get(OutboundDelivery, delivery_id)
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 0})
        if claimed is None:
            assert (delivery.state, chunk.state) == ("EXHAUSTED", "EXHAUSTED")
        else:
            assert chunk.state == "SENDING"
            assert cancelled["delivery"]["may_have_sent"]
        assert chunk.telegram_message_id is None


@pytest.mark.asyncio
async def test_historical_web_form_proposal_cannot_start_approval(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title="synthetic exact delete")
        await db.commit()
        todo_id = todo.id
    proposal = ActionProposal(
        action_id=uuid4().hex, domain="todo", operation="delete_todo",
        args={"user_id": user.id, "todo_id": todo_id, "title": "synthetic exact delete",
              "approved_todo_version": todo.version},
        human_readable_preview="Delete Todo: synthetic exact delete",
        reason="historical Web form", risk_level=RiskLevel.DESTRUCTIVE)
    run, _ = await submit_run(factory, user_id=user.id, operation="form_proposal",
                              idempotency_key=uuid4().hex,
                              body={"client": {"todo_id": todo_id},
                                    "proposal": proposal.model_dump(mode="json"),
                                    "text": "Delete selected Todo"})
    claimed = await claim_one(factory, owner="retired-form-worker", lease_seconds=120)
    assert claimed and claimed[0] == run.id
    await WebRunWorker(factory, get_settings().database_url).process(*claimed)
    async with factory() as db:
        assert (await db.get(WebRequest, run.id)).state == "FAILED"
        assert await db.get(Todo, todo_id) is not None
        assert await db.get(ActionExecution, proposal.action_id) is None

@pytest.mark.asyncio
async def test_started_send_then_cancel_keeps_remote_confirmation(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)
    async def auth(_request, *, write=False):
        return user, None
    monkeypatch.setattr(todo_api, "_authenticated", auth)
    monkeypatch.setattr(reminder_api, "_authenticated", auth)
    created = await todo_api.create_web_todo(request(key=uuid4().hex), Response(),
                                              todo_api.TodoCreate(title="synthetic sending boundary"))
    due = datetime.now(timezone.utc) + timedelta(seconds=20)
    configured = await reminder_api.set_reminder(
        request(key=uuid4().hex, etag=created["etag"]), Response(), created["id"],
        reminder_api.ReminderTime(at=due, tz="Asia/Shanghai"))
    await upsert_due_reminders(factory, bot_identity=123, timezone_name="Asia/Shanghai",
                               now=due + timedelta(seconds=1))
    async with factory() as db:
        occurrence_id = (await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.todo_id == created["id"]))).id
    render_fence = await claim_occurrence(factory, occurrence_id, "renderer",
                                          now=due + timedelta(seconds=1))
    delivery_id = await ready_occurrence(factory, occurrence_id, render_fence,
                                          text="synthetic harmless reminder", locale="zh",
                                          now=due + timedelta(seconds=1))
    chunk = await claim_next_chunk(factory, delivery_id, "sender")
    assert chunk and chunk[0] == 0
    cancelled = await reminder_api.cancel_reminder(
        request(key=uuid4().hex, etag=configured["etag"]), Response(), created["id"])
    assert cancelled["delivery"]["may_have_sent"]
    assert await finish_chunk(factory, delivery_id, 0, chunk[2],
                              message_id=123456, chat_id=user.telegram_id)
    assert await reconcile_occurrence(factory, occurrence_id) == "DELIVERED"
    async with factory() as db:
        todo = await db.get(Todo, created["id"])
        delivery = await db.get(OutboundDelivery, delivery_id)
        chunk_row = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})
        assert todo.reminder_state == "CANCELLED"
        assert todo.reminder_sent_at is None
        assert (delivery.state, chunk_row.state, chunk_row.telegram_message_id) == (
            "SENT", "SENT", 123456)


@pytest.mark.asyncio
async def test_web_calendar_form_direct_r5_create_same_action(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = FakeR5Calendar()
    monkeypatch.setattr(calendar_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    async def scoped(_request):
        return calendar

    monkeypatch.setattr(calendar_api, "_authenticated", auth)
    monkeypatch.setattr(calendar_api, "_calendar_for", scoped)
    zone = ZoneInfo(get_settings().default_timezone)
    start = (datetime.now(zone) + timedelta(days=3)).replace(hour=15, minute=0, second=0, microsecond=0)
    body = calendar_api.CalendarForm(
        operation="create", calendar_id="primary",
        fields=calendar_api.CalendarFields(title="synthetic Web calendar",
                                           start=start, end=start + timedelta(minutes=30),
                                           timeZone=get_settings().default_timezone))
    web_request = request(key=uuid4().hex)
    executor = ActionExecutor(
        calendar, SqlAlchemyActionLedger(factory), factory,
        calendar_create_coordinator=CalendarCreateCoordinator(calendar, factory))
    monkeypatch.setattr(calendar_api, "get_action_executor", lambda: executor)
    first = await calendar_api.calendar_action(web_request, Response(), body)
    again = await calendar_api.calendar_action(web_request, Response(), body)
    assert first["status"] == again["status"] == "executed"
    assert len(calendar.creates) == 1
    async with factory() as db:
        action_id = calendar.creates[0][2]
        evidence = await db.get(CalendarActionEvidence, action_id)
        ledger = await db.get(ActionExecution, action_id)
        assert (evidence.outcome, ledger.status, evidence.event_id) == (
            "CONFIRMED", "executed", calendar.creates[0][0])
        assert not (await db.scalars(select(WebRequest).where(
            WebRequest.user_id == user.id, WebRequest.operation == "form_proposal"))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update", "delete"])
async def test_web_calendar_form_direct_etag_to_r5_modification(pg_harness, monkeypatch, operation):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = FakeConditionalCalendar()
    monkeypatch.setattr(calendar_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    async def scoped(_request):
        return calendar

    monkeypatch.setattr(calendar_api, "_authenticated", auth)
    monkeypatch.setattr(calendar_api, "_calendar_for", scoped)
    body = calendar_api.CalendarForm(
        operation=operation, calendar_id="primary", event_id="synthetic-event",
        baseline_etag='"v1"',
        fields=(calendar_api.CalendarFields(
            title="After", start=datetime.fromisoformat("2026-10-15T15:00:00+08:00"),
            end=datetime.fromisoformat("2026-10-15T15:30:00+08:00"),
            timeZone="Asia/Shanghai") if operation == "update" else None))
    web_request = request(key=uuid4().hex)
    executor = ActionExecutor(
        calendar, SqlAlchemyActionLedger(factory), factory,
        calendar_modify_coordinator=CalendarModifyCoordinator(calendar, factory))
    monkeypatch.setattr(calendar_api, "get_action_executor", lambda: executor)
    first = await calendar_api.calendar_action(web_request, Response(), body)
    again = await calendar_api.calendar_action(web_request, Response(), body)
    assert first["status"] == again["status"] == "executed"
    writes = calendar.updates if operation == "update" else calendar.deletes
    assert len(writes) == 1 and writes[0][2] == '"v1"'
    async with factory() as db:
        evidence = await db.scalar(select(CalendarActionEvidence).where(
            CalendarActionEvidence.user_id == user.id))
        ledger = await db.get(ActionExecution, evidence.action_id)
        assert (evidence.outcome, ledger.status, evidence.approved_baseline_etag) == (
            "CONFIRMED", "executed", '"v1"')
