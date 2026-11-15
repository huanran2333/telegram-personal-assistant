import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import crud
from app.db.models import Todo, ScheduledOccurrence, OutboundDelivery, OutboundDeliveryChunk
from app.config import get_settings
from app.scheduler import reminders


pytestmark = pytest.mark.integration


async def create_reminder(factory, user_id, title, reminder_at):
    async with factory() as db:
        todo = await crud.create_todo(db, user_id, title, due_date=reminder_at + timedelta(days=1))
        await crud.update_todo(db, todo.id, user_id, reminder_at=reminder_at)
        await db.commit()
        return todo.id


@pytest.mark.asyncio
async def test_real_postgres_reminder_claim_once(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    todo_id = await create_reminder(factory, user.id, "itest claim", datetime.now(timezone.utc) - timedelta(seconds=2))

    async def worker():
        async with factory() as db:
            claimed = await crud.claim_due_reminders(db, datetime.now(timezone.utc))
            await db.commit()
            return [todo.id for todo in claimed]

    results = await asyncio.gather(worker(), worker())
    assert sorted(len(result) for result in results) == [0, 1]
    assert [todo_id] in results
    assert await worker() == []

    async with factory() as db:
        assert await crud.mark_reminder_sent(db, todo_id)
        await db.commit()
    async with factory() as db:
        row = await db.get(Todo, todo_id)
        assert row.reminder_sent_at is not None
        assert row.reminder_sent is True
        assert row.reminder_claimed_at is None
    assert await worker() == []


@pytest.mark.asyncio
async def test_real_postgres_reminder_dispatch_once(pg_harness, proactive_test_runtime):
    factory, new_user, _ = pg_harness
    user = await new_user()
    todo_id = await create_reminder(factory, user.id, "itest dispatch", datetime.now(timezone.utc) - timedelta(seconds=2))
    sends = []

    async def sender(chat_id, text, parse_mode=None):
        sends.append((chat_id, text))

    proactive_test_runtime(factory, sender)
    await reminders.check_and_send_reminders()
    await reminders.check_and_send_reminders()
    assert len(sends) == 1

    async with factory() as db:
        row = await db.get(Todo, todo_id)
        assert row.reminder_sent_at is not None
        assert row.reminder_sent is True


@pytest.mark.asyncio
async def test_real_postgres_reminder_retry(pg_harness, proactive_test_runtime):
    factory, new_user, _ = pg_harness
    user = await new_user()
    todo_id = await create_reminder(factory, user.id, "itest retry", datetime.now(timezone.utc) - timedelta(seconds=2))
    attempts = []

    async def failed_sender(chat_id, text, parse_mode=None):
        attempts.append("failed")
        raise RuntimeError("test sender failure")

    proactive_test_runtime(factory, failed_sender)
    await reminders.check_and_send_reminders()
    async with factory() as db:
        row = await db.get(Todo, todo_id)
        assert row.reminder_sent_at is None
        occurrence = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.todo_id == todo_id))
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrence.id))
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery.id, "chunk_index": 0})
        assert occurrence.state == "READY" and delivery.state == "UNCERTAIN"
        assert chunk.state == "UNCERTAIN" and chunk.telegram_message_id is None
        chunk.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()

    async def successful_sender(chat_id, text, parse_mode=None):
        attempts.append("sent")

    proactive_test_runtime(factory, successful_sender)
    await reminders.check_and_send_reminders()
    assert attempts == ["failed", "sent"]
    async with factory() as db:
        assert (await db.get(Todo, todo_id)).reminder_sent_at is not None


@pytest.mark.asyncio
async def test_real_postgres_reminder_completed_deleted(pg_harness, proactive_test_runtime):
    factory, new_user, _ = pg_harness
    user = await new_user()
    due = datetime.now(timezone.utc) - timedelta(seconds=2)
    done_id = await create_reminder(factory, user.id, "itest done", due)
    deleted_id = await create_reminder(factory, user.id, "itest deleted", due)
    async with factory() as db:
        await crud.update_todo(db, done_id, user.id, status="done")
        await crud.delete_todo(db, deleted_id, user.id)
        await db.commit()

    sends = []

    async def sender(chat_id, text, parse_mode=None):
        sends.append(text)

    proactive_test_runtime(factory, sender)
    await reminders.check_and_send_reminders()
    assert sends == []
    async with factory() as db:
        assert (await db.get(Todo, done_id)).reminder_sent_at is None
        assert await db.get(Todo, deleted_id) is None


@pytest.mark.asyncio
async def test_real_postgres_reminder_restart_semantics(pg_harness, monkeypatch,
                                                      proactive_test_runtime):
    factory, new_user, engine = pg_harness
    user = await new_user()
    todo_id = await create_reminder(factory, user.id, "itest persisted", datetime.now(timezone.utc) + timedelta(seconds=1))
    await engine.dispose()

    fresh_engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    fresh_factory = async_sessionmaker(fresh_engine, expire_on_commit=False)
    fresh_scheduler = AsyncIOScheduler()
    sends = []

    async def sender(chat_id, text, parse_mode=None):
        sends.append(text)

    proactive_test_runtime(fresh_factory, sender)
    monkeypatch.setattr(reminders, "scheduler", fresh_scheduler)
    reminders.start_scheduler()
    try:
        await asyncio.sleep(1.1)
        await reminders.check_and_send_reminders()
        async with fresh_factory() as db:
            row = await db.scalar(select(Todo).where(Todo.id == todo_id))
            assert row.reminder_sent_at is not None
            assert len(sends) == 1
    finally:
        reminders.stop_scheduler()
        await fresh_engine.dispose()
