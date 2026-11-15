"""E3 deterministic R4.3 crash-window tests (SQLite/fake Bot, not E5)."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app.db import crud
from app.db.models import OutboundDelivery, OutboundDeliveryChunk, ScheduledOccurrence, TodoStatus
from app.scheduler.proactive import (
    ProactiveRuntime, claim_occurrence, daily_identity, ready_occurrence,
    reconcile_occurrence, upsert_daily, upsert_due_reminders, utc_now,
)


class Bot:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
        self.calls.append((chat_id, text))
        if self.fail:
            raise TimeoutError("synthetic ambiguous send")
        return SimpleNamespace(chat_id=chat_id, message_id=len(self.calls))


async def daily_fixture(factory, *, text="daily text", now=None):
    now = now or utc_now()
    async with factory() as db:
        user = await crud.get_or_create_user(db, "r4-3-daily")
        await db.commit()
    local = now.astimezone(ZoneInfo("UTC"))
    await upsert_daily(factory, bot_identity=11, timezone_name="UTC",
                       hour=local.hour, local_date=local.date(), now=now,
                       target_user_id=user.id)
    async with factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id,
            ScheduledOccurrence.kind == "daily_rundown"))
    calls = []

    async def render(occurrence):
        calls.append(occurrence.id)
        return text, "en"

    return row, render, calls


@pytest.mark.asyncio
async def test_duplicate_fire_stable_identity_and_crash_before_render(session_factory):
    row, render, calls = await daily_fixture(session_factory)
    now = utc_now()
    for _ in range(2):
        await upsert_daily(session_factory, bot_identity=11, timezone_name="UTC",
                           hour=now.hour, local_date=now.date(), now=now,
                           target_user_id=row.user_id)
    async with session_factory() as db:
        rows = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == row.user_id))).all()
        assert len(rows) == 1
        assert rows[0].state == "PENDING"
    bot = Bot()
    restarted = ProactiveRuntime(session_factory, bot, bot_identity=11,
                                 timezone_name="UTC", daily_hour=now.hour,
                                 daily_renderer=render, reminder_renderer=None)
    await restarted.process(row.id)
    assert len(bot.calls) == 1 and calls == [row.id]
    async with session_factory() as db:
        assert (await db.get(ScheduledOccurrence, row.id)).state == "DELIVERED"


@pytest.mark.asyncio
async def test_expired_claim_fences_old_owner(session_factory):
    row, _, _ = await daily_fixture(session_factory)
    old = await claim_occurrence(session_factory, row.id, "old")
    assert old
    assert await claim_occurrence(session_factory, row.id, "new") is None
    async with session_factory() as db:
        stored = await db.get(ScheduledOccurrence, row.id)
        stored.lease_until = utc_now() - timedelta(seconds=1)
        await db.commit()
    new = await claim_occurrence(session_factory, row.id, "new")
    assert new and new != old
    assert await ready_occurrence(session_factory, row.id, old, text="stale", locale="en") is None
    assert await ready_occurrence(session_factory, row.id, new, text="fresh", locale="en")


@pytest.mark.asyncio
async def test_rendered_payload_restart_not_regenerated(session_factory):
    row, render, calls = await daily_fixture(session_factory)
    token = await claim_occurrence(session_factory, row.id, "renderer")
    delivery_id = await ready_occurrence(session_factory, row.id, token,
                                          text="immutable", locale="zh-CN")
    assert delivery_id
    bot = Bot()
    restarted = ProactiveRuntime(session_factory, bot, bot_identity=11,
                                 timezone_name="UTC", daily_hour=utc_now().hour,
                                 daily_renderer=render, reminder_renderer=None)
    await restarted.process(row.id)
    assert calls == [] and bot.calls == [(row.chat_id, "immutable")]
    async with session_factory() as db:
        delivery = await db.get(OutboundDelivery, delivery_id)
        assert delivery.text == "immutable" and delivery.locale == "zh-CN"
        assert delivery.state == "SENT"


@pytest.mark.asyncio
async def test_timeout_uncertain_then_confirmed_chunk_not_resent(session_factory):
    row, _, _ = await daily_fixture(session_factory)
    token = await claim_occurrence(session_factory, row.id, "renderer")
    delivery_id = await ready_occurrence(session_factory, row.id, token,
                                          text="a" * 4096 + "b", locale="en")
    class PartialBot(Bot):
        async def send_message(self, **kwargs):
            self.calls.append((kwargs["chat_id"], kwargs["text"]))
            if len(self.calls) == 2:
                raise TimeoutError("ambiguous second chunk")
            return SimpleNamespace(chat_id=kwargs["chat_id"], message_id=len(self.calls))
    bot = PartialBot()
    runtime = ProactiveRuntime(session_factory, bot, bot_identity=11,
                               timezone_name="UTC", daily_hour=utc_now().hour,
                               daily_renderer=None, reminder_renderer=None)
    await runtime.process(row.id)
    async with session_factory() as db:
        delivery = await db.get(OutboundDelivery, delivery_id)
        chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
            OutboundDeliveryChunk.delivery_id == delivery_id).order_by(
            OutboundDeliveryChunk.chunk_index))).all()
        assert delivery.state == "UNCERTAIN"
        assert [(chunk.state, chunk.telegram_message_id) for chunk in chunks] == [
            ("SENT", 1), ("UNCERTAIN", None)]
        chunks[1].next_attempt_at = utc_now() - timedelta(seconds=1)
        await db.commit()
    await runtime.process(row.id)
    assert len(bot.calls) == 3
    assert bot.calls[0][1] == "a" * 4096
    assert bot.calls[2][1] == "b"
    async with session_factory() as db:
        assert (await db.get(ScheduledOccurrence, row.id)).state == "DELIVERED"


@pytest.mark.asyncio
async def test_remote_success_before_aggregate_reconciles_without_resend(session_factory):
    row, _, _ = await daily_fixture(session_factory)
    token = await claim_occurrence(session_factory, row.id, "renderer")
    delivery_id = await ready_occurrence(session_factory, row.id, token,
                                          text="once", locale="en")
    from app.telegram.durable import dispatch_delivery
    first = Bot()
    await dispatch_delivery(session_factory, delivery_id, "sender", first)
    async with session_factory() as db:
        assert (await db.get(ScheduledOccurrence, row.id)).state == "READY"
    second = Bot()
    runtime = ProactiveRuntime(session_factory, second, bot_identity=11,
                               timezone_name="UTC", daily_hour=utc_now().hour,
                               daily_renderer=None, reminder_renderer=None)
    await runtime.process(row.id)
    assert len(first.calls) == 1 and second.calls == []
    async with session_factory() as db:
        assert (await db.get(ScheduledOccurrence, row.id)).state == "DELIVERED"


@pytest.mark.asyncio
async def test_confirmed_reminder_reconciles_even_if_todo_completed_after_send(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "r4-3-reminder-confirmed")
        todo = await crud.create_todo(db, user.id, "reminder already sent")
        todo.reminder_at = utc_now() - timedelta(seconds=5)
        todo.reminder_state = "ACTIVE"  # V3.2 definition, distinct from R4 delivery state.
        await db.commit()
    await upsert_due_reminders(session_factory, bot_identity=11, timezone_name="UTC")
    async with session_factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.todo_id == todo.id))
    token = await claim_occurrence(session_factory, row.id, "renderer")
    delivery_id = await ready_occurrence(session_factory, row.id, token,
                                          text="already delivered", locale="en")
    from app.telegram.durable import dispatch_delivery
    first = Bot()
    await dispatch_delivery(session_factory, delivery_id, "sender", first)
    async with session_factory() as db:
        stored = await db.get(type(todo), todo.id)
        stored.status = TodoStatus.done
        await db.commit()
    second = Bot()
    runtime = ProactiveRuntime(session_factory, second, bot_identity=11,
                               timezone_name="UTC", daily_hour=utc_now().hour,
                               daily_renderer=None, reminder_renderer=None)
    await runtime.process(row.id)
    assert len(first.calls) == 1 and second.calls == []
    async with session_factory() as db:
        assert (await db.get(ScheduledOccurrence, row.id)).state == "DELIVERED"


@pytest.mark.asyncio
async def test_exhausted_delivery_and_reminder_expiry(session_factory):
    row, _, _ = await daily_fixture(session_factory)
    token = await claim_occurrence(session_factory, row.id, "renderer")
    delivery_id = await ready_occurrence(session_factory, row.id, token,
                                          text="never confirmed", locale="en")
    async with session_factory() as db:
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 0})
        chunk.attempts = 5
        await db.commit()
    runtime = ProactiveRuntime(session_factory, Bot(), bot_identity=11,
                               timezone_name="UTC", daily_hour=utc_now().hour,
                               daily_renderer=None, reminder_renderer=None)
    await runtime.process(row.id)
    async with session_factory() as db:
        assert (await db.get(ScheduledOccurrence, row.id)).state == "EXHAUSTED"
        from app.db.models import User
        owner = await db.get(User, row.user_id)
        todo = await crud.create_todo(db, owner.id, "expired")
        await crud.update_todo(db, todo.id, owner.id,
                               reminder_at=utc_now() - timedelta(days=2))
        await db.commit()
    await upsert_due_reminders(session_factory, bot_identity=11, timezone_name="UTC")
    async with session_factory() as db:
        reminder = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.todo_id == todo.id))
        assert reminder.state == "SKIPPED" and reminder.failure_category == "ReminderExpired"


@pytest.mark.asyncio
async def test_catchup_boundary_records_skipped_not_unbounded_send(session_factory):
    now = utc_now()
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "r4-3-catchup")
        await db.commit()
    yesterday = now.date() - timedelta(days=1)
    await upsert_daily(session_factory, bot_identity=11, timezone_name="UTC",
                       hour=8, local_date=yesterday, now=now,
                       target_user_id=user.id)
    async with session_factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))
        assert row.state == "SKIPPED" and row.failure_category == "CatchupExpired"


@pytest.mark.asyncio
async def test_bounded_morning_catchup_policy(session_factory):
    planned = datetime(2026, 10, 6, 8, tzinfo=timezone.utc)
    async with session_factory() as db:
        eligible = await crud.get_or_create_user(db, "r4-3-catchup-eligible")
        stale = await crud.get_or_create_user(db, "r4-3-catchup-stale")
        await db.commit()
    await upsert_daily(session_factory, bot_identity=11, timezone_name="UTC",
                       hour=8, local_date=planned.date(),
                       now=planned + timedelta(hours=2), target_user_id=eligible.id)
    await upsert_daily(session_factory, bot_identity=11, timezone_name="UTC",
                       hour=8, local_date=planned.date(),
                       now=planned + timedelta(hours=4), target_user_id=stale.id)
    async with session_factory() as db:
        rows = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id.in_([eligible.id, stale.id])))).all()
    assert {row.user_id: row.state for row in rows} == {
        eligible.id: "PENDING", stale.id: "SKIPPED"}
