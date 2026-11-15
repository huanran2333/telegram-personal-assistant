"""E3 lifecycle faults: cooperative drain versus durable lease recovery."""
import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db import crud
from app.db.models import OutboundDelivery, ScheduledOccurrence
from app.scheduler import proactive, reminders
from app.scheduler.proactive import ProactiveRuntime, utc_now
from app.telegram.durable_runtime import DurableTelegramRuntime


@pytest.mark.asyncio
async def test_receipt_quiesce_drains_active_work_and_rejects_new_kick():
    worker = DurableTelegramRuntime(None, object(), database_url="postgresql://synthetic")
    worker._loop_task = asyncio.create_task(asyncio.sleep(100))
    entered, release = asyncio.Event(), asyncio.Event()

    async def active():
        entered.set()
        await release.wait()

    task = asyncio.create_task(active())
    worker._tasks.add(task)
    await entered.wait()
    await worker.quiesce()
    assert not task.done() and not worker.is_running
    with pytest.raises(RuntimeError, match="IngressRecoveryUnavailable"):
        worker.kick("new-receipt")
    release.set()
    await worker.drain(asyncio.get_running_loop().time() + 1)
    assert task.done() and not task.cancelled()


@pytest.mark.asyncio
async def test_receipt_deadline_cancels_active_task_not_business_outcome():
    worker = DurableTelegramRuntime(None, object(), database_url="postgresql://synthetic")
    worker._loop_task = asyncio.create_task(asyncio.sleep(100))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def active():
        entered.set()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(active())
    worker._tasks.add(task)
    await entered.wait()
    await worker.quiesce()
    await worker.drain(asyncio.get_running_loop().time() + .01)
    assert task.cancelled() and cancelled.is_set()


@pytest.mark.asyncio
async def test_scheduler_cancellation_does_not_cancel_active_runtime_tick(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    runtime_task = None

    class Runtime:
        async def tick(self):
            nonlocal runtime_task
            runtime_task = asyncio.current_task()
            entered.set()
            await release.wait()

    monkeypatch.setattr(reminders, "get_proactive_runtime", lambda: Runtime())
    scheduler_job = asyncio.create_task(reminders.check_and_send_reminders())
    await entered.wait()
    scheduler_job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await scheduler_job
    assert runtime_task is not None and not runtime_task.cancelled()
    release.set()
    await runtime_task


@pytest.mark.asyncio
async def test_proactive_quiesce_prevents_new_claim_after_active_tick(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def upsert(*_args, **_kwargs):
        calls.append("daily")
        entered.set()
        await release.wait()

    monkeypatch.setattr(proactive, "upsert_daily", upsert)
    worker = ProactiveRuntime(None, object(), bot_identity=11, timezone_name="UTC",
                              daily_hour=8, daily_renderer=None, reminder_renderer=None)
    task = asyncio.create_task(worker.tick())
    await entered.wait()
    await worker.quiesce()
    release.set()
    await worker.drain(asyncio.get_running_loop().time() + 1)
    assert task.done() and calls == ["daily"]
    await worker.tick()
    assert calls == ["daily"]


@pytest.mark.asyncio
async def test_cancelled_render_keeps_claimed_occurrence_for_lease_recovery(session_factory):
    now = utc_now()
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "shutdown-render")
        await db.commit()
    await proactive.upsert_daily(session_factory, bot_identity=11, timezone_name="UTC",
                                 hour=now.hour, local_date=now.date(), now=now,
                                 target_user_id=user.id)
    async with session_factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))
        occurrence_id = row.id
    entered = asyncio.Event()

    async def render(_row):
        entered.set()
        await asyncio.sleep(100)
        return "should never commit", "en"

    worker = ProactiveRuntime(session_factory, object(), bot_identity=11,
                              timezone_name="UTC", daily_hour=now.hour,
                              daily_renderer=render, reminder_renderer=render)
    tick = asyncio.create_task(worker.tick())
    await entered.wait()
    await worker.quiesce()
    await worker.drain(asyncio.get_running_loop().time() + .01)
    assert tick.cancelled()
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id)
        assert row.state == "CLAIMED" and row.fence_token
        assert await db.scalar(select(OutboundDelivery.id).where(
            OutboundDelivery.occurrence_id == occurrence_id)) is None
        row.lease_until = utc_now() - timedelta(seconds=1)
        await db.commit()
    assert await proactive.claim_occurrence(session_factory, occurrence_id, "restart")


@pytest.mark.asyncio
async def test_render_finishes_during_drain_but_new_send_waits_for_restart(session_factory):
    now = utc_now()
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "shutdown-render-complete")
        await db.commit()
    await proactive.upsert_daily(session_factory, bot_identity=12, timezone_name="UTC",
                                 hour=now.hour, local_date=now.date(), now=now,
                                 target_user_id=user.id)
    async with session_factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))
        occurrence_id = row.id
    entered, release = asyncio.Event(), asyncio.Event()

    async def render(_row):
        entered.set()
        await release.wait()
        return "stable rendered payload", "en"

    class Bot:
        async def send_message(self, **_kwargs):
            raise AssertionError("no new send may start after quiesce")

    worker = ProactiveRuntime(session_factory, Bot(), bot_identity=12,
                              timezone_name="UTC", daily_hour=now.hour,
                              daily_renderer=render, reminder_renderer=render)
    tick = asyncio.create_task(worker.tick())
    await entered.wait()
    await worker.quiesce()
    release.set()
    await worker.drain(asyncio.get_running_loop().time() + 1)
    assert tick.done() and not tick.cancelled()
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id)
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrence_id))
        assert row.state == "READY" and delivery.state == "PENDING"
