"""E4 isolated PG18 shutdown/restart evidence; fake Telegram transport."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.models import (OutboundDelivery, OutboundDeliveryAttempt,
                           OutboundDeliveryChunk, ScheduledOccurrence,
                           TelegramInboundReceipt, TelegramThreadLease)
from app.scheduler.proactive import (ProactiveRuntime, claim_occurrence,
                                     ready_occurrence, upsert_daily, utc_now)
from app.telegram.durable import (admit_receipt, claim_next_chunk,
                                  dispatch_delivery, finalize_receipt,
                                  thread_invocation_guard)
from app.telegram.durable_runtime import DurableTelegramRuntime

pytestmark = pytest.mark.integration


def update_payload():
    return {"update_id": int(uuid4().hex[:12], 16), "message": {
        "message_id": 1, "date": 0,
        "from": {"id": 70001, "is_bot": False, "first_name": "Synthetic"},
        "chat": {"id": 70001, "type": "private"}, "text": "synthetic restart"}}


class FakeBot:
    def __init__(self):
        self.calls = []

    async def send_message(self, **kwargs):
        self.calls.append(kwargs["text"])
        return SimpleNamespace(chat_id=kwargs["chat_id"], message_id=len(self.calls) + 900)


async def cleanup_receipt(factory, receipt_id):
    async with factory() as db:
        delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
            OutboundDelivery.receipt_id == receipt_id))).all()
        if delivery_ids:
            await db.execute(delete(OutboundDeliveryAttempt).where(
                OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDelivery).where(
                OutboundDelivery.id.in_(delivery_ids)))
        await db.execute(delete(TelegramThreadLease).where(
            TelegramThreadLease.receipt_id == receipt_id))
        await db.execute(delete(TelegramInboundReceipt).where(
            TelegramInboundReceipt.id == receipt_id))
        await db.commit()


@pytest.mark.asyncio
async def test_receipt_shutdown_timeout_then_fresh_worker_recovers(pg_harness,
                                                                    isolated_pg18_target,
                                                                    monkeypatch):
    from app.telegram import webhook
    from app.telegram.durable import enqueue_reply
    from app.telegram.processing_context import current_run

    factory, _, _ = pg_harness
    receipt = await admit_receipt(factory, bot_identity=900000 + int(uuid4().hex[:6], 16),
                                  data=update_payload(), graph_kind="telegram_session",
                                  thread_id=f"shutdown-{uuid4().hex}")
    entered = asyncio.Event()

    async def blocked(_envelope):
        entered.set()
        await asyncio.sleep(100)

    monkeypatch.setattr(webhook, "process_admitted_update", blocked)
    first = DurableTelegramRuntime(factory, FakeBot(), database_url=isolated_pg18_target)
    try:
        first._loop_task = asyncio.create_task(asyncio.sleep(100))
        first.kick(receipt.id)
        await asyncio.wait_for(entered.wait(), 5)
        await first.quiesce()
        await first.drain(asyncio.get_running_loop().time() + .01)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            old_fence = row.fence_token
            assert row.state == "PROCESSING" and old_fence
            lease = await db.get(TelegramThreadLease,
                                 {"graph_kind": row.graph_kind, "thread_id": row.thread_id})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()

        async def recovered(_envelope):
            run = current_run()
            run.record_outcome("no_business", "shutdown_recovered")
            await enqueue_reply(fresh_factory, receipt_id=run.receipt_id,
                                reply_ordinal=run.next_reply(), kind="interactive_reply",
                                chat_id="70001", locale="en", text="recovered",
                                fence_token=run.fence_token, outcome_kind=run.outcome_kind,
                                outcome_ref=run.outcome_ref,
                                required_reply_count=run.required_reply_count)

        engine = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
        fresh_factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            monkeypatch.setattr(webhook, "process_admitted_update", recovered)
            bot = FakeBot()
            fresh = DurableTelegramRuntime(fresh_factory, bot,
                                           database_url=isolated_pg18_target)
            await fresh._process(receipt.id)
            async with fresh_factory() as db:
                row = await db.get(TelegramInboundReceipt, receipt.id)
                assert row.state == "PROCESSED" and row.outcome_ref == "shutdown_recovered"
                assert row.fence_token is None or row.fence_token != old_fence
                delivery = await db.scalar(select(OutboundDelivery).where(
                    OutboundDelivery.receipt_id == receipt.id))
                chunk = await db.get(OutboundDeliveryChunk,
                                     {"delivery_id": delivery.id, "chunk_index": 0})
                assert delivery.state == "SENT" and chunk.telegram_message_id == 901
            assert bot.calls == ["recovered"]
            assert not await finalize_receipt(factory, receipt.id, old_fence)
        finally:
            await engine.dispose()
    finally:
        await cleanup_receipt(factory, receipt.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("release_guard_during_drain", [False, True])
async def test_shutdown_while_waiting_for_same_thread_guard_does_not_invoke(
        pg_harness, isolated_pg18_target, monkeypatch, release_guard_during_drain):
    from app.telegram import webhook

    factory, _, _ = pg_harness
    thread_id = f"guard-{uuid4().hex}"
    receipt = await admit_receipt(factory, bot_identity=900000 + int(uuid4().hex[:6], 16),
                                  data=update_payload(), graph_kind="telegram_session",
                                  thread_id=thread_id)
    calls = []

    async def forbidden(_envelope):
        calls.append("invoke")

    monkeypatch.setattr(webhook, "process_admitted_update", forbidden)
    worker = DurableTelegramRuntime(factory, FakeBot(), database_url=isolated_pg18_target)
    try:
        worker._loop_task = asyncio.create_task(asyncio.sleep(100))
        async with thread_invocation_guard(isolated_pg18_target,
                                           "telegram_session", thread_id):
            worker.kick(receipt.id)
            for _ in range(100):
                async with factory() as db:
                    row = await db.get(TelegramInboundReceipt, receipt.id)
                    if row.state == "PROCESSING":
                        break
                await asyncio.sleep(.01)
            assert row.state == "PROCESSING"
            await worker.quiesce()
            if not release_guard_during_drain:
                await worker.drain(asyncio.get_running_loop().time() + .01)
        if release_guard_during_drain:
            await worker.drain(asyncio.get_running_loop().time() + 1)
        assert calls == []
        async with factory() as db:
            assert (await db.get(TelegramInboundReceipt, receipt.id)).state == "PROCESSING"
    finally:
        await cleanup_receipt(factory, receipt.id)


@pytest.mark.asyncio
async def test_drain_completes_business_intent_but_starts_no_new_send(
        pg_harness, isolated_pg18_target, monkeypatch):
    from app.telegram import webhook
    from app.telegram.durable import enqueue_reply
    from app.telegram.processing_context import current_run

    factory, _, _ = pg_harness
    receipt = await admit_receipt(factory, bot_identity=900000 + int(uuid4().hex[:6], 16),
                                  data=update_payload(), graph_kind="telegram_session",
                                  thread_id=f"drain-{uuid4().hex}")
    entered, release = asyncio.Event(), asyncio.Event()

    async def process(_envelope):
        entered.set()
        await release.wait()
        run = current_run()
        run.record_outcome("no_business", "completed_during_drain")
        await enqueue_reply(factory, receipt_id=run.receipt_id,
                            reply_ordinal=run.next_reply(), kind="interactive_reply",
                            chat_id="70001", locale="en", text="after drain",
                            fence_token=run.fence_token, outcome_kind=run.outcome_kind,
                            outcome_ref=run.outcome_ref,
                            required_reply_count=run.required_reply_count)

    monkeypatch.setattr(webhook, "process_admitted_update", process)
    bot = FakeBot()
    worker = DurableTelegramRuntime(factory, bot, database_url=isolated_pg18_target)
    try:
        worker._loop_task = asyncio.create_task(asyncio.sleep(100))
        worker.kick(receipt.id)
        await asyncio.wait_for(entered.wait(), 5)
        await worker.quiesce()
        release.set()
        await worker.drain(asyncio.get_running_loop().time() + 1)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            delivery = await db.scalar(select(OutboundDelivery).where(
                OutboundDelivery.receipt_id == receipt.id))
            assert row.state == "PROCESSED" and delivery.state == "PENDING"
        assert bot.calls == []
        await dispatch_delivery(factory, delivery.id, "restart", bot)
        assert bot.calls == ["after drain"]
    finally:
        await cleanup_receipt(factory, receipt.id)


@pytest.mark.asyncio
async def test_render_shutdown_reclaim_stale_fence_and_immutable_restart(pg_harness,
                                                                          isolated_pg18_target):
    factory, new_user, _ = pg_harness
    user = await new_user()
    now = utc_now()
    await upsert_daily(factory, bot_identity=772, timezone_name="UTC",
                       hour=now.hour, local_date=now.date(), now=now,
                       target_user_id=user.id)
    async with factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))
        occurrence_id = row.id
    entered = asyncio.Event()

    async def blocked(_):
        entered.set()
        await asyncio.sleep(100)
        return "must not commit", "en"

    old_worker = ProactiveRuntime(factory, FakeBot(), bot_identity=772,
                                  timezone_name="UTC", daily_hour=now.hour,
                                  daily_renderer=blocked, reminder_renderer=blocked)
    # Isolate this user's active render. A full tick intentionally scans all
    # due occurrences in the shared isolated cluster and can begin rendering
    # an unrelated prior test row first, making this crash-point nondeterministic.
    tick = asyncio.create_task(old_worker.process(occurrence_id))
    old_worker._active_tasks.add(tick)
    tick.add_done_callback(old_worker._active_tasks.discard)
    await asyncio.wait_for(entered.wait(), 5)
    async with factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id)
        old_fence = row.fence_token
    await old_worker.quiesce()
    await old_worker.drain(asyncio.get_running_loop().time() + .01)
    assert tick.cancelled()
    async with factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id)
        assert row.state == "CLAIMED" and row.fence_token == old_fence
        row.lease_until = utc_now() - timedelta(seconds=1)
        await db.commit()
    engine = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
    fresh_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        token = await claim_occurrence(fresh_factory, occurrence_id, "new-worker")
        assert token and token != old_fence
        assert await ready_occurrence(fresh_factory, occurrence_id, old_fence,
                                      text="stale", locale="en") is None
        delivery_id = await ready_occurrence(fresh_factory, occurrence_id, token,
                                             text="immutable", locale="en")
        assert delivery_id

        async def forbidden(_):
            raise AssertionError("READY occurrence must not be regenerated")

        bot = FakeBot()
        fresh = ProactiveRuntime(fresh_factory, bot, bot_identity=772,
                                 timezone_name="UTC", daily_hour=now.hour,
                                 daily_renderer=forbidden, reminder_renderer=forbidden)
        await fresh.process(occurrence_id)
        await fresh.process(occurrence_id)
        assert bot.calls == ["immutable"]
        async with fresh_factory() as db:
            row = await db.get(ScheduledOccurrence, occurrence_id)
            chunk = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})
            assert row.state == "DELIVERED" and chunk.telegram_message_id == 901
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cancelled_sending_is_uncertain_and_confirmed_chunk_not_resent(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    now = utc_now()
    await upsert_daily(factory, bot_identity=773, timezone_name="UTC",
                       hour=now.hour, local_date=now.date(), now=now,
                       target_user_id=user.id)
    async with factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))
    token = await claim_occurrence(factory, row.id, "renderer")
    delivery_id = await ready_occurrence(factory, row.id, token,
                                          text="a" * 4096 + "b", locale="en")
    entered = asyncio.Event()

    class PartialBot(FakeBot):
        async def send_message(self, **kwargs):
            self.calls.append(kwargs["text"])
            if len(self.calls) == 2:
                entered.set()
                await asyncio.sleep(100)
            return SimpleNamespace(chat_id=kwargs["chat_id"], message_id=len(self.calls) + 900)

    bot = PartialBot()
    worker = DurableTelegramRuntime(factory, bot, database_url="postgresql://synthetic")
    worker._loop_task = asyncio.create_task(asyncio.sleep(100))
    task = asyncio.create_task(dispatch_delivery(factory, delivery_id, "old-worker", bot))
    worker._tasks.add(task)
    await asyncio.wait_for(entered.wait(), 5)
    await worker.quiesce()
    await worker.drain(asyncio.get_running_loop().time() + .01)
    assert task.cancelled()
    async with factory() as db:
        chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
            OutboundDeliveryChunk.delivery_id == delivery_id).order_by(
            OutboundDeliveryChunk.chunk_index))).all()
        assert [c.state for c in chunks] == ["SENT", "SENDING"]
        chunks[1].lease_until = utc_now() - timedelta(seconds=1)
        await db.commit()
    assert await claim_next_chunk(factory, delivery_id, "new-worker") is None
    async with factory() as db:
        chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
            OutboundDeliveryChunk.delivery_id == delivery_id).order_by(
            OutboundDeliveryChunk.chunk_index))).all()
        assert [c.state for c in chunks] == ["SENT", "UNCERTAIN"]
        assert chunks[0].telegram_message_id == 901
        chunks[1].next_attempt_at = utc_now() - timedelta(seconds=1)
        await db.commit()
    assert bot.calls == ["a" * 4096, "b"]
    recovery_bot = FakeBot()
    await dispatch_delivery(factory, delivery_id, "recovered-worker", recovery_bot)
    assert recovery_bot.calls == ["b"]
    async with factory() as db:
        assert (await db.get(OutboundDelivery, delivery_id)).state == "SENT"
