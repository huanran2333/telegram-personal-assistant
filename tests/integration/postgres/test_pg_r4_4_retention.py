"""E4 isolated PG18 retention and dedupe; no real Telegram sends."""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import crud
from app.db.models import (ActionExecution, OutboundDelivery, OutboundDeliveryAttempt, OutboundDeliveryChunk,
                           ScheduledOccurrence, Todo, TodoStatus,
                           TelegramInboundReceipt)
from app.scheduler.proactive import (ProactiveRuntime, claim_occurrence,
                                     ready_occurrence, reminder_identity)
from app.telegram.durable import (admit_receipt, claim_receipt, claim_next_chunk,
                                  enqueue_reply, finalize_receipt, finish_chunk)
from app.telegram.operational_snapshot import messaging_snapshot
from app.telegram.retention import _redact_receipt, redact_due_payloads
from app.telegram.retention import _redact_occurrence

pytestmark = pytest.mark.integration


def update_payload(update_id: int, actor: int, body: str):
    return {"update_id": update_id, "message": {
        "message_id": 1, "date": 0,
        "from": {"id": actor, "is_bot": False},
        "chat": {"id": actor, "type": "private"}, "text": body}}


async def cleanup(factory, receipt_id: str):
    async with factory() as db:
        ids = (await db.scalars(select(OutboundDelivery.id).where(
            OutboundDelivery.receipt_id == receipt_id))).all()
        if ids:
            await db.execute(delete(OutboundDeliveryAttempt).where(
                OutboundDeliveryAttempt.delivery_id.in_(ids)))
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(ids)))
            await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id.in_(ids)))
        await db.execute(delete(ActionExecution).where(
            ActionExecution.action_id.startswith(f"tgtodo:{receipt_id}:")))
        await db.execute(delete(TelegramInboundReceipt).where(
            TelegramInboundReceipt.id == receipt_id))
        await db.commit()


@pytest.mark.asyncio
async def test_pg_terminal_redaction_restart_duplicate_and_remote_proof(pg_harness,
                                                                         isolated_pg18_target):
    factory, new_user, _ = pg_harness
    user = await new_user()
    bot_id = 10**15 + int(uuid4().hex[:10], 16)
    update_id = int(uuid4().hex[:12], 16)
    actor = int(uuid4().hex[:10], 16)
    payload = update_payload(update_id, actor, "PRIVATE_LOW_ENTROPY")
    receipt = await admit_receipt(factory, bot_identity=bot_id, data=payload,
                                  graph_kind="telegram_session",
                                  thread_id=f"telegram_user_{actor}")
    old = datetime.now(timezone.utc) - timedelta(days=40)
    try:
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id, with_for_update=True)
            row.state = "PROCESSED"
            row.processed_at = row.terminal_at = old
            row.outcome_kind = "no_business"
            row.outcome_ref = "synthetic_done"
            row.required_reply_count = 1
            delivery = OutboundDelivery(receipt_id=receipt.id, reply_ordinal=0,
                                        kind="interactive_reply", chat_id=str(actor),
                                        locale="zh-CN", text="PRIVATE_REPLY", payload_sha256="a" * 64,
                                        state="SENT", sent_at=old, terminal_at=old, created_at=old)
            db.add(delivery)
            await db.flush()
            delivery_id = delivery.id
            db.add(OutboundDeliveryChunk(
                delivery_id=delivery_id, chunk_index=0, text="PRIVATE_REPLY",
                content_sha256="b" * 64, state="SENT",
                telegram_chat_id=str(actor), telegram_message_id=44001))
            action_id = f"tgtodo:{receipt.id}:abcdef123456789012"
            db.add(ActionExecution(action_id=action_id, user_id=user.id,
                                   domain="todo", operation="create_todo",
                                   status="executed", result={
                                       "receipt_id": receipt.id,
                                       "args_sha256": "c" * 64,
                                       "text": "PRIVATE_RESULT"}))
            await db.commit()

        results = await asyncio.gather(*(
            redact_due_payloads(factory, retention_days=30) for _ in range(2)))
        assert sum(result.receipts for result in results) == 1
        snapshot = await messaging_snapshot(factory)
        assert snapshot["counts"]["receipts"]["PROCESSED"] >= 1
        assert snapshot["counts"]["deliveries"]["SENT"] >= 1

        # A fresh pool represents process restart; no raw payload reappears.
        engine2 = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
        factory2 = async_sessionmaker(engine2, expire_on_commit=False)
        try:
            async with factory2() as db:
                stored = await db.get(TelegramInboundReceipt, receipt.id)
                delivery = await db.get(OutboundDelivery, delivery_id)
                chunk = await db.get(OutboundDeliveryChunk,
                                     {"delivery_id": delivery_id, "chunk_index": 0})
                assert stored.envelope is stored.envelope_sha256 is None
                assert delivery.text is delivery.payload_sha256 is None
                assert chunk.text is chunk.content_sha256 is None
                assert (await db.get(ActionExecution, action_id)).result is None
                assert chunk.telegram_message_id == 44001
            # Simulate restoring an older *logical row snapshot* containing
            # terminal payload, then reapply policy through a fresh engine.
            # Explicit-deletion tombstones predating a backup remain an
            # external deployment authority, not something this can infer.
            async with factory2() as db:
                restored = await db.get(TelegramInboundReceipt, receipt.id)
                restored.envelope = {"update_id": update_id, "message": {"text": "PRIVATE_LOW_ENTROPY"}}
                restored.envelope_sha256 = "d" * 64
                restored.payload_redacted_at = None
                restored_delivery = await db.get(OutboundDelivery, delivery_id)
                restored_delivery.text = "PRIVATE_REPLY"
                restored_delivery.payload_sha256 = "a" * 64
                restored_delivery.payload_redacted_at = None
                restored_chunk = await db.get(OutboundDeliveryChunk,
                                              {"delivery_id": delivery_id, "chunk_index": 0})
                restored_chunk.text = "PRIVATE_REPLY"
                restored_chunk.content_sha256 = "b" * 64
                await db.commit()
            current = datetime.now(timezone.utc)
            reapplied = await _redact_receipt(factory2, receipt.id, now=current,
                                               cutoff=current - timedelta(days=30))
            assert reapplied.receipts == 1
            async with factory2() as db:
                assert (await db.get(TelegramInboundReceipt, receipt.id)).envelope is None
                assert (await db.get(OutboundDeliveryChunk,
                    {"delivery_id": delivery_id, "chunk_index": 0})).text is None
            # Terminal identity, not a retired low-entropy hash, suppresses
            # duplicate delivery. Even changed content cannot replay work.
            duplicate = await admit_receipt(factory2, bot_identity=bot_id, data=payload,
                                            graph_kind="telegram_session",
                                            thread_id=f"telegram_user_{actor}")
            assert duplicate.id == receipt.id
            mismatch = await admit_receipt(
                factory2, bot_identity=bot_id,
                data=update_payload(update_id, actor, "DIFFERENT BODY"),
                graph_kind="telegram_session", thread_id=f"telegram_user_{actor}")
            assert mismatch.id == receipt.id
            assert await claim_receipt(factory2, receipt.id, "other-worker") is None
        finally:
            await engine2.dispose()
    finally:
        await cleanup(factory, receipt.id)


@pytest.mark.asyncio
async def test_pg_migration_nullable_payload_columns_and_revision(pg_harness):
    factory, _, _ = pg_harness
    async with factory() as db:
        revision = await db.scalar(text(
            "SELECT revision FROM app_schema_migrations "
            "WHERE revision = '20261006_r4_4_terminal_payload_redaction'"))
        assert revision == "20261006_r4_4_terminal_payload_redaction"
        rows = (await db.execute(text(
            "SELECT table_name,column_name,is_nullable FROM information_schema.columns "
            "WHERE (table_name,column_name) IN "
            "(('telegram_inbound_receipts','envelope'), "
            "('telegram_inbound_receipts','envelope_sha256'), "
            "('outbound_deliveries','text'), ('outbound_deliveries','payload_sha256'), "
            "('outbound_delivery_chunks','text'), "
            "('outbound_delivery_chunks','content_sha256'))"))).all()
        assert len(rows) == 6
        assert all(nullable == "YES" for _, _, nullable in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("upgrade", [False, True])
async def test_pg_fresh_and_prior_r4_upgrade_schema(pg_harness, upgrade):
    """Fresh and R4.2/R4.3-shaped schemas are isolated from public data."""
    from app.db.models import Base, User
    from app.db.migrations.r4_2_durable_ingress import apply as apply_r42
    from app.db.migrations.r4_2_outcome_proof import apply as apply_proof
    from app.db.migrations.r4_3_proactive_occurrences import (
        apply as apply_r43, create_occurrence_table)
    from app.db.migrations.r4_4_terminal_payload_redaction import apply as apply_r44

    _, _, engine = pg_harness
    schema = "r44_gate3_" + uuid4().hex
    assert schema.isidentifier() and schema.startswith("r44_gate3_")
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            await conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[User.__table__], checkfirst=True))
            await conn.run_sync(create_occurrence_table)
            await conn.run_sync(apply_r42)
            await conn.run_sync(apply_proof)
            await conn.run_sync(apply_r43)
            if upgrade:
                # R4.2/3 installations predated terminal marker columns and
                # required their replay payload columns to be NOT NULL.
                for table, columns in {
                    "telegram_inbound_receipts": ("envelope", "envelope_sha256"),
                    "outbound_deliveries": ("text", "payload_sha256"),
                    "outbound_delivery_chunks": ("text", "content_sha256"),
                    "telegram_business_operations": ("args_sha256",),
                }.items():
                    for column in columns:
                        await conn.execute(text(
                            f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET NOT NULL'))
                for table, columns in {
                    "telegram_inbound_receipts": ("terminal_at", "payload_redacted_at"),
                    "outbound_deliveries": ("terminal_at", "payload_redacted_at"),
                }.items():
                    for column in columns:
                        await conn.execute(text(
                            f'ALTER TABLE "{table}" DROP COLUMN IF EXISTS "{column}"'))
            await conn.run_sync(apply_r44)
            await conn.run_sync(apply_r44)  # idempotent restart
            rows = (await conn.execute(text(
                "SELECT table_name,column_name,is_nullable FROM information_schema.columns "
                "WHERE table_schema=:schema AND (table_name,column_name) IN "
                "(('telegram_inbound_receipts','envelope'), "
                "('outbound_deliveries','text'),('outbound_delivery_chunks','text'), "
                "('telegram_business_operations','args_sha256'))"),
                {"schema": schema})).all()
            assert len(rows) == 4 and all(item[2] == "YES" for item in rows)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.mark.asyncio
async def test_pg_redactor_finalize_and_retry_worker_races(pg_harness):
    factory, _, _ = pg_harness
    bot_id = 10**15 + int(uuid4().hex[:10], 16)
    update_id = int(uuid4().hex[:12], 16)
    actor = int(uuid4().hex[:10], 16)
    row = await admit_receipt(factory, bot_identity=bot_id,
                              data=update_payload(update_id, actor, "PRIVATE_RACE"),
                              graph_kind="telegram_session", thread_id=f"race-{uuid4().hex}")
    try:
        token = await claim_receipt(factory, row.id, "race-owner")
        assert token is not None
        delivery_id = await enqueue_reply(
            factory, receipt_id=row.id, reply_ordinal=0,
            kind="interactive_reply", chat_id=str(actor), locale="zh-CN",
            text="PRIVATE_RACE_REPLY", fence_token=token,
            outcome_kind="no_business", outcome_ref="race_done",
            required_reply_count=1)
        future = datetime.now(timezone.utc) + timedelta(days=31)
        cutoff = future - timedelta(days=30)
        finalized, initial_redaction = await asyncio.wait_for(asyncio.gather(
            finalize_receipt(factory, row.id, token),
            _redact_receipt(factory, row.id, now=future, cutoff=cutoff)), timeout=10)
        assert finalized and initial_redaction.receipts == 0
        # The sender still owns the nonterminal chunk; cleanup cannot retire
        # its immutable payload even if the receipt is old enough.
        claimed, during_send = await asyncio.wait_for(asyncio.gather(
            claim_next_chunk(factory, delivery_id, "sender"),
            _redact_receipt(factory, row.id, now=future, cutoff=cutoff)), timeout=10)
        assert claimed is not None and during_send.receipts == 0
        chunk_index, content, chunk_token = claimed
        assert content == "PRIVATE_RACE_REPLY"
        assert await finish_chunk(factory, delivery_id, chunk_index, chunk_token,
                                  message_id=901, chat_id=str(actor))
        # No actual Telegram call occurred; this is a PG state/fence test.
        assert (await _redact_receipt(factory, row.id, now=future,
                                      cutoff=cutoff)).receipts == 1
        async with factory() as db:
            receipt = await db.get(TelegramInboundReceipt, row.id)
            chunk = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})
            assert receipt.envelope is None and chunk.text is None
            assert chunk.telegram_message_id == 901
    finally:
        await cleanup(factory, row.id)


@pytest.mark.asyncio
async def test_pg_retry_closed_uncertain_reminder_redacts_without_redrive(
        pg_harness, isolated_pg18_target):
    """Actual R4.3 transition, fresh process and concurrent recovery/retention."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    now = datetime.now(timezone.utc)
    due = now - timedelta(minutes=1)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "Synthetic uncertain reminder")
        await crud.update_todo(db, todo.id, user.id, reminder_at=due)
        # This is a newly armed reminder definition, not a pre-migration
        # occurrence. Its identity must carry the current definition version.
        key = reminder_identity(todo_id=todo.id, user_id=user.id,
                                bot_identity=881, chat_id=user.telegram_id,
                                reminder_at=due, reminder_version=todo.reminder_version)
        occurrence = ScheduledOccurrence(
            occurrence_key=key, kind="reminder", user_id=user.id,
            bot_identity=881, chat_id=user.telegram_id, timezone_name="UTC",
            planned_at=due, todo_id=todo.id, reminder_at_snapshot=due,
            reminder_version_snapshot=todo.reminder_version,
            state="PENDING")
        db.add(occurrence)
        await db.commit()
        occurrence_id = occurrence.id
        todo_id = todo.id

    token = await claim_occurrence(factory, occurrence_id, "renderer")
    assert token
    delivery_id = await ready_occurrence(factory, occurrence_id, token,
                                          text="Synthetic private reminder", locale="en")
    claim = await claim_next_chunk(factory, delivery_id, "sender")
    assert claim is not None
    index, _, send_fence = claim
    assert await finish_chunk(factory, delivery_id, index, send_fence,
                              message_id=None, error_category="ReadTimeout")
    async with factory() as db:
        row = await db.get(Todo, todo_id)
        row.status = TodoStatus.done
        await db.commit()

    class NoSendBot:
        async def send_message(self, **_):
            raise AssertionError("EXHAUSTED delivery must not redrive")

    async def no_render(_):
        raise AssertionError("terminal occurrence must not render again")

    fresh_engine = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
    fresh_factory = async_sessionmaker(fresh_engine, expire_on_commit=False)
    try:
        runtime = ProactiveRuntime(fresh_factory, NoSendBot(), bot_identity=881,
                                   timezone_name="UTC", daily_hour=8,
                                   daily_renderer=no_render, reminder_renderer=no_render)
        await runtime.process(occurrence_id)  # real irrelevant-reminder transition
        async with fresh_factory() as db:
            occurrence = await db.get(ScheduledOccurrence, occurrence_id)
            delivery = await db.get(OutboundDelivery, delivery_id)
            chunk = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})
            assert (occurrence.state, delivery.state, chunk.state) == (
                "EXHAUSTED", "EXHAUSTED", "UNCERTAIN")
            # Age only the exact synthetic occurrence, never a global sweep.
            old = now - timedelta(days=40)
            occurrence.completed_at = delivery.terminal_at = old
            await db.commit()
        assert await claim_next_chunk(fresh_factory, delivery_id, "other-worker") is None
        assert await claim_occurrence(fresh_factory, occurrence_id, "other-worker") is None
        await asyncio.wait_for(asyncio.gather(
            runtime.process(occurrence_id),
            _redact_occurrence(fresh_factory, occurrence_id, now=now,
                               cutoff=now - timedelta(days=30))), timeout=10)
        async with fresh_factory() as db:
            occurrence = await db.get(ScheduledOccurrence, occurrence_id)
            delivery = await db.get(OutboundDelivery, delivery_id)
            chunk = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})
            attempt = await db.scalar(select(OutboundDeliveryAttempt).where(
                OutboundDeliveryAttempt.delivery_id == delivery_id))
            assert occurrence.state == "EXHAUSTED"
            assert delivery.state == "EXHAUSTED" and delivery.text is None
            assert chunk.state == "UNCERTAIN" and chunk.text is None
            assert attempt.state == "UNCERTAIN" and attempt.failure_category == "ReadTimeout"
            assert await db.scalar(select(OutboundDelivery.id).where(
                OutboundDelivery.occurrence_id == occurrence_id)) == delivery_id
        snapshot = await messaging_snapshot(fresh_factory)
        assert snapshot["status"] == "delivery_degraded"
        assert snapshot["counts"]["chunks"]["UNCERTAIN"] >= 1
        assert "Synthetic private reminder" not in str(snapshot)
        await runtime.process(occurrence_id)
        assert await claim_next_chunk(fresh_factory, delivery_id, "restart-worker") is None
    finally:
        await fresh_engine.dispose()
