"""E3 deterministic retention invariants; no real Telegram side effect."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.db.models import (ActionExecution, OutboundDelivery, OutboundDeliveryChunk,
                           OutboundDeliveryAttempt, ScheduledOccurrence,
                           TelegramInboundReceipt, User)
from app.telegram import retention
from app.telegram.operational_snapshot import messaging_snapshot


NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=40)


async def seeded_reply(factory, *, receipt_state="PROCESSED", delivery_state="SENT",
                       chunk_state="SENT", kind="interactive_reply", keyboard=None):
    async with factory() as db:
        user = await db.scalar(select(User).where(User.telegram_id == "7"))
        if user is None:
            user = User(telegram_id="7")
            db.add(user)
            await db.flush()
        receipt = TelegramInboundReceipt(
            bot_identity=12345, update_id=12345, graph_kind="telegram_session",
            thread_id="telegram_user_7", state=receipt_state,
            envelope={"message": {"text": "PRIVATE INPUT"}},
            envelope_sha256="a" * 64,
            outcome_kind="no_business" if receipt_state == "PROCESSED" else None,
            outcome_ref="synthetic_done" if receipt_state == "PROCESSED" else None,
            required_reply_count=1 if receipt_state == "PROCESSED" else None,
            terminal_at=OLD if receipt_state in {"PROCESSED", "DEAD"} else None,
            processed_at=OLD if receipt_state == "PROCESSED" else None,
            received_at=OLD)
        db.add(receipt)
        await db.flush()
        delivery = OutboundDelivery(
            receipt_id=receipt.id, reply_ordinal=0, kind=kind, chat_id="7",
            locale="zh-CN", text="PRIVATE REPLY", keyboard=keyboard,
            payload_sha256="b" * 64, state=delivery_state,
            terminal_at=OLD if delivery_state in {"SENT", "EXHAUSTED"} else None,
            sent_at=OLD if delivery_state == "SENT" else None,
            created_at=OLD)
        db.add(delivery)
        await db.flush()
        db.add(OutboundDeliveryChunk(
            delivery_id=delivery.id, chunk_index=0, text="PRIVATE REPLY",
            content_sha256="c" * 64, state=chunk_state,
            telegram_chat_id="7" if chunk_state == "SENT" else None,
            telegram_message_id=99 if chunk_state == "SENT" else None))
        db.add(ActionExecution(
            action_id=f"tgtodo:{receipt.id}:abcdef123456789012",
            user_id=user.id, domain="todo", operation="create_todo",
            status="executed", result={"receipt_id": receipt.id,
                                       "args_sha256": "d" * 64,
                                       "text": "PRIVATE RESULT"}))
        await db.commit()
        return receipt.id, delivery.id


@pytest.mark.asyncio
async def test_terminal_redaction_preserves_identity_state_and_remote_proof(session_factory):
    receipt_id, delivery_id = await seeded_reply(session_factory)
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult(1, 1)
    async with session_factory() as db:
        receipt = await db.get(TelegramInboundReceipt, receipt_id)
        delivery = await db.get(OutboundDelivery, delivery_id)
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 0})
        execution = await db.get(ActionExecution,
                                 f"tgtodo:{receipt_id}:abcdef123456789012")
        assert (receipt.bot_identity, receipt.update_id, receipt.state) == (12345, 12345, "PROCESSED")
        assert retention._utc(receipt.payload_redacted_at) == NOW
        assert receipt.envelope is receipt.envelope_sha256 is None
        assert delivery.text is delivery.keyboard is delivery.payload_sha256 is None
        assert chunk.text is chunk.content_sha256 is None
        assert (chunk.telegram_chat_id, chunk.telegram_message_id) == ("7", 99)
        assert execution.result is None
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult()


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_state,delivery_state,chunk_state", [
    ("RECEIVED", "SENT", "SENT"),
    ("PROCESSING", "SENT", "SENT"),
    ("FAILED", "SENT", "SENT"),
    ("PROCESSED", "PENDING", "PENDING"),
    ("PROCESSED", "UNCERTAIN", "UNCERTAIN"),
    ("PROCESSED", "SENT", "UNCERTAIN"),
])
async def test_nonterminal_dependency_never_redacts(session_factory, receipt_state,
                                                    delivery_state, chunk_state):
    receipt_id, delivery_id = await seeded_reply(
        session_factory, receipt_state=receipt_state,
        delivery_state=delivery_state, chunk_state=chunk_state)
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult()
    async with session_factory() as db:
        assert (await db.get(TelegramInboundReceipt, receipt_id)).envelope is not None
        assert (await db.get(OutboundDelivery, delivery_id)).text == "PRIVATE REPLY"


@pytest.mark.asyncio
async def test_unresolved_hitl_proposal_is_not_redacted(session_factory):
    keyboard = {"inline_keyboard": [[{"text": "Approve", "callback_data": "act:action-1:approve"}]]}
    receipt_id, _ = await seeded_reply(session_factory, kind="hitl_proposal",
                                       delivery_state="EXHAUSTED", chunk_state="UNCERTAIN",
                                       keyboard=keyboard)
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult()
    async with session_factory() as db:
        db.add(TelegramInboundReceipt(
            bot_identity=12345, update_id=12346, graph_kind="telegram_session",
            thread_id="telegram_user_8", state="PROCESSED", action_id="action-1",
            outcome_kind="no_business", outcome_ref="callback_resolved", required_reply_count=1,
            envelope={"update_id": 12346}, envelope_sha256="f" * 64,
            received_at=OLD, processed_at=OLD, terminal_at=OLD))
        await db.commit()
    result = await retention.redact_due_payloads(session_factory, retention_days=30, now=NOW)
    assert result.receipts == 0  # another user's callback cannot resolve this proposal
    async with session_factory() as db:
        callback = await db.scalar(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.update_id == 12346))
        callback.thread_id = "telegram_user_7"
        await db.commit()
    result = await retention.redact_due_payloads(session_factory, retention_days=30, now=NOW)
    assert result.receipts == 1  # malformed callback proof is kept for operator diagnosis
    async with session_factory() as db:
        assert retention._utc((await db.get(TelegramInboundReceipt, receipt_id)).payload_redacted_at) == NOW


@pytest.mark.asyncio
async def test_crash_before_commit_rolls_back_and_retry_is_idempotent(session_factory, monkeypatch):
    receipt_id, delivery_id = await seeded_reply(session_factory)
    original = retention._retire_delivery

    async def crash_after_in_memory_mutation(db, delivery, now):
        await original(db, delivery, now)
        raise RuntimeError("simulated crash before COMMIT")

    monkeypatch.setattr(retention, "_retire_delivery", crash_after_in_memory_mutation)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await retention.redact_due_payloads(session_factory, retention_days=30, now=NOW)
    async with session_factory() as db:
        assert (await db.get(TelegramInboundReceipt, receipt_id)).envelope is not None
        assert (await db.get(OutboundDelivery, delivery_id)).text == "PRIVATE REPLY"
    monkeypatch.setattr(retention, "_retire_delivery", original)
    assert (await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW)).receipts == 1


@pytest.mark.asyncio
async def test_age_without_terminal_dependency_is_not_enough(session_factory):
    receipt_id, delivery_id = await seeded_reply(session_factory)
    async with session_factory() as db:
        receipt = await db.get(TelegramInboundReceipt, receipt_id)
        receipt.terminal_at = NOW - timedelta(days=29)
        await db.commit()
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult()
    async with session_factory() as db:
        assert (await db.get(OutboundDelivery, delivery_id)).text is not None


@pytest.mark.asyncio
async def test_occurrence_delivery_requires_terminal_parent(session_factory):
    async with session_factory() as db:
        user = User(telegram_id="707", preferences={})
        db.add(user)
        await db.flush()
        occurrence = ScheduledOccurrence(
            occurrence_key="x" * 64, kind="daily_rundown", user_id=user.id,
            bot_identity=123, chat_id="707", timezone_name="UTC", planned_at=OLD,
            state="READY", completed_at=OLD)
        db.add(occurrence)
        await db.flush()
        delivery = OutboundDelivery(
            occurrence_id=occurrence.id, reply_ordinal=0, kind="daily_rundown",
            chat_id="707", locale="zh-CN", text="PRIVATE RUNDOWN",
            payload_sha256="e" * 64, state="SENT", sent_at=OLD,
            terminal_at=OLD, created_at=OLD)
        db.add(delivery)
        await db.flush()
        db.add(OutboundDeliveryChunk(
            delivery_id=delivery.id, chunk_index=0, text="PRIVATE RUNDOWN",
            content_sha256="f" * 64, state="SENT", telegram_message_id=75,
            telegram_chat_id="707"))
        await db.commit()
        occurrence_id, delivery_id = occurrence.id, delivery.id
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult()
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id)
        row.state = "DELIVERED"
        await db.commit()
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult(0, 1)
    async with session_factory() as db:
        assert (await db.get(OutboundDelivery, delivery_id)).text is None


@pytest.mark.asyncio
async def test_exhausted_reminder_retirement_preserves_uncertain_outcome(session_factory):
    """Retry-closed payload retires; the possible remote send stays uncertain."""
    async with session_factory() as db:
        user = User(telegram_id="708", preferences={})
        db.add(user)
        await db.flush()
        occurrence = ScheduledOccurrence(
            occurrence_key="y" * 64, kind="reminder", user_id=user.id,
            bot_identity=123, chat_id="708", timezone_name="UTC", planned_at=OLD,
            state="EXHAUSTED", completed_at=OLD)
        db.add(occurrence)
        await db.flush()
        delivery = OutboundDelivery(
            occurrence_id=occurrence.id, reply_ordinal=0, kind="reminder",
            chat_id="708", locale="zh-CN", text="PRIVATE REMINDER",
            payload_sha256="e" * 64, state="EXHAUSTED", terminal_at=OLD,
            created_at=OLD)
        db.add(delivery)
        await db.flush()
        db.add(OutboundDeliveryChunk(
            delivery_id=delivery.id, chunk_index=0, text="PRIVATE REMINDER",
            content_sha256="f" * 64, state="UNCERTAIN"))
        db.add(OutboundDeliveryAttempt(
            delivery_id=delivery.id, chunk_index=0, attempt_no=1,
            fence_token="fence-1", state="UNCERTAIN",
            failure_category="ReadTimeout", finished_at=OLD))
        await db.commit()
        delivery_id = delivery.id
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30,
        now=OLD + timedelta(days=29)) == retention.RedactionResult()
    async with session_factory() as db:
        assert (await db.get(OutboundDelivery, delivery_id)).text == "PRIVATE REMINDER"
    assert await retention.redact_due_payloads(
        session_factory, retention_days=30, now=NOW) == retention.RedactionResult(0, 1)
    async with session_factory() as db:
        delivery = await db.get(OutboundDelivery, delivery_id)
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 0})
        attempt = await db.scalar(select(OutboundDeliveryAttempt).where(
            OutboundDeliveryAttempt.delivery_id == delivery_id))
        assert delivery.state == "EXHAUSTED" and delivery.text is None
        assert chunk.state == "UNCERTAIN" and chunk.text is None
        assert attempt.state == "UNCERTAIN" and attempt.failure_category == "ReadTimeout"
    snapshot = await messaging_snapshot(session_factory, now=NOW)
    assert snapshot["status"] == "delivery_degraded"
    assert snapshot["counts"]["chunks"]["UNCERTAIN"] >= 1
    assert "PRIVATE REMINDER" not in str(snapshot)


@pytest.mark.asyncio
async def test_user_deletion_requires_quiescence_and_preserves_dedupe(session_factory):
    receipt_id, delivery_id = await seeded_reply(
        session_factory, receipt_state="FAILED", delivery_state="UNCERTAIN",
        chunk_state="UNCERTAIN")
    async with session_factory() as db:
        user = await db.scalar(select(User).where(User.telegram_id == "7"))
        user_id = user.id
    with pytest.raises(ValueError, match="MessagingMustBeQuiesced"):
        await retention.redact_user_messaging(session_factory, user_id=user_id,
                                              quiesced=False, now=NOW)
    result = await retention.redact_user_messaging(session_factory, user_id=user_id,
                                                   quiesced=True, now=NOW)
    assert result == retention.RedactionResult(1, 1)
    async with session_factory() as db:
        receipt = await db.get(TelegramInboundReceipt, receipt_id)
        delivery = await db.get(OutboundDelivery, delivery_id)
        assert (receipt.bot_identity, receipt.update_id, receipt.state) == (12345, 12345, "DEAD")
        assert receipt.envelope is None and delivery.text is None
        assert delivery.state == "EXHAUSTED"


@pytest.mark.asyncio
async def test_user_deletion_rejects_live_processing_lease(session_factory):
    receipt_id, _ = await seeded_reply(session_factory, receipt_state="PROCESSING",
                                       delivery_state="PENDING", chunk_state="PENDING")
    async with session_factory() as db:
        user = await db.scalar(select(User).where(User.telegram_id == "7"))
        row = await db.get(TelegramInboundReceipt, receipt_id)
        row.lease_until = NOW + timedelta(minutes=2)
        await db.commit()
        user_id = user.id
    with pytest.raises(ValueError, match="ActiveReceiptLease"):
        await retention.redact_user_messaging(session_factory, user_id=user_id,
                                              quiesced=True, now=NOW)
    async with session_factory() as db:
        assert (await db.get(TelegramInboundReceipt, receipt_id)).envelope is not None
