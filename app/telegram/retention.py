"""Terminal-only R4 payload retirement; PostgreSQL rows remain authorities.

No remote call is permitted in these short transactions. A terminal row may
still be retained indefinitely when an actionable HITL proposal is unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select

from app.db.models import (ActionExecution, OutboundDelivery, OutboundDeliveryAttempt, OutboundDeliveryChunk,
                           ScheduledOccurrence,
                           TelegramApprovalMarkupSync, TelegramInboundReceipt,
                           TelegramThreadLease, User)


TERMINAL_DELIVERY = frozenset({"SENT", "EXHAUSTED"})
TERMINAL_CHUNK = frozenset({"SENT", "EXHAUSTED"})
TERMINAL_OCCURRENCE = frozenset({"DELIVERED", "EXHAUSTED", "SKIPPED", "LEGACY_SENT"})


@dataclass(frozen=True)
class RedactionResult:
    receipts: int = 0
    deliveries: int = 0


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _aged(value: datetime | None, cutoff: datetime) -> bool:
    return value is not None and _utc(value) <= cutoff


def _proposal_action_id(delivery: OutboundDelivery) -> str | None:
    ids = _proposal_action_ids(delivery)
    return ids[0] if ids else None


def _proposal_action_ids(delivery: OutboundDelivery) -> tuple[str, ...]:
    if delivery.kind != "hitl_proposal":
        return ()
    try:
        ids = []
        for row in delivery.keyboard["inline_keyboard"]:
            for button in row:
                raw = button.get("callback_data", "")
                if raw.startswith("act:"):
                    ids.append(raw.split(":", 2)[1])
        return tuple(dict.fromkeys(ids))
    except (TypeError, KeyError, IndexError, AttributeError):
        return ()


async def _proposal_resolved(db, delivery: OutboundDelivery) -> bool:
    if delivery.kind != "hitl_proposal":
        return True
    # Mini App decisions do not produce Telegram callback receipts. The
    # fenced markup worker removes the original buttons only after Saver says
    # every action on this card has a ToolMessage; that completed removal is
    # an equally valid retention proof.
    sync = await db.get(TelegramApprovalMarkupSync, delivery.id)
    if sync is not None and sync.state == "DONE":
        return True
    action_ids = _proposal_action_ids(delivery)
    if not action_ids:
        return False
    source = await db.get(TelegramInboundReceipt, delivery.receipt_id)
    if source is None:
        return False
    for action_id in action_ids:
        callback = await db.scalar(select(TelegramInboundReceipt.id).where(
            TelegramInboundReceipt.action_id == action_id,
            TelegramInboundReceipt.bot_identity == source.bot_identity,
            TelegramInboundReceipt.thread_id == source.thread_id,
            TelegramInboundReceipt.id != delivery.receipt_id,
            TelegramInboundReceipt.state == "PROCESSED").limit(1))
        if callback is None:
            return False
    return True


async def _eligible_delivery(db, delivery: OutboundDelivery, cutoff: datetime) -> bool:
    if (delivery.payload_redacted_at is not None or
            delivery.state not in TERMINAL_DELIVERY or
            not _aged(delivery.terminal_at or delivery.sent_at, cutoff)):
        return False
    if not await _proposal_resolved(db, delivery):
        return False
    # An expired workbench button is still actionable in Telegram until its
    # exact remote markup has been removed. Keep the R4 identity/keyboard for
    # the fenced edit worker, even across a long Bot outage. This is only a
    # retention guard; the edit result never authenticates a Mini App user.
    if delivery.state == "SENT" and delivery.kind in {"workbench_launch", "interactive_reply"}:
        from app.telegram.workbench_launch import launch_ticket_hash
        if launch_ticket_hash(delivery) is not None:
            sync = await db.get(TelegramApprovalMarkupSync, delivery.id)
            if sync is None or sync.state != "DONE":
                return False
    chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
        OutboundDeliveryChunk.delivery_id == delivery.id).order_by(
        OutboundDeliveryChunk.chunk_index).with_for_update())).all()
    # EXHAUSTED closes automated retry for *all* delivery kinds: both
    # claim_next_chunk and recovery reject that parent before inspecting its
    # children. An UNCERTAIN child keeps its remote-outcome state and attempt
    # proof, but its immutable text is no longer needed for a retry. SENT does
    # not grant this exception: an uncertain child under SENT is inconsistent.
    return bool(chunks) and all(
        chunk.state in TERMINAL_CHUNK or
        (delivery.state == "EXHAUSTED" and chunk.state == "UNCERTAIN")
        for chunk in chunks)


async def _retire_delivery(db, delivery: OutboundDelivery, now: datetime) -> None:
    chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
        OutboundDeliveryChunk.delivery_id == delivery.id).with_for_update())).all()
    for chunk in chunks:
        chunk.text = None
        chunk.content_sha256 = None
    delivery.text = None
    delivery.keyboard = None
    delivery.payload_sha256 = None
    delivery.payload_redacted_at = now


async def _redact_receipt(session_factory, receipt_id: str, *, now: datetime,
                          cutoff: datetime) -> RedactionResult:
    async with session_factory() as db:
        # The dispatcher locks delivery before reading its receipt. Avoid a
        # reverse-order wait on an active delivery: terminal delivery states
        # cannot transition back to SENDING, and we recheck under locks below.
        active = await db.scalar(select(OutboundDelivery.id).where(
            OutboundDelivery.receipt_id == receipt_id,
            OutboundDelivery.state.notin_(TERMINAL_DELIVERY)).limit(1))
        if active is not None:
            return RedactionResult()
        row = await db.get(TelegramInboundReceipt, receipt_id, with_for_update=True)
        if (row is None or row.payload_redacted_at is not None or
                row.state not in {"PROCESSED", "DEAD"} or
                not _aged(row.terminal_at or row.processed_at, cutoff)):
            return RedactionResult()
        deliveries = (await db.scalars(select(OutboundDelivery).where(
            OutboundDelivery.receipt_id == row.id).order_by(
            OutboundDelivery.id).with_for_update())).all()
        if row.state == "PROCESSED":
            # Do not retire a corrupt/incomplete R4.2 outcome as though the
            # frozen PROCESSED invariant had been satisfied.
            from app.telegram.durable import has_complete_outcome_and_intents
            if not await has_complete_outcome_and_intents(db, row):
                return RedactionResult()
        for delivery in deliveries:
            if not await _eligible_delivery(db, delivery, cutoff):
                return RedactionResult()
        executions = (await db.scalars(select(ActionExecution).where(
            ActionExecution.action_id.startswith(f"tgtodo:{row.id}:")).with_for_update())).all()
        if any(execution.status != "executed" for execution in executions):
            return RedactionResult()
        for delivery in deliveries:
            await _retire_delivery(db, delivery, now)
        for execution in executions:
            execution.result = None
        row.envelope = None
        row.envelope_sha256 = None
        row.payload_redacted_at = now
        await db.commit()
        return RedactionResult(receipts=1, deliveries=len(deliveries))


async def _redact_occurrence(session_factory, occurrence_id: str, *, now: datetime,
                             cutoff: datetime) -> RedactionResult:
    async with session_factory() as db:
        active = await db.scalar(select(OutboundDelivery.id).where(
            OutboundDelivery.occurrence_id == occurrence_id,
            OutboundDelivery.state.notin_(TERMINAL_DELIVERY)).limit(1))
        if active is not None:
            return RedactionResult()
        row = await db.get(ScheduledOccurrence, occurrence_id, with_for_update=True)
        if (row is None or row.state not in TERMINAL_OCCURRENCE or
                not _aged(row.completed_at, cutoff)):
            return RedactionResult()
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrence_id).with_for_update())
        if delivery is None or not await _eligible_delivery(db, delivery, cutoff):
            return RedactionResult()
        await _retire_delivery(db, delivery, now)
        await db.commit()
        return RedactionResult(deliveries=1)


async def redact_due_payloads(session_factory, *, retention_days: int,
                              now: datetime | None = None, batch_size: int = 100) -> RedactionResult:
    """Scan bounded terminal candidates; each parent is rechecked under lock.

    Age is privacy policy, never the authority for eligibility. A concurrent
    worker can win the lock first; the later redactor then sees its new state.
    """
    if retention_days < 1 or batch_size < 1 or batch_size > 1000:
        raise ValueError("InvalidRetentionPolicy")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    receipts = deliveries = 0
    cursor = None
    while True:
        async with session_factory() as db:
            statement = select(TelegramInboundReceipt.terminal_at,
                               TelegramInboundReceipt.id).where(
                TelegramInboundReceipt.state.in_(("PROCESSED", "DEAD")),
                TelegramInboundReceipt.payload_redacted_at.is_(None),
                TelegramInboundReceipt.terminal_at <= cutoff)
            if cursor is not None:
                statement = statement.where(or_(
                    TelegramInboundReceipt.terminal_at > cursor[0],
                    and_(TelegramInboundReceipt.terminal_at == cursor[0],
                         TelegramInboundReceipt.id > cursor[1])))
            page = (await db.execute(statement.order_by(
                TelegramInboundReceipt.terminal_at,
                TelegramInboundReceipt.id).limit(batch_size))).all()
        if not page:
            break
        for terminal_at, receipt_id in page:
            result = await _redact_receipt(session_factory, receipt_id, now=now, cutoff=cutoff)
            receipts += result.receipts
            deliveries += result.deliveries
        cursor = page[-1]
    cursor = None
    while True:
        async with session_factory() as db:
            statement = select(ScheduledOccurrence.completed_at,
                               ScheduledOccurrence.id).join(
                OutboundDelivery, OutboundDelivery.occurrence_id == ScheduledOccurrence.id
            ).where(ScheduledOccurrence.state.in_(TERMINAL_OCCURRENCE),
                    ScheduledOccurrence.completed_at <= cutoff,
                    OutboundDelivery.payload_redacted_at.is_(None))
            if cursor is not None:
                statement = statement.where(or_(
                    ScheduledOccurrence.completed_at > cursor[0],
                    and_(ScheduledOccurrence.completed_at == cursor[0],
                         ScheduledOccurrence.id > cursor[1])))
            page = (await db.execute(statement.order_by(
                ScheduledOccurrence.completed_at,
                ScheduledOccurrence.id).limit(batch_size))).all()
        if not page:
            break
        for completed_at, occurrence_id in page:
            result = await _redact_occurrence(session_factory, occurrence_id,
                                               now=now, cutoff=cutoff)
            deliveries += result.deliveries
        cursor = page[-1]
    return RedactionResult(receipts=receipts, deliveries=deliveries)


async def redact_user_messaging(session_factory, *, user_id: str,
                                quiesced: bool, now: datetime | None = None) -> RedactionResult:
    """R4-only deletion step for an *offline, ingress-blocked* account workflow.

    The caller must quiesce/drain messaging, prevent later admission for this
    identity, and coordinate R3/business data deletion separately. This does
    not create a standalone account-deletion authority or erase dedupe rows.
    """
    if not quiesced:
        raise ValueError("MessagingMustBeQuiescedForUserDeletion")
    now = now or datetime.now(timezone.utc)
    async with session_factory() as db:
        user = await db.get(User, user_id, with_for_update=True)
        if user is None:
            raise ValueError("UnknownUser")
        thread_id = f"telegram_user_{user.telegram_id}"
        receipts = (await db.scalars(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.thread_id == thread_id).order_by(
            TelegramInboundReceipt.id).with_for_update())).all()
        occurrences = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user_id).order_by(
            ScheduledOccurrence.id).with_for_update())).all()
        receipt_ids = [row.id for row in receipts]
        occurrence_ids = [row.id for row in occurrences]
        scope = []
        if receipt_ids:
            scope.append(OutboundDelivery.receipt_id.in_(receipt_ids))
        if occurrence_ids:
            scope.append(OutboundDelivery.occurrence_id.in_(occurrence_ids))
        deliveries = (await db.scalars(select(OutboundDelivery).where(
            or_(*scope) if scope else OutboundDelivery.id == ""
        ).order_by(OutboundDelivery.id).with_for_update())).all()
        if any(row.state == "PROCESSING" and row.lease_until and
               _utc(row.lease_until) > now for row in receipts):
            raise ValueError("ActiveReceiptLease")
        if any(row.state == "CLAIMED" and row.lease_until and
               _utc(row.lease_until) > now for row in occurrences):
            raise ValueError("ActiveOccurrenceLease")
        for delivery in deliveries:
            active_chunk = await db.scalar(select(OutboundDeliveryChunk.chunk_index).where(
                OutboundDeliveryChunk.delivery_id == delivery.id,
                OutboundDeliveryChunk.state == "SENDING",
                OutboundDeliveryChunk.lease_until > now).limit(1))
            if active_chunk is not None:
                raise ValueError("ActiveSendLease")
        for row in receipts:
            if row.state not in {"PROCESSED", "DEAD"}:
                row.state = "DEAD"
                row.failure_category = "UserDeleted"
            row.terminal_at = row.terminal_at or now
            row.envelope = row.envelope_sha256 = None
            row.payload_redacted_at = now
            row.lease_owner = row.fence_token = row.lease_until = row.next_attempt_at = None
        for row in occurrences:
            if row.state not in TERMINAL_OCCURRENCE:
                row.state = "SKIPPED"
                row.failure_category = "UserDeleted"
            row.completed_at = row.completed_at or now
            row.lease_owner = row.fence_token = row.lease_until = row.next_attempt_at = None
        for delivery in deliveries:
            chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id == delivery.id).with_for_update())).all()
            if delivery.state not in TERMINAL_DELIVERY:
                delivery.state = "EXHAUSTED"
            delivery.terminal_at = delivery.terminal_at or now
            delivery.text = delivery.keyboard = delivery.payload_sha256 = None
            delivery.payload_redacted_at = now
            for chunk in chunks:
                if chunk.state not in TERMINAL_CHUNK:
                    chunk.state = "EXHAUSTED"
                    chunk.failure_category = "UserDeleted"
                chunk.text = chunk.content_sha256 = None
                chunk.lease_owner = chunk.fence_token = chunk.lease_until = chunk.next_attempt_at = None
            attempts = (await db.scalars(select(OutboundDeliveryAttempt).where(
                OutboundDeliveryAttempt.delivery_id == delivery.id,
                OutboundDeliveryAttempt.state == "SENDING").with_for_update())).all()
            for attempt in attempts:
                attempt.state = "UNCERTAIN"
                attempt.failure_category = "UserDeleted"
                attempt.finished_at = now
        if receipt_ids:
            executions = (await db.scalars(select(ActionExecution).where(
                ActionExecution.user_id == user_id,
                ActionExecution.action_id.startswith("tgtodo:")).with_for_update())).all()
            for execution in executions:
                execution.result = None
            leases = (await db.scalars(select(TelegramThreadLease).where(
                TelegramThreadLease.receipt_id.in_(receipt_ids)).with_for_update())).all()
            for lease in leases:
                await db.delete(lease)
        await db.commit()
        return RedactionResult(receipts=len(receipts), deliveries=len(deliveries))
