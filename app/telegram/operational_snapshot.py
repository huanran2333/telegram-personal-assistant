"""Read-only, privacy-safe operational view of sealed R4 messaging custody.

This module never claims work, advances a retry, or changes delivery authority.
Counts are per entity (not added across parent/child rows).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import and_, case, func, or_, select, text

from app.db.models import (OutboundDelivery, OutboundDeliveryChunk,
                           ScheduledOccurrence, TelegramInboundReceipt)


_STATES = {
    "receipts": frozenset({"RECEIVED", "PROCESSING", "PROCESSED", "FAILED", "DEAD"}),
    "occurrences": frozenset({"PENDING", "CLAIMED", "READY", "DELIVERED",
                              "FAILED", "EXHAUSTED", "SKIPPED"}),
    "deliveries": frozenset({"PENDING", "SENDING", "PARTIAL", "UNCERTAIN",
                             "SENT", "EXHAUSTED"}),
    "chunks": frozenset({"PENDING", "SENDING", "UNCERTAIN", "SENT", "EXHAUSTED"}),
}
_SAFE_FAILURES = frozenset({
    "BadRequest", "CatchupExpired", "Conflict", "DeliveryExhausted", "Forbidden",
    "InvalidEnvelope", "NetworkError", "ReadTimeout", "RemoteUnknown",
    "ReminderExpired", "ReminderIrrelevant", "ReminderIrrelevantAfterPossibleSend",
    "RenderFailure", "RetryAfter", "SendLeaseExpired", "StaleAction",
    "TimedOut", "TimeoutError",
})
_UNRESOLVED_RECEIPTS = ("RECEIVED", "PROCESSING", "FAILED")
_UNRESOLVED_OCCURRENCES = ("PENDING", "CLAIMED", "FAILED", "READY")
_UNRESOLVED_DELIVERIES = ("PENDING", "SENDING", "PARTIAL", "UNCERTAIN")


def _age_seconds(now: datetime, value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0, int((now - value.astimezone(timezone.utc)).total_seconds()))


async def _state_counts(db, name: str, model) -> dict[str, int]:
    rows = (await db.execute(select(model.state, func.count()).group_by(model.state))).all()
    result = {state: 0 for state in sorted(_STATES[name])}
    for state, count in rows:
        result[state if state in _STATES[name] else "OTHER"] = (
            result.get(state if state in _STATES[name] else "OTHER", 0) + int(count))
    return result


async def _failure_counts(db, model) -> dict[str, int]:
    # Normalize before grouping: an attacker-influenced category cannot cause
    # unbounded distinct groups or leak a private identifier via public health.
    safe_category = case(
        (model.failure_category.in_(_SAFE_FAILURES), model.failure_category),
        else_="Other",
    )
    rows = (await db.execute(select(safe_category, func.count()).where(
        model.failure_category.is_not(None)).group_by(safe_category))).all()
    return dict(sorted((category, int(count)) for category, count in rows))


async def messaging_snapshot(session_factory, *, now: datetime | None = None) -> dict:
    """Return one consistent PG snapshot with no user IDs, text, or raw errors."""
    now = now or datetime.now(timezone.utc)
    async with session_factory() as db:
        async with db.begin():
            if db.bind.dialect.name == "postgresql":
                await db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))

            counts = {
                "receipts": await _state_counts(db, "receipts", TelegramInboundReceipt),
                "occurrences": await _state_counts(db, "occurrences", ScheduledOccurrence),
                "deliveries": await _state_counts(db, "deliveries", OutboundDelivery),
                "chunks": await _state_counts(db, "chunks", OutboundDeliveryChunk),
            }
            failures = {
                "receipts": await _failure_counts(db, TelegramInboundReceipt),
                "occurrences": await _failure_counts(db, ScheduledOccurrence),
                "chunks": await _failure_counts(db, OutboundDeliveryChunk),
            }
            oldest_backlog = [
                await db.scalar(select(func.min(TelegramInboundReceipt.received_at)).where(
                    TelegramInboundReceipt.state.in_(_UNRESOLVED_RECEIPTS))),
                await db.scalar(select(func.min(ScheduledOccurrence.created_at)).where(
                    ScheduledOccurrence.state.in_(_UNRESOLVED_OCCURRENCES),
                    ScheduledOccurrence.planned_at <= now)),
                await db.scalar(select(func.min(OutboundDelivery.created_at)).where(
                    OutboundDelivery.state.in_(_UNRESOLVED_DELIVERIES))),
            ]
            due_receipt = or_(
                TelegramInboundReceipt.state == "RECEIVED",
                and_(TelegramInboundReceipt.state == "FAILED",
                     TelegramInboundReceipt.next_attempt_at <= now),
                and_(TelegramInboundReceipt.state == "PROCESSING",
                     TelegramInboundReceipt.lease_until <= now),
            )
            due_occurrence = or_(
                ScheduledOccurrence.state == "PENDING",
                and_(ScheduledOccurrence.state == "FAILED",
                     ScheduledOccurrence.next_attempt_at <= now),
                and_(ScheduledOccurrence.state == "CLAIMED",
                     ScheduledOccurrence.lease_until <= now),
                ScheduledOccurrence.state == "READY",
            )
            due_chunk = or_(
                OutboundDeliveryChunk.state == "PENDING",
                and_(OutboundDeliveryChunk.state == "UNCERTAIN",
                     OutboundDeliveryChunk.next_attempt_at <= now),
                and_(OutboundDeliveryChunk.state == "SENDING",
                     OutboundDeliveryChunk.lease_until <= now),
            )
            parent_ready = or_(
                TelegramInboundReceipt.state == "PROCESSED",
                ScheduledOccurrence.state == "READY",
            )
            oldest_due = [
                await db.scalar(select(func.min(TelegramInboundReceipt.received_at)).where(due_receipt)),
                await db.scalar(select(func.min(ScheduledOccurrence.planned_at)).where(
                    ScheduledOccurrence.planned_at <= now, due_occurrence)),
                await db.scalar(select(func.min(OutboundDelivery.created_at)).select_from(
                    OutboundDeliveryChunk).join(
                    OutboundDelivery, OutboundDelivery.id == OutboundDeliveryChunk.delivery_id
                ).outerjoin(
                    TelegramInboundReceipt,
                    TelegramInboundReceipt.id == OutboundDelivery.receipt_id
                ).outerjoin(
                    ScheduledOccurrence,
                    ScheduledOccurrence.id == OutboundDelivery.occurrence_id
                ).where(due_chunk, parent_ready,
                        OutboundDelivery.state.notin_(("SENT", "EXHAUSTED")))),
            ]

    backlog_age = [_age_seconds(now, item) for item in oldest_backlog if item is not None]
    due_age = [_age_seconds(now, item) for item in oldest_due if item is not None]
    degraded = (counts["receipts"]["DEAD"] > 0 or
                counts["occurrences"]["EXHAUSTED"] > 0 or
                counts["deliveries"]["EXHAUSTED"] > 0 or
                counts["chunks"]["EXHAUSTED"] > 0 or
                counts["deliveries"]["UNCERTAIN"] > 0 or
                counts["chunks"]["UNCERTAIN"] > 0)
    backlogged = bool(backlog_age)
    status_counts = {
        "pending": {"receipts": counts["receipts"]["RECEIVED"],
                    "occurrences": counts["occurrences"]["PENDING"],
                    "deliveries": counts["deliveries"]["PENDING"],
                    "chunks": counts["chunks"]["PENDING"]},
        "processing": {"receipts": counts["receipts"]["PROCESSING"],
                       "occurrences": counts["occurrences"]["CLAIMED"],
                       "deliveries": counts["deliveries"]["SENDING"],
                       "chunks": counts["chunks"]["SENDING"]},
        "uncertain": {"deliveries": counts["deliveries"]["UNCERTAIN"],
                      "chunks": counts["chunks"]["UNCERTAIN"]},
        "exhausted": {"occurrences": counts["occurrences"]["EXHAUSTED"],
                      "deliveries": counts["deliveries"]["EXHAUSTED"],
                      "chunks": counts["chunks"]["EXHAUSTED"]},
        "dead": {"receipts": counts["receipts"]["DEAD"]},
    }
    return {
        "status": "delivery_degraded" if degraded else (
            "delivery_backlogged" if backlogged else "healthy"),
        "counts": counts,
        "status_counts": status_counts,
        "safe_failure_categories": failures,
        "oldest_backlog_age_seconds": max(backlog_age) if backlog_age else None,
        "oldest_due_age_seconds": max(due_age) if due_age else None,
        "semantics": "counts_per_entity_not_additive",
    }
