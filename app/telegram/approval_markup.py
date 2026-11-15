"""Best-effort, bounded Telegram UI synchronization after Saver-owned HITL.

The edit is a projection of the approval authority, never an approval result.
R4's sent chunk supplies the exact remote message identity. Reapplying removal
after an unknown HTTP outcome is safe; no new Telegram message is sent.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from telegram.error import BadRequest

from app.db.models import (OutboundDelivery, OutboundDeliveryChunk,
                           TelegramApprovalMarkupSync, TelegramInboundReceipt)
from app.telegram.retention import _proposal_action_id, _proposal_action_ids
from app.telegram.workbench_launch import (WORKBENCH_LEGACY_TEXT,
                                           launch_expired, launch_ticket_hash)

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _resolved(snapshot, action_id: str) -> bool:
    """Only a terminal or superseded original proposal may lose its buttons."""
    if snapshot is None:
        return False
    proposal = (snapshot.values or {}).get("proposal") or {}
    if not proposal:
        return False
    if proposal.get("action_id") != action_id:
        return True  # The original thread advanced to a different proposal.
    if snapshot.next:
        return False  # Pending approval/edit or unresolved external action.
    return proposal.get("status") in {"rejected", "executed", "failed"}


def _official_card_resolved(snapshot, action_ids: tuple[str, ...]) -> bool:
    """A multi-action card may lose buttons only after every call has a result."""
    if snapshot is None or not action_ids:
        return False
    from langchain_core.messages import ToolMessage
    values = snapshot.values or {}
    prepared = values.get("prepared_actions") or {}
    completed = {item.tool_call_id for item in values.get("messages", ())
                 if isinstance(item, ToolMessage)}
    action_calls = {item.get("proposal", {}).get("action_id"): call_id
                    for call_id, item in prepared.items() if isinstance(item, dict)}
    return all(action_calls.get(action_id) in completed for action_id in action_ids)


async def _discover(factory, bot_identity: int, graph) -> None:
    """Read Saver before creating an idempotent, non-authoritative UI intent."""
    async with factory() as db:
        rows = (await db.execute(select(OutboundDelivery, TelegramInboundReceipt).join(
            TelegramInboundReceipt,
            TelegramInboundReceipt.id == OutboundDelivery.receipt_id).where(
                TelegramInboundReceipt.bot_identity == bot_identity,
                OutboundDelivery.kind == "hitl_proposal",
                OutboundDelivery.state == "SENT",
                OutboundDelivery.payload_redacted_at.is_(None),
                OutboundDelivery.created_at >= _now() - timedelta(days=30))
            .order_by(OutboundDelivery.created_at.desc()).limit(500))).all()
    for delivery, receipt in rows:
        action_id = _proposal_action_id(delivery)
        action_ids = _proposal_action_ids(delivery)
        if (not action_id or receipt.thread_id != f"telegram_user_{delivery.chat_id}" or
                not delivery.chat_id.isdigit()):
            continue
        thread = f"workflow_user_{delivery.chat_id}"
        try:
            snapshot = await graph.aget_state({"configurable": {"thread_id": thread}})
        except Exception as exc:
            logger.warning("approval_markup stage=read_saver category=%s", type(exc).__name__)
            continue
        try:
            from app.agent.official_runtime import get_graph
            agent_graph = await get_graph()
            agent_snapshot = await agent_graph.aget_state(
                {"configurable": {"thread_id": f"user_{delivery.chat_id}"}})
        except RuntimeError:
            agent_snapshot = None  # Legacy-only test/runtime; never infer Agent state.
            agent_graph = None
        except Exception as exc:
            logger.warning("approval_markup stage=read_agent category=%s", type(exc).__name__)
            # A legacy action can still be verified against its exact legacy
            # proposal. A new Agent action remains untouched without its Saver.
            agent_snapshot = None
            agent_graph = None
        official_snapshot = None
        if agent_graph is not None:
            try:
                official_snapshot = await agent_graph.aget_state(
                    {"configurable": {"thread_id": f"official-v1:user_{delivery.chat_id}"}})
            except Exception as exc:
                logger.warning("approval_markup stage=read_official category=%s",
                               type(exc).__name__)
        # Never infer a new Agent proposal's outcome from an unrelated legacy
        # workflow checkpoint. The action id must identify the exact graph.
        agent_proposal = (agent_snapshot.values or {}).get("proposal") if agent_snapshot else None
        legacy_proposal = (snapshot.values or {}).get("proposal") if snapshot else None
        agent_outcome = ((agent_snapshot.values or {}).get("agent_action_outcomes") or {}).get(
            action_id) if agent_snapshot else None
        official_prepared = ((official_snapshot.values or {}).get("prepared_actions") or {}
                             if official_snapshot else {})
        official_ids = {(item.get("proposal") or {}).get("action_id")
                        for item in official_prepared.values() if isinstance(item, dict)}
        if any(item in official_ids for item in action_ids):
            resolved = _official_card_resolved(official_snapshot, action_ids)
        elif agent_outcome:
            resolved = agent_outcome.get("status") in {"rejected", "executed", "failed", "replaced"}
        elif agent_proposal and agent_proposal.get("action_id") == action_id:
            resolved = _resolved(agent_snapshot, action_id)
        elif legacy_proposal and legacy_proposal.get("action_id") == action_id:
            resolved = _resolved(snapshot, action_id)
        else:
            resolved = False
        if not resolved:
            continue
        async with factory() as db:
            await db.execute(pg_insert(TelegramApprovalMarkupSync).values(
                delivery_id=delivery.id, action_id=action_id, state="PENDING",
                attempts=0).on_conflict_do_nothing())
            await db.commit()


async def _discover_expired_launches(factory, bot_identity: int) -> None:
    """Backfill old R4 workbench replies and custody future expirations.

    The same fenced edit queue serves approvals and launches. A launch ticket
    hash locates the exact original reply; it is not used as an auth token.
    """
    async with factory() as db:
        rows = (await db.execute(select(OutboundDelivery, TelegramInboundReceipt).join(
            TelegramInboundReceipt,
            TelegramInboundReceipt.id == OutboundDelivery.receipt_id)
            .outerjoin(TelegramApprovalMarkupSync,
                       TelegramApprovalMarkupSync.delivery_id == OutboundDelivery.id).where(
                TelegramInboundReceipt.bot_identity == bot_identity,
                TelegramApprovalMarkupSync.delivery_id.is_(None),
                OutboundDelivery.state == "SENT",
                OutboundDelivery.payload_redacted_at.is_(None),
                or_(OutboundDelivery.kind == "workbench_launch",
                    and_(OutboundDelivery.kind == "interactive_reply",
                         OutboundDelivery.text == WORKBENCH_LEGACY_TEXT)))
            .order_by(OutboundDelivery.created_at.desc()).limit(1000))).all()
        for delivery, receipt in rows:
            token_hash = launch_ticket_hash(delivery)
            if (token_hash is None or not delivery.chat_id.isdigit()
                    or receipt.thread_id != f"telegram_user_{delivery.chat_id}"
                    or not await launch_expired(db, delivery, bot_identity)):
                continue
            await db.execute(pg_insert(TelegramApprovalMarkupSync).values(
                delivery_id=delivery.id, action_id=token_hash, state="PENDING",
                attempts=0).on_conflict_do_nothing())
        await db.commit()


async def _claim(factory, delivery_id: str):
    now = _now()
    async with factory() as db:
        row = await db.get(TelegramApprovalMarkupSync, delivery_id,
                           with_for_update=True)
        if row is None or row.state in {"DONE", "FAILED"}:
            return None
        if row.attempts >= MAX_ATTEMPTS:
            row.state, row.failure_category = "FAILED", "AttemptsExhausted"
            row.fence_token = row.lease_until = None
            await db.commit()
            return None
        if (
                (row.state == "IN_FLIGHT" and row.lease_until and row.lease_until > now) or
                (row.next_attempt_at and row.next_attempt_at > now)):
            return None
        delivery = await db.get(OutboundDelivery, delivery_id)
        if delivery is None or delivery.state != "SENT":
            row.state, row.failure_category = "FAILED", "MessageIdentityUnavailable"
            await db.commit()
            return None
        if delivery.kind == "hitl_proposal":
            valid = _proposal_action_id(delivery) == row.action_id
        else:
            receipt = await db.get(TelegramInboundReceipt, delivery.receipt_id)
            valid = (receipt is not None
                     and receipt.thread_id == f"telegram_user_{delivery.chat_id}"
                     and launch_ticket_hash(delivery) == row.action_id
                     and await launch_expired(db, delivery, receipt.bot_identity))
        if not valid:
            row.state, row.failure_category = "FAILED", "MessageIdentityUnavailable"
            await db.commit()
            return None
        chunk = await db.scalar(select(OutboundDeliveryChunk).where(
            OutboundDeliveryChunk.delivery_id == delivery_id,
            OutboundDeliveryChunk.state == "SENT",
            OutboundDeliveryChunk.telegram_chat_id == delivery.chat_id,
            OutboundDeliveryChunk.telegram_message_id.is_not(None)).order_by(
                OutboundDeliveryChunk.chunk_index.desc()).limit(1))
        if chunk is None:
            row.state, row.failure_category = "FAILED", "RemoteMessageIdUnavailable"
            await db.commit()
            return None
        token = str(uuid4())
        row.state, row.fence_token = "IN_FLIGHT", token
        row.attempts += 1
        row.lease_until = now + timedelta(seconds=45)
        row.next_attempt_at = None
        await db.commit()
        return delivery.chat_id, chunk.telegram_message_id, token


async def _finish(factory, delivery_id: str, token: str, error: Exception | None):
    async with factory() as db:
        row = await db.get(TelegramApprovalMarkupSync, delivery_id,
                           with_for_update=True)
        if row is None or row.state != "IN_FLIGHT" or row.fence_token != token:
            return
        now = _now()
        row.fence_token = row.lease_until = None
        if error is None or (isinstance(error, BadRequest) and
                             "message is not modified" in str(error).lower()):
            row.state, row.failure_category, row.completed_at = "DONE", None, now
        elif isinstance(error, BadRequest) or row.attempts >= MAX_ATTEMPTS:
            row.state, row.failure_category = "FAILED", type(error).__name__
        else:
            row.state, row.failure_category = "RETRY", type(error).__name__
            row.next_attempt_at = now + timedelta(seconds=min(300, 2 ** row.attempts * 5))
        await db.commit()


async def sync_approval_markups(factory, bot, bot_identity: int, graph) -> None:
    """One fenced, bounded edit sweep for approval and expired launch buttons."""
    await _discover(factory, bot_identity, graph)
    await _discover_expired_launches(factory, bot_identity)
    async with factory() as db:
        ids = (await db.scalars(select(TelegramApprovalMarkupSync.delivery_id)
            .join(OutboundDelivery, OutboundDelivery.id == TelegramApprovalMarkupSync.delivery_id)
            .join(TelegramInboundReceipt, TelegramInboundReceipt.id == OutboundDelivery.receipt_id)
            .where(TelegramInboundReceipt.bot_identity == bot_identity,
                   TelegramApprovalMarkupSync.state.in_(("PENDING", "RETRY", "IN_FLIGHT")))
            .limit(100))).all()
    for delivery_id in ids:
        claim = await _claim(factory, delivery_id)
        if claim is None:
            continue
        chat_id, message_id, token = claim
        try:
            await bot.edit_message_reply_markup(
                chat_id=int(chat_id), message_id=message_id, reply_markup=None,
                read_timeout=15, connect_timeout=15)
        except Exception as exc:
            logger.warning("approval_markup stage=edit category=%s", type(exc).__name__)
            await _finish(factory, delivery_id, token, exc)
        else:
            await _finish(factory, delivery_id, token, None)
