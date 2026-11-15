"""PostgreSQL-owned Telegram custody and reply primitives.

R4.2: receipt/outbox state is diagnostic and recovery authority, never a
replacement for ActionLedger or LangGraph checkpoints. Each operation here
uses a short transaction; callers must not hold one during model/API calls.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import and_, or_, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import (ActionExecution, OutboundDelivery, OutboundDeliveryAttempt,
                           OutboundDeliveryChunk, TelegramInboundReceipt,
                           TelegramThreadLease,
                           ScheduledOccurrence, Todo, TodoStatus)


class DurableSafetyError(RuntimeError):
    """A receipt fence or replay proof failed; never convert it to tool text."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def telegram_todo_action_id(receipt_id: str, name: str, tool_call_id: str) -> str:
    """Stable per-Receipt tool execution identity in the shared action ledger."""
    key = f"{name}:{tool_call_id}"
    if len(key) > 128:
        raise ValueError("OversizeToolCallId")
    action_id = f"tgtodo:{receipt_id}:{hashlib.sha256(key.encode()).hexdigest()[:18]}"
    if len(action_id) > 64:
        raise DurableSafetyError("InvalidReceiptExecutionIdentity")
    return action_id


def _utc_time(value: datetime) -> datetime:
    # SQLite's isolated test fixture strips timezone metadata; PostgreSQL
    # production stores TIMESTAMPTZ and returns aware values.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _advisory_key(namespace: str, graph_kind: str, thread_id: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{graph_kind}:{thread_id}".encode()).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)


@asynccontextmanager
async def thread_invocation_guard(database_url: str, graph_kind: str, thread_id: str):
    """Session-scoped PG lock for the entire invoke/resume, without a row transaction.

    The short receipt lease fences commits. This independent lock closes the
    lease-expiry split-brain window: a paused old process and its replacement
    cannot concurrently run the same LangGraph thread. PostgreSQL releases the
    lock on connection loss, including a killed worker.
    """
    import psycopg

    parsed = database_url if isinstance(database_url, URL) else make_url(database_url)
    dsn = parsed.set(drivername="postgresql").render_as_string(hide_password=False)
    key = _advisory_key("invoke", graph_kind, thread_id)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("SELECT pg_advisory_lock(%s)", (key,))
        try:
            yield
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (key,))


@asynccontextmanager
async def telegram_workflow_guard(database_url: str, graph_kind: str, thread_id: str):
    """Bridge R4's per-DM session lock to the Saver thread shared with Web HITL.

    Telegram receipts retain their sealed ``telegram_session`` custody order.
    This second, session-scoped lock only serializes workflow read/invoke/resume
    against Web decisions and Calendar continuation; it is not a new authority.
    """
    prefix = "telegram_user_"
    if (graph_kind == "telegram_session" and thread_id.startswith(prefix)
            and thread_id[len(prefix):].isdigit()):
        identity = thread_id[len(prefix):]
        legacy = f"workflow_user_{identity}"
        agent = f"user_{identity}"
        official_agent = f"official-v1:{agent}"
        # Keep the old workflow lock for existing checkpoints, then acquire
        # the original Agent thread's lock shared with Mini App decisions.
        # A DM receipt never invokes either graph while a Web decision owns it.
        async with thread_invocation_guard(database_url, "workflow", legacy):
            async with thread_invocation_guard(database_url, "workflow", agent):
                # Mini App decisions on the versioned official thread must
                # not race Telegram callbacks or new turns on that same Saver.
                async with thread_invocation_guard(database_url, "workflow",
                                                   official_agent):
                    yield
    else:
        # Legacy/non-DM receipt fixtures have no corresponding workflow thread.
        # They retain the sealed R4 receipt guard; only canonical DM identities
        # need the additional cross-entry workflow lock.
        yield


def normalize_update(data: dict) -> tuple[int, dict, str, str]:
    """Retain the minimum replayable Bot API shape, not a raw update dump."""
    update_id = data.get("update_id")
    if not isinstance(update_id, int) or isinstance(update_id, bool) or update_id < 0:
        raise ValueError("InvalidUpdateId")
    if isinstance(data.get("callback_query"), dict):
        callback = data["callback_query"]
        actor = callback.get("from") or {}
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        callback_data = callback.get("data")
        if not isinstance(callback_data, str) or len(callback_data) > 256:
            raise ValueError("InvalidCallbackData")
        kind = "callback"
        payload = {
            "update_id": update_id,
            "callback_query": {
                "id": callback.get("id"),
                "from": {key: actor[key] for key in ("id", "is_bot", "first_name", "username") if key in actor},
                "chat_instance": callback.get("chat_instance", ""),
                "data": callback_data,
                "message": {"message_id": message.get("message_id"),
                            "date": message.get("date", 0),
                            "chat": {key: chat[key] for key in ("id", "type") if key in chat},
                            "text": ""},
            },
        }
    elif isinstance(data.get("message"), dict):
        message = data["message"]
        actor = message.get("from") or {}
        chat = message.get("chat") or {}
        voice = message.get("voice") or {}
        kind = "voice" if voice else "text"
        text = message.get("text")
        if kind == "text" and (not isinstance(text, str) or not text.strip()):
            raise ValueError("UnsupportedMessage")
        if isinstance(text, str) and len(text) > 32768:
            raise ValueError("OversizeMessage")
        if kind == "voice" and (not isinstance(voice.get("file_id"), str) or
                                len(voice["file_id"]) > 1024):
            raise ValueError("InvalidVoice")
        payload = {
            "update_id": update_id,
            "message": {"message_id": message.get("message_id"),
                        "date": message.get("date", 0),
                        "from": {key: actor[key] for key in ("id", "is_bot", "first_name", "username") if key in actor},
                        "chat": {key: chat[key] for key in ("id", "type") if key in chat},
                        **({"voice": {"file_id": voice["file_id"],
                                        "duration": voice.get("duration", 0),
                                        "file_unique_id": voice.get("file_unique_id", "")}} if voice else
                           {"text": text})},
        }
    else:
        raise ValueError("UnsupportedUpdate")
    actor_id = actor.get("id")
    chat_id = chat.get("id")
    if not isinstance(actor_id, int) or not isinstance(chat_id, int):
        raise ValueError("MissingIdentity")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return update_id, payload, kind, _sha(canonical)


async def admit_receipt(session_factory, *, bot_identity: int, data: dict,
                        graph_kind: str, thread_id: str,
                        origin_trace_id: str | None = None) -> TelegramInboundReceipt:
    update_id, envelope, _, digest = normalize_update(data)
    if bot_identity <= 0 or graph_kind not in {"ordinary", "workflow", "telegram_session"} or not thread_id:
        raise ValueError("InvalidReceiptScope")
    callback_data = (envelope.get("callback_query") or {}).get("data", "")
    action_id = callback_data.split(":", 2)[1] if callback_data.startswith("act:") else None
    async with session_factory() as db:
        # Serialize only this thread's short admission transactions, so
        # received_at reflects committed custody order. Hash collisions merely
        # over-serialize unrelated threads; they cannot mix identities.
        lock_key = _advisory_key(f"admit:{bot_identity}", graph_kind, thread_id)
        await db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
        statement = pg_insert(TelegramInboundReceipt).values(
            id=str(uuid4()), bot_identity=bot_identity, update_id=update_id,
            envelope=envelope, envelope_sha256=digest, state="RECEIVED",
            graph_kind=graph_kind, thread_id=thread_id,
            action_id=action_id, origin_trace_id=origin_trace_id,
        ).on_conflict_do_nothing(index_elements=["bot_identity", "update_id"])
        await db.execute(statement)
        row = await db.scalar(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.bot_identity == bot_identity,
            TelegramInboundReceipt.update_id == update_id))
        if row is None or (row.payload_redacted_at is None and
                           row.envelope_sha256 != digest):
            raise ValueError("ReceiptIdentityConflict")
        # After terminal redaction the authenticated bot-scoped update_id is
        # the durable duplicate key. The payload hash has been retired; never
        # recover the original body by accepting this update for replay.
        if row.payload_redacted_at is not None and row.state not in {"PROCESSED", "DEAD"}:
            raise DurableSafetyError("RedactedNonterminalReceipt")
        await db.commit()
        return row


async def admit_terminal_receipt(session_factory, *, bot_identity: int, data: dict,
                                 category: str, origin_trace_id: str | None = None):
    """Durably classify an unsupported/poison update without copying its body."""
    update_id = data.get("update_id") if isinstance(data, dict) else None
    if (not isinstance(update_id, int) or isinstance(update_id, bool) or update_id < 0
            or not category.isidentifier()):
        raise ValueError("InvalidTerminalIdentity")
    # A hash permits collision diagnostics without retaining unneeded update
    # fields (photo/contact/etc.). The sole replayable terminal envelope is its
    # transport identity; recovery deliberately never executes it.
    digest = _sha(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str,
                             separators=(",", ":")))
    async with session_factory() as db:
        await db.execute(pg_insert(TelegramInboundReceipt).values(
            id=str(uuid4()), bot_identity=bot_identity, update_id=update_id,
            envelope={"update_id": update_id, "terminal_category": category},
            envelope_sha256=digest, state="DEAD", failure_category=category,
            graph_kind="terminal", thread_id=None, origin_trace_id=origin_trace_id,
            outcome_kind="terminal_unsupported", processed_at=_now(), terminal_at=_now(),
        ).on_conflict_do_nothing(index_elements=["bot_identity", "update_id"]))
        row = await db.scalar(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.bot_identity == bot_identity,
            TelegramInboundReceipt.update_id == update_id))
        if row is None or (row.payload_redacted_at is None and
                           row.envelope_sha256 != digest):
            raise ValueError("ReceiptIdentityConflict")
        if row.payload_redacted_at is not None and row.state not in {"PROCESSED", "DEAD"}:
            raise DurableSafetyError("RedactedNonterminalReceipt")
        await db.commit()
        return row


async def claim_receipt(session_factory, receipt_id: str, owner: str,
                        *, lease_seconds: int = 120) -> str | None:
    """Claim receipt and thread in one short transaction; no LLM runs inside it."""
    now = _now()
    until = now + timedelta(seconds=lease_seconds)
    token = str(uuid4())
    async with session_factory() as db:
        row = await db.scalar(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.id == receipt_id).with_for_update(skip_locked=True))
        if row is None or row.state in {"PROCESSED", "DEAD"}:
            return None
        if row.state == "PROCESSING" and row.lease_until and row.lease_until > now:
            return None
        if row.state == "FAILED" and row.next_attempt_at and row.next_attempt_at > now:
            return None
        earlier = await db.scalar(select(TelegramInboundReceipt.id).where(
            TelegramInboundReceipt.graph_kind == row.graph_kind,
            TelegramInboundReceipt.thread_id == row.thread_id,
            TelegramInboundReceipt.state.notin_(("PROCESSED", "DEAD")),
            or_(TelegramInboundReceipt.received_at < row.received_at,
                and_(TelegramInboundReceipt.received_at == row.received_at,
                     TelegramInboundReceipt.id < row.id))).limit(1))
        if earlier is not None:
            return None
        # A row-locked lease exists per graph/thread. ON CONFLICT avoids a
        # read/insert race; UPDATE compares the old lease deadline.
        await db.execute(pg_insert(TelegramThreadLease).values(
            graph_kind=row.graph_kind, thread_id=row.thread_id,
            receipt_id=receipt_id, lease_owner=owner,
            fence_token=token, lease_until=until,
        ).on_conflict_do_nothing())
        lease = await db.scalar(select(TelegramThreadLease).where(
            TelegramThreadLease.graph_kind == row.graph_kind,
            TelegramThreadLease.thread_id == row.thread_id).with_for_update())
        if lease is None:
            raise RuntimeError("MissingThreadLease")
        if lease.receipt_id != receipt_id and lease.lease_until > now:
            return None
        # Keep an already reclaimed same-receipt lease fenced, too.
        lease.receipt_id = receipt_id
        lease.lease_owner = owner
        lease.fence_token = token
        lease.lease_until = until
        row.state = "PROCESSING"
        row.attempts += 1
        row.lease_owner = owner
        row.fence_token = token
        row.lease_until = until
        row.next_attempt_at = None
        await db.commit()
        return token


async def has_complete_outcome_and_intents(db, row: TelegramInboundReceipt) -> bool:
    """Require a durable business proof and every declared logical reply."""
    if (row.outcome_kind not in {"no_business", "ordinary_checkpoint",
                                 "workflow_checkpoint", "tool_result", "profile_store"} or
            not row.outcome_ref or row.required_reply_count is None or
            not 1 <= row.required_reply_count <= 10 or
            (row.required_reply_count > 1 and row.outcome_kind != "ordinary_checkpoint")):
        return False
    intents = (await db.execute(select(OutboundDelivery.reply_ordinal,
                                       OutboundDelivery.kind).where(
        OutboundDelivery.receipt_id == row.id).order_by(
        OutboundDelivery.reply_ordinal))).all()
    ordinals = [ordinal for ordinal, _ in intents]
    if ordinals != list(range(row.required_reply_count)):
        return False
    if row.required_reply_count > 1 and any(kind != "hitl_proposal" for _, kind in intents):
        return False
    deliveries = (await db.scalars(select(OutboundDelivery.id).where(
        OutboundDelivery.receipt_id == row.id))).all()
    for delivery_id in deliveries:
        first_chunk = await db.scalar(select(OutboundDeliveryChunk.chunk_index).where(
            OutboundDeliveryChunk.delivery_id == delivery_id).limit(1))
        if first_chunk is None:
            return False
    return True


async def record_resume_origin(session_factory, receipt_id: str, token: str,
                               checkpoint_id: str) -> str:
    """Fence and persist the pre-resume Saver cursor before Command(resume).

    On recovery a changed Saver cursor proves that this receipt already
    advanced the workflow (same-thread receipts cannot overtake it).
    """
    if not checkpoint_id or len(checkpoint_id) > 128:
        raise DurableSafetyError("InvalidResumeCheckpoint")
    async with session_factory() as db:
        row = await db.get(TelegramInboundReceipt, receipt_id, with_for_update=True)
        if (row is None or row.state != "PROCESSING" or row.fence_token != token or
                row.lease_until is None or row.lease_until <= _now()):
            raise DurableSafetyError("LostReceiptFence")
        if row.resume_from_checkpoint_id is None:
            row.resume_from_checkpoint_id = checkpoint_id
            await db.commit()
        elif row.resume_from_checkpoint_id != checkpoint_id:
            raise DurableSafetyError("ResumeCheckpointReplayMismatch")
        return row.resume_from_checkpoint_id


async def finalize_receipt(session_factory, receipt_id: str, token: str) -> bool:
    """Mark PROCESSED only with current fences, business proof and all intents."""
    now = _now()
    async with session_factory() as db:
        row = await db.scalar(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.id == receipt_id).with_for_update())
        if row is None or row.state != "PROCESSING" or row.fence_token != token or row.lease_until <= now:
            return False
        lease = await db.scalar(select(TelegramThreadLease).where(
            TelegramThreadLease.graph_kind == row.graph_kind,
            TelegramThreadLease.thread_id == row.thread_id).with_for_update())
        if lease is None or lease.receipt_id != receipt_id or lease.fence_token != token or lease.lease_until <= now:
            return False
        if not await has_complete_outcome_and_intents(db, row):
            raise DurableSafetyError("MissingBusinessOutcomeOrReplyIntent")
        row.state = "PROCESSED"
        row.processed_at = now
        row.terminal_at = now
        row.lease_owner = None
        row.fence_token = None
        row.lease_until = None
        await db.delete(lease)
        await db.commit()
        return True


async def fail_receipt(session_factory, receipt_id: str, token: str, category: str,
                       *, retry_after_seconds: int = 30, max_attempts: int = 5) -> bool:
    if not category.isidentifier():
        category = "Unavailable"
    now = _now()
    async with session_factory() as db:
        row = await db.scalar(select(TelegramInboundReceipt).where(
            TelegramInboundReceipt.id == receipt_id).with_for_update())
        if row is None or row.state != "PROCESSING" or row.fence_token != token:
            return False
        lease = await db.scalar(select(TelegramThreadLease).where(
            TelegramThreadLease.graph_kind == row.graph_kind,
            TelegramThreadLease.thread_id == row.thread_id).with_for_update())
        if lease is None or lease.fence_token != token:
            return False
        row.state = "DEAD" if row.attempts >= max_attempts else "FAILED"
        if row.state == "DEAD":
            row.terminal_at = now
        row.next_attempt_at = None if row.state == "DEAD" else now + timedelta(seconds=retry_after_seconds)
        row.failure_category = category
        row.lease_owner = row.fence_token = row.lease_until = None
        await db.delete(lease)
        await db.commit()
        return True


def render_chunks(text: str, *, parse_mode: str | None, keyboard: dict | None) -> list[str]:
    """Immutable plain chunks; an actionable keyboard is on a bounded final preview."""
    if not text:
        raise ValueError("EmptyDelivery")
    if parse_mode:
        # Splitting HTML/Markdown at raw character boundaries can create
        # malformed entities. Fail closed until a structured renderer exists.
        if len(text) > 4096:
            raise ValueError("FormattedDeliveryTooLong")
        return [text]
    chunks = [text[index:index + 4096] for index in range(0, len(text), 4096)]
    if keyboard and len(chunks) > 1:
        # Preserve the entire explanation first; the final actionable message
        # repeats a bounded preview next to its buttons. Never attach one
        # action's keyboard to several chunks.
        chunks.append(text[:3500] + "\n…")
    return chunks


async def enqueue_reply(session_factory, *, receipt_id: str, reply_ordinal: int,
                        kind: str, chat_id: str, locale: str, text: str,
                        fence_token: str | None = None,
                        outcome_kind: str | None = None,
                        outcome_ref: str | None = None,
                        required_reply_count: int | None = None,
                        parse_mode: str | None = None, keyboard: dict | None = None) -> str:
    chunks = render_chunks(text, parse_mode=parse_mode, keyboard=keyboard)
    canonical = json.dumps({"kind": kind, "chat_id": chat_id, "locale": locale,
                            "text": text, "parse_mode": parse_mode, "keyboard": keyboard},
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = _sha(canonical)
    async with session_factory() as db:
        if fence_token is not None:
            receipt = await db.get(TelegramInboundReceipt, receipt_id, with_for_update=True)
            if (receipt is None or receipt.state != "PROCESSING" or
                    receipt.fence_token != fence_token or
                    receipt.lease_until is None or receipt.lease_until <= _now()):
                raise DurableSafetyError("LostReceiptFence")
            if (outcome_kind not in {"no_business", "ordinary_checkpoint",
                                     "workflow_checkpoint", "tool_result", "profile_store"} or
                    not outcome_ref or required_reply_count is None or
                    not 1 <= required_reply_count <= 10 or
                    (required_reply_count > 1 and (
                        outcome_kind != "ordinary_checkpoint" or kind != "hitl_proposal")) or
                    reply_ordinal not in range(required_reply_count)):
                raise DurableSafetyError("MissingBusinessOutcomeProof")
            if receipt.outcome_kind is None:
                receipt.outcome_kind = outcome_kind
                receipt.outcome_ref = outcome_ref
                receipt.required_reply_count = required_reply_count
            elif (receipt.outcome_kind, receipt.outcome_ref,
                  receipt.required_reply_count) != (outcome_kind, outcome_ref,
                                                    required_reply_count):
                raise DurableSafetyError("BusinessOutcomeReplayMismatch")
        existing = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.receipt_id == receipt_id,
            OutboundDelivery.reply_ordinal == reply_ordinal))
        if existing is not None:
            if existing.payload_sha256 != digest:
                raise DurableSafetyError("ReplyReplayMismatch")
            # A pre-proof orphan intent can only become complete after the
            # handler has replayed and supplied the same immutable reply plus
            # a verified outcome. Persist that proof under this receipt lock.
            await db.commit()
            return existing.id
        delivery = OutboundDelivery(receipt_id=receipt_id, reply_ordinal=reply_ordinal,
                                    kind=kind, chat_id=chat_id, locale=locale, text=text,
                                    parse_mode=parse_mode, keyboard=keyboard,
                                    payload_sha256=digest, state="PENDING")
        db.add(delivery)
        await db.flush()
        for index, chunk in enumerate(chunks):
            db.add(OutboundDeliveryChunk(delivery_id=delivery.id, chunk_index=index,
                                         text=chunk, content_sha256=_sha(chunk), state="PENDING"))
        await db.commit()
        return delivery.id


async def claim_next_chunk(session_factory, delivery_id: str, owner: str,
                           *, lease_seconds: int = 45,
                           retry_after_seconds: int = 30,
                           max_attempts: int = 5) -> tuple[int, str, str] | None:
    """Fence one ordered chunk; an expired send is uncertain, not unsent."""
    now = _now()
    token = str(uuid4())
    async with session_factory() as db:
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.id == delivery_id).with_for_update(skip_locked=True))
        if delivery is None or delivery.state in {"SENT", "EXHAUSTED"}:
            return None
        if delivery.receipt_id is not None:
            receipt = await db.get(TelegramInboundReceipt, delivery.receipt_id)
            if receipt is None or receipt.state != "PROCESSED":
                return None
        else:
            occurrence = await db.get(ScheduledOccurrence, delivery.occurrence_id)
            if occurrence is None or occurrence.state != "READY":
                return None
        if delivery.kind == "hitl_proposal" and delivery.keyboard:
            try:
                callback_data = delivery.keyboard["inline_keyboard"][0][0]["callback_data"]
                action_id = callback_data.split(":", 2)[1] if callback_data.startswith("act:") else None
            except (KeyError, IndexError, TypeError, AttributeError):
                action_id = None
            if action_id:
                execution = await db.get(ActionExecution, action_id)
                resolved_callback = await db.scalar(select(TelegramInboundReceipt.id).where(
                    TelegramInboundReceipt.action_id == action_id,
                    TelegramInboundReceipt.id != delivery.receipt_id,
                    TelegramInboundReceipt.state == "PROCESSED").limit(1))
                if (execution is not None and execution.status in {"executed", "failed"}) or resolved_callback:
                    delivery.state = "EXHAUSTED"
                    delivery.terminal_at = now
                    pending = (await db.scalars(select(OutboundDeliveryChunk).where(
                        OutboundDeliveryChunk.delivery_id == delivery_id,
                        OutboundDeliveryChunk.state != "SENT"))).all()
                    for item in pending:
                        item.state = "EXHAUSTED"
                        item.failure_category = "StaleAction"
                    await db.commit()
                    return None
        chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
            OutboundDeliveryChunk.delivery_id == delivery_id).order_by(
                OutboundDeliveryChunk.chunk_index).with_for_update())).all()
        if delivery.kind in {"workbench_launch", "interactive_reply"}:
            from app.telegram.workbench_launch import launch_ticket_hash, launch_sendable
            launch_hash = launch_ticket_hash(delivery)
            is_launch = delivery.kind == "workbench_launch" or launch_hash is not None
            unsendable = bool(is_launch and (
                not delivery.receipt_id
                or receipt.thread_id != f"telegram_user_{delivery.chat_id}"
                or not await launch_sendable(db, delivery, receipt.bot_identity, at=now)))
            if unsendable and not any(
                    c.state == "SENDING" for c in chunks):
                # Do not put an already-expired launch button into Telegram.
                # An in-flight send may have crossed the remote boundary;
                # retain its SENDING/UNCERTAIN proof until the R4 lease resolves.
                for item in chunks:
                    if item.state not in {"SENT", "UNCERTAIN"}:
                        item.state, item.failure_category = "EXHAUSTED", "ExpiredWorkbenchLaunch"
                delivery.state, delivery.terminal_at = "EXHAUSTED", now
                await db.commit()
                return None
        if delivery.kind == "reminder" and not all(c.state == "SENT" for c in chunks):
            # This Todo lock shares the exact commit boundary with Web edits.
            # Whichever transaction commits first wins the PENDING -> SENDING
            # race; a past SENDING/UNCERTAIN remote outcome is never rewritten.
            occurrence = await db.get(ScheduledOccurrence, delivery.occurrence_id)
            todo = await db.get(Todo, occurrence.todo_id, with_for_update=True) if occurrence else None
            relevant = bool(todo and todo.status != TodoStatus.done and
                            todo.reminder_state == "ACTIVE" and
                            todo.reminder_version == (occurrence.reminder_version_snapshot or 0) and
                            todo.reminder_at is not None and
                            _utc_time(todo.reminder_at) == _utc_time(occurrence.reminder_at_snapshot))
            if not relevant:
                if any(c.state == "SENDING" and c.lease_until and
                       _utc_time(c.lease_until) > now for c in chunks):
                    return None
                for item in chunks:
                    if item.state == "PENDING":
                        item.state, item.failure_category = "EXHAUSTED", "ReminderIrrelevant"
                delivery.state, delivery.terminal_at = "EXHAUSTED", now
                await db.commit()
                return None
        for chunk in chunks:
            if chunk.state == "SENT":
                continue
            if chunk.state == "SENDING" and chunk.lease_until and _utc_time(chunk.lease_until) > now:
                return None
            if chunk.state == "SENDING":
                chunk.state = "UNCERTAIN"
                chunk.failure_category = "SendLeaseExpired"
                chunk.next_attempt_at = now + timedelta(seconds=retry_after_seconds)
                prior = await db.scalar(select(OutboundDeliveryAttempt).where(
                    OutboundDeliveryAttempt.delivery_id == delivery_id,
                    OutboundDeliveryAttempt.chunk_index == chunk.chunk_index,
                    OutboundDeliveryAttempt.fence_token == chunk.fence_token).with_for_update())
                if prior is not None:
                    prior.state = "UNCERTAIN"
                    prior.failure_category = "SendLeaseExpired"
                    prior.finished_at = now
                chunk.lease_owner = chunk.fence_token = chunk.lease_until = None
                delivery.state = "UNCERTAIN"
                await db.commit()
                return None
            if chunk.next_attempt_at and _utc_time(chunk.next_attempt_at) > now:
                await db.commit()
                return None
            if chunk.attempts >= max_attempts:
                chunk.state = "EXHAUSTED"
                delivery.state = "EXHAUSTED"
                delivery.terminal_at = now
                await db.commit()
                return None
            chunk.attempts += 1
            chunk.state = "SENDING"
            chunk.lease_owner = owner
            chunk.fence_token = token
            chunk.lease_until = now + timedelta(seconds=lease_seconds)
            chunk.send_started_at = now
            chunk.next_attempt_at = None
            delivery.state = "SENDING"
            db.add(OutboundDeliveryAttempt(delivery_id=delivery_id,
                                           chunk_index=chunk.chunk_index,
                                           attempt_no=chunk.attempts,
                                           fence_token=token, state="SENDING"))
            await db.commit()
            return chunk.chunk_index, chunk.text, token
        # A prior worker may have confirmed every chunk before crashing before
        # the aggregate transition. Reconcile from chunk proof.
        delivery.state = "SENT"
        delivery.sent_at = now
        delivery.terminal_at = now
        await db.commit()
        return None


async def finish_chunk(session_factory, delivery_id: str, chunk_index: int,
                       token: str, *, message_id: int | None,
                       chat_id: str | None = None, error_category: str | None = None,
                       retry_after_seconds: int = 30) -> bool:
    now = _now()
    async with session_factory() as db:
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.id == delivery_id).with_for_update())
        chunk = await db.get(OutboundDeliveryChunk, {"delivery_id": delivery_id,
                                                     "chunk_index": chunk_index},
                             with_for_update=True)
        if (delivery is None or chunk is None or chunk.state != "SENDING" or
                chunk.fence_token != token or _utc_time(chunk.lease_until) <= now):
            return False
        attempt = await db.scalar(select(OutboundDeliveryAttempt).where(
            OutboundDeliveryAttempt.delivery_id == delivery_id,
            OutboundDeliveryAttempt.chunk_index == chunk_index,
            OutboundDeliveryAttempt.fence_token == token).with_for_update())
        if attempt is None:
            raise RuntimeError("MissingChunkAttempt")
        attempt.finished_at = now
        chunk.lease_owner = chunk.fence_token = chunk.lease_until = None
        if message_id is not None:
            if chat_id is None:
                raise ValueError("MissingRemoteChatId")
            chunk.state = attempt.state = "SENT"
            chunk.telegram_chat_id = chat_id
            chunk.telegram_message_id = attempt.telegram_message_id = message_id
            all_states = (await db.scalars(select(OutboundDeliveryChunk.state).where(
                OutboundDeliveryChunk.delivery_id == delivery_id))).all()
            if all(state == "SENT" for state in all_states):
                delivery.state = "SENT"
                delivery.sent_at = now
                delivery.terminal_at = now
            else:
                delivery.state = "PARTIAL"
        else:
            category = error_category if error_category and error_category.isidentifier() else "RemoteUnknown"
            chunk.state = attempt.state = "UNCERTAIN"
            chunk.failure_category = attempt.failure_category = category
            chunk.next_attempt_at = now + timedelta(seconds=retry_after_seconds)
            delivery.state = "UNCERTAIN"
        await db.commit()
        return True


async def dispatch_delivery(session_factory, delivery_id: str, owner: str, bot) -> None:
    """Send only unconfirmed chunks. A Bot API exception has unknown outcome."""
    from telegram import InlineKeyboardMarkup

    while True:
        claim = await claim_next_chunk(session_factory, delivery_id, owner)
        if claim is None:
            return
        index, text, token = claim
        async with session_factory() as db:
            delivery = await db.get(OutboundDelivery, delivery_id)
            if delivery is None:
                raise RuntimeError("MissingDelivery")
            chat_id, parse_mode = delivery.chat_id, delivery.parse_mode
            keyboard = delivery.keyboard
            last_index = await db.scalar(select(OutboundDeliveryChunk.chunk_index).where(
                OutboundDeliveryChunk.delivery_id == delivery_id).order_by(
                    OutboundDeliveryChunk.chunk_index.desc()).limit(1))
        try:
            markup = InlineKeyboardMarkup.de_json(keyboard, bot) if keyboard and index == last_index else None
            message = await bot.send_message(chat_id=chat_id, text=text,
                                             parse_mode=parse_mode, reply_markup=markup)
        except Exception as exc:
            await finish_chunk(session_factory, delivery_id, index, token,
                               message_id=None, error_category=type(exc).__name__)
            return
        if not await finish_chunk(session_factory, delivery_id, index, token,
                                  message_id=message.message_id,
                                  chat_id=str(message.chat_id)):
            # A lease was lost while the remote request was in flight. Do not
            # claim success under a stale fence; recovery will mark uncertainty.
            return


async def commit_todo_tool_operation(db, *, name: str, args: dict,
                                     tool_call_id: str | None = None, perform):
    """Atomically commit a direct Todo mutation and its replay proof.

    LangGraph has checkpointed the AI tool-call ID before execute_tools runs.
    A replay of that superstep therefore sees the same call ID. If a replay
    proposes different arguments under one key, fail closed instead of
    silently treating a different user request as the completed mutation.
    """
    from app.telegram.processing_context import current_run, current_tool_call_id

    # Existing non-production callers still supply their call ID through the
    # older scope. The production Agent passes ToolRuntime.tool_call_id directly.
    if tool_call_id is None:
        tool_call_id = current_tool_call_id()

    run = current_run()
    if run is None:
        from app.web.processing_context import current_web_run
        web_run = current_web_run()
        if web_run is not None:
            return await _commit_web_todo_operation(
                db, run=web_run, name=name, args=args,
                tool_call_id=tool_call_id, perform=perform)
        result = await perform()
        await db.commit()
        return result
    receipt = await db.scalar(select(TelegramInboundReceipt).where(
        TelegramInboundReceipt.id == run.receipt_id).with_for_update())
    if (receipt is None or receipt.state != "PROCESSING" or
            receipt.fence_token != run.fence_token or
            receipt.lease_until is None or receipt.lease_until <= _now()):
        raise DurableSafetyError("LostReceiptFence")
    if not tool_call_id:
        raise DurableSafetyError("MissingCheckpointedToolCallId")
    key = f"{name}:{tool_call_id}"
    if len(key) > 128:
        raise ValueError("OversizeToolCallId")
    digest = _sha(json.dumps(args, ensure_ascii=False, sort_keys=True, default=str,
                             separators=(",", ":")))
    user_id = args.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise DurableSafetyError("MissingTodoExecutionOwner")
    # A Receipt can contain several Tool Calls. Keep its identity in the key
    # for retention, but hash the per-call part to fit ActionExecution(64).
    action_id = telegram_todo_action_id(run.receipt_id, name, tool_call_id)
    inserted = await db.scalar(pg_insert(ActionExecution).values(
        action_id=action_id, user_id=user_id, domain="todo", operation=name,
        status="executing", result={"args_sha256": digest, "receipt_id": run.receipt_id}
    ).on_conflict_do_nothing().returning(ActionExecution.action_id))
    if inserted is None:
        record = await db.get(ActionExecution, action_id)
        if (record is None or record.user_id != user_id or record.domain != "todo" or
                record.operation != name or record.status != "executed" or
                not record.result or record.result.get("receipt_id") != run.receipt_id or
                record.result.get("args_sha256") != digest or
                "text" not in record.result):
            raise DurableSafetyError("TodoReplayProofMismatch")
        return record.result["text"]
    result = await perform()
    record = await db.get(ActionExecution, action_id)
    record.status = "executed"
    record.result = {"args_sha256": digest, "receipt_id": run.receipt_id,
                     "text": result}
    record.executed_at = _now()
    await db.commit()
    return result


async def _commit_web_todo_operation(db, *, run, name: str, args: dict,
                                     tool_call_id: str | None, perform):
    """Todo mutation and ActionExecution proof share one short PG transaction.

    The graph's checkpointed tool-call ID is the stable per-run operation key.
    A missing/lost Web lease cannot start a new mutation. PostgreSQL's unique
    action ID serializes an identical replay even across worker processes.
    """
    from app.db.models import ActionExecution, WebRequest
    import hashlib
    import json

    request = await db.scalar(select(WebRequest).where(WebRequest.id == run.request_id).with_for_update())
    if (request is None or request.user_id != run.user_id or request.state != "RUNNING"
            or request.fence_token != run.fence_token or request.lease_until is None
            or request.lease_until <= _now()):
        raise DurableSafetyError("LostWebRequestFence")
    call_id = tool_call_id
    if not call_id:
        raise DurableSafetyError("MissingCheckpointedToolCallId")
    action_id = "webtodo:" + hashlib.sha256(
        f"{run.request_id}:{call_id}:{name}".encode()).hexdigest()[:48]
    args_hash = _sha(json.dumps(args, ensure_ascii=False, sort_keys=True,
                                default=str, separators=(",", ":")))
    inserted = await db.scalar(pg_insert(ActionExecution).values(
        action_id=action_id, user_id=run.user_id, domain="todo", operation=name,
        status="executing", result={"args_sha256": args_hash}
    ).on_conflict_do_nothing().returning(ActionExecution.action_id))
    proof = await db.get(ActionExecution, action_id)
    if inserted is None:
        if (proof is None or proof.user_id != run.user_id or proof.operation != name or
                proof.status != "executed" or not proof.result or
                proof.result.get("args_sha256") != args_hash or
                "text" not in proof.result):
            raise DurableSafetyError("WebTodoReplayProofMismatch")
        return proof.result["text"]
    result = await perform()
    proof.status = "executed"
    proof.result = {"args_sha256": args_hash, "text": result}
    proof.executed_at = _now()
    await db.commit()
    return result
