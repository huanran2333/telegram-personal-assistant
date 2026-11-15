"""R4.3 durable occurrence handoff into the R4.2 Telegram chunk outbox.

The scheduler is a wakeup source, never an execution or delivery authority.
All ownership changes use short PostgreSQL transactions; rendering and Bot API
calls run outside them. The existing shared chunk dispatcher owns remote sends.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload

from app.db import crud
from app.db.models import (DailyBriefDelivery, OutboundDelivery,
                           OutboundDeliveryChunk, ScheduledOccurrence, Todo,
                           TodoStatus, User)
from app.telegram.durable import dispatch_delivery, render_chunks

logger = logging.getLogger(__name__)

# Product policy: a morning brief is useful for a bounded period after the
# planned morning send. Older missed days are recorded SKIPPED, not spammed.
RUNDOWN_CATCHUP = timedelta(hours=3)
REMINDER_RELEVANCE = timedelta(days=1)
OCCURRENCE_LEASE = timedelta(minutes=5)
MAX_RENDER_ATTEMPTS = 5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def observe_detached_tick(task: asyncio.Task) -> None:
    """Consume a tick's error if its scheduler/wakeup parent was cancelled."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("proactive stage=detached_tick category=%s", type(exc).__name__)


def as_utc(value: datetime) -> datetime:
    """SQLite test fixtures drop tzinfo; PostgreSQL production retains it."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def stable_key(*parts: object) -> str:
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _insert_occurrence(db, values: dict) -> None:
    # SQLite is an explicit isolated test backend. Production uses PostgreSQL
    # ON CONFLICT and the same stable unique key; no read-then-insert race.
    if db.bind.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as insert_fn
    else:
        insert_fn = pg_insert
    await db.execute(insert_fn(ScheduledOccurrence).values(**values)
                     .on_conflict_do_nothing(index_elements=["occurrence_key"]))


def daily_identity(*, user_id: str, bot_identity: int, chat_id: str,
                   local_date: date, timezone_name: str,
                   planned_at: datetime) -> str:
    return stable_key("daily_rundown", user_id, bot_identity, chat_id,
                      local_date.isoformat(), timezone_name,
                      planned_at.astimezone(timezone.utc).isoformat())


def reminder_identity(*, todo_id: str, user_id: str, bot_identity: int,
                      chat_id: str, reminder_at: datetime,
                      reminder_version: int = 0) -> str:
    if reminder_version:
        return stable_key("reminder", todo_id, user_id, bot_identity, chat_id,
                          reminder_at.astimezone(timezone.utc).isoformat(), reminder_version)
    return stable_key("reminder", todo_id, user_id, bot_identity, chat_id,
                      reminder_at.astimezone(timezone.utc).isoformat())


async def upsert_daily(session_factory, *, bot_identity: int, timezone_name: str,
                       hour: int, local_date: date, now: datetime | None = None,
                       target_user_id: str | None = None,
                       allowed_chat_ids: frozenset[str] | None = None) -> list[str]:
    """Idempotent multi-worker wakeup; legacy *sent* rows are imported once."""
    now = now or utc_now()
    planned = datetime.combine(local_date, time(hour), ZoneInfo(timezone_name)).astimezone(timezone.utc)
    async with session_factory() as db:
        stmt = select(User)
        if target_user_id:
            stmt = stmt.where(User.id == target_user_id)
        if allowed_chat_ids is not None:
            stmt = stmt.where(User.telegram_id.in_(allowed_chat_ids))
        users = (await db.scalars(stmt)).all()
        keys: list[str] = []
        for user in users:
            key = daily_identity(user_id=user.id, bot_identity=bot_identity,
                                 chat_id=user.telegram_id, local_date=local_date,
                                 timezone_name=timezone_name, planned_at=planned)
            legacy = await db.get(DailyBriefDelivery,
                                  {"user_id": user.id, "brief_date": local_date.isoformat()})
            state = ("LEGACY_SENT" if legacy and legacy.sent_at is not None else
                     "SKIPPED" if now > planned + RUNDOWN_CATCHUP else "PENDING")
            await _insert_occurrence(db, dict(
                id=str(uuid4()), occurrence_key=key, kind="daily_rundown",
                user_id=user.id, bot_identity=bot_identity, chat_id=user.telegram_id,
                local_date=local_date.isoformat(), timezone_name=timezone_name,
                planned_at=planned, state=state,
                failure_category="CatchupExpired" if state == "SKIPPED" else None,
                completed_at=now if state in {"SKIPPED", "LEGACY_SENT"} else None,
            ))
            keys.append(key)
        await db.commit()
        return keys


async def upsert_due_reminders(session_factory, *, bot_identity: int,
                               timezone_name: str, now: datetime | None = None,
                               allowed_chat_ids: frozenset[str] | None = None) -> list[str]:
    """Reuse Todo's SKIP LOCKED discovery; occurrence, not claim timestamp, owns delivery."""
    now = now or utc_now()
    async with session_factory() as db:
        todos = await crud.claim_due_reminders(
            db, before=now + timedelta(minutes=1),
            allowed_telegram_ids=allowed_chat_ids)
        keys: list[str] = []
        for todo in todos:
            if todo.user is None or todo.reminder_at is None:
                continue
            due = as_utc(todo.reminder_at)
            key = reminder_identity(todo_id=todo.id, user_id=todo.user_id,
                                    bot_identity=bot_identity,
                                    chat_id=todo.user.telegram_id, reminder_at=due,
                                    reminder_version=todo.reminder_version)
            state = "SKIPPED" if now > due + REMINDER_RELEVANCE else "PENDING"
            await _insert_occurrence(db, dict(
                id=str(uuid4()), occurrence_key=key, kind="reminder",
                user_id=todo.user_id, bot_identity=bot_identity,
                chat_id=todo.user.telegram_id, timezone_name=timezone_name,
                planned_at=due, todo_id=todo.id, reminder_at_snapshot=due,
                reminder_version_snapshot=todo.reminder_version,
                state=state,
                failure_category="ReminderExpired" if state == "SKIPPED" else None,
                completed_at=now if state == "SKIPPED" else None,
            ))
            # The old claim remains a short row-lock discovery aid, not a
            # second delivery authority or a five-minute retry throttle.
            todo.reminder_claimed_at = None
            keys.append(key)
        await db.commit()
        return keys


async def claim_occurrence(session_factory, occurrence_id: str, owner: str,
                           *, now: datetime | None = None) -> str | None:
    now = now or utc_now()
    token = str(uuid4())
    async with session_factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.id == occurrence_id).with_for_update(skip_locked=True))
        if (row is None or as_utc(row.planned_at) > now or
                row.state not in {"PENDING", "FAILED", "CLAIMED"}):
            return None
        if row.state == "CLAIMED" and row.lease_until and as_utc(row.lease_until) > now:
            return None
        if row.state == "FAILED" and row.next_attempt_at and as_utc(row.next_attempt_at) > now:
            return None
        if row.kind == "daily_rundown" and now > as_utc(row.planned_at) + RUNDOWN_CATCHUP:
            row.state, row.failure_category, row.completed_at = "SKIPPED", "CatchupExpired", now
            await db.commit()
            return None
        if row.kind == "reminder" and now > as_utc(row.planned_at) + REMINDER_RELEVANCE:
            row.state, row.failure_category, row.completed_at = "SKIPPED", "ReminderExpired", now
            await db.commit()
            return None
        row.state = "CLAIMED"
        row.attempts += 1
        row.lease_owner, row.fence_token = owner, token
        row.lease_until = now + OCCURRENCE_LEASE
        row.next_attempt_at = None
        await db.commit()
        return token


async def fail_occurrence(session_factory, occurrence_id: str, token: str,
                          category: str, *, now: datetime | None = None) -> bool:
    now = now or utc_now()
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id, with_for_update=True)
        if row is None or row.state != "CLAIMED" or row.fence_token != token:
            return False
        row.state = "EXHAUSTED" if row.attempts >= MAX_RENDER_ATTEMPTS else "FAILED"
        row.next_attempt_at = None if row.state == "EXHAUSTED" else now + timedelta(seconds=30)
        row.failure_category = category if category.isidentifier() else "RenderFailure"
        row.lease_owner = row.fence_token = row.lease_until = None
        row.completed_at = now if row.state == "EXHAUSTED" else None
        await db.commit()
        return True


async def ready_occurrence(session_factory, occurrence_id: str, token: str,
                           *, text: str, locale: str, parse_mode: str | None = None,
                           now: datetime | None = None) -> str | None:
    """Fence and atomically commit immutable rendered intent and READY state."""
    now = now or utc_now()
    chunks = render_chunks(text, parse_mode=parse_mode, keyboard=None)
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id, with_for_update=True)
        if (row is None or row.state != "CLAIMED" or row.fence_token != token or
                row.lease_until is None or as_utc(row.lease_until) <= now):
            return None
        if row.kind == "reminder":
            todo = await db.get(Todo, row.todo_id, with_for_update=True)
            if (todo is None or todo.status == TodoStatus.done or
                    todo.reminder_state != "ACTIVE" or
                    todo.reminder_version != (row.reminder_version_snapshot or 0) or
                    todo.reminder_at is None or
                    as_utc(todo.reminder_at) != as_utc(row.reminder_at_snapshot) or
                    now > as_utc(row.planned_at) + REMINDER_RELEVANCE):
                row.state, row.failure_category, row.completed_at = "SKIPPED", "ReminderIrrelevant", now
                row.lease_owner = row.fence_token = row.lease_until = None
                await db.commit()
                return None
        existing = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrence_id))
        if existing is not None:
            # The delivery insert and READY are one transaction; this branch
            # catches a corrupted/incompatible replay rather than redrafting.
            raise RuntimeError("UnexpectedExistingOccurrenceDelivery")
        digest = stable_key(row.kind, row.chat_id, locale, text, parse_mode)
        delivery = OutboundDelivery(
            id=str(uuid4()), receipt_id=None, occurrence_id=occurrence_id,
            reply_ordinal=0, kind=row.kind, chat_id=row.chat_id, locale=locale,
            text=text, parse_mode=parse_mode, keyboard=None,
            payload_sha256=digest, state="PENDING")
        db.add(delivery)
        for index, chunk_text in enumerate(chunks):
            db.add(OutboundDeliveryChunk(
                delivery_id=delivery.id, chunk_index=index, text=chunk_text,
                content_sha256=hashlib.sha256(chunk_text.encode()).hexdigest(),
                state="PENDING"))
        row.state = "READY"
        row.lease_owner = row.fence_token = row.lease_until = None
        await db.commit()
        return delivery.id


async def reconcile_occurrence(session_factory, occurrence_id: str,
                               *, now: datetime | None = None) -> str | None:
    """Aggregate only confirmed chunk proof; never resend a confirmed chunk."""
    now = now or utc_now()
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id, with_for_update=True)
        if row is None:
            return None
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrence_id))
        if row.state != "READY" or delivery is None:
            return row.state
        if delivery.state == "SENT":
            chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id == delivery.id))).all()
            if chunks and all(chunk.state == "SENT" and chunk.telegram_message_id is not None
                              for chunk in chunks):
                row.state, row.completed_at = "DELIVERED", now
                if row.kind == "reminder" and row.todo_id:
                    todo = await db.get(Todo, row.todo_id, with_for_update=True)
                    if (todo is not None and todo.reminder_at is not None and
                            todo.reminder_state == "ACTIVE" and
                            todo.reminder_version == (row.reminder_version_snapshot or 0) and
                            as_utc(todo.reminder_at) == as_utc(row.reminder_at_snapshot)):
                        todo.reminder_sent_at = now
                        todo.reminder_sent = True
                        todo.reminder_claimed_at = None
        elif delivery.state == "EXHAUSTED":
            row.state, row.failure_category, row.completed_at = "EXHAUSTED", "DeliveryExhausted", now
        await db.commit()
        return row.state


async def reminder_relevant(session_factory, row: ScheduledOccurrence,
                            *, now: datetime | None = None) -> bool:
    now = now or utc_now()
    if row.kind != "reminder":
        return True
    if now > as_utc(row.planned_at) + REMINDER_RELEVANCE:
        return False
    async with session_factory() as db:
        todo = await db.get(Todo, row.todo_id)
        return bool(todo and todo.status != TodoStatus.done and
                    todo.reminder_state == "ACTIVE" and
                    todo.reminder_version == (row.reminder_version_snapshot or 0) and
                    todo.reminder_at is not None and
                    as_utc(todo.reminder_at) == as_utc(row.reminder_at_snapshot))


async def skip_irrelevant_reminder(session_factory, occurrence_id: str,
                                   *, now: datetime | None = None) -> bool:
    now = now or utc_now()
    async with session_factory() as db:
        row = await db.get(ScheduledOccurrence, occurrence_id, with_for_update=True)
        if row is None or row.kind != "reminder" or row.state not in {"READY", "PENDING", "FAILED", "CLAIMED"}:
            return False
        if row.state == "CLAIMED" and row.lease_until and as_utc(row.lease_until) > now:
            return False
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrence_id))
        if delivery is not None:
            chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id == delivery.id).with_for_update())).all()
            if any(chunk.state == "SENDING" and chunk.lease_until and as_utc(chunk.lease_until) > now
                   for chunk in chunks):
                return False
            ambiguous = any(chunk.state in {"UNCERTAIN", "SENDING", "SENT"}
                            for chunk in chunks)
            for chunk in chunks:
                if chunk.state == "PENDING":
                    chunk.state, chunk.failure_category = "EXHAUSTED", "ReminderIrrelevant"
            delivery.state = "EXHAUSTED"
            delivery.terminal_at = now
        else:
            ambiguous = False
        row.state = "EXHAUSTED" if ambiguous else "SKIPPED"
        row.failure_category = ("ReminderIrrelevantAfterPossibleSend" if ambiguous
                                else "ReminderIrrelevant")
        row.completed_at = now
        row.lease_owner = row.fence_token = row.lease_until = None
        await db.commit()
        return True


class ProactiveRuntime:
    """One process's bounded recovery worker; PostgreSQL owns all work."""

    def __init__(self, session_factory, bot, *, bot_identity: int,
                 timezone_name: str, daily_hour: int,
                 daily_renderer, reminder_renderer, on_failure=None,
                  daily_enabled: bool = True,
                  reminder_allowed_chat_ids: frozenset[str] | None = None,
                  daily_allowed_chat_ids: frozenset[str] | None = None):
        self.session_factory = session_factory
        self.bot = bot
        self.bot_identity = bot_identity
        self.timezone_name = timezone_name
        self.daily_hour = daily_hour
        self.daily_renderer = daily_renderer
        self.daily_enabled = daily_enabled
        self.reminder_allowed_chat_ids = reminder_allowed_chat_ids
        self.daily_allowed_chat_ids = daily_allowed_chat_ids
        self.reminder_renderer = reminder_renderer
        self.on_failure = on_failure
        self.owner = str(uuid4())
        self._closing = False
        self._task = None
        self._active_tasks: set[asyncio.Task] = set()

    async def preflight(self) -> None:
        async with self.session_factory() as db:
            await db.scalar(select(ScheduledOccurrence.id).limit(1))
            await db.scalar(select(OutboundDelivery.occurrence_id).limit(1))

    async def start(self) -> None:
        await self.preflight()
        await self.tick()
        self._task = asyncio.create_task(self._loop())

    async def tick(self) -> None:
        if self._closing:
            return
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.add(task)
        try:
            now = utc_now()
            local = now.astimezone(ZoneInfo(self.timezone_name))
            if self.daily_enabled:
                await upsert_daily(self.session_factory, bot_identity=self.bot_identity,
                                   timezone_name=self.timezone_name, hour=self.daily_hour,
                                   local_date=local.date(), now=now,
                                   allowed_chat_ids=self.daily_allowed_chat_ids)
                if self._closing:
                    return
                await upsert_daily(self.session_factory, bot_identity=self.bot_identity,
                                   timezone_name=self.timezone_name, hour=self.daily_hour,
                                   local_date=local.date() - timedelta(days=1), now=now,
                                   allowed_chat_ids=self.daily_allowed_chat_ids)
                if self._closing:
                    return
            await upsert_due_reminders(self.session_factory, bot_identity=self.bot_identity,
                                        timezone_name=self.timezone_name, now=now,
                                        allowed_chat_ids=self.reminder_allowed_chat_ids)
            if self._closing:
                return
            async with self.session_factory() as db:
                stmt = select(ScheduledOccurrence.id).where(or_(
                    ScheduledOccurrence.state == "PENDING",
                    and_(ScheduledOccurrence.state == "FAILED",
                         ScheduledOccurrence.next_attempt_at <= now),
                    and_(ScheduledOccurrence.state == "CLAIMED",
                         ScheduledOccurrence.lease_until <= now),
                    ScheduledOccurrence.state == "READY",
                ), ScheduledOccurrence.planned_at <= now)
                if self.daily_allowed_chat_ids is not None:
                    stmt = stmt.where(or_(
                        ScheduledOccurrence.kind != "daily_rundown",
                        ScheduledOccurrence.chat_id.in_(self.daily_allowed_chat_ids)))
                if self.reminder_allowed_chat_ids is not None:
                    stmt = stmt.where(or_(
                        ScheduledOccurrence.kind != "reminder",
                        ScheduledOccurrence.chat_id.in_(self.reminder_allowed_chat_ids)))
                ids = (await db.scalars(stmt.order_by(
                    ScheduledOccurrence.planned_at, ScheduledOccurrence.id).limit(100))).all()
            for occurrence_id in ids:
                if self._closing:
                    break
                await self.process(occurrence_id)
        finally:
            if task is not None:
                self._active_tasks.discard(task)

    async def process(self, occurrence_id: str) -> None:
        if self._closing:
            return
        async with self.session_factory() as db:
            row = await db.get(ScheduledOccurrence, occurrence_id)
            if row is None:
                return
            allowed = (self.daily_allowed_chat_ids if row.kind == "daily_rundown"
                       else self.reminder_allowed_chat_ids)
            if allowed is not None and row.chat_id not in allowed:
                return
        if row.state == "READY":
            # Remote proof wins over a later source change. A crash can happen
            # after Telegram confirms the final chunk but before aggregate
            # occurrence finalization; reconcile that proof before deciding a
            # now-completed/rescheduled reminder is irrelevant.
            if await reconcile_occurrence(self.session_factory, occurrence_id) != "READY":
                return
        if row.kind == "reminder" and not await reminder_relevant(self.session_factory, row):
            await skip_irrelevant_reminder(self.session_factory, occurrence_id)
            return
        if row.state == "READY":
            if not self._closing:
                await self._dispatch_and_reconcile(occurrence_id)
            return
        if self._closing:
            return
        token = await claim_occurrence(self.session_factory, occurrence_id, self.owner)
        if token is None:
            return
        try:
            async with self.session_factory() as db:
                claimed = await db.get(ScheduledOccurrence, occurrence_id)
            if claimed.kind == "daily_rundown":
                text, locale = await self.daily_renderer(claimed)
            else:
                text, locale = await self.reminder_renderer(claimed)
            delivery_id = await ready_occurrence(self.session_factory, occurrence_id,
                                                  token, text=text, locale=locale)
            if delivery_id is not None and not self._closing:
                await self._dispatch_and_reconcile(occurrence_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("proactive stage=render category=%s", type(exc).__name__)
            await fail_occurrence(self.session_factory, occurrence_id, token,
                                  type(exc).__name__)

    async def _dispatch_and_reconcile(self, occurrence_id: str) -> None:
        async with self.session_factory() as db:
            delivery_id = await db.scalar(select(OutboundDelivery.id).where(
                OutboundDelivery.occurrence_id == occurrence_id))
        if delivery_id is None:
            return
        await dispatch_delivery(self.session_factory, delivery_id, self.owner, self.bot)
        await reconcile_occurrence(self.session_factory, occurrence_id)

    async def _loop(self) -> None:
        failures = 0
        while not self._closing:
            try:
                # Cancellation of the recovery wakeup must not cancel a
                # render/send already in progress. tick owns its own task.
                work = asyncio.create_task(self.tick())
                work.add_done_callback(observe_detached_tick)
                await asyncio.shield(work)
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("proactive stage=recovery category=%s", type(exc).__name__)
                failures += 1
                if failures >= 3:
                    if self.on_failure:
                        self.on_failure(exc)
                    return
            await asyncio.sleep(30)

    async def quiesce(self) -> None:
        """Stop wakeups and future claims; leave active ticks to drain."""
        self._closing = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def drain(self, deadline: float) -> None:
        tasks = {task for task in self._active_tasks if not task.done()}
        if tasks:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            _, pending = await asyncio.wait(tasks, timeout=remaining)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()

    async def stop(self, drain_seconds: float = 30.0) -> None:
        await self.quiesce()
        await self.drain(asyncio.get_running_loop().time() + drain_seconds)
