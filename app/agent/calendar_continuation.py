"""Durable wakeup locator for a Saver-owned, already-approved dependent workflow.

The locator is not an approval, Todo payload or provider outcome authority.  A
worker checks R5 evidence before claiming and the original graph checks it
again when resumed.  PostgreSQL thread leases share R4's graph-thread scope.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app.db.models import (ActionExecution, CalendarActionEvidence,
                           CalendarDependentContinuation, TelegramThreadLease)
from app.telegram.durable import thread_invocation_guard

logger = logging.getLogger(__name__)
_LEASE_SECONDS = 120


def _now():
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    # SQLite is only an isolated E3 harness and strips timezone metadata.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class CalendarContinuationStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def register_waiting(self, parent_action_id: str, thread_id: str,
                               *, resume_checkpoint_id: str | None = None) -> None:
        child_action_id = f"{parent_action_id}:todo"
        async with self.session_factory() as db:
            row = await db.get(CalendarDependentContinuation, parent_action_id)
            if row is not None:
                if row.thread_id != thread_id or row.child_action_id != child_action_id:
                    raise ValueError("Calendar continuation identity changed")
                if row.checkpoint_id is None and resume_checkpoint_id:
                    row.checkpoint_id = resume_checkpoint_id
                    await db.commit()
                return
            db.add(CalendarDependentContinuation(
                parent_action_id=parent_action_id, thread_id=thread_id,
                child_action_id=child_action_id, state="WAITING",
                checkpoint_id=resume_checkpoint_id))
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                row = await db.get(CalendarDependentContinuation, parent_action_id)
                if row is None or row.thread_id != thread_id or row.child_action_id != child_action_id:
                    raise

    async def bind_checkpoint(self, parent_action_id: str, thread_id: str,
                              checkpoint_id: str) -> None:
        async with self.session_factory() as db:
            row = await db.get(CalendarDependentContinuation, parent_action_id,
                               with_for_update=True)
            if row is None or row.thread_id != thread_id or row.state == "COMPLETED":
                return
            row.checkpoint_id = checkpoint_id
            await db.commit()

    async def due_ids(self, limit: int = 20) -> list[str]:
        now = _now()
        async with self.session_factory() as db:
            return list((await db.scalars(select(CalendarDependentContinuation.parent_action_id)
                .join(CalendarActionEvidence, CalendarActionEvidence.action_id ==
                      CalendarDependentContinuation.parent_action_id)
                .join(ActionExecution, ActionExecution.action_id ==
                      CalendarDependentContinuation.parent_action_id)
                .where(CalendarDependentContinuation.state.in_(("WAITING", "READY")),
                       CalendarActionEvidence.outcome == "CONFIRMED",
                       ActionExecution.status == "executed",
                       ((CalendarDependentContinuation.lease_until.is_(None)) |
                        (CalendarDependentContinuation.lease_until <= now)))
                .order_by(CalendarDependentContinuation.created_at).limit(limit))).all())

    async def claim(self, action_id: str, owner: str) -> tuple[str, str, str | None, str] | None:
        now = _now()
        until = now + timedelta(seconds=_LEASE_SECONDS)
        token = str(uuid4())
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarDependentContinuation).where(
                CalendarDependentContinuation.parent_action_id == action_id)
                .with_for_update(skip_locked=True))
            if row is None or row.state not in {"WAITING", "READY"} or (
                    row.lease_until is not None and _utc(row.lease_until) > now):
                return None
            evidence = await db.get(CalendarActionEvidence, action_id)
            ledger = await db.get(ActionExecution, action_id)
            if evidence is None or ledger is None or evidence.user_id != ledger.user_id or (
                    evidence.outcome != "CONFIRMED" or ledger.status != "executed"):
                return None
            if db.bind.dialect.name == "postgresql":
                await db.execute(pg_insert(TelegramThreadLease).values(
                    graph_kind="workflow", thread_id=row.thread_id,
                    receipt_id=row.id, lease_owner=owner, fence_token=token,
                    lease_until=until).on_conflict_do_nothing())
            else:
                existing = await db.get(TelegramThreadLease, {
                    "graph_kind": "workflow", "thread_id": row.thread_id})
                if existing is None:
                    db.add(TelegramThreadLease(
                        graph_kind="workflow", thread_id=row.thread_id,
                        receipt_id=row.id, lease_owner=owner, fence_token=token,
                        lease_until=until))
                    await db.flush()
            lease = await db.scalar(select(TelegramThreadLease).where(
                TelegramThreadLease.graph_kind == "workflow",
                TelegramThreadLease.thread_id == row.thread_id).with_for_update())
            if lease.receipt_id != row.id and _utc(lease.lease_until) > now:
                return None
            lease.receipt_id = row.id
            lease.lease_owner = owner
            lease.fence_token = token
            lease.lease_until = until
            row.lease_owner = owner
            row.fence_token = token
            row.lease_until = until
            row.state = "READY"
            await db.commit()
            return token, row.thread_id, row.checkpoint_id, ledger.user_id

    async def renew(self, action_id: str, token: str) -> bool:
        now = _now()
        async with self.session_factory() as db:
            row = await db.get(CalendarDependentContinuation, action_id, with_for_update=True)
            if row is None or row.fence_token != token or _utc(row.lease_until) <= now:
                return False
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": row.thread_id},
                with_for_update=True)
            if lease is None or lease.fence_token != token or _utc(lease.lease_until) <= now:
                return False
            until = now + timedelta(seconds=_LEASE_SECONDS)
            row.lease_until = lease.lease_until = until
            await db.commit()
            return True

    async def is_current(self, action_id: str, token: str) -> bool:
        async with self.session_factory() as db:
            row = await db.get(CalendarDependentContinuation, action_id)
            if row is None or row.fence_token != token or _utc(row.lease_until) <= _now():
                return False
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": row.thread_id})
            return bool(lease and lease.fence_token == token and
                        _utc(lease.lease_until) > _now())

    async def finish(self, action_id: str, token: str, *, completed: bool = False,
                     checkpoint_id: str | None = None) -> bool:
        async with self.session_factory() as db:
            row = await db.get(CalendarDependentContinuation, action_id, with_for_update=True)
            if (row is None or row.fence_token != token or row.lease_until is None or
                    _utc(row.lease_until) <= _now()):
                return False
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": row.thread_id},
                with_for_update=True)
            if (lease is None or lease.fence_token != token or
                    _utc(lease.lease_until) <= _now()):
                return False
            if completed:
                child = await db.get(ActionExecution, row.child_action_id)
                if child is None:
                    raise RuntimeError("Continuation lacks completed child proof")
                row.state = "COMPLETED" if child.status == "executed" else "REVIEW_REQUIRED"
                if row.state == "COMPLETED":
                    row.completed_at = _now()
            else:
                row.state = "WAITING"
            if checkpoint_id:
                row.checkpoint_id = checkpoint_id
            row.lease_owner = row.fence_token = row.lease_until = None
            await db.delete(lease)
            await db.commit()
            return row.state == "COMPLETED"


class CalendarContinuationRuntime:
    def __init__(self, graph, session_factory):
        self.graph = graph
        self.store = CalendarContinuationStore(session_factory)
        self.owner = f"calendar-continuation:{uuid4().hex}"
        bind = session_factory.kw.get("bind")
        self.database_url = bind.url if bind is not None and (
            bind.dialect.name == "postgresql") else None

    async def preflight(self) -> None:
        if self.graph.checkpointer is None:
            raise RuntimeError("ContinuationRequiresSaver")
        async with self.store.session_factory() as db:
            await db.execute(select(CalendarDependentContinuation.parent_action_id).limit(1))

    @asynccontextmanager
    async def _thread_guard(self, thread_id: str):
        if self.database_url:
            async with thread_invocation_guard(self.database_url, "workflow", thread_id):
                yield
        else:
            # SQLite/InMemorySaver is an isolated E3 test path only.
            yield

    async def _heartbeat(self, action_id: str, token: str) -> None:
        while True:
            await asyncio.sleep(30)
            if not await self.store.renew(action_id, token):
                return

    async def recover_due(self, limit: int = 20) -> int:
        ids = await self.store.due_ids(limit)
        for action_id in ids:
            try:
                await self.recover(action_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("action_id=%s continuation category=%s",
                               action_id, type(exc).__name__)
        return len(ids)

    async def recover(self, action_id: str) -> bool:
        claim = await self.store.claim(action_id, self.owner)
        if claim is None:
            return False
        token, thread_id, checkpoint_id, user_id = claim
        heartbeat = asyncio.create_task(self._heartbeat(action_id, token))
        try:
            async with self._thread_guard(thread_id):
                if not await self.store.is_current(action_id, token):
                    return False
                config = {"configurable": {"thread_id": thread_id}}
                if checkpoint_id:
                    config["configurable"]["checkpoint_id"] = checkpoint_id
                snapshot = await self.graph.aget_state(config)
                if (not snapshot or snapshot.values.get("proposal", {}).get("action_id") != action_id
                        or snapshot.values.get("user_id") != user_id):
                    # Never resume a different approval on this thread.
                    return False
                if "wait_for_calendar" in snapshot.next:
                    from langgraph.types import Command
                    await self.graph.ainvoke(Command(resume={"calendar_confirmation": action_id}), config)
                elif "execute" in snapshot.next or "execute_child_todo" in snapshot.next:
                    await self.graph.ainvoke(None, config)
                final = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
                completed = bool(final and not final.next and
                                 final.values.get("proposal", {}).get("action_id") == action_id)
                return await self.store.finish(
                    action_id, token, completed=completed,
                    checkpoint_id=(final.config.get("configurable", {}).get("checkpoint_id")
                                   if final else None))
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
