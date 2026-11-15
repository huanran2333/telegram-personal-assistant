"""R4.2 recoverable Telegram update and interactive-reply worker."""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from sqlalchemy import and_, or_, select, text

from app.db.models import OutboundDelivery, TelegramInboundReceipt
from app.telegram.durable import (claim_receipt, dispatch_delivery, fail_receipt,
                                  finalize_receipt, has_complete_outcome_and_intents,
                                  thread_invocation_guard, telegram_workflow_guard)
from app.telegram.processing_context import receipt_scope

logger = logging.getLogger(__name__)


class DurableTelegramRuntime:
    def __init__(self, session_factory, bot, *, database_url: str | None = None,
                 on_recovery_failure=None, retention_days: int = 30):
        self.session_factory = session_factory
        self.bot = bot
        self.bot_identity: int | None = None
        self.owner = str(uuid4())
        self._closing = False
        self.admission_lock = asyncio.Lock()
        self._loop_task: asyncio.Task | None = None
        self._markup_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        if database_url is None:
            from app.config import get_settings
            database_url = get_settings().database_url_sync
        self.database_url = database_url
        self.on_recovery_failure = on_recovery_failure
        self.retention_days = retention_days

    @property
    def is_running(self) -> bool:
        return not self._closing and self._loop_task is not None and not self._loop_task.done()

    async def verify_bot_identity(self) -> int:
        # getMe is an authenticated Bot API identity check. Never store token
        # or its hash as the receipt namespace.
        me = await self.bot.get_me(read_timeout=20, connect_timeout=20,
                                   pool_timeout=20)
        if not isinstance(me.id, int) or me.id <= 0:
            raise RuntimeError("InvalidBotIdentity")
        self.bot_identity = me.id
        logger.info("telegram_preflight stage=bot_identity status=verified")
        return me.id

    async def start(self) -> None:
        if self.bot_identity is None:
            await self.verify_bot_identity()
        async with self.session_factory() as db:
            await db.execute(text("SELECT 1 FROM telegram_inbound_receipts LIMIT 1"))
            await db.execute(text("SELECT 1 FROM outbound_deliveries LIMIT 1"))
            await db.execute(text("SELECT 1 FROM telegram_thread_leases LIMIT 1"))
            await db.execute(text("SELECT 1 FROM telegram_approval_markup_sync LIMIT 1"))
        logger.info("telegram_preflight stage=durable_schema status=verified")
        self._closing = False
        self._loop_task = asyncio.create_task(self._recover_loop())
        self._markup_task = asyncio.create_task(self._markup_loop())

    async def _markup_loop(self) -> None:
        """A UI-only R4 edit sweep; failure cannot revoke a Saver decision."""
        from app.agent.workflow_runtime import get_productivity_graph
        from app.telegram.approval_markup import sync_approval_markups
        while not self._closing:
            try:
                graph = await get_productivity_graph()
                await sync_approval_markups(
                    self.session_factory, self.bot, self.bot_identity, graph)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("approval_markup stage=sweep category=%s", type(exc).__name__)
            await asyncio.sleep(5)

    def kick(self, receipt_id: str) -> None:
        if self._closing or self._loop_task is None or self._loop_task.done():
            raise RuntimeError("IngressRecoveryUnavailable")
        task = asyncio.create_task(self._process(receipt_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _heartbeat(self, receipt_id: str, token: str) -> None:
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import select
        from app.db.models import TelegramThreadLease
        # An already-claimed receipt may finish during graceful drain. Its
        # heartbeat belongs to that task, not to the recovery/admission loop.
        while True:
            await asyncio.sleep(30)
            now = datetime.now(timezone.utc)
            async with self.session_factory() as db:
                receipt = await db.scalar(select(TelegramInboundReceipt).where(
                    TelegramInboundReceipt.id == receipt_id).with_for_update())
                if receipt is None or receipt.state != "PROCESSING" or receipt.fence_token != token:
                    return
                lease = await db.get(TelegramThreadLease,
                                     {"graph_kind": receipt.graph_kind,
                                      "thread_id": receipt.thread_id}, with_for_update=True)
                if lease is None or lease.fence_token != token:
                    return
                # The worker must renew before expiry. A delayed/stale owner
                # cannot resurrect a lease after another worker may claim it.
                if receipt.lease_until <= now or lease.lease_until <= now:
                    return
                receipt.lease_until = lease.lease_until = now + timedelta(seconds=120)
                await db.commit()

    async def _process(self, receipt_id: str) -> None:
        if self._closing:
            return
        token = await claim_receipt(self.session_factory, receipt_id, self.owner)
        if token is None:
            return
        heartbeat = asyncio.create_task(self._heartbeat(receipt_id, token))
        try:
            async with self.session_factory() as db:
                receipt = await db.get(TelegramInboundReceipt, receipt_id)
                if receipt is None:
                    raise RuntimeError("MissingReceipt")
                envelope = receipt.envelope
            from app.telegram.webhook import process_admitted_update
            async with (thread_invocation_guard(self.database_url, receipt.graph_kind,
                                                receipt.thread_id),
                        telegram_workflow_guard(self.database_url, receipt.graph_kind,
                                                receipt.thread_id)):
                # A prior owner may have lost the receipt fence while waiting
                # for the session lock. Never invoke on its stale authority.
                async with self.session_factory() as db:
                    current = await db.get(TelegramInboundReceipt, receipt_id)
                    if current is None or current.state != "PROCESSING" or current.fence_token != token:
                        return
                    complete_outcome = await has_complete_outcome_and_intents(db, current)
                # A receipt already claimed but waiting for the same-thread
                # guard must not *begin* an Agent invoke during shutdown.
                # Its lease remains available for restart recovery.
                if self._closing and not complete_outcome:
                    return
                # Reuse an immutable reply only when its business proof was
                # atomically committed with the complete set of intents.
                if not complete_outcome:
                    with receipt_scope(receipt_id, token):
                        await process_admitted_update(envelope)
                if not await finalize_receipt(self.session_factory, receipt_id, token):
                    logger.warning("telegram_receipt stage=finalize category=LostFence")
                    return
            # The intent is durable. A shutdown that began during the Agent
            # run must not start a *new* send; restart recovery will dispatch.
            if not self._closing:
                await self._dispatch_receipt(receipt_id)
        except asyncio.CancelledError:
            # Lease remains durable for recovery. Never mark a cancelled run
            # processed merely because the Python task has exited.
            raise
        except Exception as exc:
            logger.warning("telegram_receipt stage=process category=%s", type(exc).__name__)
            try:
                await fail_receipt(self.session_factory, receipt_id, token,
                                   type(exc).__name__)
            except Exception as failure:
                logger.error("telegram_receipt stage=record_failure category=%s",
                             type(failure).__name__)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("telegram_receipt stage=heartbeat category=%s", type(exc).__name__)

    async def _dispatch_receipt(self, receipt_id: str) -> None:
        async with self.session_factory() as db:
            ids = (await db.scalars(select(OutboundDelivery.id).where(
                OutboundDelivery.receipt_id == receipt_id).order_by(
                    OutboundDelivery.reply_ordinal))).all()
        for delivery_id in ids:
            if self._closing:
                break
            await dispatch_delivery(self.session_factory, delivery_id, self.owner, self.bot)

    async def _recover_loop(self) -> None:
        from datetime import datetime, timezone
        from app.telegram.retention import redact_due_payloads
        consecutive_failures = 0
        last_retention = asyncio.get_running_loop().time()
        while not self._closing:
            try:
                clock = asyncio.get_running_loop().time()
                if clock - last_retention >= 3600:
                    result = await redact_due_payloads(
                        self.session_factory, retention_days=self.retention_days)
                    logger.info("telegram_retention receipts=%d deliveries=%d",
                                result.receipts, result.deliveries)
                    last_retention = clock
                now = datetime.now(timezone.utc)
                async with self.session_factory() as db:
                    receipts = (await db.scalars(select(TelegramInboundReceipt.id).where(
                        or_(TelegramInboundReceipt.state == "RECEIVED",
                            and_(TelegramInboundReceipt.state == "FAILED",
                                 TelegramInboundReceipt.next_attempt_at <= now),
                            and_(TelegramInboundReceipt.state == "PROCESSING",
                                 TelegramInboundReceipt.lease_until <= now)))
                        .order_by(TelegramInboundReceipt.received_at).limit(100))).all()
                    deliveries = (await db.scalars(select(OutboundDelivery.id).join(
                        TelegramInboundReceipt,
                        TelegramInboundReceipt.id == OutboundDelivery.receipt_id).where(
                        TelegramInboundReceipt.state == "PROCESSED",
                        OutboundDelivery.state.notin_(("SENT", "EXHAUSTED")))
                        .order_by(OutboundDelivery.created_at).limit(100))).all()
                for receipt_id in receipts:
                    if self._closing:
                        break
                    self.kick(receipt_id)
                for delivery_id in deliveries:
                    if self._closing:
                        break
                    task = asyncio.create_task(dispatch_delivery(
                        self.session_factory, delivery_id, self.owner, self.bot))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("telegram_recovery stage=sweep category=%s", type(exc).__name__)
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    if self.on_recovery_failure is not None:
                        self.on_recovery_failure(exc)
                    return
            await asyncio.sleep(5)

    async def quiesce(self) -> None:
        """Stop admission/recovery claims without cancelling active work."""
        async with self.admission_lock:
            self._closing = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        if self._markup_task is not None:
            self._markup_task.cancel()
            try:
                await self._markup_task
            except asyncio.CancelledError:
                pass

    async def drain(self, deadline: float) -> None:
        """Give active work until the shared monotonic deadline, then cancel it."""
        tasks = {task for task in self._tasks if not task.done()}
        if tasks:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            _, pending = await asyncio.wait(tasks, timeout=remaining)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def stop(self, drain_seconds: float = 30.0) -> None:
        await self.quiesce()
        await self.drain(asyncio.get_running_loop().time() + drain_seconds)
