"""E4 isolated PG18 checks; no Telegram or model-provider calls."""
import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.requests import Request

from app.db.models import (ActionExecution, OutboundDelivery, OutboundDeliveryAttempt,
                           OutboundDeliveryChunk, TelegramBusinessOperation,
                           TelegramInboundReceipt, TelegramThreadLease, Todo, User)
from app.db.migrations.telegram_todo_execution_backfill import retire
from app.telegram.durable import (admit_receipt, claim_receipt, dispatch_delivery,
                                  enqueue_reply, finalize_receipt, admit_terminal_receipt,
                                  thread_invocation_guard, fail_receipt, DurableSafetyError,
                                  record_resume_origin, telegram_todo_action_id)
from app.telegram.processing_context import receipt_scope, tool_call_scope

pytestmark = pytest.mark.integration


def update_payload(update_id: int, *, actor: int = 70001, text: str = "synthetic hello"):
    return {"update_id": update_id, "message": {
        "message_id": 1, "date": 0,
        "from": {"id": actor, "is_bot": False, "first_name": "Synthetic"},
        "chat": {"id": actor, "type": "private"}, "text": text}}


async def cleanup(factory, receipt_ids):
    async with factory() as db:
        delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
            OutboundDelivery.receipt_id.in_(receipt_ids)))).all()
        if delivery_ids:
            await db.execute(delete(OutboundDeliveryAttempt).where(
                OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id.in_(delivery_ids)))
        for receipt_id in receipt_ids:
            await db.execute(delete(ActionExecution).where(
                ActionExecution.action_id.startswith(f"tgtodo:{receipt_id}:")))
        await db.execute(delete(TelegramThreadLease).where(
            TelegramThreadLease.receipt_id.in_(receipt_ids)))
        await db.execute(delete(TelegramInboundReceipt).where(
            TelegramInboundReceipt.id.in_(receipt_ids)))
        await db.commit()


@pytest.mark.asyncio
async def test_pg_action_execution_telegram_proof_replays_without_second_todo(pg_harness):
    factory, new_user, engine = pg_harness
    user = await new_user()
    actor = 700000 + int(uuid4().hex[:6], 16)
    receipt_ids = []
    try:
        async with factory() as db:
            owner = await db.get(User, user.id)
            owner.telegram_id = str(actor)
            await db.commit()
        receipt = await admit_receipt(
            factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
            data=update_payload(int(uuid4().hex[:12], 16), actor=actor),
            graph_kind="telegram_session", thread_id=f"legacy-proof-{actor}")
        receipt_ids.append(receipt.id)
        fence = await claim_receipt(factory, receipt.id, "legacy-proof-worker")
        args = {"user_id": user.id, "title": "legacy proof committed task"}
        digest = hashlib.sha256(json.dumps(args, ensure_ascii=False, sort_keys=True,
                                          default=str, separators=(",", ":")).encode()).hexdigest()
        async with factory() as db:
            db.add(Todo(user_id=user.id, title=args["title"]))
            db.add(ActionExecution(
                action_id=telegram_todo_action_id(receipt.id, "create_todo", "legacy-call"),
                user_id=user.id, domain="todo", operation="create_todo",
                status="executed", result={"receipt_id": receipt.id,
                                           "args_sha256": digest,
                                           "text": "created previously"}))
            await db.commit()
        from app.telegram.durable import commit_todo_tool_operation
        async def must_not_perform():
            raise AssertionError("Backfilled operation must never create another Todo")
        with receipt_scope(receipt.id, fence), tool_call_scope("legacy-call"):
            async with factory() as db:
                assert await commit_todo_tool_operation(
                    db, name="create_todo", args=args,
                    perform=must_not_perform) == "created previously"
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(Todo).where(
                Todo.user_id == user.id, Todo.title == args["title"])) == 1
            proof = await db.get(ActionExecution, telegram_todo_action_id(
                receipt.id, "create_todo", "legacy-call"))
            assert proof.status == "executed" and proof.result["args_sha256"] == digest
    finally:
        await cleanup(factory, receipt_ids)


@pytest.mark.asyncio
async def test_pg_explicit_retirement_backfills_and_drops_legacy_table(pg_harness):
    from app.db.models import Base

    _, _, engine = pg_harness
    schema = "retire_tg_" + uuid4().hex
    user_id, receipt_id = str(uuid4()), str(uuid4())
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            await connection.run_sync(lambda sync: Base.metadata.create_all(sync, tables=[
                User.__table__, TelegramInboundReceipt.__table__,
                TelegramBusinessOperation.__table__, ActionExecution.__table__]))
            await connection.execute(User.__table__.insert().values(
                id=user_id, telegram_id="77101"))
            await connection.execute(TelegramInboundReceipt.__table__.insert().values(
                id=receipt_id, bot_identity=7, update_id=17, state="FAILED",
                envelope={"message": {"from": {"id": 77101}}}))
            await connection.execute(TelegramBusinessOperation.__table__.insert().values(
                receipt_id=receipt_id, operation_key="create_todo:call-1",
                args_sha256="a" * 64, status="done", result={"text": "already created"}))
            result = await connection.run_sync(retire)
            assert result == {"copied": 1, "already_present": 0,
                              "skipped_terminal": 0, "dropped": True}
            assert await connection.scalar(text(
                "SELECT to_regclass('telegram_business_operations')")) is None
            proof = (await connection.execute(select(ActionExecution.__table__).where(
                ActionExecution.action_id == telegram_todo_action_id(
                    receipt_id, "create_todo", "call-1")))).mappings().one()
            assert proof["result"]["text"] == "already created"
            assert (await connection.run_sync(retire))["dropped"] is False
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.mark.asyncio
async def test_pg_retirement_refuses_unowned_active_proof_without_dropping(pg_harness):
    from app.db.models import Base

    _, _, engine = pg_harness
    schema = "retire_guard_" + uuid4().hex
    receipt_id = str(uuid4())
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            await connection.run_sync(lambda sync: Base.metadata.create_all(sync, tables=[
                User.__table__, TelegramInboundReceipt.__table__,
                TelegramBusinessOperation.__table__, ActionExecution.__table__]))
            await connection.execute(TelegramInboundReceipt.__table__.insert().values(
                id=receipt_id, bot_identity=8, update_id=18, state="FAILED",
                envelope={"message": {"from": {"id": 77102}}}))
            await connection.execute(TelegramBusinessOperation.__table__.insert().values(
                receipt_id=receipt_id, operation_key="create_todo:call-2",
                args_sha256="b" * 64, status="done", result={"text": "already created"}))
        with pytest.raises(RuntimeError, match="ActiveTelegramProofOwnerUnknown"):
            async with engine.begin() as connection:
                await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
                await connection.run_sync(retire)
        async with engine.begin() as connection:
            await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            assert await connection.scalar(text(
                "SELECT to_regclass('telegram_business_operations')")) is not None
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.mark.asyncio
async def test_pg_unique_admission_restart_and_cross_worker_claim(pg_harness, isolated_pg18_target):
    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    update_id = int(uuid4().hex[:12], 16)
    ids = []
    try:
        data = update_payload(update_id)
        first = await admit_receipt(factory, bot_identity=bot_id, data=data,
                                    graph_kind="telegram_session", thread_id="synthetic-thread-a")
        ids.append(first.id)
        duplicate = await admit_receipt(factory, bot_identity=bot_id, data=data,
                                        graph_kind="telegram_session", thread_id="synthetic-thread-a")
        assert duplicate.id == first.id
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(TelegramInboundReceipt).where(
                TelegramInboundReceipt.bot_identity == bot_id,
                TelegramInboundReceipt.update_id == update_id)) == 1
        with pytest.raises(ValueError, match="ReceiptIdentityConflict"):
            await admit_receipt(factory, bot_identity=bot_id,
                                data=update_payload(update_id, text="different payload"),
                                graph_kind="telegram_session", thread_id="synthetic-thread-a")

        # A new engine/process sees the committed replay envelope, not RAM state.
        engine2 = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
        factory2 = async_sessionmaker(engine2, expire_on_commit=False)
        try:
            async with factory2() as db:
                restored = await db.get(TelegramInboundReceipt, first.id)
                assert restored.envelope["message"]["text"] == "synthetic hello"
            claimed = await asyncio.gather(
                claim_receipt(factory, first.id, "worker-a"),
                claim_receipt(factory2, first.id, "worker-b"))
            assert sum(token is not None for token in claimed) == 1
        finally:
            await engine2.dispose()
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_webhook_duplicate_ack_requires_committed_custody(pg_harness):
    """Authenticated HTTP duplicate before and after terminal state is E4, not Telegram E5."""
    from app.telegram import webhook

    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    data = update_payload(int(uuid4().hex[:12], 16))
    original = await admit_receipt(factory, bot_identity=bot_id, data=data,
                                   graph_kind="telegram_session", thread_id="telegram_user_70001")
    kicked = []

    class Recovery:
        bot_identity = bot_id
        is_running = True
        admission_lock = asyncio.Lock()

        def kick(self, receipt_id):
            kicked.append(receipt_id)

    async def invoke_duplicate():
        async def receive():
            return {"type": "http.request", "body": json.dumps(data).encode(),
                    "more_body": False}

        request = Request({"type": "http", "method": "POST", "path": "/webhook",
                           "app": SimpleNamespace(state=SimpleNamespace(
                               runtime=SimpleNamespace(ready=True),
                               telegram_durable=Recovery())),
                           "headers": [(b"x-telegram-bot-api-secret-token",
                                        webhook.webhook_secret_token().encode())]}, receive)
        return await webhook.telegram_webhook(request)

    try:
        assert (await invoke_duplicate())["ok"]
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(TelegramInboundReceipt).where(
                TelegramInboundReceipt.bot_identity == bot_id,
                TelegramInboundReceipt.update_id == data["update_id"])) == 1
            row = await db.get(TelegramInboundReceipt, original.id)
            assert row.state == "RECEIVED"
            row.state = "PROCESSED"
            row.processed_at = datetime.now(timezone.utc)
            await db.commit()
        assert (await invoke_duplicate())["ok"]
        assert kicked == [original.id, original.id]
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(TelegramInboundReceipt).where(
                TelegramInboundReceipt.bot_identity == bot_id,
                TelegramInboundReceipt.update_id == data["update_id"])) == 1
    finally:
        await cleanup(factory, [original.id])


@pytest.mark.asyncio
async def test_attended_webhook_scope_defers_other_chat_without_receipt(pg_harness, monkeypatch):
    from app.telegram import webhook

    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    allowed_update = update_payload(int(uuid4().hex[:12], 16), actor=70001)
    other_update = update_payload(int(uuid4().hex[:12], 16), actor=70002)
    kicked = []

    class Recovery:
        bot_identity = bot_id
        is_running = True
        admission_lock = asyncio.Lock()

        def kick(self, receipt_id):
            kicked.append(receipt_id)

    monkeypatch.setattr(webhook.settings, "telegram_allowed_ingress_ids", "70001")
    monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)

    async def invoke(data):
        async def receive():
            return {"type": "http.request", "body": json.dumps(data).encode(),
                    "more_body": False}

        request = Request({"type": "http", "method": "POST", "path": "/webhook",
                           "app": SimpleNamespace(state=SimpleNamespace(
                               runtime=SimpleNamespace(ready=True),
                               telegram_durable=Recovery())),
                           "headers": [(b"x-telegram-bot-api-secret-token",
                                        webhook.webhook_secret_token().encode())]}, receive)
        return await webhook.telegram_webhook(request)

    receipt_id = None
    try:
        with pytest.raises(Exception) as rejected:
            await invoke(other_update)
        assert getattr(rejected.value, "status_code", None) == 503
        assert rejected.value.headers["Retry-After"] == "60"
        assert kicked == []
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(
                TelegramInboundReceipt).where(
                    TelegramInboundReceipt.bot_identity == bot_id)) == 0

        assert (await invoke(allowed_update))["ok"] is True
        assert len(kicked) == 1
        receipt_id = kicked[0]
    finally:
        if receipt_id is not None:
            await cleanup(factory, [receipt_id])


@pytest.mark.asyncio
async def test_pg_thread_singleflight_distinct_thread_and_expired_fence(pg_harness):
    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    base = int(uuid4().hex[:10], 16)
    thread_one = f"thread-one-{uuid4().hex}"
    thread_two = f"thread-two-{uuid4().hex}"
    ids = []
    try:
        receipts = []
        for index, thread in enumerate((thread_one, thread_one, thread_two)):
            receipt = await admit_receipt(factory, bot_identity=bot_id,
                                          data=update_payload(base + index),
                                          graph_kind="telegram_session", thread_id=thread)
            receipts.append(receipt)
            ids.append(receipt.id)
        # Admission is ordered by (received_at, id), not by update_id or the
        # order in which this test's Python loop happens to return objects.
        first_receipt, second_receipt = sorted(
            receipts[:2], key=lambda row: (row.received_at, row.id))
        first = await claim_receipt(factory, first_receipt.id, "owner-a")
        assert first is not None
        assert await claim_receipt(factory, second_receipt.id, "owner-b") is None
        assert await claim_receipt(factory, receipts[2].id, "owner-c") is not None

        # Simulate dead owner on first thread, then reclaim. Old token cannot
        # finalize a stale receipt even if it later resumes.
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, first_receipt.id)
            lease = await db.get(TelegramThreadLease,
                                 {"graph_kind": "telegram_session", "thread_id": thread_one})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        second = await claim_receipt(factory, first_receipt.id, "owner-recovery")
        assert second is not None and second != first
        await enqueue_reply(factory, receipt_id=first_receipt.id, reply_ordinal=0,
                            kind="interactive_reply", chat_id="70001", locale="en", text="hello",
                            fence_token=second, outcome_kind="no_business",
                            outcome_ref="synthetic_hello", required_reply_count=1)
        assert not await finalize_receipt(factory, first_receipt.id, first)
        assert await finalize_receipt(factory, first_receipt.id, second)
        assert await claim_receipt(factory, second_receipt.id, "owner-b") is not None
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_partial_chunk_timeout_and_restart_retry_only_unknown_chunk(pg_harness):
    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    ids = []
    try:
        receipt = await admit_receipt(factory, bot_identity=bot_id,
                                      data=update_payload(int(uuid4().hex[:12], 16)),
                                      graph_kind="telegram_session", thread_id="chunk-thread")
        ids.append(receipt.id)
        token = await claim_receipt(factory, receipt.id, "processor")
        delivery_id = await enqueue_reply(factory, receipt_id=receipt.id, reply_ordinal=0,
                                          kind="interactive_reply", chat_id="70001", locale="en",
                                          text="x" * 4097, fence_token=token,
                                          outcome_kind="no_business", outcome_ref="synthetic_chunk",
                                          required_reply_count=1)
        assert await finalize_receipt(factory, receipt.id, token)

        class FirstBot:
            def __init__(self):
                self.calls = []
            async def send_message(self, **kwargs):
                self.calls.append(kwargs["text"])
                if len(self.calls) == 2:
                    raise TimeoutError("synthetic unknown result")
                return type("Message", (), {"message_id": 101, "chat_id": 70001})()
        first_bot = FirstBot()
        await dispatch_delivery(factory, delivery_id, "sender-a", first_bot)
        async with factory() as db:
            chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id == delivery_id).order_by(
                    OutboundDeliveryChunk.chunk_index))).all()
            assert [chunk.state for chunk in chunks] == ["SENT", "UNCERTAIN"]
            assert chunks[0].telegram_message_id == 101
            assert (await db.get(OutboundDelivery, delivery_id)).state != "SENT"
            chunks[1].next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()

        class RetryBot:
            def __init__(self):
                self.calls = []
            async def send_message(self, **kwargs):
                self.calls.append(kwargs["text"])
                return type("Message", (), {"message_id": 102, "chat_id": 70001})()
        retry_bot = RetryBot()
        await dispatch_delivery(factory, delivery_id, "sender-restarted", retry_bot)
        assert [len(item) for item in retry_bot.calls] == [1]
        async with factory() as db:
            assert (await db.get(OutboundDelivery, delivery_id)).state == "SENT"
            attempts = (await db.scalars(select(OutboundDeliveryAttempt).where(
                OutboundDeliveryAttempt.delivery_id == delivery_id))).all()
            assert len(attempts) == 3
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_direct_todo_commit_before_receipt_finalization_replays_once(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    ids = []
    try:
        receipt = await admit_receipt(factory, bot_identity=bot_id,
                                      data=update_payload(int(uuid4().hex[:12], 16)),
                                      graph_kind="telegram_session", thread_id="todo-thread")
        ids.append(receipt.id)
        first = await claim_receipt(factory, receipt.id, "owner-original")
        from app.agent.tools import make_todo_tools
        create_tool = next(tool for tool in make_todo_tools(factory) if tool.name == "create_todo")
        args = {"user_id": user.id, "title": "synthetic receipt replay task"}
        with receipt_scope(receipt.id, first), tool_call_scope("checkpointed-tool-call-1"):
            created = await create_tool.ainvoke(args)
        # No receipt finalization: model/process crashes after the tool's
        # independently committed transaction.
        with pytest.raises(DurableSafetyError, match="MissingBusinessOutcomeOrReplyIntent"):
            await finalize_receipt(factory, receipt.id, first)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            lease = await db.get(TelegramThreadLease,
                                 {"graph_kind": "telegram_session", "thread_id": "todo-thread"})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        second = await claim_receipt(factory, receipt.id, "owner-restarted")
        with receipt_scope(receipt.id, second), tool_call_scope("checkpointed-tool-call-1"):
            replayed = await create_tool.ainvoke(args)
        assert replayed == created
        await enqueue_reply(factory, receipt_id=receipt.id, reply_ordinal=0,
                            kind="interactive_reply", chat_id="70001", locale="en",
                            text=replayed, fence_token=second,
                            outcome_kind="ordinary_checkpoint",
                            outcome_ref="synthetic-checkpoint-after-replay",
                            required_reply_count=1)
        assert await finalize_receipt(factory, receipt.id, second)
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(Todo).where(
                Todo.user_id == user.id, Todo.title == "synthetic receipt replay task")) == 1
            assert (await db.get(TelegramInboundReceipt, receipt.id)).state == "PROCESSED"
            proof = await db.get(ActionExecution, telegram_todo_action_id(
                receipt.id, "create_todo", "checkpointed-tool-call-1"))
            assert proof.status == "executed"
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_todo_delete_replay_uses_committed_business_proof(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    ids = []
    try:
        async with factory() as db:
            task = Todo(user_id=user.id, title="synthetic delete replay task")
            db.add(task)
            await db.commit()
            task_id = task.id
        receipt = await admit_receipt(factory, bot_identity=bot_id,
                                      data=update_payload(int(uuid4().hex[:12], 16)),
                                      graph_kind="telegram_session", thread_id="delete-thread")
        ids.append(receipt.id)
        first = await claim_receipt(factory, receipt.id, "owner-original")
        from app.agent.tools import make_todo_tools
        delete_tool = next(tool for tool in make_todo_tools(factory) if tool.name == "delete_todo")
        args = {"user_id": user.id, "todo_id": task_id}
        with receipt_scope(receipt.id, first), tool_call_scope("checkpointed-delete-call"):
            original = await delete_tool.ainvoke(args)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            lease = await db.get(TelegramThreadLease,
                                 {"graph_kind": "telegram_session", "thread_id": "delete-thread"})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        second = await claim_receipt(factory, receipt.id, "owner-restarted")
        with receipt_scope(receipt.id, second), tool_call_scope("checkpointed-delete-call"):
            replayed = await delete_tool.ainvoke(args)
        assert replayed == original
        async with factory() as db:
            assert await db.get(Todo, task_id) is None
            proof = await db.get(ActionExecution, telegram_todo_action_id(
                receipt.id, "delete_todo", "checkpointed-delete-call"))
            assert proof.status == "executed"
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_failed_receipt_bounded_retry_then_dead(pg_harness):
    factory, _, _ = pg_harness
    receipt = await admit_receipt(factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
                                  data=update_payload(int(uuid4().hex[:12], 16)),
                                  graph_kind="telegram_session", thread_id="poison-thread")
    try:
        first = await claim_receipt(factory, receipt.id, "owner-one")
        assert await fail_receipt(factory, receipt.id, first, "SyntheticFailure",
                                  retry_after_seconds=0, max_attempts=2)
        second = await claim_receipt(factory, receipt.id, "owner-two")
        assert second and second != first
        assert await fail_receipt(factory, receipt.id, second, "SyntheticFailure",
                                  retry_after_seconds=0, max_attempts=2)
        assert await claim_receipt(factory, receipt.id, "owner-three") is None
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            assert row.state == "DEAD"
            assert row.attempts == 2
            assert row.failure_category == "SyntheticFailure"
    finally:
        await cleanup(factory, [receipt.id])


@pytest.mark.asyncio
async def test_pg_session_lock_serializes_same_thread_invoke_and_resume_only(isolated_pg18_target):
    """Two independent DB sessions cannot overlap one graph thread's work.

    The labels model ordinary invoke and HITL resume at the shared admission
    boundary; no model, external write or Telegram send runs in this test.
    """
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    resume_entered = asyncio.Event()
    distinct_entered = asyncio.Event()

    async def ordinary_invoke():
        async with thread_invocation_guard(isolated_pg18_target, "telegram_session", "same-thread"):
            first_entered.set()
            await release_first.wait()

    async def hitl_resume():
        async with thread_invocation_guard(isolated_pg18_target, "telegram_session", "same-thread"):
            resume_entered.set()

    async def other_thread():
        async with thread_invocation_guard(isolated_pg18_target, "telegram_session", "other-thread"):
            distinct_entered.set()

    original = asyncio.create_task(ordinary_invoke())
    await asyncio.wait_for(first_entered.wait(), timeout=5)
    resumed = asyncio.create_task(hitl_resume())
    unrelated = asyncio.create_task(other_thread())
    try:
        await asyncio.wait_for(distinct_entered.wait(), timeout=5)
        await asyncio.sleep(0.1)
        assert not resume_entered.is_set()
    finally:
        release_first.set()
        await asyncio.gather(original, resumed, unrelated)
    assert resume_entered.is_set()


@pytest.mark.asyncio
async def test_pg_unsupported_update_is_terminal_without_private_payload(pg_harness):
    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    update_id = int(uuid4().hex[:12], 16)
    ids = []
    try:
        data = {"update_id": update_id, "message": {
            "photo": [{"file_id": "synthetic-private-photo-id"}],
            "contact": {"phone_number": "synthetic-private-phone"}}}
        row = await admit_terminal_receipt(factory, bot_identity=bot_id, data=data,
                                           category="UnsupportedMessage")
        ids.append(row.id)
        duplicate = await admit_terminal_receipt(factory, bot_identity=bot_id, data=data,
                                                 category="UnsupportedMessage")
        assert duplicate.id == row.id
        async with factory() as db:
            stored = await db.get(TelegramInboundReceipt, row.id)
            assert stored.state == "DEAD"
            assert stored.outcome_kind == "terminal_unsupported"
            assert "synthetic-private" not in str(stored.envelope)
        assert await claim_receipt(factory, row.id, "worker") is None
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_restart_worker_processes_custody_and_persists_remote_identity(
        pg_harness, isolated_pg18_target, monkeypatch):
    from app.telegram.durable_runtime import DurableTelegramRuntime
    from app.telegram.processing_context import current_run
    from app.telegram import webhook

    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    ids = []
    try:
        receipt = await admit_receipt(factory, bot_identity=bot_id,
                                      data=update_payload(int(uuid4().hex[:12], 16)),
                                      graph_kind="telegram_session", thread_id="restart-worker-thread")
        ids.append(receipt.id)

        async def process(_envelope):
            run = current_run()
            assert run is not None and run.receipt_id == receipt.id
            run.record_outcome("no_business", "synthetic_restart")
            await enqueue_reply(factory, receipt_id=run.receipt_id,
                                reply_ordinal=run.next_reply(), kind="interactive_reply",
                                chat_id="70001", locale="en", text="durable synthetic reply",
                                fence_token=run.fence_token, outcome_kind=run.outcome_kind,
                                outcome_ref=run.outcome_ref,
                                required_reply_count=run.required_reply_count)

        class FakeBot:
            async def send_message(self, **_kwargs):
                return type("Message", (), {"message_id": 911, "chat_id": 70001})()

        monkeypatch.setattr(webhook, "process_admitted_update", process)
        restarted = DurableTelegramRuntime(factory, FakeBot(), database_url=isolated_pg18_target)
        await restarted._process(receipt.id)
        async with factory() as db:
            stored = await db.get(TelegramInboundReceipt, receipt.id)
            assert stored.state == "PROCESSED" and stored.outcome_kind == "no_business"
            delivery = await db.scalar(select(OutboundDelivery).where(
                OutboundDelivery.receipt_id == receipt.id))
            chunk = await db.get(OutboundDeliveryChunk, {"delivery_id": delivery.id,
                                                         "chunk_index": 0})
            assert delivery.state == "SENT"
            assert (chunk.telegram_chat_id, chunk.telegram_message_id) == ("70001", 911)
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_crash_after_reply_intent_recovers_without_reinvoking_business(
        pg_harness, isolated_pg18_target, monkeypatch):
    from app.telegram.durable_runtime import DurableTelegramRuntime
    from app.telegram import webhook

    factory, _, _ = pg_harness
    bot_id = 800000 + int(uuid4().hex[:6], 16)
    ids = []
    try:
        receipt = await admit_receipt(factory, bot_identity=bot_id,
                                      data=update_payload(int(uuid4().hex[:12], 16)),
                                      graph_kind="telegram_session", thread_id="intent-crash-thread")
        ids.append(receipt.id)
        original = await claim_receipt(factory, receipt.id, "original-worker")
        delivery_id = await enqueue_reply(factory, receipt_id=receipt.id,
                                          reply_ordinal=0, kind="interactive_reply",
                                          chat_id="70001", locale="en",
                                          text="original durable outcome", fence_token=original,
                                          outcome_kind="no_business",
                                          outcome_ref="synthetic_original",
                                          required_reply_count=1)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            lease = await db.get(TelegramThreadLease,
                                 {"graph_kind": "telegram_session", "thread_id": "intent-crash-thread"})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()

        async def forbidden(_envelope):
            raise AssertionError("stored reply intent must prevent a second business invoke")

        class FakeBot:
            async def send_message(self, **_kwargs):
                return type("Message", (), {"message_id": 912, "chat_id": 70001})()

        monkeypatch.setattr(webhook, "process_admitted_update", forbidden)
        restarted = DurableTelegramRuntime(factory, FakeBot(), database_url=isolated_pg18_target)
        await restarted._process(receipt.id)
        async with factory() as db:
            assert (await db.get(TelegramInboundReceipt, receipt.id)).state == "PROCESSED"
            stored = await db.get(TelegramInboundReceipt, receipt.id)
            assert (stored.outcome_kind, stored.outcome_ref, stored.required_reply_count) == (
                "no_business", "synthetic_original", 1)
            assert (await db.get(OutboundDelivery, delivery_id)).state == "SENT"
            chunk = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})
            assert chunk.telegram_message_id == 912
    finally:
        await cleanup(factory, ids)


@pytest.mark.asyncio
async def test_pg_orphan_reply_without_outcome_cannot_shortcut_recovery(
        pg_harness, isolated_pg18_target, monkeypatch):
    """An intent alone is never proof that its business path completed."""
    from app.telegram.durable_runtime import DurableTelegramRuntime
    from app.telegram.processing_context import current_run
    from app.telegram import webhook

    factory, _, _ = pg_harness
    receipt = await admit_receipt(factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
                                  data=update_payload(int(uuid4().hex[:12], 16)),
                                  graph_kind="telegram_session", thread_id="orphan-intent-thread")
    try:
        first = await claim_receipt(factory, receipt.id, "crashed-owner")
        # Simulate the old unsafe state: a committed reply, but no business
        # proof. The new sender never creates this state in production.
        await enqueue_reply(factory, receipt_id=receipt.id, reply_ordinal=0,
                            kind="interactive_reply", chat_id="70001", locale="en",
                            text="synthetic result")
        with pytest.raises(DurableSafetyError, match="MissingBusinessOutcomeOrReplyIntent"):
            await finalize_receipt(factory, receipt.id, first)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            lease = await db.get(TelegramThreadLease,
                                 {"graph_kind": "telegram_session", "thread_id": "orphan-intent-thread"})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()

        calls = []
        async def replay(_envelope):
            calls.append("business_rechecked")
            run = current_run()
            run.record_outcome("no_business", "synthetic_replay_checked")
            await enqueue_reply(factory, receipt_id=run.receipt_id,
                                reply_ordinal=run.next_reply(), kind="interactive_reply",
                                chat_id="70001", locale="en", text="synthetic result",
                                fence_token=run.fence_token, outcome_kind=run.outcome_kind,
                                outcome_ref=run.outcome_ref,
                                required_reply_count=run.required_reply_count)

        class FakeBot:
            async def send_message(self, **_kwargs):
                return type("Message", (), {"message_id": 913, "chat_id": 70001})()

        monkeypatch.setattr(webhook, "process_admitted_update", replay)
        await DurableTelegramRuntime(factory, FakeBot(),
                                     database_url=isolated_pg18_target)._process(receipt.id)
        assert calls == ["business_rechecked"]
        async with factory() as db:
            stored = await db.get(TelegramInboundReceipt, receipt.id)
            assert stored.state == "PROCESSED"
            assert stored.outcome_ref == "synthetic_replay_checked"
    finally:
        await cleanup(factory, [receipt.id])


@pytest.mark.asyncio
async def test_pg_crash_before_reply_enqueue_replays_business_once(
        pg_harness, isolated_pg18_target, monkeypatch):
    """E4 deterministic crash just before reply intent, after Todo commit."""
    from app.telegram.durable_runtime import DurableTelegramRuntime
    from app.telegram.processing_context import current_run
    from app.telegram import webhook
    from app.agent.tools import make_todo_tools

    factory, new_user, _ = pg_harness
    user = await new_user()
    receipt = await admit_receipt(factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
                                  data=update_payload(int(uuid4().hex[:12], 16)),
                                  graph_kind="telegram_session", thread_id="pre-enqueue-crash-thread")
    title = f"synthetic-pre-enqueue-{uuid4().hex[:8]}"
    create_tool = next(item for item in make_todo_tools(factory) if item.name == "create_todo")
    handler_calls = []

    async def process(_envelope):
        run = current_run()
        handler_calls.append(run.receipt_id)
        with tool_call_scope("stable-checkpointed-call"):
            answer = await create_tool.ainvoke({"user_id": user.id, "title": title})
        if len(handler_calls) == 1:
            raise RuntimeError("synthetic crash before reply enqueue")
        run.record_outcome("ordinary_checkpoint", "synthetic-saver-after-replay")
        await enqueue_reply(factory, receipt_id=run.receipt_id,
                            reply_ordinal=run.next_reply(), kind="interactive_reply",
                            chat_id="70001", locale="en", text=answer,
                            fence_token=run.fence_token, outcome_kind=run.outcome_kind,
                            outcome_ref=run.outcome_ref,
                            required_reply_count=run.required_reply_count)

    class FakeBot:
        async def send_message(self, **_kwargs):
            return type("Message", (), {"message_id": 914, "chat_id": 70001})()

    monkeypatch.setattr(webhook, "process_admitted_update", process)
    runtime = DurableTelegramRuntime(factory, FakeBot(), database_url=isolated_pg18_target)
    try:
        await runtime._process(receipt.id)
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            assert row.state == "FAILED" and row.outcome_kind is None
            assert await db.scalar(select(func.count()).select_from(OutboundDelivery).where(
                OutboundDelivery.receipt_id == receipt.id)) == 0
            row.next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        await runtime._process(receipt.id)
        assert handler_calls == [receipt.id, receipt.id]
        async with factory() as db:
            row = await db.get(TelegramInboundReceipt, receipt.id)
            assert row.state == "PROCESSED" and row.outcome_ref == "synthetic-saver-after-replay"
            assert await db.scalar(select(func.count()).select_from(Todo).where(
                Todo.user_id == user.id, Todo.title == title)) == 1
            assert await db.scalar(select(func.count()).select_from(ActionExecution).where(
                ActionExecution.action_id.startswith(f"tgtodo:{receipt.id}:"))) == 1
    finally:
        await cleanup(factory, [receipt.id])


@pytest.mark.asyncio
async def test_pg_callback_resume_cursor_prevents_duplicate_command_after_crash(
        pg_harness, monkeypatch):
    """E4 cursor persists before resume; newer Saver state is reused on retry."""
    from app.telegram import webhook

    factory, _, _ = pg_harness
    receipt = await admit_receipt(factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
        data={"update_id": int(uuid4().hex[:12], 16), "callback_query": {
            "id": "synthetic-callback", "from": {"id": 70001, "is_bot": False,
            "first_name": "Synthetic"}, "data": "act:synthetic-action:reject",
            "message": {"message_id": 1, "date": 0,
                        "chat": {"id": 70001, "type": "private"}}}},
        graph_kind="telegram_session", thread_id="callback-cursor-thread")
    try:
        token = await claim_receipt(factory, receipt.id, "callback-owner")
        assert await record_resume_origin(factory, receipt.id, token, "before-checkpoint") == "before-checkpoint"
        async with factory() as db:
            assert (await db.get(TelegramInboundReceipt, receipt.id)).resume_from_checkpoint_id == "before-checkpoint"

        class Graph:
            async def ainvoke(self, *_args, **_kwargs):
                raise AssertionError("completed callback must not receive a second Command(resume)")

        delivered = []
        async def capture(*args, **kwargs):
            delivered.append((args, kwargs))

        monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
        monkeypatch.setattr(webhook, "_send_workflow_result", capture)
        newer = SimpleNamespace(config={"configurable": {"checkpoint_id": "after-checkpoint"}},
                                tasks=(), next=(), values={"response": "rejected", "user_id": "synthetic"})
        with receipt_scope(receipt.id, token):
            assert await webhook._replay_completed_resume(Graph(), "workflow_user_70001",
                                                           newer, 70001)
        assert len(delivered) == 1
        assert delivered[0][1]["prior_checkpoint_id"] == "before-checkpoint"
    finally:
        await cleanup(factory, [receipt.id])
