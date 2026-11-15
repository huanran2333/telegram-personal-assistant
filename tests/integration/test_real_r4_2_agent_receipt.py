"""Real model + isolated PG18 receipt replay; fake Telegram transport (not E5)."""
import asyncio
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select

from app.agent.graph import get_graph, run_agent
from app.agent.planner import route_request
from app.agent.workflow import start_workflow
from app.agent.workflow_runtime import get_productivity_graph
from app.config import get_settings
from app.db import crud
from app.db.models import (OutboundDelivery, OutboundDeliveryAttempt,
                           OutboundDeliveryChunk, TelegramBusinessOperation,
                           TelegramInboundReceipt, Todo, TranscriptProjection, User,
                           ScheduledOccurrence)
from app.db.session import AsyncSessionLocal, async_engine
from app.main import app, lifespan
from app.telegram.durable import admit_receipt, claim_receipt, finalize_receipt
from app.telegram.processing_context import receipt_scope
from app.telegram.sender import send_message

pytestmark = pytest.mark.llm_integration


async def cleanup_test_user_occurrences(db, user_id: str) -> None:
    """Remove only this synthetic user's scheduler rows before FK parent cleanup."""
    occurrence_ids = (await db.scalars(select(ScheduledOccurrence.id).where(
        ScheduledOccurrence.user_id == user_id))).all()
    if not occurrence_ids:
        return
    delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
        OutboundDelivery.occurrence_id.in_(occurrence_ids)))).all()
    if delivery_ids:
        await db.execute(delete(OutboundDeliveryAttempt).where(
            OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
        await db.execute(delete(OutboundDeliveryChunk).where(
            OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
        await db.execute(delete(OutboundDelivery).where(
            OutboundDelivery.id.in_(delivery_ids)))
    await db.execute(delete(ScheduledOccurrence).where(
        ScheduledOccurrence.id.in_(occurrence_ids)))


@pytest.fixture(scope="session")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.mark.asyncio
@pytest.mark.provider_lifespan
async def test_real_provider_normal_chat_reaches_durable_reply(
        isolated_pg18_target, monkeypatch):
    """E4 real route/main models + PG18; Telegram transport remains fake."""
    from app import main as app_main
    assert isolated_pg18_target.database == "assistant"
    monkeypatch.setattr(app_main.settings, "telegram_webhook_url", "https://example.invalid/r4-chat")
    monkeypatch.setattr(app_main.settings, "telegram_bot_token", "synthetic-test-token")

    class FakeBot:
        async def get_me(self, **_kwargs):
            return type("BotIdentity", (), {"id": 123456789})()

        async def set_webhook(self, **_kwargs):
            return True

        async def send_message(self, **_kwargs):
            return type("Message", (), {"message_id": 779, "chat_id": 70001})()

    monkeypatch.setattr("app.telegram.sender.get_bot", lambda: FakeBot())
    telegram_id = str(int(uuid4().hex[:8], 16))
    receipt_id = user_id = None
    try:
        async with lifespan(app):
            prompt = "你好，请只用一句话介绍你自己。"
            route = await route_request(prompt, get_settings().default_timezone)
            assert route.intent == "chat"
            receipt = await admit_receipt(AsyncSessionLocal, bot_identity=123456789,
                data={"update_id": int(uuid4().hex[:12], 16), "message": {
                    "message_id": 1, "date": 0,
                    "from": {"id": int(telegram_id), "is_bot": False,
                             "first_name": "Synthetic"},
                    "chat": {"id": 70001, "type": "private"}, "text": prompt}},
                graph_kind="telegram_session", thread_id=f"telegram_user_{telegram_id}")
            receipt_id = receipt.id
            app.state.telegram_durable.kick(receipt_id)
            for _ in range(80):
                await asyncio.sleep(0.5)
                async with AsyncSessionLocal() as db:
                    current = await db.get(TelegramInboundReceipt, receipt_id)
                    delivery = await db.scalar(select(OutboundDelivery).where(
                        OutboundDelivery.receipt_id == receipt_id))
                    if current.state == "PROCESSED" and delivery and delivery.state == "SENT":
                        break
            else:
                raise AssertionError("Real chat did not reach durable remote acceptance")
            async with AsyncSessionLocal() as db:
                user = await db.scalar(select(User).where(User.telegram_id == telegram_id))
                assert user is not None
                user_id = user.id
                assert await db.scalar(select(func.count()).select_from(Todo).where(
                    Todo.user_id == user_id)) == 0
                chunk = await db.scalar(select(OutboundDeliveryChunk).where(
                    OutboundDeliveryChunk.delivery_id == delivery.id))
                assert chunk.telegram_message_id == 779
            graph = await get_graph()
            await graph.checkpointer.adelete_thread(f"user_{telegram_id}")
    finally:
        if receipt_id or user_id:
            async with AsyncSessionLocal() as db:
                if receipt_id:
                    delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
                        OutboundDelivery.receipt_id == receipt_id))).all()
                    if delivery_ids:
                        await db.execute(delete(OutboundDeliveryAttempt).where(
                            OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDeliveryChunk).where(
                            OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDelivery).where(
                            OutboundDelivery.id.in_(delivery_ids)))
                    await db.execute(delete(TelegramInboundReceipt).where(
                        TelegramInboundReceipt.id == receipt_id))
                if user_id:
                    await cleanup_test_user_occurrences(db, user_id)
                    await db.execute(delete(TranscriptProjection).where(
                        TranscriptProjection.user_id == user_id))
                    await db.execute(delete(User).where(User.id == user_id))
                await db.commit()
        await async_engine.dispose()


@pytest.mark.asyncio
@pytest.mark.provider_lifespan
async def test_real_provider_todo_receipt_replay_does_not_mutate_twice(
        isolated_pg18_target, monkeypatch):
    assert isolated_pg18_target.database == "assistant"
    settings = get_settings()
    assert settings.openai_api_key and settings.llm_provider == "openai"
    from app import main as app_main
    monkeypatch.setattr(app_main.settings, "telegram_webhook_url", "https://example.invalid/r4-test")
    monkeypatch.setattr(app_main.settings, "telegram_bot_token", "synthetic-test-token")

    class FakeBot:
        async def get_me(self, **_kwargs):
            return type("BotIdentity", (), {"id": 123456789})()

        async def set_webhook(self, **_kwargs):
            return True

        async def send_message(self, **_kwargs):
            return type("Message", (), {"message_id": 777, "chat_id": 70001})()

    monkeypatch.setattr("app.telegram.sender.get_bot", lambda: FakeBot())
    telegram_id = f"r4-itest-{uuid4().hex}"
    update_id = int(uuid4().hex[:12], 16)
    title = f"R4 synthetic replay {uuid4().hex[:8]}"
    user_id = receipt_id = None
    try:
        async with lifespan(app):
            assert app.state.runtime.ready
            async with AsyncSessionLocal() as db:
                user = await crud.get_or_create_user(db, telegram_id)
                await db.commit()
                user_id = user.id
            payload = {"update_id": update_id, "message": {
                "message_id": 1, "date": 0,
                "from": {"id": int(uuid4().hex[:8], 16), "is_bot": False,
                         "first_name": "Synthetic"},
                "chat": {"id": 70001, "type": "private"},
                "text": f"创建一个待办：{title}"}}
            receipt = await admit_receipt(
                AsyncSessionLocal, bot_identity=123456789, data=payload,
                graph_kind="telegram_session", thread_id=f"telegram_user_{telegram_id}")
            receipt_id = receipt.id
            token = await claim_receipt(AsyncSessionLocal, receipt_id, "synthetic-provider-worker")
            assert token is not None
            graph = await get_graph()
            from app.telegram import webhook
            with receipt_scope(receipt_id, token):
                answer = await run_agent(graph, user_id, telegram_id, "70001", payload["message"]["text"])
                await webhook._record_checkpoint_outcome(graph, f"user_{telegram_id}", ordinary=True)
                await send_message("70001", answer)
            # Crash window: graph + Todo + reply intent are committed but the
            # receipt is still PROCESSING. The same update must reuse Saver.
            with receipt_scope(receipt_id, token):
                replay = await run_agent(graph, user_id, telegram_id, "70001", payload["message"]["text"])
            assert replay == answer
            async with AsyncSessionLocal() as db:
                assert await db.scalar(select(func.count()).select_from(Todo).where(
                    Todo.user_id == user_id, Todo.title == title)) == 1
                proofs = (await db.scalars(select(TelegramBusinessOperation).where(
                    TelegramBusinessOperation.receipt_id == receipt_id))).all()
                assert len(proofs) == 1 and proofs[0].status == "done"
            assert await finalize_receipt(AsyncSessionLocal, receipt_id, token)
            await graph.checkpointer.adelete_thread(f"user_{telegram_id}")
    finally:
        if receipt_id or user_id:
            async with AsyncSessionLocal() as db:
                if receipt_id:
                    delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
                        OutboundDelivery.receipt_id == receipt_id))).all()
                    if delivery_ids:
                        await db.execute(delete(OutboundDeliveryAttempt).where(
                            OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDeliveryChunk).where(
                            OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDelivery).where(
                            OutboundDelivery.id.in_(delivery_ids)))
                    await db.execute(delete(TelegramBusinessOperation).where(
                        TelegramBusinessOperation.receipt_id == receipt_id))
                    await db.execute(delete(TelegramInboundReceipt).where(
                        TelegramInboundReceipt.id == receipt_id))
                if user_id:
                    await cleanup_test_user_occurrences(db, user_id)
                    await db.execute(delete(TranscriptProjection).where(
                        TranscriptProjection.user_id == user_id))
                    await db.execute(delete(Todo).where(Todo.user_id == user_id))
                    await db.execute(delete(User).where(User.id == user_id))
                await db.commit()
        await async_engine.dispose()


@pytest.mark.asyncio
@pytest.mark.provider_lifespan
async def test_real_provider_hitl_proposal_and_callback_reject_are_two_durable_receipts(
        isolated_pg18_target, monkeypatch):
    """E4 PG18 + real route model; transport is synthetic, no external write."""
    from app import main as app_main
    from app.telegram import webhook
    monkeypatch.setattr(app_main.settings, "telegram_webhook_url", "https://example.invalid/r4-hitl")
    monkeypatch.setattr(app_main.settings, "telegram_bot_token", "synthetic-test-token")

    class FakeBot:
        async def get_me(self, **_kwargs):
            return type("BotIdentity", (), {"id": 123456789})()

        async def set_webhook(self, **_kwargs):
            return True

        async def send_message(self, **_kwargs):
            return type("Message", (), {"message_id": 778, "chat_id": 70001})()

    monkeypatch.setattr("app.telegram.sender.get_bot", lambda: FakeBot())
    telegram_id = str(int(uuid4().hex[:8], 16))
    thread_id = f"workflow_user_{telegram_id}"
    title = f"R4 synthetic HITL {uuid4().hex[:8]}"
    user_id = None
    receipt_ids = []
    try:
        async with lifespan(app):
            assert app.state.runtime.ready
            async with AsyncSessionLocal() as db:
                user = await crud.get_or_create_user(db, telegram_id)
                todo = await crud.create_todo(db, user_id=user.id, title=title)
                await db.commit()
                user_id, todo_id = user.id, todo.id
            request_text = f"删除任务：{title}"
            route = await route_request(request_text, get_settings().default_timezone)
            assert route.intent == "todo" and route.todo_operation == "delete"
            original = await admit_receipt(AsyncSessionLocal, bot_identity=123456789,
                data={"update_id": int(uuid4().hex[:12], 16), "message": {
                    "message_id": 1, "date": 0,
                    "from": {"id": int(telegram_id), "is_bot": False,
                             "first_name": "Synthetic"},
                    "chat": {"id": 70001, "type": "private"}, "text": request_text}},
                graph_kind="telegram_session", thread_id=f"telegram_user_{telegram_id}")
            receipt_ids.append(original.id)
            token = await claim_receipt(AsyncSessionLocal, original.id, "hitl-model-worker")
            graph = await get_productivity_graph()
            with receipt_scope(original.id, token):
                result = await start_workflow(graph, user_id, request_text,
                    get_settings().default_timezone, thread_id, route=route,
                    telegram_id=telegram_id)
                await webhook._send_workflow_result(70001, thread_id, result)
            assert result["__interrupt__"][0].value["kind"] == "action_proposal"
            assert await finalize_receipt(AsyncSessionLocal, original.id, token)
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            action_id = state.values["proposal"]["action_id"]
            callback_data = f"act:{action_id}:reject"
            callback_receipt = await admit_receipt(AsyncSessionLocal, bot_identity=123456789,
                data={"update_id": int(uuid4().hex[:12], 16), "callback_query": {
                    "id": "synthetic-callback", "from": {"id": int(telegram_id),
                    "is_bot": False, "first_name": "Synthetic"},
                    "chat_instance": "synthetic-instance", "data": callback_data,
                    "message": {"message_id": 2, "date": 0,
                    "chat": {"id": 70001, "type": "private"}, "text": "Proposal"}}},
                graph_kind="telegram_session", thread_id=f"telegram_user_{telegram_id}")
            receipt_ids.append(callback_receipt.id)
            callback_token = await claim_receipt(AsyncSessionLocal, callback_receipt.id,
                                                 "hitl-callback-worker")

            class Callback:
                data = callback_data
                from_user = SimpleNamespace(id=int(telegram_id))
                message = SimpleNamespace(chat_id=70001)

                async def answer(self, *_args, **_kwargs):
                    return True

            with receipt_scope(callback_receipt.id, callback_token):
                await webhook._handle_action_callback(Callback())
            assert await finalize_receipt(AsyncSessionLocal, callback_receipt.id,
                                          callback_token)
            async with AsyncSessionLocal() as db:
                assert await db.get(Todo, todo_id) is not None  # Reject is no write.
                deliveries = (await db.scalars(select(OutboundDelivery).where(
                    OutboundDelivery.receipt_id.in_(receipt_ids)))).all()
                assert len(deliveries) == 2
                assert {delivery.kind for delivery in deliveries} == {
                    "hitl_proposal", "interactive_reply"}
            await graph.checkpointer.adelete_thread(thread_id)
    finally:
        if receipt_ids or user_id:
            async with AsyncSessionLocal() as db:
                if receipt_ids:
                    delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
                        OutboundDelivery.receipt_id.in_(receipt_ids)))).all()
                    if delivery_ids:
                        await db.execute(delete(OutboundDeliveryAttempt).where(
                            OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDeliveryChunk).where(
                            OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDelivery).where(
                            OutboundDelivery.id.in_(delivery_ids)))
                    await db.execute(delete(TelegramBusinessOperation).where(
                        TelegramBusinessOperation.receipt_id.in_(receipt_ids)))
                    await db.execute(delete(TelegramInboundReceipt).where(
                        TelegramInboundReceipt.id.in_(receipt_ids)))
                if user_id:
                    await cleanup_test_user_occurrences(db, user_id)
                    await db.execute(delete(TranscriptProjection).where(
                        TranscriptProjection.user_id == user_id))
                    await db.execute(delete(Todo).where(Todo.user_id == user_id))
                    await db.execute(delete(User).where(User.id == user_id))
                await db.commit()
        await async_engine.dispose()
