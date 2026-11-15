"""Isolated PG18 + mock Bot API; no real Telegram calls."""
import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.db.models import (OutboundDelivery, OutboundDeliveryChunk,
                           TelegramApprovalMarkupSync, TelegramInboundReceipt,
                           TelegramMiniAppLaunchTicket)
from app.telegram.approval_markup import sync_approval_markups
from app.telegram.durable import claim_next_chunk
from app.telegram.retention import _eligible_delivery
from app.telegram.workbench_launch import WORKBENCH_LEGACY_TEXT

pytestmark = pytest.mark.integration


class Graph:
    def __init__(self, statuses):
        self.statuses = statuses

    async def aget_state(self, config):
        action, status, next_nodes = self.statuses[config["configurable"]["thread_id"]]
        return SimpleNamespace(values={"proposal": {
            "action_id": action, "status": status}}, next=next_nodes)


class Bot:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def edit_message_reply_markup(self, **kwargs):
        self.calls.append((kwargs["chat_id"], kwargs["message_id"], kwargs["reply_markup"]))
        if self.fail:
            raise RuntimeError("synthetic transport failure")


@pytest.mark.asyncio
async def test_pg_agent_terminal_markup_only_edits_resolved_action(
        pg_harness, isolated_pg18_target, monkeypatch):
    from app.agent import official_runtime
    from langchain_core.messages import ToolMessage

    factory, _, _ = pg_harness
    bot_id = int(uuid4().hex[:10], 16)
    chat = str(int(uuid4().hex[:9], 16))
    actions = [uuid4().hex, uuid4().hex]
    receipt_ids, delivery_ids = [], []
    try:
        async with factory() as db:
            for index, action in enumerate(actions):
                receipt = TelegramInboundReceipt(
                    bot_identity=bot_id, update_id=int(uuid4().hex[:12], 16),
                    state="PROCESSED", graph_kind="telegram_session",
                    thread_id=f"telegram_user_{chat}")
                db.add(receipt)
                await db.flush()
                delivery = OutboundDelivery(
                    receipt_id=receipt.id, reply_ordinal=index,
                    kind="hitl_proposal", chat_id=chat, locale="en", state="SENT",
                    keyboard={"inline_keyboard": [[{
                        "callback_data": f"act:{action}:approve", "text": "Approve"}]]})
                db.add(delivery)
                await db.flush()
                db.add(OutboundDeliveryChunk(
                    delivery_id=delivery.id, chunk_index=0, state="SENT",
                    telegram_chat_id=chat, telegram_message_id=800 + index))
                receipt_ids.append(receipt.id)
                delivery_ids.append(delivery.id)
            await db.commit()

        class AgentGraph:
            async def aget_state(self, config):
                thread = config["configurable"]["thread_id"]
                if thread == f"user_{chat}":
                    return SimpleNamespace(values={}, next=())
                assert thread == f"official-v1:user_{chat}"
                return SimpleNamespace(values={
                    "prepared_actions": {
                        "call-a": {"proposal": {"action_id": actions[0]}},
                        "call-b": {"proposal": {"action_id": actions[1]}},
                    },
                    "messages": [ToolMessage(content="已处理", tool_call_id="call-a")],
                }, next=("tools",))

        monkeypatch.setattr(official_runtime, "get_graph", AsyncMock(return_value=AgentGraph()))
        legacy = Graph({f"workflow_user_{chat}": ("unrelated-action", "proposed",
                                                     ("approval_gate",))})
        bot = Bot()
        await sync_approval_markups(factory, bot, bot_id, legacy)
        assert bot.calls == [(int(chat), 800, None)]
        async with factory() as db:
            assert (await db.get(TelegramApprovalMarkupSync, delivery_ids[0])).state == "DONE"
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[1]) is None
    finally:
        async with factory() as db:
            await db.execute(delete(TelegramApprovalMarkupSync).where(
                TelegramApprovalMarkupSync.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id.in_(delivery_ids)))
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id.in_(receipt_ids)))
            await db.commit()


@pytest.mark.asyncio
async def test_pg_expired_workbench_button_uses_fenced_edit_and_skips_live_or_other_messages(
        pg_harness, isolated_pg18_target):
    factory, _, _ = pg_harness
    bot_id = int(uuid4().hex[:10], 16)
    chat = str(int(uuid4().hex[:9], 16))
    other_chat = str(int(uuid4().hex[:9], 16))
    receipt_ids, delivery_ids, ticket_hashes = [], [], []
    clock = datetime.now(timezone.utc)
    try:
        async with factory() as db:
            for index, (kind, age, expiry, owner, text) in enumerate((
                    ("interactive_reply", 12, -7, chat, WORKBENCH_LEGACY_TEXT),
                    ("workbench_launch", 6, -1, chat, "请在 5 分钟内打开工作台"),
                    ("workbench_launch", 1, 4, other_chat, "请在 5 分钟内打开工作台"),
                    ("interactive_reply", 12, -7, chat, "Unrelated web app"),
                    ("interactive_reply", 31 * 24 * 60, -20, chat, WORKBENCH_LEGACY_TEXT),
                    ("workbench_launch", 12, -7, chat, "请在 5 分钟内打开工作台"))):
                token = secrets.token_urlsafe(32)
                ticket_hash = hashlib.sha256(token.encode()).hexdigest()
                ticket_hashes.append(ticket_hash)
                receipt = TelegramInboundReceipt(
                    bot_identity=bot_id, update_id=int(uuid4().hex[:12], 16),
                    state="PROCESSED", graph_kind="telegram_session",
                    thread_id=f"telegram_user_{owner}")
                db.add(receipt)
                await db.flush()
                if index != 4:  # Historical ticket already pruned; R4 message remains.
                    db.add(TelegramMiniAppLaunchTicket(
                        token_hash=ticket_hash, bot_identity=bot_id,
                        telegram_user_id=other_chat if index == 5 else owner,
                        expires_at=clock + timedelta(minutes=expiry),
                        consumed_at=clock if index == 2 else None))
                delivery = OutboundDelivery(
                    receipt_id=receipt.id, reply_ordinal=0, kind=kind, chat_id=owner,
                    locale="zh", text=text, state="SENT",
                    created_at=clock - timedelta(minutes=age),
                    sent_at=clock - timedelta(minutes=age),
                    terminal_at=clock - timedelta(minutes=age),
                    keyboard={"inline_keyboard": [[{"text": "打开工作台", "web_app": {
                        "url": "https://example.test/workbench/#launchTicket=" + token}}]]})
                db.add(delivery)
                await db.flush()
                db.add(OutboundDeliveryChunk(
                    delivery_id=delivery.id, chunk_index=0, state="SENT",
                    telegram_chat_id=owner, telegram_message_id=900 + index))
                receipt_ids.append(receipt.id)
                delivery_ids.append(delivery.id)
            await db.commit()
        bot = Bot(fail=True)
        await asyncio.gather(sync_approval_markups(factory, bot, bot_id, Graph({})),
                             sync_approval_markups(factory, bot, bot_id, Graph({})))
        assert set(bot.calls) == {(int(chat), 900, None), (int(chat), 901, None),
                                  (int(chat), 904, None)}
        async with factory() as db:
            for index in (0, 1, 4):
                intent = await db.get(TelegramApprovalMarkupSync, delivery_ids[index])
                assert intent.state == "RETRY" and intent.attempts == 1
                intent.next_attempt_at = None
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[2]) is None
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[3]) is None
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[5]) is None
            assert not await _eligible_delivery(
                db, await db.get(OutboundDelivery, delivery_ids[4]),
                clock - timedelta(days=30))
            await db.commit()
        bot.fail = False
        await sync_approval_markups(factory, bot, bot_id, Graph({}))
        async with factory() as db:
            for i in (0, 1, 4):
                assert (await db.get(TelegramApprovalMarkupSync, delivery_ids[i])).state == "DONE"
            assert await _eligible_delivery(
                db, await db.get(OutboundDelivery, delivery_ids[4]),
                clock - timedelta(days=30))
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[2]) is None
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[3]) is None
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[5]) is None
    finally:
        async with factory() as db:
            await db.execute(delete(TelegramApprovalMarkupSync).where(
                TelegramApprovalMarkupSync.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id.in_(delivery_ids)))
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id.in_(receipt_ids)))
            await db.execute(delete(TelegramMiniAppLaunchTicket).where(
                TelegramMiniAppLaunchTicket.token_hash.in_(ticket_hashes)))
            await db.commit()


@pytest.mark.asyncio
async def test_pg_r4_does_not_start_sending_an_expired_launch(
        pg_harness, isolated_pg18_target):
    factory, _, _ = pg_harness
    bot_id = int(uuid4().hex[:10], 16)
    chat = str(int(uuid4().hex[:9], 16))
    token = secrets.token_urlsafe(32)
    ticket_hash = hashlib.sha256(token.encode()).hexdigest()
    clock = datetime.now(timezone.utc)
    async with factory() as db:
        receipt = TelegramInboundReceipt(
            bot_identity=bot_id, update_id=int(uuid4().hex[:12], 16),
            state="PROCESSED", graph_kind="telegram_session",
            thread_id=f"telegram_user_{chat}")
        db.add(receipt)
        db.add(TelegramMiniAppLaunchTicket(
            token_hash=ticket_hash, bot_identity=bot_id,
            telegram_user_id=chat, expires_at=clock - timedelta(seconds=1)))
        await db.flush()
        delivery = OutboundDelivery(
            receipt_id=receipt.id, reply_ordinal=0, kind="workbench_launch",
            chat_id=chat, locale="zh", text="入口已过期", state="PENDING",
            keyboard={"inline_keyboard": [[{"text": "打开工作台", "web_app": {
                "url": "https://example.test/workbench/#launchTicket=" + token}}]]})
        db.add(delivery)
        await db.flush()
        db.add(OutboundDeliveryChunk(
            delivery_id=delivery.id, chunk_index=0, state="PENDING", text="入口已过期"))
        await db.commit()
        receipt_id, delivery_id = receipt.id, delivery.id
    try:
        assert await claim_next_chunk(factory, delivery_id, "synthetic-owner") is None
        async with factory() as db:
            assert (await db.get(OutboundDelivery, delivery_id)).state == "EXHAUSTED"
            assert (await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 0})).state == "EXHAUSTED"
    finally:
        async with factory() as db:
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id == delivery_id))
            await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id == delivery_id))
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id == receipt_id))
            await db.execute(delete(TelegramMiniAppLaunchTicket).where(
                TelegramMiniAppLaunchTicket.token_hash == ticket_hash))
            await db.commit()


@pytest.mark.asyncio
async def test_pg_terminal_markup_exact_message_retry_and_pending_isolation(
        pg_harness, isolated_pg18_target):
    factory, _, _ = pg_harness
    bot_id = int(uuid4().hex[:10], 16)
    chats = [str(int(uuid4().hex[:9], 16)) for _ in range(2)]
    actions = [uuid4().hex for _ in range(2)]
    receipt_ids, delivery_ids = [], []
    try:
        async with factory() as db:
            for i, action in enumerate(actions):
                receipt = TelegramInboundReceipt(
                    bot_identity=bot_id, update_id=int(uuid4().hex[:12], 16),
                    state="PROCESSED", graph_kind="telegram_session",
                    thread_id=f"telegram_user_{chats[i]}")
                db.add(receipt)
                await db.flush()
                delivery = OutboundDelivery(
                    receipt_id=receipt.id, reply_ordinal=0,
                    kind="hitl_proposal", chat_id=chats[i], locale="en",
                    state="SENT", keyboard={"inline_keyboard": [[{
                        "callback_data": f"act:{action}:approve", "text": "Approve"}]]})
                db.add(delivery)
                await db.flush()
                db.add(OutboundDeliveryChunk(
                    delivery_id=delivery.id, chunk_index=0, state="SENT",
                    telegram_chat_id=chats[i], telegram_message_id=700 + i))
                receipt_ids.append(receipt.id)
                delivery_ids.append(delivery.id)
            await db.commit()
        graph = Graph({
            f"workflow_user_{chats[0]}": (actions[0], "rejected", ()),
            f"workflow_user_{chats[1]}": (actions[1], "proposed", ("approval_gate",)),
        })
        bot = Bot(fail=True)
        await asyncio.gather(
            sync_approval_markups(factory, bot, bot_id, graph),
            sync_approval_markups(factory, bot, bot_id, graph))
        assert bot.calls == [(int(chats[0]), 700, None)]
        async with factory() as db:
            first = await db.get(TelegramApprovalMarkupSync, delivery_ids[0])
            assert first.state == "RETRY" and first.attempts == 1
            # Another user's still-pending proposal is untouched.
            assert await db.get(TelegramApprovalMarkupSync, delivery_ids[1]) is None
            first.next_attempt_at = None
            await db.commit()
        bot.fail = False
        await sync_approval_markups(factory, bot, bot_id, graph)
        assert bot.calls.count((int(chats[0]), 700, None)) == 2
        async with factory() as db:
            assert (await db.get(TelegramApprovalMarkupSync, delivery_ids[0])).state == "DONE"
    finally:
        async with factory() as db:
            await db.execute(delete(TelegramApprovalMarkupSync).where(
                TelegramApprovalMarkupSync.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id.in_(delivery_ids)))
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id.in_(receipt_ids)))
            await db.commit()
