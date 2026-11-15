"""Opt-in E5: one harmless proactive message via the sealed shared outbox.

Never run in ordinary/integration suites without PPA_R44_E5_TELEGRAM=1.
No crash-after-send ambiguity is manufactured against the real Bot API.
"""
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from telegram import Bot

from app.config import get_settings
from app.db.models import (OutboundDelivery, OutboundDeliveryAttempt,
                           OutboundDeliveryChunk, ScheduledOccurrence, User)
from app.telegram.durable import dispatch_delivery
from app.telegram.operational_snapshot import messaging_snapshot
from app.scheduler.proactive import reconcile_occurrence, stable_key

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_one_real_proactive_delivery_and_gate1_observation(pg_harness):
    if os.environ.get("PPA_R44_E5_TELEGRAM") != "1":
        pytest.skip("Explicit real Telegram send opt-in required")
    factory, _, _ = pg_harness
    settings = get_settings()
    recipient = settings.google_calendar_owner_telegram_id
    if not recipient.isdecimal() or not settings.telegram_bot_token:
        pytest.skip("Authorized test Bot/recipient not configured")
    created_user = False
    async with factory() as db:
        user = await db.scalar(select(User).where(User.telegram_id == recipient))
        if user is None:
            # This is the isolated PG18 acceptance DB, not the product account
            # database. Remove only this test-created row in the finalizer.
            user = User(telegram_id=recipient, preferences={})
            db.add(user)
            await db.commit()
            created_user = True
        user_id = user.id
    async with Bot(settings.telegram_bot_token) as bot:
        identity = await bot.get_me()
        now = datetime.now(timezone.utc)
        test_key = stable_key("r4_4_gate3_e5", user_id, uuid4().hex)
        occurrence_id = delivery_id = None
        try:
            async with factory() as db:
                occurrence = ScheduledOccurrence(
                    occurrence_key=test_key, kind="daily_rundown", user_id=user_id,
                    bot_identity=identity.id, chat_id=recipient,
                    local_date=now.date().isoformat(), timezone_name=settings.default_timezone,
                    planned_at=now, state="READY", created_at=now)
                db.add(occurrence)
                await db.flush()
                occurrence_id = occurrence.id
                message = "R4.4 Gate 3 无敏感内容的持久化消息验收。"
                delivery = OutboundDelivery(
                    occurrence_id=occurrence_id, reply_ordinal=0,
                    kind="daily_rundown", chat_id=recipient, locale="zh-CN",
                    text=message, payload_sha256=stable_key(message),
                    state="PENDING", created_at=now)
                db.add(delivery)
                await db.flush()
                delivery_id = delivery.id
                db.add(OutboundDeliveryChunk(
                    delivery_id=delivery_id, chunk_index=0, text=message,
                    content_sha256=stable_key(message), state="PENDING"))
                await db.commit()
            # Intent is already committed before any remote Bot API call.
            await dispatch_delivery(factory, delivery_id, "r44-e5", bot)
            assert await reconcile_occurrence(factory, occurrence_id) == "DELIVERED"
            async with factory() as db:
                delivery = await db.get(OutboundDelivery, delivery_id)
                chunk = await db.get(OutboundDeliveryChunk,
                                     {"delivery_id": delivery_id, "chunk_index": 0})
                assert delivery.state == "SENT"
                assert chunk.telegram_chat_id == recipient
                assert isinstance(chunk.telegram_message_id, int) and chunk.telegram_message_id > 0
            snapshot = await messaging_snapshot(factory)
            assert snapshot["counts"]["occurrences"]["DELIVERED"] >= 1
            assert snapshot["counts"]["deliveries"]["SENT"] >= 1
        finally:
            if occurrence_id or created_user:
                async with factory() as db:
                    if delivery_id:
                        await db.execute(delete(OutboundDeliveryAttempt).where(
                            OutboundDeliveryAttempt.delivery_id == delivery_id))
                        await db.execute(delete(OutboundDeliveryChunk).where(
                            OutboundDeliveryChunk.delivery_id == delivery_id))
                        await db.execute(delete(OutboundDelivery).where(
                            OutboundDelivery.id == delivery_id))
                    if occurrence_id:
                        await db.execute(delete(ScheduledOccurrence).where(
                            ScheduledOccurrence.id == occurrence_id))
                    if created_user:
                        await db.execute(delete(User).where(User.id == user_id))
                    await db.commit()
