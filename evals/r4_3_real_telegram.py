"""Explicit E5 opt-in: one synthetic-content proactive Telegram send.

Requires the verified isolated PG18 URL/data directory and the user's existing
Bot/chat configuration. A stable key makes rerunning this script no-spam.
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.db.models import (OutboundDelivery, OutboundDeliveryChunk,
                           ScheduledOccurrence, User)
from app.db.session import AsyncSessionLocal
from app.scheduler.proactive import (claim_occurrence, ready_occurrence,
                                     reconcile_occurrence, stable_key, utc_now)
from app.telegram.durable import dispatch_delivery
from app.telegram.sender import close_bot, get_bot


async def run() -> None:
    expected = os.environ.get("PPA_ISOLATED_PG18_DATA_DIR")
    if not expected:
        raise RuntimeError("IsolatedPG18IdentityRequired")
    async with AsyncSessionLocal() as db:
        version = await db.scalar(text("SHOW server_version"))
        directory = await db.scalar(text("SHOW data_directory"))
    if not str(version).startswith("18.") or Path(directory).resolve() != Path(expected).resolve():
        raise RuntimeError("WrongPostgreSQLTarget")

    settings = get_settings()
    chat_id = settings.google_calendar_owner_telegram_id
    if not chat_id:
        raise RuntimeError("ConfiguredTestChatRequired")
    bot = get_bot()
    try:
        identity = (await bot.get_me()).id
        now = utc_now()
        key = stable_key("r4_3_e5_test", identity, chat_id, now.date().isoformat())
        async with AsyncSessionLocal() as db:
            user = await db.scalar(select(User).where(User.telegram_id == chat_id))
            if user is None:
                raise RuntimeError("ConfiguredTestUserNotFound")
            await db.execute(pg_insert(ScheduledOccurrence).values(
                id=str(uuid4()), occurrence_key=key, kind="daily_rundown",
                user_id=user.id, bot_identity=identity, chat_id=chat_id,
                local_date=now.date().isoformat(), timezone_name="Asia/Shanghai",
                planned_at=now - timedelta(minutes=1), state="PENDING",
            ).on_conflict_do_nothing(index_elements=["occurrence_key"]))
            await db.commit()
            row = await db.scalar(select(ScheduledOccurrence).where(
                ScheduledOccurrence.occurrence_key == key))
            occurrence_id = row.id
            state = row.state
        if state in {"PENDING", "FAILED", "CLAIMED"}:
            token = await claim_occurrence(AsyncSessionLocal, occurrence_id, "r4_3_e5")
            if token:
                await ready_occurrence(
                    AsyncSessionLocal, occurrence_id, token,
                    text="R4.3 主动简报验收\n今日重点：验证持久化投递、远端消息 ID 与重启恢复。\n本条只含合成测试内容。",
                    locale="zh-CN")
        async with AsyncSessionLocal() as db:
            row = await db.get(ScheduledOccurrence, occurrence_id)
            delivery = await db.scalar(select(OutboundDelivery).where(
                OutboundDelivery.occurrence_id == occurrence_id))
        if row.state == "READY" and delivery is not None:
            await dispatch_delivery(AsyncSessionLocal, delivery.id, "r4_3_e5", bot)
            await reconcile_occurrence(AsyncSessionLocal, occurrence_id)
        async with AsyncSessionLocal() as db:
            row = await db.get(ScheduledOccurrence, occurrence_id)
            delivery = await db.scalar(select(OutboundDelivery).where(
                OutboundDelivery.occurrence_id == occurrence_id))
            chunks = (await db.scalars(select(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id == delivery.id))).all() if delivery else []
        print("R4_3_E5_STATE=" + row.state)
        print("DELIVERY_STATE=" + (delivery.state if delivery else "MISSING"))
        print("CONFIRMED_CHUNKS=" + str(sum(chunk.telegram_message_id is not None for chunk in chunks)))
        print("REMOTE_MESSAGE_ID=" + str(chunks[0].telegram_message_id) if chunks else "REMOTE_MESSAGE_ID=NONE")
    finally:
        await close_bot()


if __name__ == "__main__":
    asyncio.run(run())
