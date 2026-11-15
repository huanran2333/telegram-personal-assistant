from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.db import crud
from app.db.models import OutboundDelivery, OutboundDeliveryChunk, ScheduledOccurrence
from app.scheduler.daily_brief import send_daily_briefs

pytestmark = pytest.mark.integration


class CalendarForBrief:
    async def search_events(self, query, start=None, end=None):
        assert query == ""
        return [{"summary": "itest meeting", "start": {"dateTime": start}}]


@pytest.mark.asyncio
async def test_real_postgres_daily_brief_claim_and_content(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    now = datetime.now(ZoneInfo(get_settings().default_timezone))
    sent = []

    async def sender(chat_id, text):
        sent.append((chat_id, text))

    async with factory() as db:
        await crud.create_todo(db, user.id, "itest due today", due_date=now.replace(hour=23, minute=0))
        await crud.create_todo(db, user.id, "itest overdue", due_date=now - timedelta(days=1))
        completed = await crud.create_todo(db, user.id, "itest completed", due_date=now)
        await crud.update_todo(db, completed.id, user.id, status="done")
        await db.commit()

    await send_daily_briefs(CalendarForBrief(), factory, sender, target_user_id=user.id)
    await send_daily_briefs(CalendarForBrief(), factory, sender, target_user_id=user.id)
    assert len(sent) == 1
    text = sent[0][1]
    assert "itest meeting" in text
    assert "itest due today" in text and "due today" in text
    assert "itest overdue" in text and "overdue" in text
    assert "Top priorities" in text and "Next step" in text
    assert "itest completed" not in text
    async with factory() as db:
        rows = (await db.execute(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id,
            ScheduledOccurrence.kind == "daily_rundown"))).scalars().all()
        assert len(rows) == 1
        assert rows[0].state == "DELIVERED"
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == rows[0].id))
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery.id, "chunk_index": 0})
        assert delivery.state == "SENT" and chunk.telegram_message_id is not None
