from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.db import crud
from app.config import get_settings


@pytest.mark.asyncio
async def test_reminder_dispatch_once(session_factory, proactive_test_runtime):
    import app.scheduler.reminders as reminders

    sent = []

    async def fake_send(chat_id, text, parse_mode=None):
        sent.append((chat_id, text))

    proactive_test_runtime(session_factory, fake_send)
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-reminder")
        due = datetime(2027, 1, 1, 16, 0, tzinfo=timezone.utc)
        todo = await crud.create_todo(db, user.id, "Call mom", due_date=due)
        await crud.update_todo(db, todo.id, user.id, reminder_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        await db.commit()

    await reminders.check_and_send_reminders()
    await reminders.check_and_send_reminders()
    assert len(sent) == 1
    expected_due = due.astimezone(ZoneInfo(get_settings().default_timezone)).strftime("%b %d at %H:%M %Z")
    assert f"Due: {expected_due}" in sent[0][1]


@pytest.mark.asyncio
async def test_reminder_survives_restart_semantics(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-restart")
        todo = await crud.create_todo(db, user.id, "Persisted reminder")
        await crud.update_todo(db, todo.id, user.id, reminder_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        await db.commit()

    # A fresh scheduler/session factory can reclaim the persisted row.
    async with session_factory() as db:
        claimed = await crud.claim_due_reminders(db, datetime.now(timezone.utc))
        await db.commit()
    assert [item.title for item in claimed] == ["Persisted reminder"]
