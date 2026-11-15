import pytest
from datetime import datetime, timezone

from app.agent.calendar import CalendarError
from app.db import crud
from app.scheduler.daily_brief import send_daily_briefs
from app.scheduler.daily_brief import format_daily_brief


class BriefCalendar:
    async def search_events(self, query, start=None, end=None):
        assert datetime.fromisoformat(start).hour == 0
        assert datetime.fromisoformat(start).second == 0
        assert datetime.fromisoformat(start).microsecond == 0
        assert datetime.fromisoformat(end).hour == 0
        return [{"summary": "Team meeting", "start": "10:00"}]


class FailingBriefCalendar:
    async def search_events(self, query, start=None, end=None):
        raise CalendarError("OAuth expired")


@pytest.mark.asyncio
async def test_daily_brief_deduplicated(session_factory):
    sent = []

    async def sender(chat_id, text):
        sent.append(text)

    async with session_factory() as db:
        await crud.get_or_create_user(db, "brief-user")
        await db.commit()
    await send_daily_briefs(BriefCalendar(), session_factory, sender)
    await send_daily_briefs(BriefCalendar(), session_factory, sender)
    assert len(sent) == 1
    assert "Team meeting" in sent[0]


@pytest.mark.asyncio
async def test_daily_brief_calendar_failure_degrades(session_factory):
    sent = []

    async def sender(chat_id, text):
        sent.append(text)

    async with session_factory() as db:
        await crud.get_or_create_user(db, "brief-fallback")
        await db.commit()
    await send_daily_briefs(FailingBriefCalendar(), session_factory, sender)
    assert "Calendar unavailable" in sent[0]


def test_daily_brief_formats_real_mcp_event_shape_and_due_labels():
    class Todo:
        def __init__(self, title, due_date):
            self.title = title
            self.due_date = due_date
            self.priority = type("Priority", (), {"value": "high"})()

    today = datetime(2026, 10, 1, tzinfo=timezone.utc).date()
    text = format_daily_brief(
        [{"summary": "测试会议", "start": {"dateTime": "2026-10-01T16:00:00+00:00"}}],
        [Todo("今天的任务", datetime(2026, 10, 1, 12, tzinfo=timezone.utc)),
         Todo("逾期任务", datetime(2026, 9, 29, 12, tzinfo=timezone.utc))],
        timezone_name="UTC", today=today,
    )
    assert "16:00 测试会议" in text
    assert "今天的任务 (due today)" in text
    assert "逾期任务 (overdue)" in text
