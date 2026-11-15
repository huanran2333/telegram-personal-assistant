from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.db import crud
from app.scheduler.daily_brief import send_daily_briefs
from app.scheduler.rundown import (RundownSynthesis, build_snapshot, format_rundown,
                                   prioritize, synthesize_order)


ZONE = ZoneInfo("Asia/Shanghai")
TODAY = date(2026, 10, 1)


def todo(id, title, due, priority="high", reminder_at=None):
    return SimpleNamespace(id=id, title=title, due_date=due,
                           priority=SimpleNamespace(value=priority),
                           status=SimpleNamespace(value="pending"), reminder_at=reminder_at)


def test_snapshot_priorities_conflict_and_source_attribution():
    events = [
        {"id": "meeting-1", "summary": "论文讨论", "start": {"dateTime": "2026-10-01T15:00:00+08:00"},
         "end": {"dateTime": "2026-10-01T16:00:00+08:00"}},
        {"id": "meeting-2", "summary": "组会", "start": {"dateTime": "2026-10-01T15:30:00+08:00"},
         "end": {"dateTime": "2026-10-01T16:30:00+08:00"}},
    ]
    tasks = [todo("todo-1", "修改论文", datetime(2026, 10, 2, 20, tzinfo=ZONE)),
             todo("todo-2", "整理数据", datetime(2026, 9, 30, 20, tzinfo=ZONE))]
    snapshot = build_snapshot("user-a", TODAY, "Asia/Shanghai", events, tasks, [], {})
    assert [item.id for item in snapshot.overdue] == ["todo-2"]
    assert [item.id for item in snapshot.due_soon] == ["todo-1"]
    assert snapshot.conflicts == [("meeting-1", "meeting-2")]
    ranked = prioritize(snapshot, datetime(2026, 10, 1, 12, tzinfo=ZONE))
    meeting = next(item for item in ranked if item.source_id == "event:meeting-1")
    assert "related unfinished Todo" in meeting.reason
    assert "todo-1" in meeting.reason
    assert "overlaps" in meeting.reason
    output = format_rundown(snapshot, ranked)
    assert "Top priorities" in output
    assert "Next step" in output
    assert "[todo:todo-2]" in output or "[event:meeting-1]" in output
    assert "meeting-1 overlaps meeting-2" in output


def test_empty_and_unavailable_sources_are_not_confused():
    snapshot = build_snapshot("u", TODAY, "Asia/Shanghai", [], [], [],
                              {"Calendar": "read failed"})
    text = format_rundown(snapshot, [])
    assert "Calendar unavailable" in text
    assert "No calendar events found" not in text


@pytest.mark.asyncio
async def test_llm_can_only_reorder_known_ids_and_malformed_output_falls_back():
    ranked = prioritize(build_snapshot("u", TODAY, "Asia/Shanghai", [], [
        todo("one", "Task one", datetime(2026, 10, 1, 17, tzinfo=ZONE)),
        todo("two", "Task two", datetime(2026, 10, 1, 18, tzinfo=ZONE))], [], {}),
        datetime(2026, 10, 1, 10, tzinfo=ZONE))

    class FakeLLM:
        def __init__(self, output):
            self.output = output

        def with_structured_output(self, schema):
            assert schema is RundownSynthesis
            return self

        async def ainvoke(self, messages):
            return self.output

    expected = [item.source_id for item in ranked]
    assert await synthesize_order(ranked, FakeLLM({"ordered_source_ids": expected[::-1]})) == expected[::-1]
    assert await synthesize_order(ranked, FakeLLM({"ordered_source_ids": ["invented:event"]})) == expected
    assert await synthesize_order(ranked, FakeLLM({"not_the_schema": True})) == expected


class EventCalendar:
    async def search_events(self, query, start=None, end=None):
        local = datetime.fromisoformat(start)
        return [{"id": "real-shape-event", "summary": "Project review",
                 "start": {"dateTime": (local + timedelta(hours=15)).isoformat()},
                 "end": {"dateTime": (local + timedelta(hours=16)).isoformat()}}]


@pytest.mark.asyncio
async def test_daily_rundown_multiuser_and_delivery_dedupe(session_factory):
    messages = []

    async def sender(chat_id, text):
        messages.append((chat_id, text))

    local_now = datetime.now(ZONE)
    async with session_factory() as db:
        alice = await crud.get_or_create_user(db, "alice-telegram")
        bob = await crud.get_or_create_user(db, "bob-telegram")
        await crud.create_todo(db, alice.id, "Alice report", due_date=local_now + timedelta(hours=2),
                               priority="high")
        await crud.create_todo(db, bob.id, "Bob plan", due_date=local_now + timedelta(hours=3))
        await db.commit()
    await send_daily_briefs(EventCalendar(), session_factory, sender)
    await send_daily_briefs(EventCalendar(), session_factory, sender)
    assert len(messages) == 2
    by_chat = dict(messages)
    assert "Alice report" in by_chat["alice-telegram"]
    assert "Bob plan" not in by_chat["alice-telegram"]
    assert "Bob plan" in by_chat["bob-telegram"]
    assert "Project review" in by_chat["alice-telegram"]


@pytest.mark.asyncio
async def test_calendar_source_not_shared_with_other_telegram_users(session_factory):
    messages = []

    async def sender(chat_id, text):
        messages.append((chat_id, text))

    async with session_factory() as db:
        await crud.get_or_create_user(db, "owner-telegram")
        await crud.get_or_create_user(db, "other-telegram")
        await db.commit()
    await send_daily_briefs(EventCalendar(), session_factory, sender,
                            calendar_owner_telegram_id="owner-telegram")
    by_chat = dict(messages)
    assert "Project review" in by_chat["owner-telegram"]
    assert "Project review" not in by_chat["other-telegram"]
    assert "Calendar unavailable" in by_chat["other-telegram"]
