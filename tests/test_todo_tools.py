"""User-facing Todo tool behavior; real PostgreSQL has separate opt-in tests."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.agent.tools import make_todo_tools, parse_cjk_absolute_clock
from app.agent.nodes import messages_router
from app.db import crud
from app.config import get_settings


def tool_by_name(session_factory, name):
    return next(tool for tool in make_todo_tools(session_factory) if tool.name == name)


def test_todo_deadline_does_not_trigger_calendar_reservation_check():
    assert messages_router({"intent": "create", "confidence": 0.95,
                            "entities": {"due_date_str": "周五"}}) == "todo_llm"


@pytest.mark.parametrize("text,hour,minute", [
    ("2030年1月15日上午9点", 9, 0),
    ("2030年1月15日下午3点30分", 15, 30),
    ("2030年1月15日晚上8时", 20, 0),
])
def test_bounded_chinese_absolute_clock(text, hour, minute):
    parsed = parse_cjk_absolute_clock(text, "Asia/Shanghai")
    assert parsed is not None
    assert (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute) == (2030, 1, 15, hour, minute)
    assert parsed.utcoffset().total_seconds() == 8 * 3600


@pytest.mark.parametrize("text,zone", [
    ("2030年2月30日上午9点", "Asia/Shanghai"),
    ("2030年1月15日下午15点", "Asia/Shanghai"),
    ("2030年1月15日上午9点后随便执行", "Asia/Shanghai"),
    ("2030年3月10日凌晨2点", "Asia/Shanghai"),
    ("2026年3月8日上午2点", "America/New_York"),
    ("2026年11月1日上午1点30分", "America/New_York"),
])
def test_chinese_absolute_clock_rejects_invalid_or_unsafe(text, zone):
    assert parse_cjk_absolute_clock(text, zone) is None


@pytest.mark.asyncio
async def test_set_reminder_accepts_chinese_absolute_clock_without_retry(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-task-cjk-clock")
        todo = await crud.create_todo(db, user.id, "Synthetic CJK reminder")
        await db.commit()
        user_id, todo_id = user.id, todo.id
    result = await tool_by_name(session_factory, "set_reminder").ainvoke({
        "user_id": user_id, "todo_id": todo_id,
        "remind_at_str": "2030年1月15日上午9点",
    })
    assert not result.startswith("❌")
    async with session_factory() as db:
        updated = await crud.get_todo(db, todo_id, user_id)
        assert updated is not None
        assert updated.reminder_version == 1
        # SQLite drops timezone information; PostgreSQL returns an aware value.
        saved = updated.reminder_at
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=timezone.utc)
        assert saved.astimezone(ZoneInfo(get_settings().default_timezone)).strftime("%Y-%m-%d %H:%M") == "2030-01-15 09:00"


@pytest.mark.asyncio
async def test_create_with_independent_chinese_deadline_and_reminder(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-task-tools")
        await db.commit()
        user_id = user.id

    response = await tool_by_name(session_factory, "create_todo").ainvoke({
        "title": "交论文", "user_id": user_id,
        # A fixed weekday becomes past after its local 20:00; use a relative
        # future day so this behavior test is not tied to CI wall-clock time.
        "due_date_str": "明天", "reminder_at_str": "明天晚上8点",
        "priority": "high", "tags": "work, thesis",
    })
    assert "Created todo" in response
    async with session_factory() as db:
        todos = await crud.list_todos(db, user_id)
    assert len(todos) == 1
    todo = todos[0]
    tomorrow = datetime.now(ZoneInfo(get_settings().default_timezone)) + timedelta(days=1)
    assert todo.due_date.weekday() == tomorrow.weekday()
    assert todo.reminder_at.weekday() == tomorrow.weekday()
    assert todo.reminder_at.hour == 20
    assert todo.due_date != todo.reminder_at
    assert todo.priority.value == "high"
    assert todo.tags == ["work", "thesis"]


@pytest.mark.asyncio
async def test_update_complete_delete_by_unique_title(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-task-title")
        todo = await crud.create_todo(db, user.id, "修改论文")
        await db.commit()
        user_id = user.id

    update = tool_by_name(session_factory, "update_todo")
    result = await update.ainvoke({"user_id": user_id, "todo_id": "修改论文", "priority": "high"})
    assert "Updated todo" in result
    result = await update.ainvoke({"user_id": user_id, "todo_id": "修改论文", "status": "done"})
    assert "status: done" in result
    async with session_factory() as db:
        stored = await crud.get_todo(db, todo.id, user_id)
        assert stored.priority.value == "high"
        assert stored.status.value == "done"

    result = await tool_by_name(session_factory, "delete_todo").ainvoke({
        "user_id": user_id, "todo_id": "修改论文",
    })
    assert "Deleted todo" in result
    async with session_factory() as db:
        assert await crud.list_todos(db, user_id) == []


@pytest.mark.asyncio
async def test_ambiguous_title_does_not_mutate(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-task-ambiguous")
        await crud.create_todo(db, user.id, "修改论文初稿")
        await crud.create_todo(db, user.id, "修改论文终稿")
        await db.commit()
        user_id = user.id
    result = await tool_by_name(session_factory, "delete_todo").ainvoke({
        "user_id": user_id, "todo_id": "修改论文",
    })
    assert "Multiple todos" in result
    async with session_factory() as db:
        assert len(await crud.list_todos(db, user_id)) == 2


@pytest.mark.asyncio
async def test_colon_reminder_and_date_only_deadline(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-tonight")
        await db.commit()
    result = await tool_by_name(session_factory, "create_todo").ainvoke({
        "user_id": user.id, "title": "明晚提醒测试", "due_date_str": "后天",
        "reminder_at_str": "明天20:00",
    })
    assert "Created todo" in result
    async with session_factory() as db:
        todo = (await crud.list_todos(db, user.id))[0]
    assert (todo.due_date.hour, todo.due_date.minute) == (23, 59)
    assert (todo.reminder_at.hour, todo.reminder_at.minute) == (20, 0)


@pytest.mark.asyncio
async def test_past_reminder_is_rejected_before_todo_write(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-past-reminder")
        await db.commit()
    result = await tool_by_name(session_factory, "create_todo").ainvoke({
        "user_id": user.id, "title": "不能补发过去的新提醒",
        "due_date_str": "明天", "reminder_at_str": "2020-01-01 20:00",
    })
    assert "already passed" in result
    async with session_factory() as db:
        assert await crud.list_todos(db, user.id) == []


@pytest.mark.asyncio
async def test_past_reminder_update_does_not_change_existing_todo(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-past-reminder-update")
        todo = await crud.create_todo(db, user.id, "保留任务")
        await db.commit()
    result = await tool_by_name(session_factory, "set_reminder").ainvoke({
        "user_id": user.id, "todo_id": todo.id,
        "remind_at_str": "2020-01-01 20:00",
    })
    assert "already passed" in result
    async with session_factory() as db:
        persisted = await crud.get_todo(db, todo.id, user.id)
        assert persisted.reminder_at is None
