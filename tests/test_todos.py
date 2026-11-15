from datetime import datetime, timedelta, timezone

import pytest

from app.db import crud
from app.db.models import TodoStatus


@pytest.mark.asyncio
async def test_todo_create(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-a")
        todo = await crud.create_todo(db, user.id, "Write thesis", priority="high", tags=["work"])
        await db.commit()
        listed = await crud.list_todos(db, user.id, tags=["work"])
    assert [item.id for item in listed] == [todo.id]
    assert listed[0].priority.value == "high"


@pytest.mark.asyncio
async def test_tag_filter_applies_before_result_limit(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-tag-limit")
        now = datetime.now(timezone.utc)
        for index in range(25):
            await crud.create_todo(db, user.id, f"Unrelated {index}",
                                   due_date=now + timedelta(hours=index), tags=["other"])
        matching = await crud.create_todo(db, user.id, "Tagged task",
                                          due_date=now + timedelta(days=10), tags=["work"])
        await db.commit()
        result = await crud.list_todos(db, user.id, tags=["work"], limit=1)
    assert [todo.id for todo in result] == [matching.id]


@pytest.mark.asyncio
async def test_todo_due_and_reminder_are_separate(session_factory):
    due = datetime.now(timezone.utc) + timedelta(days=2)
    reminder = due - timedelta(days=1)
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-b")
        todo = await crud.create_todo(db, user.id, "Submit thesis", due_date=due)
        await crud.update_todo(db, todo.id, user.id, reminder_at=reminder, reminder_sent=False)
        await db.commit()
        refreshed = await crud.get_todo(db, todo.id, user.id)
    # SQLite test storage drops timezone metadata; PostgreSQL TIMESTAMPTZ keeps
    # it.  The regression here is field independence, not the test dialect.
    assert refreshed.due_date.replace(tzinfo=timezone.utc) == due
    assert refreshed.reminder_at.replace(tzinfo=timezone.utc) == reminder


@pytest.mark.asyncio
async def test_user_isolation(session_factory):
    async with session_factory() as db:
        first = await crud.get_or_create_user(db, "telegram-first")
        second = await crud.get_or_create_user(db, "telegram-second")
        await crud.create_todo(db, first.id, "Private task")
        await db.commit()
        assert len(await crud.list_todos(db, first.id)) == 1
        assert await crud.list_todos(db, second.id) == []


@pytest.mark.asyncio
async def test_complete_and_delete_do_not_remain_dispatchable(session_factory):
    now = datetime.now(timezone.utc)
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "telegram-c")
        done = await crud.create_todo(db, user.id, "Done", due_date=now)
        deleted = await crud.create_todo(db, user.id, "Delete", due_date=now)
        await crud.update_todo(db, done.id, user.id, reminder_at=now, status=TodoStatus.done.value)
        await crud.update_todo(db, deleted.id, user.id, reminder_at=now)
        await crud.delete_todo(db, deleted.id, user.id)
        await db.commit()
        claimed = await crud.claim_due_reminders(db, now + timedelta(minutes=1))
    assert claimed == []
