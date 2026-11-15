"""Web and Agent writes to one Todo must observe one row order."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.db import crud
from app.db.models import TodoStatus
from app.agent.tools import make_todo_operations
from app.localization import tr
from app.web.todos import require_version
from app.web import reminders as reminder_api
from app.web import todos as todo_api
from fastapi import Response
from types import SimpleNamespace
from uuid import uuid4
from app.agent.executor import SqlAlchemyActionLedger
from app.agent.unified_tools import prepare_todo_delete
from app.agent.actions import ActionStatus


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_web_commits_first_agent_updates_latest_row(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial")
        todo_id = todo.id
        await db.commit()

    async with factory() as web_db:
        web_todo = await crud.get_todo(web_db, todo_id, user.id, lock=True)
        web_todo.title = "changed on Web"
        web_todo.version += 1
        await web_db.flush()

        async def agent_update():
            async with factory() as agent_db:
                await crud.update_todo(agent_db, todo_id, user.id, status="done")
                await agent_db.commit()

        task = asyncio.create_task(agent_update())
        await asyncio.sleep(0.05)
        assert not task.done()
        await web_db.commit()
        await task

    async with factory() as db:
        final = await crud.get_todo(db, todo_id, user.id)
        assert (final.title, final.status, final.version) == (
            "changed on Web", TodoStatus.done, 3)


@pytest.mark.asyncio
async def test_agent_commits_first_stale_web_version_is_rejected(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial")
        todo_id = todo.id
        original_etag = f'"{todo.version}"'
        await db.commit()

    async with factory() as agent_db:
        await crud.update_todo(agent_db, todo_id, user.id, status="done")
        await agent_db.commit()

    class WebRequest:
        headers = {"if-match": original_etag}

    async with factory() as web_db:
        web_todo = await crud.get_todo(web_db, todo_id, user.id, lock=True)
        with pytest.raises(HTTPException) as error:
            require_version(WebRequest(), web_todo)
        assert error.value.status_code == 412


@pytest.mark.asyncio
async def test_web_completion_first_agent_cannot_set_reminder(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial")
        todo_id = todo.id
        await db.commit()
    async with factory() as db:
        row = await crud.get_todo(db, todo_id, user.id, lock=True)
        row.status = TodoStatus.done
        row.version += 1
        await db.commit()

    when = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    result = await make_todo_operations(factory)["set_reminder"](
        user_id=user.id, todo_id=todo_id, remind_at_str=when)
    assert result == tr("todo_completed_reminder")
    async with factory() as db:
        row = await crud.get_todo(db, todo_id, user.id)
        assert row.reminder_at is None and row.status == TodoStatus.done


@pytest.mark.asyncio
async def test_agent_completion_cancels_existing_web_reminder(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    when = datetime.now(timezone.utc) + timedelta(days=2)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial", reminder_at=when)
        todo_id = todo.id
        await db.commit()
    async with factory() as db:
        await crud.update_todo(db, todo_id, user.id, status="done")
        await db.commit()
    async with factory() as db:
        row = await crud.get_todo(db, todo_id, user.id)
        assert row.status == TodoStatus.done
        assert row.reminder_state == "CANCELLED" and row.reminder_at is None
        assert row.reminder_version == 1


@pytest.mark.asyncio
async def test_web_first_agent_stale_update_and_reminder_do_not_overwrite(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial")
        todo_id, observed_version = todo.id, todo.version
        await db.commit()
    async with factory() as db:
        changed = await crud.update_todo(db, todo_id, user.id, title="Web revision")
        await db.commit()
        assert changed.version == observed_version + 1
    tools = make_todo_operations(factory)
    update = await tools["update_todo"](
        user_id=user.id, todo_id=todo_id[:8], title="stale Telegram revision",
        expected_version=observed_version)
    reminder = await tools["set_reminder"](
        user_id=user.id, todo_id=todo_id[:8],
        remind_at_str=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
        expected_version=observed_version)
    assert update == reminder == tr("todo_version_conflict")
    async with factory() as db:
        final = await crud.get_todo(db, todo_id, user.id)
        assert (final.title, final.version, final.reminder_at) == (
            "Web revision", observed_version + 1, None)


@pytest.mark.asyncio
async def test_web_commit_during_telegram_update_waits_then_rejects_stale_version(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial")
        todo_id, version = todo.id, todo.version
        await db.commit()
    async with factory() as web_db:
        web_todo = await crud.get_todo(web_db, todo_id, user.id, lock=True)
        web_todo.title = "Web wins"
        web_todo.version += 1
        await web_db.flush()
        pending = asyncio.create_task(make_todo_operations(factory)["update_todo"](
            user_id=user.id, todo_id=todo_id[:8], title="Telegram stale",
            expected_version=version))
        await asyncio.sleep(0.05)
        assert not pending.done()
        await web_db.commit()
        assert await pending == tr("todo_version_conflict")
    async with factory() as db:
        final = await crud.get_todo(db, todo_id, user.id)
        assert (final.title, final.version) == ("Web wins", version + 1)


@pytest.mark.asyncio
async def test_telegram_first_web_patch_and_reminder_stale_are_rejected(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    monkeypatch.setattr(todo_api, "_authenticated", auth)
    monkeypatch.setattr(reminder_api, "_authenticated", auth)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "initial")
        todo_id, stale_etag = todo.id, f'"{todo.version}"'
        await db.commit()
    tools = make_todo_operations(factory)
    updated = await tools["update_todo"](
        user_id=user.id, todo_id=todo_id[:8], title="Telegram revision",
        expected_version=1)
    assert "Telegram revision" in updated
    request = SimpleNamespace(headers={"if-match": stale_etag,
                                       "idempotency-key": uuid4().hex})
    with pytest.raises(HTTPException) as patch_error:
        await todo_api.patch_web_todo(request, Response(), todo_id,
                                      todo_api.TodoPatch(title="stale Web revision"))
    assert patch_error.value.status_code == 412
    with pytest.raises(HTTPException) as reminder_error:
        await reminder_api.set_reminder(
            request, Response(), todo_id,
            reminder_api.ReminderTime(
                at=datetime.now(timezone.utc) + timedelta(days=2), tz="Asia/Shanghai"))
    assert reminder_error.value.status_code == 412
    async with factory() as db:
        final = await crud.get_todo(db, todo_id, user.id)
        assert final.title == "Telegram revision" and final.reminder_at is None


@pytest.mark.asyncio
async def test_telegram_reminder_first_stale_web_cancel_does_not_undo_it(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    monkeypatch.setattr(reminder_api, "_authenticated", auth)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "target")
        todo_id, old_etag = todo.id, f'"{todo.version}"'
        await db.commit()
    when = datetime.now(timezone.utc) + timedelta(days=2)
    result = await make_todo_operations(factory)["set_reminder"](
        user_id=user.id, todo_id=todo_id[:8], remind_at_str=when.isoformat(),
        expected_version=1)
    assert "target" in result
    with pytest.raises(HTTPException) as stale:
        await reminder_api.cancel_reminder(
            SimpleNamespace(headers={"if-match": old_etag,
                                     "idempotency-key": uuid4().hex}),
            Response(), todo_id)
    assert stale.value.status_code == 412
    async with factory() as db:
        final = await crud.get_todo(db, todo_id, user.id)
        assert final.reminder_at == when and final.reminder_state == "ACTIVE"


@pytest.mark.asyncio
async def test_web_reminder_change_preserves_plan_and_versioned_key(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(reminder_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    monkeypatch.setattr(reminder_api, "_authenticated", auth)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "unscheduled")
        todo_id, etag = todo.id, f'"{todo.version}"'
        await db.commit()
    when = datetime.now(timezone.utc) + timedelta(days=2)
    key = uuid4().hex
    request = SimpleNamespace(headers={"if-match": etag, "idempotency-key": key})
    result = await reminder_api.set_reminder(
        request, Response(), todo_id,
        reminder_api.ReminderTime(at=when, tz="Asia/Shanghai"))
    assert result["todo_version"] == 2
    async with factory() as db:
        row = await crud.get_todo(db, todo_id, user.id)
        assert row.planned_start_at is None and row.planned_end_at is None
    changed_request = SimpleNamespace(headers={"if-match": '"2"',
                                               "idempotency-key": key})
    with pytest.raises(HTTPException) as conflict:
        await reminder_api.set_reminder(
            changed_request, Response(), todo_id,
            reminder_api.ReminderTime(at=when, tz="Asia/Shanghai"))
    assert conflict.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["web", "telegram"])
async def test_pending_telegram_delete_and_web_patch_do_not_cross_execute(
        pg_harness, monkeypatch, first):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    monkeypatch.setattr(todo_api, "_authenticated", auth)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "candidate")
        todo_id, etag = todo.id, f'"{todo.version}"'
        await db.commit()
    proposal = await prepare_todo_delete(factory, user_id=user.id, target=todo_id[:8])
    ledger = SqlAlchemyActionLedger(factory)
    if first == "web":
        changed = await todo_api.patch_web_todo(
            SimpleNamespace(headers={"if-match": etag}), Response(), todo_id,
            todo_api.TodoPatch(title="changed on Web"))
        assert changed["version"] == 2
        result = await ledger.delete_todo_atomically(proposal)
        assert result.status == ActionStatus.FAILED
        assert result.result["failure_category"] == "todo_target_changed"
        async with factory() as db:
            assert (await crud.get_todo(db, todo_id, user.id)).title == "changed on Web"
    else:
        result = await ledger.delete_todo_atomically(proposal)
        assert result.status == ActionStatus.EXECUTED
        with pytest.raises(HTTPException) as missing:
            await todo_api.patch_web_todo(
                SimpleNamespace(headers={"if-match": etag}), Response(), todo_id,
                todo_api.TodoPatch(title="stale Web edit"))
        assert missing.value.status_code == 404
