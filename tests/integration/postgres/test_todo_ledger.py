import asyncio
from datetime import datetime, time, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select, text

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.tools import make_todo_tools
from app.config import get_settings
from app.db import crud
from app.db.models import ActionExecution, Todo, TodoStatus


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_postgres_todo_tools_with_reminder_and_title_resolution(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    tools = {tool.name: tool for tool in make_todo_tools(factory)}
    result = await tools["create_todo"].ainvoke({
        "user_id": user.id, "title": "itest 完成论文", "due_date_str": "明天",
        "reminder_at_str": "明天晚上8点", "priority": "high", "tags": "work",
    })
    assert "Created todo" in result
    async with factory() as db:
        rows = await crud.list_todos(db, user.id)
        assert len(rows) == 1
        todo_id = rows[0].id
        assert rows[0].due_date != rows[0].reminder_at
        assert rows[0].reminder_at.astimezone(ZoneInfo(get_settings().default_timezone)).hour == 20
        assert rows[0].tags == ["work"]
    result = await tools["update_todo"].ainvoke({
        "user_id": user.id, "todo_id": "itest 完成论文", "status": "done",
    })
    assert "status: done" in result
    async with factory() as db:
        assert (await crud.get_todo(db, todo_id, user.id)).status == TodoStatus.done


@pytest.mark.asyncio
async def test_real_postgres_todo_crud(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "itest thesis", priority="high", tags=["work"])
        todo_id = todo.id
        await db.commit()

    async with factory() as db:
        assert await db.scalar(select(func.count()).select_from(Todo).where(Todo.id == todo_id)) == 1
        assert [item.id for item in await crud.list_todos(db, user.id)] == [todo_id]
        changed = await crud.update_todo(db, todo_id, user.id, title="itest revised thesis")
        assert changed.title == "itest revised thesis"
        await db.commit()

    async with factory() as db:
        assert (await crud.get_todo(db, todo_id, user.id)).title == "itest revised thesis"
        await crud.update_todo(db, todo_id, user.id, status="done")
        await db.commit()

    async with factory() as db:
        assert (await crud.get_todo(db, todo_id, user.id)).status == TodoStatus.done
        assert await crud.delete_todo(db, todo_id, user.id)
        await db.commit()

    async with factory() as db:
        assert await crud.get_todo(db, todo_id, user.id) is None
        assert await db.scalar(select(func.count()).select_from(Todo).where(Todo.id == todo_id)) == 0


@pytest.mark.asyncio
async def test_real_postgres_user_isolation(pg_harness):
    factory, new_user, _ = pg_harness
    first, second = await new_user(), await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, first.id, "itest private")
        todo_id = todo.id
        await db.commit()

    async with factory() as db:
        assert await crud.list_todos(db, second.id) == []
        assert await crud.search_todos(db, second.id, "private") == []
        assert await crud.get_todo(db, todo_id, second.id) is None
        assert await crud.update_todo(db, todo_id, second.id, title="stolen") is None
        assert not await crud.delete_todo(db, todo_id, second.id)
        await db.commit()

    async with factory() as db:
        assert (await crud.get_todo(db, todo_id, first.id)).title == "itest private"


@pytest.mark.asyncio
async def test_real_postgres_due_and_reminder(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    local_tz = ZoneInfo("Asia/Shanghai")
    today = datetime.now(local_tz).date()
    days_to_friday = (4 - today.weekday()) % 7 or 7
    friday = today + timedelta(days=days_to_friday)
    due = datetime.combine(friday, time(23, 0), local_tz).astimezone(timezone.utc)
    reminder = datetime.combine(friday - timedelta(days=1), time(20, 0), local_tz).astimezone(timezone.utc)

    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "itest deadline", due_date=due)
        await crud.update_todo(db, todo.id, user.id, reminder_at=reminder)
        todo_id = todo.id
        await db.commit()

    async with factory() as db:
        row = (await db.execute(text("SELECT due_date, reminder_at FROM todos WHERE id = :id"), {"id": todo_id})).one()
        assert row.due_date == due
        assert row.reminder_at == reminder
        assert row.due_date != row.reminder_at


def proposal_for(user_id, action_id):
    return ActionProposal(
        action_id=action_id, domain="todo", operation="create_todo",
        args={"user_id": user_id, "title": "itest ledger side effect"},
        human_readable_preview="Create test Todo", reason="PostgreSQL integration",
        risk_level=RiskLevel.INTERNAL_LOW_RISK,
    )


@pytest.mark.asyncio
async def test_real_postgres_action_idempotency(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    action_id = str(uuid5(NAMESPACE_URL, f"itest-ledger-{user.id}"))
    executor = ActionExecutor(None, SqlAlchemyActionLedger(factory), factory)
    proposal = proposal_for(user.id, action_id)
    first = await executor.execute(proposal)
    second = await executor.execute(proposal)
    assert first.status == second.status == ActionStatus.EXECUTED

    async with factory() as db:
        row = await db.get(ActionExecution, action_id)
        assert row.status == "executed"
        assert row.executed_at is not None
        assert row.result["id"] == first.result["id"]
        assert row.external_id == first.result["id"]
        assert await db.scalar(select(func.count()).select_from(Todo).where(Todo.user_id == user.id)) == 1


@pytest.mark.asyncio
async def test_real_postgres_concurrent_action_claim(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    action_id = str(uuid5(NAMESPACE_URL, f"itest-concurrent-{user.id}"))
    executor = ActionExecutor(None, SqlAlchemyActionLedger(factory), factory)
    proposal = proposal_for(user.id, action_id)
    attempts = await asyncio.gather(executor.execute(proposal), executor.execute(proposal))

    async with factory() as db:
        row = await db.get(ActionExecution, action_id)
        assert row.status == "executed"
        assert sum(item.status == ActionStatus.EXECUTED for item in attempts) >= 1
        assert await db.scalar(select(func.count()).select_from(ActionExecution).where(ActionExecution.action_id == action_id)) == 1
        assert await db.scalar(select(func.count()).select_from(Todo).where(Todo.user_id == user.id)) == 1


@pytest.mark.asyncio
async def test_real_postgres_approved_bulk_tag_delete_is_idempotent_and_scoped(pg_harness):
    factory, new_user, _ = pg_harness
    user, other = await new_user(), await new_user()
    async with factory() as db:
        first = await crud.create_todo(db, user.id, "itest first", tags=["work"])
        second = await crud.create_todo(db, user.id, "itest second", tags=["work"])
        private = await crud.create_todo(db, other.id, "itest private", tags=["work"])
        await db.commit()
    action_id = str(uuid5(NAMESPACE_URL, f"itest-bulk-delete-{user.id}"))
    proposal = ActionProposal(
        action_id=action_id, domain="todo", operation="delete_todos_by_tag",
        args={"user_id": user.id, "tag": "work", "todo_ids": [first.id, second.id]},
        human_readable_preview="Delete two work Todos", reason="PostgreSQL integration",
        risk_level=RiskLevel.DESTRUCTIVE,
    )
    executor = ActionExecutor(None, SqlAlchemyActionLedger(factory), factory)
    first_result = await executor.execute(proposal)
    second_result = await executor.execute(proposal)
    assert first_result.status == second_result.status == ActionStatus.EXECUTED
    assert second_result.message == "Action was already processed."
    async with factory() as db:
        assert await crud.get_todo(db, first.id, user.id) is None
        assert await crud.get_todo(db, second.id, user.id) is None
        assert await crud.get_todo(db, private.id, other.id) is not None
        ledger = await db.get(ActionExecution, action_id)
        assert ledger.status == "executed" and ledger.result["count"] == 2


@pytest.mark.asyncio
async def test_bulk_delete_does_not_split_business_commit_from_ledger(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "itest atomic bulk", tags=["work"])
        await db.commit()
    proposal = ActionProposal(
        action_id=str(uuid5(NAMESPACE_URL, f"itest-atomic-bulk-{user.id}")),
        domain="todo", operation="delete_todos_by_tag",
        args={"user_id": user.id, "tag": "work", "todo_ids": [todo.id]},
        human_readable_preview="Delete test Todo", reason="fault injection",
        risk_level=RiskLevel.DESTRUCTIVE,
    )
    ledger = SqlAlchemyActionLedger(factory)

    async def fail_split_complete(*_args, **_kwargs):
        raise RuntimeError("injected split ledger failure")

    monkeypatch.setattr(ledger, "complete", fail_split_complete)
    result = await ActionExecutor(None, ledger, factory).execute(proposal)
    async with factory() as db:
        row = await db.get(ActionExecution, proposal.action_id)
        remaining = await crud.get_todo(db, todo.id, user.id)
    assert result.status == ActionStatus.EXECUTED
    assert row.status == "executed" and row.result["ids"] == [todo.id]
    assert remaining is None


@pytest.mark.asyncio
async def test_bulk_delete_crash_before_commit_rolls_back_then_retries(pg_harness, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        first = await crud.create_todo(db, user.id, "itest atomic first", tags=["work"])
        second = await crud.create_todo(db, user.id, "itest atomic second", tags=["work"])
        await db.commit()
    proposal = ActionProposal(
        action_id=str(uuid5(NAMESPACE_URL, f"itest-bulk-crash-{user.id}")),
        domain="todo", operation="delete_todos_by_tag",
        args={"user_id": user.id, "tag": "work", "todo_ids": [first.id, second.id]},
        human_readable_preview="Delete test Todos", reason="fault injection",
        risk_level=RiskLevel.DESTRUCTIVE,
    )
    original_delete = AsyncSession.delete
    calls = 0

    async def crash_after_first_delete(db, row):
        nonlocal calls
        await original_delete(db, row)
        calls += 1
        if calls == 1:
            raise RuntimeError("injected crash after first deletion")

    monkeypatch.setattr(AsyncSession, "delete", crash_after_first_delete)
    executor = ActionExecutor(None, SqlAlchemyActionLedger(factory), factory)
    with pytest.raises(RuntimeError, match="injected crash"):
        await executor.execute(proposal)
    monkeypatch.setattr(AsyncSession, "delete", original_delete)
    async with factory() as db:
        assert await db.get(ActionExecution, proposal.action_id) is None
        assert await crud.get_todo(db, first.id, user.id) is not None
        assert await crud.get_todo(db, second.id, user.id) is not None
    assert (await executor.execute(proposal)).status == ActionStatus.EXECUTED
    assert (await executor.execute(proposal)).status == ActionStatus.EXECUTED


@pytest.mark.asyncio
async def test_bulk_delete_concurrent_same_action_has_one_result(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "itest concurrent bulk", tags=["work"])
        await db.commit()
    proposal = ActionProposal(
        action_id=str(uuid5(NAMESPACE_URL, f"itest-bulk-concurrent-{user.id}")),
        domain="todo", operation="delete_todos_by_tag",
        args={"user_id": user.id, "tag": "work", "todo_ids": [todo.id]},
        human_readable_preview="Delete test Todo", reason="concurrent approval",
        risk_level=RiskLevel.DESTRUCTIVE,
    )
    executor = ActionExecutor(None, SqlAlchemyActionLedger(factory), factory)
    results = await asyncio.gather(executor.execute(proposal), executor.execute(proposal))
    assert all(result.status == ActionStatus.EXECUTED for result in results)
    assert results[0].result == results[1].result
    async with factory() as db:
        assert await db.get(ActionExecution, proposal.action_id) is not None
        assert await crud.get_todo(db, todo.id, user.id) is None


@pytest.mark.asyncio
async def test_bulk_delete_rejects_approved_version_after_todo_edit(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "itest approved title", tags=["work"])
        await db.commit()
        approved_version = todo.version
    proposal = ActionProposal(
        domain="todo", operation="delete_todos_by_tag",
        args={"user_id": user.id, "tag": "work", "todo_ids": [todo.id],
              "approved_todo_versions": {todo.id: approved_version}},
        human_readable_preview="Delete approved test Todo", reason="stale approval",
        risk_level=RiskLevel.DESTRUCTIVE,
    )
    async with factory() as db:
        changed = await crud.update_todo(db, todo.id, user.id, title="itest changed after approval")
        await db.commit()
        assert changed.version != approved_version
    result = await ActionExecutor(None, SqlAlchemyActionLedger(factory), factory).execute(proposal)
    assert result.status == ActionStatus.FAILED
    assert result.result["failure_category"] == "todo_target_changed"
    async with factory() as db:
        assert await crud.get_todo(db, todo.id, user.id) is not None
        row = await db.get(ActionExecution, proposal.action_id)
        assert row.status == "failed"
