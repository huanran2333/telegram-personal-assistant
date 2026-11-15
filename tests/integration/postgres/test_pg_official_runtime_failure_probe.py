"""Read failures injected inside actual production tools, not proposal preflight."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from app.agent import official_read_tools, unified_tools
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError, GmailUnavailable
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.config import get_settings
from app.db import crud
from app.db.models import Todo
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

pytestmark = pytest.mark.integration


def _raise_read_error(domain: str, failure: str) -> None:
    if domain == "calendar":
        if failure == "bad_response":
            raise CalendarError("invalid response token=SECRET")
        cause = (httpx.ReadTimeout("token=SECRET") if failure == "timeout" else
                 CalendarProviderError("provider_unavailable", int(failure), "0"))
        raise CalendarError("token=SECRET internal URL") from cause
    if failure == "bad_response":
        raise GmailError("invalid response token=SECRET")
    cause = httpx.ReadTimeout("token=SECRET") if failure == "timeout" else None
    if cause is None:
        request = httpx.Request("GET", "https://internal.example/token=SECRET")
        response = httpx.Response(int(failure), request=request,
                                  headers={"Retry-After": "0"})
        cause = httpx.HTTPStatusError("token=SECRET", request=request, response=response)
    raise GmailUnavailable("token=SECRET") from cause


class FailingCalendar:
    def __init__(self, failure):
        self.failure = failure
        self.calls = 0

    async def search_events(self, *_):
        self.calls += 1
        _raise_read_error("calendar", self.failure)


class FailingGmail:
    def __init__(self, failure):
        self.failure = failure
        self.calls = 0

    async def unread(self, *_args, **_kwargs):
        self.calls += 1
        _raise_read_error("gmail", self.failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain,failure,attempts,category", [
    ("calendar", "timeout", 2, "connection_timeout"),
    ("calendar", "403", 1, "permission_denied"),
    ("calendar", "429", 2, "rate_limited"),
    ("calendar", "503", 2, "service_unavailable"),
    ("calendar", "bad_response", 1, "read_failed"),
    ("gmail", "timeout", 2, "connection_timeout"),
    ("gmail", "403", 1, "permission_denied"),
    ("gmail", "503", 2, "service_unavailable"),
    ("gmail", "404", 1, "resource_unavailable"),
    ("gmail", "bad_response", 1, "read_failed"),
])
async def test_runtime_read_failure_reaches_paired_error_without_write(
        pg_harness, isolated_pg18_target, monkeypatch,
        domain, failure, attempts, category):
    factory, new_user, _ = pg_harness
    user = await new_user()
    adapter = FailingCalendar(failure) if domain == "calendar" else FailingGmail(failure)
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": domain == "calendar",
        "google_calendar_owner_telegram_id": user.telegram_id if domain == "calendar" else "",
        "gmail_enabled": domain == "gmail",
        "gmail_owner_telegram_id": user.telegram_id if domain == "gmail" else "",
    })
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    if domain == "calendar":
        monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: adapter)
        call = {"name": "calendar_events", "id": "runtime-read",
                "args": {"query": "test", "start": "2026-12-10T00:00:00+08:00",
                         "end": "2026-12-10T23:59:59+08:00"}}
    else:
        monkeypatch.setattr(official_read_tools, "get_gmail_source", lambda: adapter)
        call = {"name": "gmail_unread", "id": "runtime-read", "args": {}}
    model = ScriptedModel([AIMessage(content="", tool_calls=[call]),
                           AIMessage(content=f"{domain} 读取失败，不能据此判断结果为空。")])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(model=model, session_factory=factory,
                                        settings=settings, checkpointer=saver,
                                        interrupt_mode="per_call")
        reply, state = await run_agent(
            graph, user.id, user.telegram_id, "", f"只读查询 {domain}，如实报告结果。",
            origin="web", turn_id=f"runtime-read-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = [item for item in state["messages"] if isinstance(item, ToolMessage)]
        assert len(messages) == 1
        assert messages[0].tool_call_id == "runtime-read"
        assert messages[0].status == "error"
        assert category in messages[0].content and "未获得可靠结果" in messages[0].content
        assert "SECRET" not in messages[0].content
        assert "读取失败" in reply
        assert adapter.calls == attempts
        async with factory() as db:
            assert not (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_unhandled_runtime_error_after_independent_todo_commit_aborts_turn(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"mixed-unknown-{uuid4().hex[:10]}"
    committed = asyncio.Event()
    from app.telegram import durable
    original_commit = durable.commit_todo_tool_operation

    async def signal_after_commit(*args, **kwargs):
        result = await original_commit(*args, **kwargs)
        committed.set()
        return result

    class CrashingCalendar:
        async def search_events(self, *_):
            await asyncio.wait_for(committed.wait(), timeout=5)
            raise RuntimeError("synthetic programming failure")

    monkeypatch.setattr(durable, "commit_todo_tool_operation", signal_after_commit)
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", CrashingCalendar)
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id,
    })
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "calendar_events", "id": "unknown-read",
             "args": {"query": "test", "start": "2026-12-10T00:00:00+08:00",
                      "end": "2026-12-10T23:59:59+08:00"}},
            {"name": "create_todo", "id": "committed-todo", "args": {"title": title}},
        ]),
        AIMessage(content="This final reply must never be generated."),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(model=model, session_factory=factory,
                                        settings=settings, checkpointer=saver,
                                        interrupt_mode="per_call")
        thread = f"web:ordinary:{user.id}"
        with pytest.raises(RuntimeError, match="synthetic programming failure"):
            await run_agent(graph, user.id, user.telegram_id, "",
                            f"创建无时间待办 {title}，并查日历。", origin="web",
                            turn_id=f"unknown-{uuid4().hex}", thread_id=thread)
        assert committed.is_set()
        snapshot = await graph.aget_state({"configurable": {
            "thread_id": f"official-v1:{thread}"}})
        assert snapshot.next
        assert not any(isinstance(item, AIMessage) and
                       "This final reply" in str(item.content)
                       for item in snapshot.values["messages"])
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            assert len(rows) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel_reminder", "update_todo"])
async def test_todo_business_not_found_is_error_tool_message(
        pg_harness, isolated_pg18_target, operation):
    factory, new_user, _ = pg_harness
    user = await new_user()
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": operation, "id": "missing-reminder",
             "args": {"todo_id": "no-such-owned-todo"}},
        ]),
        AIMessage(content="没有找到可取消提醒的待办。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(model=model, session_factory=factory,
                                        settings=get_settings(), checkpointer=saver,
                                        interrupt_mode="per_call")
        reply, state = await run_agent(
            graph, user.id, user.telegram_id, "", "取消那个不存在待办的提醒。",
            origin="web", turn_id=f"missing-reminder-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = [item for item in state["messages"] if isinstance(item, ToolMessage)]
        assert len(messages) == 1
        assert messages[0].tool_call_id == "missing-reminder"
        assert messages[0].status == "error"
        assert "no-such-owned-todo" in messages[0].content
        assert "没有找到" in reply
        async with factory() as db:
            assert not (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel_reminder", "update_todo"])
async def test_todo_success_result_remains_success_tool_message(
        pg_harness, isolated_pg18_target, operation):
    factory, new_user, _ = pg_harness
    user = await new_user()
    original_title = "runtime-success-" + uuid4().hex[:8]
    async with factory() as db:
        todo = await crud.create_todo(
            db, user.id, original_title,
            reminder_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
        await db.commit()
        todo_id = todo.id
    call_id = "successful-" + operation
    args = {"todo_id": todo_id[:8]}
    if operation == "update_todo":
        args["title"] = "renamed-" + uuid4().hex[:8]
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": operation, "id": call_id,
                                            "args": args}]),
        AIMessage(content="操作已完成。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(model=model, session_factory=factory,
                                        settings=get_settings(), checkpointer=saver,
                                        interrupt_mode="per_call")
        _, state = await run_agent(
            graph, user.id, user.telegram_id, "",
            f"请处理我的待办 {original_title}：{operation}",
            origin="web", turn_id=f"success-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = [item for item in state["messages"] if isinstance(item, ToolMessage)
                    and item.tool_call_id == call_id]
        assert len(messages) == 1 and messages[0].status == "success"
        async with factory() as db:
            row = await db.get(Todo, todo_id)
            assert row is not None
            if operation == "cancel_reminder":
                assert row.reminder_at is None and row.title == original_title
            else:
                assert row.title == args["title"]
    finally:
        await conn.close()
