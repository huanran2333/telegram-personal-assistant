"""A proposal read failure must not abort an independent ToolNode write."""

from uuid import uuid4

import httpx
import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from app.agent import unified_tools, workflow_runtime
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.config import get_settings
from app.db.models import Todo
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

pytestmark = pytest.mark.integration


class UnavailableCalendar:
    def __init__(self):
        self.reads = 0

    async def availability(self, *_):
        self.reads += 1
        try:
            raise CalendarProviderError("provider_unavailable", 503)
        except CalendarProviderError as cause:
            raise CalendarError("private OAuth token=SECRET") from cause


class ForbiddenGmail:
    def __init__(self):
        self.reads = 0

    async def thread(self, *_):
        self.reads += 1
        request = httpx.Request("GET", "https://internal.example/token=SECRET")
        response = httpx.Response(403, request=request)
        try:
            raise httpx.HTTPStatusError("private token=SECRET", request=request,
                                        response=response)
        except httpx.HTTPStatusError as cause:
            raise GmailError("private token=SECRET") from cause


@pytest.mark.asyncio
async def test_calendar_preflight_503_keeps_independent_todo_and_final_feedback(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"preflight-independent-{uuid4().hex[:10]}"
    calendar = UnavailableCalendar()
    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: calendar)
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id,
    })
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "propose_calendar_create", "id": "calendar-proposal",
             "args": {"title": "test event", "start": "2026-12-10T15:00:00+08:00",
                      "end": "2026-12-10T16:00:00+08:00"}},
            {"name": "create_todo", "id": "independent-todo", "args": {"title": title}},
        ]),
        AIMessage(content=f"待办 {title} 已创建；日历预检服务不可用，日程未创建。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, session_factory=factory, settings=settings,
            checkpointer=saver, interrupt_mode="per_call")
        reply, state = await run_agent(
            graph, user.id, user.telegram_id, "",
            f"创建无时间待办 {title}，并申请新建 12 月 10 日 15:00 的日程。",
            origin="web", turn_id=f"preflight-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = {item.tool_call_id: item for item in state["messages"]
                    if isinstance(item, ToolMessage)}
        assert state.get("__interrupt__") is None
        assert set(messages) == {"calendar-proposal", "independent-todo"}
        assert messages["calendar-proposal"].status == "error"
        assert "服务暂时不可用" in messages["calendar-proposal"].content
        assert "未进入审批" in messages["calendar-proposal"].content
        assert "SECRET" not in messages["calendar-proposal"].content
        assert messages["independent-todo"].status == "success"
        assert title in reply and "日程未创建" in reply
        assert calendar.reads == 1
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            assert len(rows) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_gmail_preflight_403_has_no_approval_or_write(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    gmail = ForbiddenGmail()
    monkeypatch.setattr(workflow_runtime, "get_gmail_source", lambda: gmail)
    settings = get_settings().model_copy(update={
        "gmail_enabled": True, "gmail_owner_telegram_id": user.telegram_id,
    })
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "propose_email_todo", "id": "gmail-proposal",
             "args": {"thread_id": "thread", "message_id": "message",
                      "evidence_quote": "private evidence", "title": "from email"}},
        ]),
        AIMessage(content="邮件访问权限不足，未取得来源证据；没有提出审批，也没有创建待办。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, session_factory=factory, settings=settings,
            checkpointer=saver, interrupt_mode="per_call")
        reply, state = await run_agent(
            graph, user.id, user.telegram_id, "",
            "请根据那封邮件提出创建待办，先核对邮件来源。",
            origin="web", turn_id=f"preflight-gmail-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = [item for item in state["messages"] if isinstance(item, ToolMessage)]
        assert len(messages) == 1
        assert messages[0].tool_call_id == "gmail-proposal"
        assert messages[0].status == "error"
        assert "认证或访问权限不足" in messages[0].content
        assert "未进入审批" in messages[0].content
        assert "SECRET" not in messages[0].content
        assert "没有创建待办" in reply
        assert gmail.reads == 1
        async with factory() as db:
            assert not (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
    finally:
        await conn.close()
