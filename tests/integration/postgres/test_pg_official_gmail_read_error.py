"""A failed authorized Gmail read must not hide a committed independent Todo."""

from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from app.agent import official_read_tools, unified_tools
from app.agent.gmail import GmailError
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.config import get_settings
from app.db.models import Todo
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel


pytestmark = pytest.mark.integration


class FailingGmail:
    async def unread(self, query, limit=5):
        raise GmailError("private OAuth refresh detail")


@pytest.mark.asyncio
async def test_gmail_read_failure_is_paired_with_independent_todo_commit(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Gmail-read-failure-{uuid4().hex[:10]}"
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "gmail_unread", "id": "mail-read", "args": {}},
            {"name": "create_todo", "id": "todo-create", "args": {"title": title}},
        ]),
        AIMessage(content=f"已创建 {title}；邮件读取失败，不能确认未读数量。"),
    ])
    settings = get_settings().model_copy(update={
        "gmail_enabled": True, "gmail_owner_telegram_id": user.telegram_id})
    monkeypatch.setattr(official_read_tools, "get_gmail_source", lambda: FailingGmail())
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
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
            f"查未读邮件，并创建不设时间的待办『{title}』；分别报告结果。",
            origin="web", turn_id=f"gmail-failure-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = {item.tool_call_id: item for item in state["messages"]
                    if isinstance(item, ToolMessage)}
        assert messages["mail-read"].status == "error"
        assert "未获得可靠结果" in messages["mail-read"].content
        assert "gmail_unread" in messages["mail-read"].content
        assert "read_failed" in messages["mail-read"].content
        assert "OAuth" not in messages["mail-read"].content
        assert messages["todo-create"].status != "error"
        assert "邮件读取失败" in reply and title in reply
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            assert len(rows) == 1
            assert rows[0].due_date is None and rows[0].reminder_at is None
    finally:
        await conn.close()
