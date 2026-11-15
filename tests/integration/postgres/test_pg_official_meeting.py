"""Current create_agent meeting draft, Todo continuation, and real PG Saver.

The Calendar and synthesis model are synthetic; all local business rows and
checkpoints use the identity-checked disposable PostgreSQL fixture.
"""
from __future__ import annotations

from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import func, select

from app.agent import official_read_tools, unified_tools, workflow_runtime
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.official_approval_view import (official_decision_for_action,
                                              pending_official_approvals)
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.agent.tools import make_todo_tools
from app.config import get_settings
from app.db import crud
from app.db.models import ActionExecution, MeetingDraft, Todo
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel
from tests.integration.postgres.test_pg_meeting_drafts import Calendar, Model

pytestmark = [pytest.mark.integration,
              pytest.mark.skip(reason="Retired meeting draft Agent tool and Mini App review page")]


@pytest.mark.asyncio
async def test_official_meeting_draft_then_todo_and_same_turn_replay(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    title = f"Meeting follow-up {uuid4().hex[:8]}"
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": owner.telegram_id,
        "gmail_enabled": False,
    })
    synthesis = Model()
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "_build_chat_llm", lambda *_a, **_k: synthesis)
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: Calendar())
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "prepare_meeting", "id": "prepare-1", "args": {"event_id": "event-1"}},
            {"name": "create_todo", "id": "create-1", "args": {"title": title}},
        ]),
        AIMessage(content=f"会议草稿待审核，已创建待办 {title}。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, tools = build_official_agent(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=settings, checkpointer=saver)
        assert "prepare_meeting" in {tool.name for tool in tools}
        assert not {"meeting_evidence", "submit_meeting_brief"} & {
            tool.name for tool in tools}
        thread = f"web:ordinary:{owner.id}"
        turn = f"web-request:{uuid4().hex}"
        request = f"为 event-1 准备会议草稿并创建待办 {title}，不设置时间"
        reply, result = await run_agent(
            graph, owner.id, owner.telegram_id, "", request,
            thread_id=thread, turn_id=turn, origin="web", return_state=True)
        replay = await run_agent(
            graph, owner.id, owner.telegram_id, "", request,
            thread_id=thread, turn_id=turn, origin="web")
        assert replay == reply
        assert "待审核" in reply and title in reply
        messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
        assert {item.tool_call_id for item in messages} == {"prepare-1", "create-1"}
        assert all(item.status != "error" for item in messages)
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(MeetingDraft).where(
                MeetingDraft.user_id == owner.id,
                MeetingDraft.status == "NEEDS_REVIEW")) == 1
            assert await db.scalar(select(func.count()).select_from(Todo).where(
                Todo.user_id == owner.id, Todo.title == title)) == 1
        snapshot = await graph.aget_state({
            "configurable": {"thread_id": f"official-v1:{thread}"}})
        assert not snapshot.next and not snapshot.interrupts
        assert synthesis.calls == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_official_meeting_draft_remains_reviewable_after_delete_decision(
        pg_harness, isolated_pg18_target, monkeypatch, decision):
    from langgraph.types import Command

    factory, new_user, _ = pg_harness
    owner = await new_user()
    title = f"Meeting delete target {uuid4().hex[:8]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=owner.id, title=title)
        await db.commit()
        todo_id = todo.id
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": owner.telegram_id,
        "gmail_enabled": False,
    })
    synthesis = Model()
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "_build_chat_llm", lambda *_a, **_k: synthesis)
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: Calendar())
    monkeypatch.setattr(workflow_runtime, "_action_executor",
                        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "prepare_meeting", "id": "prepare-1", "args": {"event_id": "event-1"}},
            {"name": "propose_todo_delete", "id": "delete-1", "args": {"target": title}},
            {"name": "list_todos", "id": "list-1", "args": {}},
        ]),
        AIMessage(content=f"会议草稿待审核；删除决定 {decision}；待办已查询。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=settings, checkpointer=saver)
        thread = f"web:ordinary:{owner.id}"
        _, initial = await run_agent(
            graph, owner.id, owner.telegram_id, "",
            f"为 event-1 准备会议草稿，删除待办 {title}，再列待办",
            thread_id=thread, turn_id=f"web-request:{uuid4().hex}",
            origin="web", return_state=True)
        assert initial.get("__interrupt__")
        config = {"configurable": {"thread_id": f"official-v1:{thread}"}}
        paused = await graph.aget_state(config)
        approvals = pending_official_approvals(paused, user_id=owner.id)
        assert len(approvals) == 1
        async with factory() as db:
            draft = await db.scalar(select(MeetingDraft).where(MeetingDraft.user_id == owner.id))
            assert draft is not None and draft.status == "NEEDS_REVIEW"
            assert await db.get(Todo, todo_id) is not None
        payload = official_decision_for_action(
            paused, user_id=owner.id, action_id=approvals[0].proposal.action_id,
            decision=decision,
            checkpoint_id=paused.config["configurable"]["checkpoint_id"])
        finished = await graph.ainvoke(
            Command(resume=payload), config=config,
            context={"user_id": owner.id, "telegram_id": owner.telegram_id,
                     "chat_id": "", "origin": "web"})
        messages = [item for item in finished["messages"] if isinstance(item, ToolMessage)]
        assert {item.tool_call_id for item in messages} == {
            "prepare-1", "delete-1", "list-1"}
        assert "待审核" in finished["messages"][-1].content
        assert synthesis.calls == 1
        async with factory() as db:
            assert (await db.get(Todo, todo_id) is None) == (decision == "approve")
            assert (await db.get(MeetingDraft, draft.id)).status == "NEEDS_REVIEW"
            execution = await db.get(ActionExecution, approvals[0].proposal.action_id)
            assert (execution is not None and execution.status == "executed") == (
                decision == "approve")
    finally:
        await conn.close()
