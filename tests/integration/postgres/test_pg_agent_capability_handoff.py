"""Agent high-level action handoff against an identity-checked isolated PG18."""
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.agent.capability_tools import prepare_action_request, capability_scope
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.planner import RoutedRequest
from app.agent.workflow import build_productivity_graph, read_workflow_turn
from app.config import get_settings
from app.db import crud
from app.db.models import ActionExecution, Todo
from tests.test_workflow import FakeCalendar


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skip(reason=(
        "Historical Router/Workflow handoff only; current create_agent "
        "approval and replay are covered by test_pg_official_approval_entry.py")),
]


@pytest.mark.asyncio
async def test_agent_handoff_creates_one_pending_proposal_without_delete(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"synthetic agent handoff {uuid4().hex[:8]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id = todo.id

    calendar = FakeCalendar()
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        workflow = build_productivity_graph(
            calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(factory), factory), saver)
        monkeypatch.setattr("app.agent.workflow_runtime.get_productivity_graph",
                            lambda: _return(workflow))
        turn = f"web-request:{uuid4()}"
        route = RoutedRequest(intent="todo", todo_operation="delete", title=title)
        with capability_scope(route=route.model_dump(), request=f"删除任务：{title}",
                              user_id=user.id, telegram_id=user.telegram_id,
                              turn_id=turn, origin="web"):
            first = await prepare_action_request.ainvoke({})
            state = await read_workflow_turn(workflow, first["thread_id"],
                                             turn_id=turn, user_id=user.id)
            assert state["__interrupt__"][0].value["kind"] == "action_proposal"
            action_id = state["proposal"]["action_id"]
            replay = await prepare_action_request.ainvoke({})
            assert replay == first
            replay_state = await read_workflow_turn(workflow, replay["thread_id"],
                                                    turn_id=turn, user_id=user.id)
            assert replay_state["proposal"]["action_id"] == action_id

    async with factory() as db:
        assert await db.get(Todo, todo_id) is not None
        assert await db.scalar(select(ActionExecution).where(
            ActionExecution.action_id == action_id)) is None


async def _return(value):
    return value


@pytest.mark.asyncio
async def test_real_agent_graph_handoff_replays_without_second_action(pg_harness, monkeypatch):
    from app.agent import graph as agent_graph
    from app.agent import nodes

    class CapabilityModel:
        def __init__(self):
            self.calls = 0
            self.bindings = []

        def bind_tools(self, tools):
            self.bindings.append([tool.name for tool in tools])
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            return AIMessage(content="", tool_calls=[{
                "name": "prepare_action_request", "args": {},
                "id": f"synthetic-call-{self.calls}", "type": "tool_call",
            }])

    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"synthetic graph handoff {uuid4().hex[:8]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id = todo.id
    model = CapabilityModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(agent_graph, "discover_calendar_tools", AsyncMock(return_value=[]))
    calendar = FakeCalendar()
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        workflow = build_productivity_graph(
            calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(factory), factory), saver)
        monkeypatch.setattr("app.agent.workflow_runtime.get_productivity_graph",
                            lambda: _return(workflow))
        ordinary = await agent_graph.build_graph(factory, durable=True)
        turn = f"web-request:{uuid4()}"
        route = RoutedRequest(intent="todo", todo_operation="delete", title=title)
        thread = f"web:ordinary:{user.id}"
        try:
            args = (ordinary, user.id, user.telegram_id, "", f"删除任务：{title}")
            _, first = await agent_graph.run_agent(
                *args, thread_id=thread, turn_id=turn, origin="web",
                route=route, return_state=True)
            _, replay = await agent_graph.run_agent(
                *args, thread_id=thread, turn_id=turn, origin="web",
                route=route, return_state=True)
            assert model.bindings[-1] == ["prepare_action_request"]
            assert model.calls == 1
            assert first["workflow_handoff"] == replay["workflow_handoff"]
            handoff = first["workflow_handoff"]
            state = await read_workflow_turn(workflow, handoff["thread_id"],
                                             turn_id=turn, user_id=user.id)
            assert state["__interrupt__"][0].value["kind"] == "action_proposal"
            async with factory() as db:
                assert await db.get(Todo, todo_id) is not None
                assert await db.scalar(select(ActionExecution).where(
                    ActionExecution.action_id == state["proposal"]["action_id"])) is None
        finally:
            await ordinary.checkpointer.conn.close()
            agent_graph.clear_graph_runtime()
