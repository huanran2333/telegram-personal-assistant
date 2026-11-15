"""Official complete Agent graph against real disposable PG Saver and Todo."""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import psycopg
import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from sqlalchemy import select

from app.agent.official_graph import build_candidate
from app.agent import official_runtime, graph as legacy_graph
from app.agent.official_runtime import run_agent
from app.agent.tools import make_todo_tools
from app.config import get_settings
from app.db.models import Todo
from app.db.models import WebRequest
from app.web import runs
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel
from experiments.official_agent_migration_20261020.verify_per_call_pg import _OldConversation

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "interrupt_mode" not in inspect.signature(HumanInTheLoopMiddleware).parameters,
        reason="Requires isolated LangChain >=1.4.6 per-call HITL"),
]


@pytest.mark.asyncio
async def test_official_graph_creates_real_owned_todo_once(
        pg_harness, isolated_pg18_target):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Official integration {uuid4().hex[:12]}"
    call_id = f"create-{uuid4().hex}"
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "create_todo",
            "args": {"title": title, "description": "Synthetic integration",
                     "due_date_str": "", "planned_start_str": "",
                     "planned_end_str": "", "reminder_at_str": ""},
            "id": call_id}]),
        AIMessage(content=f"Created {title} without a time"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_candidate(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=get_settings(),
            checkpointer=saver, interrupt_mode="per_call")
        turn = f"turn-{uuid4().hex}"
        thread = f"web:ordinary:{user.id}"
        reply, result = await run_agent(
            graph, user.id, user.telegram_id, "", f"创建待办：{title}，不要设置时间",
            thread_id=thread, turn_id=turn, origin="web", return_state=True)
        assert title in reply
        messages = [message for message in result["messages"]
                    if isinstance(message, ToolMessage) and message.tool_call_id == call_id]
        assert len(messages) == 1 and messages[0].status != "error"
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            assert len(rows) == 1
            assert rows[0].due_date is None
            assert rows[0].planned_start_at is None
            assert rows[0].reminder_at is None
        replay = await run_agent(
            graph, user.id, user.telegram_id, "", f"创建待办：{title}，不要设置时间",
            thread_id=thread, turn_id=turn, origin="web")
        assert replay == reply
        async with factory() as db:
            assert len((await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_official_production_builder_has_durable_saver_and_legacy_reader(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    sync_url = isolated_pg18_target.set(drivername="postgresql").render_as_string(
        hide_password=False)
    settings = get_settings().model_copy(update={
        "database_url_sync": sync_url, "app_env": "test"})
    monkeypatch.setattr(official_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(legacy_graph, "settings", settings)
    model = ScriptedModel([AIMessage(content="Official production graph ready")])
    graph = await official_runtime.build_graph(
        factory, durable=True, model=model)
    try:
        assert isinstance(graph.checkpointer, AsyncPostgresSaver)
        assert official_runtime._legacy_reader is not None
        # Readiness scans *all* users in the disposable clone. Earlier
        # integration runs intentionally leave old interrupted checkpoints;
        # their presence must block cutover, not make this builder test depend
        # on unrelated test history. The guard has separate deterministic
        # contract tests.
        answer = await run_agent(
            graph, user.id, user.telegram_id, "", "Hello", origin="web",
            turn_id=f"turn-{uuid4().hex}", thread_id=f"web:ordinary:{user.id}")
        assert answer == "Official production graph ready"
        snapshot = await graph.aget_state({"configurable": {
            "thread_id": f"official-v1:web:ordinary:{user.id}"}})
        assert snapshot.values["user_id"] == user.id
    finally:
        await graph.checkpointer.conn.close()
        await official_runtime._legacy_reader.checkpointer.conn.close()
        official_runtime.clear_graph_runtime()


@pytest.mark.asyncio
async def test_official_first_turn_migrates_recent_old_messages_with_owner_check(
        pg_harness, isolated_pg18_target):
    factory, new_user, _ = pg_harness
    user = await new_user()
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        old_thread = f"user_{user.telegram_id}"
        old = StateGraph(_OldConversation)
        old.add_node("done", lambda state: {})
        old.set_entry_point("done")
        old_graph = old.compile(checkpointer=saver)
        old_turn = f"old-{uuid4().hex}"
        await old_graph.ainvoke({
            "messages": [HumanMessage(id=old_turn, content="记住我的合成偏好：回答简洁"),
                         AIMessage(content="知道了")],
            "user_id": user.id, "telegram_id": user.telegram_id,
            "conversation_thread_id": old_thread},
            config={"configurable": {"thread_id": old_thread}})
        model = ScriptedModel([AIMessage(content="已接上之前的对话")])
        official, _ = build_candidate(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=get_settings(),
            checkpointer=saver, interrupt_mode="per_call")
        class Suppression:
            async def suppressed_message_ids(self, *, user_id, thread_id):
                assert user_id == user.id and thread_id == old_thread
                return set()
            async def suppressed_terms(self, *, user_id):
                assert user_id == user.id
                return ()
        official_runtime._legacy_reader = old_graph
        official_runtime._migration_memory_orchestrator = Suppression()
        reply, state = await run_agent(
            official, user.id, user.telegram_id, user.telegram_id,
            "继续聊刚才的偏好", origin="telegram", return_state=True)
        assert reply == "已接上之前的对话"
        assert any(isinstance(message, HumanMessage) and message.id == old_turn
                   for message in state["messages"])
        assert len([message for message in state["messages"]
                    if isinstance(message, HumanMessage)]) == 2
        assert any(isinstance(message, HumanMessage) and message.id == old_turn
                   for message in model._received[0])
        assert await saver.aget_tuple({"configurable": {"thread_id": old_thread}})
    finally:
        official_runtime.clear_graph_runtime()
        await conn.close()


@pytest.mark.asyncio
async def test_webrequest_ordinary_chat_reaches_official_graph_and_real_todo(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Web official {uuid4().hex[:10]}"
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "create_todo", "id": "web-create-1",
            "args": {"title": title, "description": "", "due_date_str": "",
                     "planned_start_str": "", "planned_end_str": "",
                     "reminder_at_str": ""}}]),
        AIMessage(content=f"已创建 {title}"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_candidate(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=get_settings(),
            checkpointer=saver, interrupt_mode="per_call")
        monkeypatch.setattr(official_runtime, "get_graph",
                            AsyncMock(return_value=graph))
        from app.agent import workflow_runtime
        monkeypatch.setattr(
            workflow_runtime, "get_productivity_graph",
            AsyncMock(return_value=SimpleNamespace(
                aget_state=AsyncMock(return_value=SimpleNamespace(
                    values={}, next=())))))
        row, _ = await runs.submit_run(
            factory, user_id=user.id, operation="chat",
            idempotency_key=f"official-{uuid4().hex}",
            body={"text": f"创建待办 {title}，不要设置时间"})
        fence = str(uuid4())
        async with factory() as db:
            persisted = await db.get(WebRequest, row.id)
            persisted.state = "RUNNING"
            persisted.fence_token = fence
            await db.commit()
        result_state, result = await runs._execute(factory, row, fence)
        assert result_state == "COMPLETED"
        assert title in result["response"]
        async with factory() as db:
            updated = await db.get(WebRequest, row.id)
            assert updated.route_payload == {"agent_mode": "official-v1"}
            assert updated.thread_id == f"official-v1:web:ordinary:{user.id}"
            todos = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            assert len(todos) == 1 and todos[0].due_date is None
    finally:
        await conn.close()
