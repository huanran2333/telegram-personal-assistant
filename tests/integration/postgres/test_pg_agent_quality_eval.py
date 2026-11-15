"""Optional real-model synthetic task-effect checks against the isolated PG18.

The shared fixture proves PG18 database, port and data_directory before DDL.
No Telegram sender, Scheduler, Gmail or Google provider is started.
"""
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.agent.graph import build_graph, run_agent
from app.db.models import Todo
from evals.quality_graders import grade_todo_environment, grade_tool_trajectory

pytestmark = [pytest.mark.integration, pytest.mark.llm_integration]


def snapshot(rows):
    return [{"id": row.id, "user_id": row.user_id, "title": row.title,
             "status": row.status.value} for row in rows]


@pytest.mark.asyncio
@pytest.mark.parametrize("turns,expected_calls,expected_status", [
    (["创建任务：{title}"], ["create_todo"], "pending"),
    (["创建任务：{title}", "列出我的任务。"], ["create_todo", "list_todos"], "pending"),
    (["创建任务：{title}", "把“{title}”标记完成。"], ["create_todo", "update_todo"], "done"),
])
async def test_real_agent_todo_postcondition_and_trace(pg_harness, monkeypatch,
                                                        turns, expected_calls, expected_status):
    import app.agent.graph as graph_module
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Synthetic eval {uuid4().hex[:12]}"
    thread_id = f"agent-quality-{uuid4().hex}"

    # Actual model + Todo tools + PostgreSQL checkpoint/business state remain
    # under test. The current graph does not discover Calendar MCP at build time.
    graph = await build_graph(factory, durable=True)
    try:
        async with factory() as db:
            before = snapshot((await db.scalars(select(Todo).where(Todo.user_id == user.id))).all())
        for index, turn in enumerate(turns):
            await run_agent(graph, user.id, user.telegram_id, user.telegram_id,
                            turn.format(title=title), thread_id=thread_id,
                            turn_id=f"synthetic-eval-{index}")
        async with factory() as db:
            after = snapshot((await db.scalars(select(Todo).where(Todo.user_id == user.id))).all())
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        actual_calls = [{"name": call["name"], "args": call.get("args", {})}
                        for message in state.values["messages"] if isinstance(message, AIMessage)
                        for call in (message.tool_calls or [])]
        environment = grade_todo_environment(before, after, owner_id=user.id,
                                             expected={title: expected_status})
        trajectory = grade_tool_trajectory(actual_calls,
                                           [{"name": name, "args": {}}
                                            for name in expected_calls], ordered=True)
        assert environment["passed"], {"environment": environment, "tools": [x["name"] for x in actual_calls]}
        assert trajectory["passed"], trajectory
    finally:
        checkpointer = graph.checkpointer
        connection = getattr(checkpointer, "conn", None)
        if connection is not None:
            await connection.close()
        graph_module.clear_graph_runtime()
