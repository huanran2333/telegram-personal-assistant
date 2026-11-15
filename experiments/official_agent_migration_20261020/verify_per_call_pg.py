"""Isolated PG18 verification of official per-call HITL across Saver restarts.

Synthetic tools only. Refuses a connection unless both configured URLs and the
live server identity match the known disposable clone.
"""
from __future__ import annotations

import asyncio
import os
import selectors
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, TypedDict
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import psycopg
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.db.isolated_test_target import (
    validate_isolated_pg18_configuration, validate_isolated_pg18_target,
)
from experiments.official_agent_migration_20261020.test_official_runtime_contract import (
    ScriptedModel, _synthetic_writes, owner_scoped_read, owner_scoped_write,
)
from app.agent import unified_tools, workflow_runtime
from app.agent.actions import ActionResult, ActionStatus
from app.agent.official_approval_view import (pending_official_approvals,
                                              official_decision_for_action,
                                              committed_official_web_decision)
from app.web.runs import _agent_approval_result
from experiments.official_agent_migration_20261020.official_todo_approval import (
    ApprovalState, TodoApprovalPreflight, make_approved_todo_delete_tool,
    needs_todo_review, todo_review_description,
)
from experiments.official_agent_migration_20261020.recent_history_migration import (
    select_recent_history,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIR = (ROOT / ".local-tools" / "r3_ablation_pg18_copy_20261017").resolve()
EXPECTED_SYSTEM_ID = "7696790601313522276"


class _OldConversation(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    telegram_id: str
    conversation_thread_id: str


def _graph(model, saver):
    return create_agent(
        model=model,
        tools=[owner_scoped_read, owner_scoped_write],
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={"owner_scoped_write": {
                "allowed_decisions": ["approve", "reject"]}},
            interrupt_mode="per_call")],
        checkpointer=saver,
    )


@asynccontextmanager
async def _synthetic_session():
    yield object()


def _project_graph(model, saver):
    return create_agent(
        model=model,
        tools=[make_approved_todo_delete_tool()],
        state_schema=ApprovalState,
        context_schema=dict,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={
            "propose_todo_delete": {
                "allowed_decisions": ["approve", "reject"],
                "when": needs_todo_review,
                "description": todo_review_description}},
            interrupt_mode="per_call"),
            TodoApprovalPreflight(_synthetic_session)],
        checkpointer=saver,
    )


async def _checked_connection():
    async_url = os.environ["DATABASE_URL"]
    sync_url = os.environ["DATABASE_URL_SYNC"]
    validate_isolated_pg18_configuration(async_url, sync_url, str(EXPECTED_DIR),
                                         repository_root=ROOT)
    conn = await psycopg.AsyncConnection.connect(sync_url, autocommit=True)
    try:
        row = await (await conn.execute(
            "SELECT current_setting('data_directory'), current_database(), "
            "current_setting('port'), current_setting('server_version')")).fetchone()
        validate_isolated_pg18_target(async_url, sync_url, str(EXPECTED_DIR),
                                      *row, repository_root=ROOT)
        actual_id = await (await conn.execute(
            "SELECT system_identifier FROM pg_control_system()" )).fetchone()
        if str(actual_id[0]) != EXPECTED_SYSTEM_ID:
            raise RuntimeError("Isolated PG18 system identifier mismatch")
        print(f"identity=PASS database={row[1]} port={row[2]} version={row[3]} "
              f"system_identifier={actual_id[0]}")
        return conn
    except BaseException:
        await conn.close()
        raise


async def main():
    _synthetic_writes.clear()
    thread = f"official-per-call-pg-verification-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread}}
    # Refuse to overwrite an earlier run of this exact test thread.
    conn = await _checked_connection()
    try:
        saver = AsyncPostgresSaver(conn)
        if await saver.aget_tuple(config) is not None:
            raise RuntimeError("Verification thread already exists")
        print(f"synthetic_thread={thread}")
        model = ScriptedModel([AIMessage(content="", tool_calls=[
            {"name": "owner_scoped_read", "args": {"value": "check"}, "id": "read-1"},
            {"name": "owner_scoped_write", "args": {"value": "first"}, "id": "write-1"},
            {"name": "owner_scoped_write", "args": {"value": "second"}, "id": "write-2"},
        ])])
        graph = _graph(model, saver)
        await graph.ainvoke({"messages": [HumanMessage(content="synthetic goals")]},
                            config=config, context={"user_id": "synthetic-owner"})
        paused = await graph.aget_state(config)
        assert {item.value["tool_call_id"] for item in paused.interrupts} == {
            "write-1", "write-2"}
        assert {item.tool_call_id for item in paused.values["messages"]
                if isinstance(item, ToolMessage)} == {"read-1"}
        assert _synthetic_writes == []
        first_id = next(item.id for item in paused.interrupts
                        if item.value["tool_call_id"] == "write-1")
        second_id = next(item.id for item in paused.interrupts
                         if item.value["tool_call_id"] == "write-2")
    finally:
        await conn.close()

    await _generic_resume(config, first_id, second_id)
    # The project-specific preflight and UI projection must also survive an
    # actual PostgreSQL Saver restart, not just InMemorySaver unit tests.
    await _project_approval_restart()
    await _history_migration_pg()


async def _project_approval_restart():
    thread = f"official-project-approval-pg-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread}}
    context = {"user_id": "synthetic-owner", "telegram_id": "synthetic-tg",
               "chat_id": "synthetic-chat", "origin": "telegram"}
    row = SimpleNamespace(id="synthetic-todo-1", title="Synthetic Todo", version=7)

    async def execute(proposal):
        return ActionResult(action_id=proposal.action_id,
                            status=ActionStatus.EXECUTED, message="synthetic done")

    executor = SimpleNamespace(execute=AsyncMock(side_effect=execute))
    with (patch.object(unified_tools.crud, "list_todos", AsyncMock(return_value=[row])),
          patch.object(workflow_runtime, "get_action_executor", return_value=executor)):
        conn = await _checked_connection()
        try:
            graph = _project_graph(ScriptedModel([AIMessage(content="", tool_calls=[
                {"name": "propose_todo_delete", "args": {"target": "Synthetic Todo"},
                 "id": "delete-1"}])]), AsyncPostgresSaver(conn))
            await graph.ainvoke({
                "messages": [HumanMessage(id="synthetic-turn", content="Delete Synthetic Todo")],
                "user_id": "synthetic-owner", "telegram_id": "synthetic-tg",
                "chat_id": "synthetic-chat", "request_origin": "telegram",
            }, config=config, context=context)
            paused = await graph.aget_state(config)
            pending = pending_official_approvals(paused, user_id="synthetic-owner")
            assert len(pending) == 1
            status, presentation = await _agent_approval_result(
                graph, thread, {"__interrupt__": paused.interrupts})
            assert status == "WAITING_APPROVAL"
            assert [item["action_id"] for item in presentation["pending_approvals"]] == [
                pending[0].proposal.action_id]
            assert pending[0].tool_call_id == "delete-1"
            assert pending[0].proposal.args["todo_id"] == "synthetic-todo-1"
            assert pending[0].proposal.args["approved_todo_version"] == 7
            assert executor.execute.await_count == 0
            interrupt_id = pending[0].interrupt_id
            checkpoint_id = paused.config["configurable"]["checkpoint_id"]
            action_id = pending[0].proposal.action_id
        finally:
            await conn.close()

        conn = await _checked_connection()
        try:
            graph = _project_graph(ScriptedModel([AIMessage(content="finished")]),
                                   AsyncPostgresSaver(conn))
            paused = await graph.aget_state(config)
            assert pending_official_approvals(paused, user_id="synthetic-owner")[0].interrupt_id == interrupt_id
            try:
                pending_official_approvals(paused, user_id="different-owner")
            except PermissionError:
                pass
            else:
                raise AssertionError("Cross-owner approval projection succeeded")
            try:
                await graph.ainvoke(Command(resume={interrupt_id: {"type": "approve"}}),
                                    config=config,
                                    context={**context, "user_id": "different-owner"})
            except PermissionError:
                pass
            else:
                raise AssertionError("Cross-owner approval resume succeeded")
            assert executor.execute.await_count == 0
            decision = official_decision_for_action(
                paused, user_id="synthetic-owner",
                action_id=pending[0].proposal.action_id, decision="approve",
                checkpoint_id=paused.config["configurable"]["checkpoint_id"])
            assert decision == {interrupt_id: {"type": "approve"}}
            result = await graph.ainvoke(Command(resume=decision),
                                         config={**config, "metadata": {
                                             "web_request_id": "synthetic-web-decision",
                                             "approval_action_id": action_id,
                                             "approval_decision": "approve"}},
                                         context=context)
            assert executor.execute.await_count == 1
            proposal = executor.execute.await_args.args[0]
            assert proposal.args["user_id"] == "synthetic-owner"
            assert proposal.args["approved_todo_version"] == 7
            assert {item.tool_call_id for item in result["messages"]
                    if isinstance(item, ToolMessage)} == {"delete-1"}
            print("project_pg_restart=PASS owner-bound=1 version-bound=7 "
                  "cross-owner-denied=1 approved-once=1")
        finally:
            await conn.close()

        conn = await _checked_connection()
        try:
            graph = _project_graph(ScriptedModel([AIMessage(content="finished")]),
                                   AsyncPostgresSaver(conn))
            committed = await committed_official_web_decision(
                graph, thread, user_id="synthetic-owner", action_id=action_id,
                decision="approve", request_id="synthetic-web-decision",
                checkpoint_id=checkpoint_id)
            assert committed is not None
            assert await committed_official_web_decision(
                graph, thread, user_id="synthetic-owner", action_id=action_id,
                decision="approve", request_id="other-entry",
                checkpoint_id=checkpoint_id) is None
            assert executor.execute.await_count == 1
            print("official_web_crash_reconciliation_pg=PASS "
                  "origin-marker=1 paired-result=1 no-replay=1")
        finally:
            await conn.close()

async def _generic_resume(config, first_id, second_id):
    conn = await _checked_connection()
    try:
        graph = _graph(ScriptedModel([AIMessage(content="finished")]),
                       AsyncPostgresSaver(conn))
        persisted = await graph.aget_state(config)
        assert {item.id for item in persisted.interrupts} == {first_id, second_id}
        await graph.ainvoke(Command(resume={first_id: {"type": "approve"}}),
                            config=config, context={"user_id": "synthetic-owner"})
        middle = await graph.aget_state(config)
        assert _synthetic_writes == [("synthetic-owner", "first")]
        assert {item.tool_call_id for item in middle.values["messages"]
                if isinstance(item, ToolMessage)} == {"read-1", "write-1"}
        assert any(item.id == second_id for item in middle.interrupts)
        assert middle.config["configurable"]["checkpoint_id"] == (
            persisted.config["configurable"]["checkpoint_id"])
        print("partial_resume_checkpoint_unchanged=1 paired_tool_result=1")
    finally:
        await conn.close()

    conn = await _checked_connection()
    try:
        graph = _graph(ScriptedModel([AIMessage(content="finished")]),
                       AsyncPostgresSaver(conn))
        result = await graph.ainvoke(
            Command(resume={second_id: {"type": "reject"}}),
            config=config, context={"user_id": "synthetic-owner"})
        replies = {item.tool_call_id: item for item in result["messages"]
                   if isinstance(item, ToolMessage)}
        assert set(replies) == {"read-1", "write-1", "write-2"}
        assert replies["write-2"].status == "error"
        assert _synthetic_writes == [("synthetic-owner", "first")]
        assert not (await graph.aget_state(config)).interrupts
        print("per_call_pg_restart=PASS read-before-review=1 approved-once=1 "
              "rejected-without-write=1 paired-tool-messages=3")
    finally:
        await conn.close()


async def _history_migration_pg():
    old_thread = f"user_synthetic_migration_{uuid4().hex}"
    new_thread = f"official-v1:{old_thread}"
    old_config = {"configurable": {"thread_id": old_thread}}
    new_config = {"configurable": {"thread_id": new_thread}}
    old_messages = [
        HumanMessage(id="old-human", content="Find a synthetic fact"),
        AIMessage(id="old-model-call", content="", tool_calls=[{
            "name": "owner_scoped_read", "args": {"value": "old"}, "id": "old-call"}]),
        ToolMessage(id="old-tool-result", content="synthetic-owner:old",
                    tool_call_id="old-call"),
        AIMessage(id="old-final", content="The old value is old"),
    ]
    conn = await _checked_connection()
    try:
        saver = AsyncPostgresSaver(conn)
        builder = StateGraph(_OldConversation)
        builder.add_node("done", lambda state: {})
        builder.set_entry_point("done")
        legacy = builder.compile(checkpointer=saver)
        await legacy.ainvoke({"messages": old_messages, "user_id": "synthetic-owner",
                              "telegram_id": "synthetic-tg",
                              "conversation_thread_id": old_thread}, config=old_config)
        assert not (await saver.aget_tuple(new_config))
        migration = select_recent_history(await legacy.aget_state(old_config),
                                          user_id="synthetic-owner", thread_id=old_thread,
                                          telegram_id="synthetic-tg",
                                          suppressed_message_ids=frozenset(),
                                          suppressed_terms=())
        assert list(migration.messages) == old_messages
    finally:
        await conn.close()

    conn = await _checked_connection()
    try:
        saver = AsyncPostgresSaver(conn)
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[{
                "name": "owner_scoped_read", "args": {"value": "fresh"},
                "id": "new-call"}]),
            AIMessage(content="New answer"),
        ])
        graph = create_agent(model=model, tools=[owner_scoped_read],
                             checkpointer=saver)
        # The new graph drops old-schema owner channels. Migration therefore
        # MUST use the old graph as reader; raw Saver presence blocks fallback.
        inspected = await graph.aget_state(old_config)
        assert inspected.values.get("user_id") is None
        assert await saver.aget_tuple(old_config) is not None
        result = await graph.ainvoke({"messages": [
            *migration.messages, HumanMessage(id="new-human", content="Continue")]},
            config=new_config, context={"user_id": "synthetic-owner"})
        tool_ids = [item.tool_call_id for item in result["messages"]
                    if isinstance(item, ToolMessage)]
        assert tool_ids == ["old-call", "new-call"]
        assert [item.tool_call_id for item in model._received[0]
                if isinstance(item, ToolMessage)] == ["old-call"]
        assert result["messages"][-1].content == "New answer"
        assert (await saver.aget_tuple(old_config)) is not None
        print("history_migration_pg=PASS old-tool-not-replayed=1 "
              "recent-messages-present=1 old-saver-intact=1")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main(), loop_factory=lambda: asyncio.SelectorEventLoop(
        selectors.SelectSelector()))
