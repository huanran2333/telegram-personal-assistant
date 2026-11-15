"""The guarded tool pauses and resumes the original Agent graph."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.runtime import Runtime

from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel


def web_context():
    return {"user_id": "synthetic-user", "telegram_id": "owner",
            "chat_id": "", "origin": "web"}


def telegram_context():
    return {**web_context(), "chat_id": "owner", "origin": "telegram"}


@pytest.mark.asyncio
async def test_edit_replaces_only_prior_action_and_requires_fresh_approval(monkeypatch):
    from app.agent import agent_approval, planner, workflow, workflow_runtime
    old = ActionProposal(
        action_id="old-action", domain="calendar", operation="update_event",
        args={"user_id": "synthetic-user", "eventId": "synthetic-event",
              "start": "2030-01-01T12:00:00+08:00", "end": "2030-01-01T13:00:00+08:00"},
        human_readable_preview="Old time", reason="synthetic", risk_level=RiskLevel.EXTERNAL_WRITE)
    edited = old.model_copy(update={"action_id": "new-action", "human_readable_preview": "New time"})
    monkeypatch.setattr(planner, "apply_edit", AsyncMock(return_value=edited))
    monkeypatch.setattr(workflow, "bind_approved_calendar_version", AsyncMock())
    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: object())
    state = {"pending_agent_proposal": old.model_dump(),
             "pending_agent_call_id": "synthetic-call", "user_id": "synthetic-user",
             "telegram_id": "owner", "chat_id": "", "request_origin": "web",
             "agent_edit_instruction": "改时间", "edit_web_request_id": "web-edit"}
    result = await agent_approval.apply_agent_edit(state, Runtime(context=web_context()))
    assert result["pending_agent_proposal"]["action_id"] == "new-action"
    assert result["agent_action_outcomes"]["old-action"]["status"] == "replaced"
    assert result["approval_decision"] == ""
    assert "messages" not in result  # No tool result or write before the new approval.


@asynccontextmanager
async def no_db():
    yield object()


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.observed = []

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.observed.append(list(messages))
        return self.responses.pop(0)


async def setup_graph(monkeypatch, *, with_followup=False,
                      with_second_guarded=False, model_responses=None):
    from app.agent import graph, nodes, unified_loop

    proposal = ActionProposal(
        action_id="synthetic-action", domain="todo", operation="delete_todo",
        args={"user_id": "synthetic-user", "todo_id": "synthetic-todo",
              "approved_todo_version": 2, "title": "Synthetic task"},
        human_readable_preview="Delete Synthetic task", reason="synthetic test",
        risk_level=RiskLevel.DESTRUCTIVE)

    proposals = 0

    async def propose_todo_delete(target: str):
        nonlocal proposals
        assert target in {"Synthetic task", "synthetic-todo", "Other synthetic task"}
        proposals += 1
        next_proposal = (proposal if proposals == 1 else
                         proposal.model_copy(update={"action_id": "synthetic-action-2"}))
        if target == "Other synthetic task":
            next_proposal = next_proposal.model_copy(update={"args": {
                **next_proposal.args, "todo_id": "other-synthetic-todo",
                "title": "Other synthetic task"}})
        return {"approval_proposal": next_proposal.model_dump(),
                "response": "Awaiting approval"}

    async def list_todos():
        return "Synthetic remaining tasks"

    tools = [StructuredTool.from_function(
        coroutine=propose_todo_delete, name="propose_todo_delete",
        description="Propose deletion of one task")]
    if with_followup:
        tools.append(StructuredTool.from_function(
            coroutine=list_todos, name="list_todos", description="List current tasks"))
    calls = [{"name": "propose_todo_delete", "args": {"target": "Synthetic task"},
              "id": "delete-call", "type": "tool_call"}]
    if with_followup:
        calls.append({"name": "list_todos", "args": {}, "id": "read-call", "type": "tool_call"})
    if with_second_guarded:
        calls.append({"name": "propose_todo_delete", "args": {"target": "Other synthetic task"},
                      "id": "delete-call-2", "type": "tool_call"})
    model = Model(*(model_responses or (
        AIMessage(content="", tool_calls=calls), AIMessage(content="Finished"))))
    monkeypatch.setattr(graph.settings, "app_env", "test")
    monkeypatch.setattr(graph, "make_todo_tools", lambda _factory: [])
    monkeypatch.setattr(graph, "make_unified_tools", lambda *_a, **_k: tools)
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: Model())
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    compiled = await graph.build_graph(no_db, durable=False,
                                       test_checkpointer=InMemorySaver())
    return compiled, model, proposal


@pytest.mark.asyncio
async def test_reject_resumes_original_agent_without_executing(monkeypatch):
    from app.agent import graph
    from app.agent import workflow_runtime
    executor = AsyncMock()
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    compiled, model, proposal = await setup_graph(monkeypatch)
    _, first = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "删除 Synthetic task",
        thread_id="web:ordinary:synthetic-user", turn_id="web-request:one",
        origin="web", return_state=True)
    assert first["__interrupt__"][0].value["action_id"] == proposal.action_id
    snapshot = await compiled.aget_state(
        {"configurable": {"thread_id": "web:ordinary:synthetic-user"}})
    assert "agent_approval_gate" in snapshot.next
    assert not any(isinstance(message, ToolMessage) for message in snapshot.values["messages"])
    result = await compiled.ainvoke(Command(resume={"decision": "reject"}),
                                    {"configurable": {"thread_id": "web:ordinary:synthetic-user"}},
                                    context=web_context())
    assert result["proposal"]["status"] == "rejected"
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "delete-call"
               for message in result["messages"])
    assert model.observed[-1][-1].content != "删除 Synthetic task"
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_approval_resume_rejects_mismatched_runtime_identity(monkeypatch):
    from app.agent import graph, workflow_runtime
    executor = AsyncMock()
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    compiled, _, proposal = await setup_graph(monkeypatch)
    thread = "web:ordinary:synthetic-user:runtime-owner"
    await graph.run_agent(compiled, "synthetic-user", "owner", "", "删除 Synthetic task",
                          thread_id=thread, turn_id="web-request:runtime-owner", origin="web")
    with pytest.raises(PermissionError, match="AgentRuntimeOwnerMismatch"):
        await compiled.ainvoke(Command(resume={"decision": "approve"}),
                               {"configurable": {"thread_id": thread}},
                               context={**web_context(), "user_id": "other-user"})
    snapshot = await compiled.aget_state({"configurable": {"thread_id": thread}})
    assert "agent_approval_gate" in snapshot.next
    assert snapshot.values["proposal"]["action_id"] == proposal.action_id
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_turn_rejected_todo_cannot_be_reproposed_by_id(monkeypatch):
    from app.agent import graph, workflow_runtime
    executor = AsyncMock()
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    model_responses = (
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "args": {"target": "Synthetic task"},
            "id": "delete-by-title", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "args": {"target": "synthetic-todo"},
            "id": "delete-by-id", "type": "tool_call"}]),
        AIMessage(content="删除已被用户拒绝，任务保留。"),
    )
    compiled, model, proposal = await setup_graph(
        monkeypatch, model_responses=model_responses)
    thread = "web:ordinary:synthetic-user:reject-retry"
    _, pending = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "删除 Synthetic task",
        thread_id=thread, turn_id="web-request:reject-retry", origin="web",
        return_state=True)
    assert pending["__interrupt__"][0].value["action_id"] == proposal.action_id
    result = await compiled.ainvoke(Command(resume={"decision": "reject"}),
                                    {"configurable": {"thread_id": thread}}, context=web_context())
    assert not result.get("__interrupt__")
    replies = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert [item.tool_call_id for item in replies] == ["delete-by-title", "delete-by-id"]
    assert replies[0].status == replies[1].status == "error"
    assert "user rejected" in replies[0].content.lower()
    assert "do not propose" in replies[0].content.lower()
    assert result["agent_action_outcomes"][proposal.action_id]["status"] == "rejected"
    assert len(model.observed) == 3
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_delete_in_same_batch_is_blocked_but_other_target_is_allowed(monkeypatch):
    from app.agent import graph, workflow_runtime
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: AsyncMock())
    responses = (
        AIMessage(content="", tool_calls=[
            {"name": "propose_todo_delete", "args": {"target": "Synthetic task"},
             "id": "first-delete", "type": "tool_call"},
            {"name": "propose_todo_delete", "args": {"target": "synthetic-todo"},
             "id": "repeat-delete", "type": "tool_call"},
            {"name": "propose_todo_delete", "args": {"target": "Other synthetic task"},
             "id": "other-delete", "type": "tool_call"}]),
        AIMessage(content="Other target was reviewed separately."),
    )
    compiled, _, _ = await setup_graph(monkeypatch, model_responses=responses)
    thread = "web:ordinary:synthetic-user:batch-reject"
    await graph.run_agent(compiled, "synthetic-user", "owner", "", "删除两个待办",
                          thread_id=thread, turn_id="web-request:batch-reject", origin="web")
    result = await compiled.ainvoke(Command(resume={"decision": "reject"}),
                                    {"configurable": {"thread_id": thread}}, context=web_context())
    assert result["__interrupt__"][0].value["action_id"] != "synthetic-action"
    assert result["pending_agent_call_id"] == "other-delete"
    repeated = [item for item in result["messages"] if isinstance(item, ToolMessage)
                and item.tool_call_id == "repeat-delete"]
    assert len(repeated) == 1 and repeated[0].status == "error"


@pytest.mark.asyncio
async def test_new_explicit_user_turn_can_request_previously_rejected_delete(monkeypatch):
    from app.agent import graph, workflow_runtime
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: AsyncMock())
    responses = (
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "args": {"target": "Synthetic task"},
            "id": "first-turn", "type": "tool_call"}]),
        AIMessage(content="删除已拒绝。"),
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "args": {"target": "synthetic-todo"},
            "id": "new-turn", "type": "tool_call"}]),
    )
    compiled, _, _ = await setup_graph(monkeypatch, model_responses=responses)
    thread = "web:ordinary:synthetic-user:new-authorization"
    await graph.run_agent(compiled, "synthetic-user", "owner", "", "删除 Synthetic task",
                          thread_id=thread, turn_id="web-request:old", origin="web")
    await compiled.ainvoke(Command(resume={"decision": "reject"}),
                           {"configurable": {"thread_id": thread}}, context=web_context())
    _, pending = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "我现在明确要求再次删除 Synthetic task",
        thread_id=thread, turn_id="web-request:new", origin="web", return_state=True)
    assert pending["__interrupt__"][0].value["action_id"] == "synthetic-action-2"
    assert pending["pending_agent_call_id"] == "new-turn"


@pytest.mark.asyncio
async def test_approve_returns_tool_result_then_processes_remaining_call(monkeypatch):
    from app.agent import graph, workflow_runtime
    executor = AsyncMock()
    executor.execute.return_value = ActionResult(
        action_id="synthetic-action", status=ActionStatus.EXECUTED,
        message="Action completed")
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    compiled, model, proposal = await setup_graph(monkeypatch, with_followup=True)
    _, first = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "删除 Synthetic task，再列任务",
        thread_id="web:ordinary:synthetic-user", turn_id="web-request:two",
        origin="web", return_state=True)
    assert first["__interrupt__"][0].value["action_id"] == proposal.action_id
    result = await compiled.ainvoke(Command(resume={"decision": "approve"}),
                                    {"configurable": {"thread_id": "web:ordinary:synthetic-user"}},
                                    context=web_context())
    assert result["proposal"]["status"] == "executed"
    tool_results = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert [item.tool_call_id for item in tool_results] == ["delete-call", "read-call"]
    assert result["messages"][-1].content == "Finished"
    executor.execute.assert_awaited_once()
    assert model.observed[-1][-1].tool_call_id == "read-call"


@pytest.mark.asyncio
async def test_old_agent_approval_button_is_disabled_after_official_cutover(monkeypatch):
    from app.agent import graph, workflow_runtime
    from app.telegram import webhook
    executor = AsyncMock()
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    compiled, _, proposal = await setup_graph(monkeypatch)
    _, pending = await graph.run_agent(
        compiled, "synthetic-user", "owner", "owner", "删除 Synthetic task",
        thread_id="user_owner", turn_id="synthetic-turn", return_state=True)
    assert pending["__interrupt__"][0].value["action_id"] == proposal.action_id
    monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=compiled))
    monkeypatch.setattr(webhook, "get_user_by_telegram_id", AsyncMock(
        return_value=SimpleNamespace(id="synthetic-user")))
    legacy = AsyncMock(side_effect=AssertionError("Legacy approval graph must not run"))
    monkeypatch.setattr(webhook, "get_productivity_graph", legacy)
    sent = AsyncMock()
    monkeypatch.setattr(webhook, "_send_agent_result", sent)
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None
    monkeypatch.setattr(webhook, "AsyncSessionLocal", Session)
    notified = AsyncMock()
    monkeypatch.setattr(webhook, "_durable_callback_reply", notified)
    monkeypatch.setattr(webhook, "_record_no_business", lambda *_args: None)
    callback = SimpleNamespace(
        data=f"act:{proposal.action_id}:reject",
        from_user=SimpleNamespace(id="owner"),
        message=SimpleNamespace(chat_id="owner"), answer=AsyncMock())
    await webhook._handle_action_callback(callback)
    sent.assert_not_awaited()
    notified.assert_awaited_once()
    assert "旧版审批按钮已停用" in notified.await_args.args[1]
    snapshot = await compiled.aget_state({"configurable": {"thread_id": "user_owner"}})
    assert "agent_approval_gate" in snapshot.next
    executor.execute.assert_not_awaited()
    legacy.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_callback_stays_terminal_while_next_proposal_is_pending(monkeypatch):
    from app.agent import graph, workflow_runtime
    from app.telegram import webhook
    executor = AsyncMock()
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    compiled, _, first_proposal = await setup_graph(monkeypatch,
                                                     with_second_guarded=True)
    await graph.run_agent(
        compiled, "synthetic-user", "owner", "owner", "删除两项合成任务",
        thread_id="user_owner", turn_id="synthetic-two-actions")
    result = await compiled.ainvoke(Command(resume={"decision": "reject"}),
                                    {"configurable": {"thread_id": "user_owner"}},
                                    context=telegram_context())
    assert result["__interrupt__"][0].value["action_id"] == "synthetic-action-2"
    assert result["agent_action_outcomes"][first_proposal.action_id]["status"] == "rejected"
    monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=compiled))
    monkeypatch.setattr(webhook, "get_user_by_telegram_id", AsyncMock(
        return_value=SimpleNamespace(id="synthetic-user")))
    monkeypatch.setattr(webhook, "get_productivity_graph", AsyncMock(
        side_effect=AssertionError("Do not use legacy workflow")))
    monkeypatch.setattr(webhook, "_durable_callback_reply", AsyncMock())
    callback = SimpleNamespace(
        data=f"act:{first_proposal.action_id}:approve",
        from_user=SimpleNamespace(id="owner"),
        message=SimpleNamespace(chat_id="owner"), answer=AsyncMock())
    await webhook._handle_action_callback(callback)
    callback.answer.assert_awaited_once()
    assert callback.answer.await_args.kwargs["show_alert"] is True
    assert "agent_approval_gate" in (await compiled.aget_state(
        {"configurable": {"thread_id": "user_owner"}})).next
    executor.execute.assert_not_awaited()
