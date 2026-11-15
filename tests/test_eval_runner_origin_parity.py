"""Offline Stage 1A checks against the production create_agent assembly.

The model and Todo business storage are synthetic. No provider, Bot, or DB
connection is made; these tests do not replace the isolated-PG business suite.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent import official_todo_approval
from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel
from app.agent.official_approval_view import (
    official_decision_for_action, pending_official_approvals,
)
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import official_thread_id, run_agent
from app.config import get_settings
from experiments.official_agent_eval_rebuild_20261023.test_current_eval import (
    _synthetic_telegram_identity,
)
from experiments.official_agent_migration_20261020.test_official_runtime_contract import (
    ScriptedModel,
)


def _user(name: str):
    return SimpleNamespace(id=f"eval-owner-{name}", telegram_id=f"eval-chat-{name}")


def _graph(responses, *, google_owner=None):
    model = ScriptedModel(responses)
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": bool(google_owner),
        "google_calendar_owner_telegram_id": google_owner,
        "gmail_enabled": bool(google_owner),
        "gmail_owner_telegram_id": google_owner,
        "tavily_api_key": "", "memory_enabled": False,
    })
    # These visibility/HITL tests patch only the Todo business boundary below.
    # No tool tries to open this session factory.
    graph, _ = build_official_agent(
        model=model, session_factory=lambda: None, settings=settings,
        checkpointer=InMemorySaver(), interrupt_mode="per_call")
    return graph, model


@pytest.mark.asyncio
async def test_eval_synthetic_telegram_sees_protected_tool_while_web_does_not():
    user = _user("visibility")
    thread_id, context = _synthetic_telegram_identity(user)
    assert thread_id == f"user_{user.telegram_id}"
    assert context == {"user_id": user.id, "telegram_id": user.telegram_id,
                       "chat_id": str(user.telegram_id), "origin": "telegram"}

    telegram_graph, telegram_model = _graph(
        [AIMessage(content="done")], google_owner=user.telegram_id)
    await run_agent(telegram_graph, user.id, user.telegram_id,
                    context["chat_id"], "列出我的待办", thread_id=thread_id,
                    origin=context["origin"], return_state=True)
    telegram_tools = set(telegram_model._bound_tool_names[0])
    assert {"list_todos", "create_todo", "propose_todo_delete",
            "calendar_events", "gmail_search", "propose_calendar_create"} <= telegram_tools

    web_graph, web_model = _graph(
        [AIMessage(content="done")], google_owner=user.telegram_id)
    await run_agent(web_graph, user.id, user.telegram_id, "",
                    "列出我的待办", thread_id=f"web:ordinary:{user.id}",
                    turn_id="eval-web-visibility", origin="web", return_state=True)
    web_tools = set(web_model._bound_tool_names[0])
    assert {"list_todos", "create_todo", "calendar_events", "gmail_search"} <= web_tools
    assert "propose_todo_delete" not in web_tools
    assert not any(name.startswith("propose_") for name in web_tools)

    bystander = _user("unconfigured-google")
    bystander_graph, bystander_model = _graph(
        [AIMessage(content="done")], google_owner=user.telegram_id)
    await run_agent(bystander_graph, bystander.id, bystander.telegram_id,
                    bystander.telegram_id, "列出我的待办", origin="telegram")
    bystander_tools = set(bystander_model._bound_tool_names[0])
    assert {"list_todos", "create_todo", "propose_todo_delete"} <= bystander_tools
    assert "calendar_events" not in bystander_tools
    assert "gmail_search" not in bystander_tools
    assert "propose_calendar_create" not in bystander_tools


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.asyncio
async def test_eval_synthetic_telegram_formal_hitl_owner_bound_resume(
        monkeypatch, decision):
    owner, bystander = _user(f"{decision}-owner"), _user(f"{decision}-bystander")
    title = f"Delete target for {decision}"
    todos = {(owner.id, title): {"id": f"todo-{decision}", "version": 1}}
    executed = []

    async def prepare(_factory, *, user_id, target):
        row = todos.get((user_id, target))
        if row is None:
            raise ValueError("TodoTargetNotUnique")
        return ActionProposal(
            domain="todo", operation="delete_todo", risk_level=RiskLevel.DESTRUCTIVE,
            args={"user_id": user_id, "todo_id": row["id"],
                  "title": target, "approved_todo_version": row["version"]},
            human_readable_preview=f"Delete Todo: {target}", reason="mock owner check")

    async def execute(proposal, identity):
        assert proposal.args["user_id"] == identity["user_id"]
        assert proposal.args["todo_id"] == todos[(owner.id, title)]["id"]
        executed.append(proposal.action_id)
        del todos[(owner.id, title)]
        return ActionResult(action_id=proposal.action_id, status=ActionStatus.EXECUTED,
                            message="deleted")

    monkeypatch.setattr(official_todo_approval, "prepare_todo_delete", prepare)
    monkeypatch.setattr(official_todo_approval, "execute_approved_proposal", execute)
    graph, model = _graph([
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "args": {"target": title},
            "id": f"eval-delete-{decision}"}]),
        AIMessage(content="审批处理完成"),
    ])
    thread_id, context = _synthetic_telegram_identity(owner)
    _, initial = await run_agent(
        graph, owner.id, owner.telegram_id, context["chat_id"],
        f"请确认删除 {title}", thread_id=thread_id, origin=context["origin"],
        return_state=True)
    assert initial.get("__interrupt__")
    assert (owner.id, title) in todos and not executed
    config = {"configurable": {"thread_id": official_thread_id(thread_id)}}
    paused = await graph.aget_state(config)
    pending = pending_official_approvals(paused, user_id=owner.id)
    assert len(pending) == 1
    assert pending[0].tool_call_id == f"eval-delete-{decision}"
    assert pending[0].proposal.args["user_id"] == owner.id
    with pytest.raises(PermissionError, match="OfficialApprovalOwnerMismatch"):
        pending_official_approvals(paused, user_id=bystander.id)
    with pytest.raises(PermissionError, match="CheckpointOwnerMismatch"):
        await run_agent(graph, bystander.id, bystander.telegram_id,
                        bystander.telegram_id, "查看待办", thread_id=thread_id,
                        origin="telegram")

    resume = official_decision_for_action(
        paused, user_id=owner.id, action_id=pending[0].proposal.action_id,
        decision=decision,
        checkpoint_id=paused.config["configurable"]["checkpoint_id"])
    _, foreign_context = _synthetic_telegram_identity(bystander)
    with pytest.raises(PermissionError, match="AgentRuntimeOwnerMismatch"):
        await graph.ainvoke(Command(resume=resume), config=config,
                            context=foreign_context)
    assert (owner.id, title) in todos and not executed
    await graph.ainvoke(Command(resume=resume), config=config, context=context)
    finished = await graph.aget_state(config)
    assert not pending_official_approvals(finished, user_id=owner.id)
    paired = [message for message in finished.values["messages"]
              if isinstance(message, ToolMessage) and
              message.tool_call_id == f"eval-delete-{decision}"]
    assert len(paired) == 1
    if decision == "approve":
        assert (owner.id, title) not in todos
    else:
        assert (owner.id, title) in todos
    assert len(executed) == (1 if decision == "approve" else 0)
    assert len(model._received) == 2
