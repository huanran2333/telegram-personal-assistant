"""Approved business result must be explicit in the original tool call's reply."""
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.runtime import Runtime

from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel
from app.agent.agent_approval import execute_action


def _state():
    proposal = ActionProposal(
        action_id="synthetic-action-123", domain="todo", operation="delete_todo",
        args={"user_id": "synthetic-user", "todo_id": "synthetic-todo", "title": "旧发票"},
        human_readable_preview="Delete Todo: 旧发票", reason="owner-scoped target",
        risk_level=RiskLevel.DESTRUCTIVE)
    return {"user_id": "synthetic-user", "telegram_id": "synthetic-telegram",
            "chat_id": "", "request_origin": "web",
            "pending_agent_proposal": proposal.model_dump(),
            "pending_agent_call_id": "original-model-call", "approval_decision": "approve",
            "pending_tool_index": 1, "agent_action_outcomes": {}}, proposal


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [ActionStatus.EXECUTED, ActionStatus.EXECUTING,
                                    ActionStatus.FAILED])
async def test_approval_feedback_has_verified_action_status_target_and_call_binding(status):
    state, proposal = _state()
    executor = AsyncMock()
    executor.execute.return_value = ActionResult(
        action_id=proposal.action_id, status=status, message="server result",
        result={"remote_state": "PARKED_REVIEW"} if status == ActionStatus.EXECUTING else {})
    with patch("app.agent.workflow_runtime.get_action_executor", return_value=executor):
        update = await execute_action(state, Runtime(context={
            "user_id": "synthetic-user", "telegram_id": "synthetic-telegram",
            "chat_id": "", "origin": "web"}))
    msg = update["messages"][0]
    assert msg.tool_call_id == "original-model-call"
    assert update["pending_tool_index"] == 2
    assert update["agent_action_outcomes"][proposal.action_id]["status"] == status.value
    assert proposal.action_id in msg.content
    assert "delete_todo" in msg.content
    assert "旧发票" in msg.content
    assert status.value.upper() in msg.content
    if status == ActionStatus.EXECUTED:
        assert "no longer awaiting approval" in msg.content
    else:
        assert "no success is claimed" in msg.content
    executor.execute.assert_awaited_once_with(proposal)
