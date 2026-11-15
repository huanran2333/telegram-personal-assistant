"""The same persisted approval reply carries completed read results once."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_telegram_proposal_receives_prior_result_in_same_reply(monkeypatch):
    from app.telegram import webhook
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(
        values={"proposal": {"action_id": "synthetic-action"}})))
    monkeypatch.setattr(webhook, "get_productivity_graph", AsyncMock(return_value=graph))
    sender = AsyncMock()
    monkeypatch.setattr(webhook, "send_action_proposal", sender)
    result = {"__interrupt__": [SimpleNamespace(value={"kind": "action_proposal"})]}
    await webhook._send_workflow_result(
        1, "synthetic-workflow", result,
        completed_summary="已完成的查询：日历查询：合成会议")
    sender.assert_awaited_once_with(
        1, {"action_id": "synthetic-action"},
        completed_summary="已完成的查询：日历查询：合成会议")


@pytest.mark.asyncio
async def test_web_pending_approval_contains_prior_result(monkeypatch):
    from app.agent import workflow
    from app.web import runs
    monkeypatch.setattr(workflow, "bind_continuation_checkpoint", AsyncMock())
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(
        values={"proposal": {"action_id": "synthetic-action", "risk_level": "destructive"}},
        config={"configurable": {"checkpoint_id": "synthetic-checkpoint"}})))
    result = {"__interrupt__": [SimpleNamespace(value={"kind": "action_proposal"})]}
    state, response = await runs._workflow_result(
        graph, "synthetic-workflow", result, None,
        completed_summary="已完成的查询：日历查询：合成会议")
    assert state == "WAITING_APPROVAL"
    assert response["pending_approval"]["completed_summary"].endswith("合成会议")
    assert response["pending_approval"]["action_id"] == "synthetic-action"
