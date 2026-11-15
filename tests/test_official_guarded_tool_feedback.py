"""The current official graph must not turn an uncertain remote write into a retry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel
from app.agent import official_guarded_tools


@pytest.mark.asyncio
async def test_unconfirmed_calendar_result_tells_model_not_to_repeat_action(monkeypatch):
    proposal = ActionProposal(
        action_id="synthetic-action", domain="calendar", operation="create_event",
        args={"user_id": "owner"}, human_readable_preview="Synthetic meeting",
        reason="User requested it", risk_level=RiskLevel.EXTERNAL_WRITE,
    )
    execute = AsyncMock(return_value=ActionResult(
        action_id=proposal.action_id, status=ActionStatus.EXECUTING,
        message="Calendar outcome is not yet confirmed; do not submit a replacement action.",
        result={"remote_state": "UNCERTAIN"},
    ))
    monkeypatch.setattr(official_guarded_tools, "execute_approved_proposal", execute)
    state = {
        "messages": [HumanMessage(id="synthetic-turn", content="Create meeting")],
        "user_id": "owner", "telegram_id": "telegram-owner",
        "chat_id": "telegram-owner", "request_origin": "telegram",
        "prepared_actions": {
            "synthetic-call": {
                "turn_id": "synthetic-turn", "proposal": proposal.model_dump(mode="json"),
            },
        },
    }
    runtime = SimpleNamespace(
        state=state, tool_call_id="synthetic-call",
        context={"user_id": "owner", "telegram_id": "telegram-owner",
                 "chat_id": "telegram-owner", "origin": "telegram"},
    )

    reply = await official_guarded_tools._execute(runtime, "create_event")

    assert reply.status == "error"
    assert reply.tool_call_id == "synthetic-call"
    assert "status=EXECUTING" in reply.content
    assert "outcome=UNCONFIRMED" in reply.content
    assert "Do not retry or propose a replacement" in reply.content
    execute.assert_awaited_once()
