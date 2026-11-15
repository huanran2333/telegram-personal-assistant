"""The new meeting tool receives authority from the server, not its schema."""
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock

import pytest

from app.agent.unified_tools import AgentToolScope, agent_tool_scope, make_unified_tools


@asynccontextmanager
async def no_db():
    raise AssertionError("No database should be reached in the tool boundary test")
    yield


@pytest.mark.asyncio
async def test_prepare_tool_injects_scope_and_returns_only_resource_metadata(monkeypatch):
    from app.agent import meeting_drafts, nodes, unified_tools, workflow_runtime
    from app.config import get_settings
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True, "google_calendar_owner_telegram_id": "owner",
        "gmail_enabled": False})
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    calendar, llm = object(), object()
    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: calendar)
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: llm)
    prepare = AsyncMock(return_value={"draft_id": "draft-1", "revision": 1,
                                     "status": "NEEDS_REVIEW", "sources": [{"key": "C1"}]})
    monkeypatch.setattr(meeting_drafts, "prepare_meeting_draft", prepare)
    tool = next(item for item in make_unified_tools([], no_db) if item.name == "prepare_meeting")
    assert set(tool.args_schema.model_fields) == {"event_id"}
    with agent_tool_scope(AgentToolScope("server-user", "owner", "prepare meeting", "turn-1",
                                        "web", "server-thread")):
        result = json.loads(await tool.ainvoke({"event_id": "event-1"}))
    assert result["draft_id"] == "draft-1" and "content" not in result
    assert prepare.await_args.kwargs["user_id"] == "server-user"
    assert prepare.await_args.kwargs["turn_id"] == "turn-1"
    assert prepare.await_args.kwargs["calendar"] is calendar
    assert prepare.await_args.kwargs["llm"] is llm
    with agent_tool_scope(AgentToolScope("other-user", "outsider", "prepare meeting", "turn-2",
                                        "web", "other-thread")):
        with pytest.raises(PermissionError, match="CalendarOwnerMismatch"):
            await tool.ainvoke({"event_id": "event-1"})
    assert prepare.await_count == 1
