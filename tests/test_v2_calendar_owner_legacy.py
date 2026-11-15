"""The old Todo graph must honor the same single-account Calendar boundary."""
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from app.agent import nodes


class CalendarReadTool:
    name = "list-events"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        return [{"type": "text", "text": "private event"}]


class StubLLM:
    def bind_tools(self, tools):
        return self


@pytest.mark.asyncio
async def test_legacy_calendar_tool_does_not_read_another_users_calendar():
    tool = CalendarReadTool()
    with patch.object(nodes, "_build_chat_llm", return_value=StubLLM()), patch.object(
        nodes.settings, "google_calendar_owner_telegram_id", "owner-telegram"
    ):
        execute_tools = nodes.make_nodes([tool])[3]
        message = AIMessage(content="", tool_calls=[{
            "name": "list-events", "args": {}, "id": "read-1", "type": "tool_call",
        }])
        result = await execute_tools({"messages": [message], "telegram_id": "other-telegram"})
        assert "unavailable for this user" in result["messages"][0].content
        assert tool.calls == 0
        await execute_tools({"messages": [message], "telegram_id": "owner-telegram"})
        assert tool.calls == 1


@pytest.mark.asyncio
async def test_legacy_conflict_check_does_not_read_another_users_calendar():
    tool = CalendarReadTool()
    with patch.object(nodes, "_build_chat_llm", return_value=StubLLM()), patch.object(
        nodes.settings, "google_calendar_owner_telegram_id", "owner-telegram"
    ):
        check_calendar_node = nodes.make_nodes([tool])[7]
        result = await check_calendar_node({
            "intent": "create", "entities": {"due_date_str": "tomorrow 16:00"},
            "telegram_id": "other-telegram",
        })
        assert result["calendar_check_done"]
        assert tool.calls == 0
