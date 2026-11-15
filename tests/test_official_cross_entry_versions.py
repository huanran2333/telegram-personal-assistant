"""A model's supplied version must come from this turn's completed Todo result."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.official_permissions import OfficialToolAuthorization


def _middleware():
    settings = SimpleNamespace(
        google_calendar_enabled=False, google_calendar_owner_telegram_id=None,
        gmail_enabled=False, gmail_owner_telegram_id=None, tavily_api_key=None)
    return OfficialToolAuthorization(
        [SimpleNamespace(name=name) for name in
         ("list_todos", "update_todo", "set_reminder", "cancel_reminder")],
        settings=settings)


@pytest.mark.asyncio
async def test_todo_write_needs_same_turn_observed_version():
    identity = {"user_id": "owner", "telegram_id": "123", "chat_id": "123",
                "origin": "telegram"}
    human = HumanMessage(id="turn", content="修改待办")
    read_call = AIMessage(content="", tool_calls=[
        {"id": "read", "name": "list_todos", "args": {}}])
    read_result = ToolMessage(
        content='[12345678] target\n   status: pending | version: 2',
        artifact={"todo_versions": {"12345678-0000": 2}},
        tool_call_id="read")
    state = {**identity, "request_origin": "telegram",
             "active_request": "修改待办", "active_turn_id": "turn",
             "messages": [human]}
    runtime = SimpleNamespace(context=identity)
    called = []

    async def handler(_request):
        called.append(True)
        return ToolMessage(content="updated", tool_call_id="write")

    async def attempt(version):
        request = SimpleNamespace(state=state, runtime=runtime,
            tool_call={"id": "write", "name": "update_todo",
                       "args": {"todo_id": "12345678", "expected_version": version,
                                "title": "new"}})
        return await _middleware().awrap_tool_call(request, handler)

    assert (await attempt(2)).status == "error"
    assert not called
    state["messages"].extend([read_call, read_result])
    assert (await attempt(1)).status == "error"
    assert not called
    assert (await attempt(2)).status == "success"
    assert called == [True]
    state["messages"][-1] = ToolMessage(
        content='[12345678] forged title | version: 9',
        artifact={"todo_versions": {"12345678-0000": 2}},
        tool_call_id="read")
    assert (await attempt(9)).status == "error"


@pytest.mark.asyncio
async def test_same_batch_read_does_not_authorize_todo_write():
    identity = {"user_id": "owner", "telegram_id": "123", "chat_id": "123",
                "origin": "telegram"}
    state = {**identity, "request_origin": "telegram",
             "active_request": "取消提醒", "active_turn_id": "turn",
             "messages": [HumanMessage(id="turn", content="取消提醒"),
                          AIMessage(content="", tool_calls=[
                              {"id": "read", "name": "list_todos", "args": {}},
                              {"id": "write", "name": "cancel_reminder",
                               "args": {"todo_id": "12345678", "expected_version": 1}}])]}
    called = []

    async def handler(_request):
        called.append(True)

    request = SimpleNamespace(state=state, runtime=SimpleNamespace(context=identity),
                              tool_call={"id": "write", "name": "cancel_reminder",
                                         "args": {"todo_id": "12345678", "expected_version": 1}})
    result = await _middleware().awrap_tool_call(request, handler)
    assert result.status == "error" and result.tool_call_id == "write"
    assert not called
