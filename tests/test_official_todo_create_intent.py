"""Normal Todo creation no longer gets a second natural-language intent judge."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from app.agent.official_permissions import OfficialToolAuthorization


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", [
    "请分别创建两个无时间待办：A 和 B",
    "只读，不创建任何待办",
    "不创建 A，只创建 B",
    "只创建 B",
])
async def test_create_boundary_does_not_reinterpret_user_words(monkeypatch, request_text):
    boundary = OfficialToolAuthorization([], settings=SimpleNamespace())
    monkeypatch.setattr(boundary, "_allowed", lambda *_: {"create_todo"})
    request = SimpleNamespace(
        state={"active_request": request_text}, runtime=SimpleNamespace(),
        tool_call={"name": "create_todo", "id": "create-call", "args": {"title": "A"}},
    )
    called = False

    async def handler(_):
        nonlocal called
        called = True
        return ToolMessage(content="executed", tool_call_id="create-call")

    result = await boundary.awrap_tool_call(request, handler)
    assert called and result.status == "success"


@pytest.mark.asyncio
async def test_create_boundary_still_rejects_unauthorized_time(monkeypatch):
    boundary = OfficialToolAuthorization([], settings=SimpleNamespace())
    monkeypatch.setattr(boundary, "_allowed", lambda *_: {"create_todo"})
    request = SimpleNamespace(
        state={"active_request": "创建不设时间的待办 A"}, runtime=SimpleNamespace(),
        tool_call={"name": "create_todo", "id": "timed-call", "args": {
            "title": "A", "planned_start_str": "明天 16:30", "time_source_quote": ""}},
    )

    async def handler(_):
        raise AssertionError("Tool must not execute with unauthorized time")

    result = await boundary.awrap_tool_call(request, handler)
    assert result.status == "error" and result.tool_call_id == "timed-call"
    assert "TodoTimeNotAuthorized" in result.content
