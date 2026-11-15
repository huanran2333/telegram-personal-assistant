"""Execution-boundary regressions for reminders on existing Todos."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import official_permissions
from app.agent.official_permissions import OfficialToolAuthorization


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _Db:
    def __init__(self, rows):
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def execute(self, _):
        return _Rows(self.rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,quote,remind_at,expected", [
    ("给已有待办『U15-准备周会』设置2026年11月12日08:30（北京时间）的提醒，只设提醒，不修改标题或计划时间。",
     "2026年11月12日08:30（北京时间）的提醒", "2026-11-12T08:30:00+08:00", True),
    ("给待办『E2E-20261103-提醒闭环-2341』设置2026年11月3日23:55（北京时间）的提醒；只设置提醒，不修改计划时间。",
     "给待办『E2E-20261103-提醒闭环-2341』设置2026年11月3日23:55（北京时间）的提醒",
     "2026-11-03T23:55:00+08:00", True),
    ("给已有待办『U15-准备周会』设置2026年11月12日08:30的提醒。",
     "2026年11月12日08:30的提醒", "2026-11-12T09:30:00+08:00", False),
    ("不要给待办『U15-准备周会』设置2026年11月12日08:30的提醒。",
     "2026年11月12日08:30的提醒", "2026-11-12T08:30:00+08:00", False),
    ("给待办『U15-准备周会』设置2026年11月12日08:30的提醒；会议在2026年11月12日09:30。",
     "2026年11月12日09:30", "2026-11-12T09:30:00+08:00", False),
    ("给待办『另一个目标』设置2026年11月12日08:30的提醒。",
     "2026年11月12日08:30的提醒", "2026-11-12T08:30:00+08:00", False),
    ("给待办『U15-准备周会』设置2026年11月12日08:30的提醒；给待办『另一个目标』设置2026年11月12日09:30的提醒。",
     "2026年11月12日09:30的提醒", "2026-11-12T09:30:00+08:00", False),
    ("给待办『U15-准备周会』和『另一个目标』设置2026年11月12日08:30的提醒。",
     "2026年11月12日08:30的提醒", "2026-11-12T08:30:00+08:00", False),
])
async def test_reminder_requires_owned_target_explicit_intent_and_matching_time(
        monkeypatch, text, quote, remind_at, expected):
    boundary = OfficialToolAuthorization(
        [], settings=SimpleNamespace(default_timezone="Asia/Shanghai"),
        session_factory=lambda: _Db([SimpleNamespace(
            title="E2E-20261103-提醒闭环-2341" if "E2E-" in text else "U15-准备周会")]))
    monkeypatch.setattr(boundary, "_allowed", lambda *_: {"set_reminder"})
    monkeypatch.setattr(official_permissions, "require_agent_runtime_identity",
                        lambda *_: {"user_id": "owner"})
    request = SimpleNamespace(
        state={"active_request": text, "active_turn_id": "turn",
               "messages": [HumanMessage(id="turn", content=text),
                            AIMessage(content="", tool_calls=[
                                {"name": "list_todos", "id": "read", "args": {}}]),
                            ToolMessage(content='[8ca15e37] owned target\n   version: 1',
                                        artifact={"todo_versions": {"8ca15e37-0000": 1}},
                                        tool_call_id="read")]}, runtime=SimpleNamespace(),
        tool_call={"name": "set_reminder", "id": "u15-call", "args": {
            "todo_id": "8ca15e37", "remind_at_str": remind_at,
            "expected_version": 1,
            "time_source_quote": quote,
        }},
    )
    called = False

    async def handler(_):
        nonlocal called
        called = True
        return ToolMessage(content="executed", tool_call_id="u15-call")

    result = await boundary.awrap_tool_call(request, handler)
    assert called is expected
    assert result.status == ("success" if expected else "error")
    assert result.tool_call_id == "u15-call"
