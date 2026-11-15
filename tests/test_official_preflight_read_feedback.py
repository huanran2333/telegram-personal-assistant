"""Known proposal reads fail as paired results before HITL or remote writes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import official_guarded_tools as guarded
from app.agent import official_todo_approval as todo_approval
from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError
from app.agent.official_permissions import OfficialToolAuthorization


def _calendar_503():
    try:
        raise CalendarProviderError("provider_unavailable", 503)
    except CalendarProviderError as cause:
        raise CalendarError("token=SECRET") from cause


def _gmail_403():
    request = httpx.Request("GET", "https://internal.example/token=SECRET")
    response = httpx.Response(403, request=request)
    try:
        raise httpx.HTTPStatusError("token=SECRET", request=request, response=response)
    except httpx.HTTPStatusError as cause:
        raise GmailError("token=SECRET") from cause


@pytest.mark.asyncio
@pytest.mark.parametrize("name,failure,reason", [
    ("propose_calendar_create", _calendar_503, "服务暂时不可用"),
    ("propose_email_todo", _gmail_403, "认证或访问权限不足"),
])
async def test_known_preflight_read_failure_is_paired_without_interrupt(
        monkeypatch, name, failure, reason):
    async def fail(**_):
        failure()

    target = "prepare_calendar_create" if name == "propose_calendar_create" else "prepare_email_todo"
    monkeypatch.setattr(guarded, target, fail)
    state = {"messages": [HumanMessage(id="turn", content="test"),
                          AIMessage(content="", tool_calls=[
                              {"name": name, "args": {}, "id": "failed-call"},
                              {"name": "create_todo", "args": {"title": "independent"},
                               "id": "independent-call"}])],
             "prepared_actions": {}, "user_id": "owner", "telegram_id": "telegram",
             "chat_id": "chat", "request_origin": "telegram"}
    boundary = guarded.AllGuardedPreflight(None)
    runtime = SimpleNamespace(context={"user_id": "owner", "telegram_id": "telegram",
                                       "chat_id": "chat", "origin": "telegram"})
    update = await boundary.aafter_model(state, runtime)
    state.update(update)
    failed = SimpleNamespace(state=state, tool_call={"name": name, "id": "failed-call"})
    assert not todo_approval.needs_todo_review(failed)

    async def must_not_execute(_):
        raise AssertionError("failed proposal entered execution")

    result = await boundary.awrap_tool_call(failed, must_not_execute)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "failed-call" and result.status == "error"
    assert "预检读取失败" in result.content and "本次写入未执行" in result.content
    assert reason in result.content
    assert "SECRET" not in result.content

    async def independent(_):
        return ToolMessage(content="created", tool_call_id="independent-call")

    other = await boundary.awrap_tool_call(
        SimpleNamespace(state=state, tool_call={"name": "create_todo", "id": "independent-call"}),
        independent)
    assert other.status == "success" and other.tool_call_id == "independent-call"


@pytest.mark.asyncio
async def test_managed_todo_calendar_block_is_rejected_before_telegram_approval(monkeypatch):
    async def managed(**_):
        raise CalendarError("ManagedTodoCalendarEvent")

    monkeypatch.setattr(guarded, "prepare_calendar_update", managed)
    state = {"messages": [HumanMessage(id="turn", content="修改日历事件"),
                          AIMessage(content="", tool_calls=[{
                              "name": "propose_calendar_update", "id": "managed-call",
                              "args": {"event_id": "managed", "title": "new"}}])],
             "prepared_actions": {}, "user_id": "owner", "telegram_id": "telegram",
             "chat_id": "chat", "request_origin": "telegram"}
    runtime = SimpleNamespace(context={"user_id": "owner", "telegram_id": "telegram",
                                       "chat_id": "chat", "origin": "telegram"})
    boundary = guarded.AllGuardedPreflight(None)
    state.update(await boundary.aafter_model(state, runtime))
    request = SimpleNamespace(state=state, tool_call={
        "name": "propose_calendar_update", "id": "managed-call"})
    assert not todo_approval.needs_todo_review(request)

    async def must_not_execute(_):
        raise AssertionError("managed mirror write reached ToolNode")

    result = await boundary.awrap_tool_call(request, must_not_execute)
    assert result.status == "error" and result.tool_call_id == "managed-call"
    assert "待办同步管理" in result.content and "本次写入未执行" in result.content


@pytest.mark.asyncio
async def test_unknown_preflight_program_error_still_propagates(monkeypatch):
    async def fail(**_):
        raise RuntimeError("program failure")

    monkeypatch.setattr(guarded, "prepare_calendar_create", fail)
    state = {"messages": [HumanMessage(id="turn", content="test"),
                          AIMessage(content="", tool_calls=[
                              {"name": "propose_calendar_create", "args": {}, "id": "call"}])],
             "user_id": "owner", "telegram_id": "telegram", "chat_id": "chat",
             "request_origin": "telegram"}
    with pytest.raises(RuntimeError, match="program failure"):
        await guarded.AllGuardedPreflight(None).aafter_model(
            state, SimpleNamespace(context={"user_id": "owner", "telegram_id": "telegram",
                                            "chat_id": "chat", "origin": "telegram"}))


@pytest.mark.asyncio
async def test_approved_todo_version_conflict_is_explained(monkeypatch):
    proposal = ActionProposal(
        action_id="action", domain="todo", operation="delete_todo",
        args={"user_id": "owner", "todo_id": "hidden-id", "title": "test"},
        human_readable_preview="Delete test", reason="user asked",
        risk_level=RiskLevel.DESTRUCTIVE)
    monkeypatch.setattr(todo_approval, "require_agent_runtime_identity",
                        lambda *_: {"user_id": "owner"})
    monkeypatch.setattr(todo_approval, "execute_approved_proposal", AsyncMock(
        return_value=ActionResult(action_id="action", status=ActionStatus.FAILED,
                                  message="changed", result={"failure_category": "todo_target_changed"})))
    state = {"messages": [HumanMessage(id="turn", content="delete")],
             "prepared_actions": {"call": {"turn_id": "turn", "proposal": proposal.model_dump(mode="json")}}}
    runtime = SimpleNamespace(state=state, tool_call_id="call", context={"user_id": "owner"})
    result = await todo_approval.make_approved_todo_delete_tool().coroutine(
        target="test", runtime=runtime)
    assert result.status == "error" and "目标已变化" in result.content


@pytest.mark.asyncio
async def test_guarded_approved_target_change_is_not_reported_as_unknown(monkeypatch):
    proposal = ActionProposal(
        action_id="action", domain="todo", operation="delete_todos_by_tag",
        args={"user_id": "owner", "tag": "test"},
        human_readable_preview="Delete tagged Todos", reason="user asked",
        risk_level=RiskLevel.DESTRUCTIVE)
    monkeypatch.setattr(guarded, "execute_approved_proposal", AsyncMock(
        return_value=ActionResult(action_id="action", status=ActionStatus.FAILED,
                                  message="changed", result={"failure_category": "todo_target_changed"})))
    state = {"messages": [HumanMessage(id="turn", content="delete")],
             "prepared_actions": {"call": {"turn_id": "turn",
                                           "proposal": proposal.model_dump(mode="json")}},
             "user_id": "owner", "telegram_id": "telegram", "chat_id": "chat",
             "request_origin": "telegram"}
    runtime = SimpleNamespace(state=state, tool_call_id="call",
                              context={"user_id": "owner", "telegram_id": "telegram",
                                       "chat_id": "chat", "origin": "telegram"})
    result = await guarded._execute(runtime, "delete_todos_by_tag")
    assert result.status == "error" and result.tool_call_id == "call"
    assert "目标已变化" in result.content and "本次删除未执行" in result.content
    assert "UNCONFIRMED" not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("quote,expected,next_step", [
    ("", "缺少该待办时间或提醒的原话依据", "未给出则先询问用户"),
    ("明天 16:30 处理", "所给时间依据不在本轮用户原话中", "不能补造时间依据"),
    ("把复习 FastAPI 安排到合适的时间", "没有与该待办和时段匹配的成功日历候选结果",
     "根据成功的候选结果排期"),
])
async def test_time_rejection_explains_only_verified_branch(
        monkeypatch, quote, expected, next_step):
    boundary = OfficialToolAuthorization([], settings=SimpleNamespace())
    monkeypatch.setattr(boundary, "_allowed", lambda *_: {"create_todo"})
    request = SimpleNamespace(
        state={"active_request": "先查空闲，再把复习 FastAPI 安排到合适的时间。"},
        runtime=SimpleNamespace(),
        tool_call={"name": "create_todo", "id": "timed-call", "args": {
            "title": "复习 FastAPI", "planned_start_str": "2026-12-10T15:00:00+08:00",
            "planned_end_str": "2026-12-10T16:00:00+08:00",
            "time_source_quote": quote}},
    )

    async def must_not_write(_):
        raise AssertionError("unauthorized time reached Todo write")

    result = await boundary.awrap_tool_call(request, must_not_write)
    assert result.tool_call_id == "timed-call" and result.status == "error"
    assert expected in result.content and "本次工具写入未执行" in result.content
    assert next_step in result.content
