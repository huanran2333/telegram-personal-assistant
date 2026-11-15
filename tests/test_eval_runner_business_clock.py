"""Offline Stage 1B: task-local time through the production Agent graph."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent import agent_approval, official_read_tools, unified_tools, workflow_runtime
from app.agent.actions import ActionResult, ActionStatus
from app.agent.business_clock import business_now, isolated_evaluation_time
from app.agent.official_permissions import OfficialToolAuthorization
from app.agent.official_approval_view import official_decision_for_action, pending_official_approvals
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import official_thread_id, run_agent
from app.agent.tools import make_todo_operations
from app.config import get_settings
from experiments.official_agent_eval_rebuild_20261023.test_current_eval import (
    _case_evaluation_time, _invoke_at_evaluation_time, _synthetic_telegram_identity,
)
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel


ZONE = ZoneInfo("Asia/Shanghai")
BEFORE = datetime(2026, 11, 9, 12, 0, tzinfo=ZONE)
EVENT_START = "2026-11-10T13:00:00+08:00"
EVENT_END = "2026-11-10T13:45:00+08:00"


def _settings(owner="clock-user"):
    return get_settings().model_copy(update={
        "app_env": "test", "default_timezone": "Asia/Shanghai",
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": owner,
        "gmail_enabled": False, "tavily_api_key": "", "memory_enabled": False,
    })


def _graph(responses, settings):
    model = ScriptedModel(responses)
    graph, _ = build_official_agent(
        model=model, session_factory=lambda: None, settings=settings,
        checkpointer=InMemorySaver(), interrupt_mode="per_call")
    return graph, model


def test_case_clock_requires_an_exact_aware_instant_and_matching_iana_zone():
    case = {"evaluation_now": BEFORE.isoformat(),
            "evaluation_timezone": "Asia/Shanghai"}
    assert _case_evaluation_time(case, default_timezone="Asia/Shanghai") == (
        BEFORE, "Asia/Shanghai")
    with pytest.raises(ValueError, match="ExplicitAware"):
        _case_evaluation_time({"time_reference": "2026-11-09 Asia/Shanghai"},
                              default_timezone="Asia/Shanghai")
    assert _case_evaluation_time({**case, "evaluation_now": "2026-11-09T04:00:00+00:00"},
                                 default_timezone="Asia/Shanghai")[0].astimezone(ZONE) == BEFORE
    with pytest.raises(ValueError, match="ZoneMismatch"):
        _case_evaluation_time({**case, "evaluation_timezone": "UTC"},
                              default_timezone="Asia/Shanghai")
    with pytest.raises(PermissionError, match="RequiresTestEnvironment"):
        with isolated_evaluation_time(BEFORE, "Asia/Shanghai", app_env="production"):
            pass


def test_default_clock_is_live_and_scoped_clock_restores_it():
    lower = datetime.now(timezone.utc)
    normal = business_now(timezone.utc)
    upper = datetime.now(timezone.utc)
    assert lower <= normal <= upper
    with isolated_evaluation_time(BEFORE, "Asia/Shanghai", app_env="test"):
        assert business_now(timezone.utc) == BEFORE.astimezone(timezone.utc)
    assert abs((business_now(timezone.utc) - datetime.now(timezone.utc)).total_seconds()) < 2


@pytest.mark.asyncio
async def test_tomorrow_todo_time_is_independent_of_wall_date(monkeypatch):
    import app.config as config
    from app.telegram import durable

    monkeypatch.setattr(config, "get_settings", lambda: _settings())
    captured = []

    class EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    async def capture(_db, **kwargs):
        captured.append(kwargs["args"])
        return "captured"

    monkeypatch.setattr(durable, "commit_todo_tool_operation", capture)
    create = make_todo_operations(lambda: EmptySession())["create_todo"]
    # UTC Nov 9, 16:30 is already Nov 10 in Shanghai. "Tomorrow" is Nov 11.
    instant = datetime(2026, 11, 9, 16, 30, tzinfo=timezone.utc)
    with isolated_evaluation_time(instant, "Asia/Shanghai", app_env="test"):
        assert await create(title="复习", user_id="owner",
                            planned_start_str="明天 09:00") == "captured"
    assert captured[0]["planned_start_at"].astimezone(ZONE).date().isoformat() == "2026-11-11"
    with isolated_evaluation_time(BEFORE, "Asia/Shanghai", app_env="test"):
        assert await create(title="有提醒", user_id="owner",
                            reminder_at_str=EVENT_START) == "captured"
    with isolated_evaluation_time(BEFORE + timedelta(days=2), "Asia/Shanghai", app_env="test"):
        assert await create(title="过期提醒", user_id="owner",
                            reminder_at_str=EVENT_START) != "captured"
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_reminder_authorization_uses_fixed_relative_base():
    settings = _settings()
    row = SimpleNamespace(title="复习")

    class Result:
        def scalars(self):
            return self

        def all(self):
            return [row]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _query):
            return Result()

    middleware = OfficialToolAuthorization(
        [], settings=settings, session_factory=lambda: Session())
    quote = "给『复习』设置明天09:00提醒"
    identity = {"user_id": "owner", "telegram_id": "clock-user",
                "chat_id": "clock-user", "origin": "telegram"}
    state = {**identity, "request_origin": "telegram", "active_request": quote}
    args = {"todo_id": "12345678", "time_source_quote": quote,
            "remind_at_str": "2026-11-10T09:00:00+08:00"}
    runtime = SimpleNamespace(context=identity)
    with isolated_evaluation_time(BEFORE, "Asia/Shanghai", app_env="test"):
        assert await middleware._reminder_authorized(state, runtime, args)
    with isolated_evaluation_time(BEFORE + timedelta(days=1), "Asia/Shanghai", app_env="test"):
        assert not await middleware._reminder_authorized(state, runtime, args)


@pytest.mark.asyncio
async def test_calendar_future_preflight_uses_business_instant(monkeypatch):
    settings = _settings()
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "require_calendar_owner", lambda _id: None)

    class Calendar:
        async def availability(self, *_):
            return {"calendars": {"primary": {"busy": []}}}

    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: Calendar())
    with isolated_evaluation_time(BEFORE, "Asia/Shanghai", app_env="test"):
        proposal = await unified_tools.prepare_calendar_create(
            user_id="owner", telegram_id="clock-user", title="复习",
            start=EVENT_START, end=EVENT_END)
        assert proposal.operation == "create_event"
    with isolated_evaluation_time(BEFORE + timedelta(days=2), "Asia/Shanghai", app_env="test"):
        with pytest.raises(ValueError, match="CalendarStartInPast"):
            await unified_tools.prepare_calendar_create(
                user_id="owner", telegram_id="clock-user", title="复习",
                start=EVENT_START, end=EVENT_END)

    async def verified_email(**_):
        return SimpleNamespace(message_id="mail-1")

    monkeypatch.setattr(unified_tools, "verified_email_source", verified_email)
    with isolated_evaluation_time(BEFORE, "Asia/Shanghai", app_env="test"):
        proposal = await unified_tools.prepare_email_calendar(
            user_id="owner", telegram_id="clock-user", thread_id="thread-1",
            message_id="mail-1", evidence_quote="confirmed", title="复习",
            start=EVENT_START, end=EVENT_END)
        assert proposal.operation == "create_event"
    with isolated_evaluation_time(BEFORE + timedelta(days=2), "Asia/Shanghai", app_env="test"):
        with pytest.raises(ValueError, match="CalendarStartInPast"):
            await unified_tools.prepare_email_calendar(
                user_id="owner", telegram_id="clock-user", thread_id="thread-1",
                message_id="mail-1", evidence_quote="confirmed", title="复习",
                start=EVENT_START, end=EVENT_END)


@pytest.mark.asyncio
async def test_production_candidate_slot_tool_uses_fixed_now(monkeypatch):
    settings = _settings()
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)

    class Calendar:
        async def availability(self, *_):
            return {"calendars": {"primary": {"busy": []}}}

    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: Calendar())
    call = {"name": "calendar_candidate_slots", "id": "candidate-clock",
            "args": {"title": "复习", "duration_minutes": 45,
                     "range_start": "2026-11-10T13:00:00+08:00",
                     "range_end": "2026-11-10T17:00:00+08:00"}}
    graph, _ = _graph([AIMessage(content="", tool_calls=[call]),
                       AIMessage(content="已查询")], settings)
    await _invoke_at_evaluation_time(
        run_agent(graph, "owner", "clock-user", "clock-user", "明天下午找空档",
                  thread_id="clock-candidate", origin="telegram"),
        now=BEFORE, zone_name="Asia/Shanghai", settings=settings)
    snapshot = await graph.aget_state({"configurable": {
        "thread_id": official_thread_id("clock-candidate")}})
    result = next(item for item in snapshot.values["messages"]
                  if isinstance(item, ToolMessage) and item.tool_call_id == "candidate-clock")
    assert result.status == "success"
    assert "2026-11-10T13:00:00+08:00" in str(result.content)


@pytest.mark.asyncio
async def test_model_context_and_human_timestamp_use_same_fixed_instant():
    user = SimpleNamespace(id="clock-owner", telegram_id="clock-user")
    thread_id, context = _synthetic_telegram_identity(user)
    settings = _settings(user.telegram_id)
    graph, model = _graph([AIMessage(content="完成")], settings)
    await _invoke_at_evaluation_time(
        run_agent(graph, user.id, user.telegram_id, context["chat_id"],
                  "明天有什么安排？", thread_id=thread_id, origin=context["origin"]),
        now=BEFORE, zone_name="Asia/Shanghai", settings=settings)
    prompt = next(str(item.content) for item in model._received[0]
                  if isinstance(item, SystemMessage))
    assert f"Current local time: {BEFORE.isoformat()}" in prompt
    saved = await graph.aget_state({"configurable": {"thread_id": official_thread_id(thread_id)}})
    human = next(item for item in saved.values["messages"] if isinstance(item, HumanMessage))
    assert human.additional_kwargs["occurred_at_utc"] == BEFORE.astimezone(timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_unbound_production_graph_uses_live_time():
    settings = _settings()
    graph, model = _graph([AIMessage(content="完成")], settings)
    lower = datetime.now(timezone.utc)
    await run_agent(graph, "live-owner", "clock-user", "clock-user",
                    "现在几点？", thread_id="live-clock", origin="telegram")
    upper = datetime.now(timezone.utc)
    prompt = next(str(item.content) for item in model._received[0]
                  if isinstance(item, SystemMessage))
    stamped = prompt.split("Current local time: ", 1)[1].split(";", 1)[0]
    assert lower <= datetime.fromisoformat(stamped).astimezone(timezone.utc) <= upper


@pytest.mark.parametrize("expired_on_resume", [False, True])
@pytest.mark.asyncio
async def test_calendar_preflight_and_hitl_resume_share_clock(monkeypatch, expired_on_resume):
    settings = _settings()
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_approval, "get_settings", lambda: settings)
    availability_times = []
    executed = []

    class Calendar:
        async def availability(self, *_):
            availability_times.append(business_now(timezone.utc))
            return {"calendars": {"primary": {"busy": []}}}

    class Executor:
        async def calendar_create_prepared(self, _action_id):
            return False

        async def execute(self, proposal):
            executed.append(proposal.action_id)
            return ActionResult(action_id=proposal.action_id,
                                status=ActionStatus.EXECUTED, message="mock created")

    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: Calendar())
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: Executor())
    graph, model = _graph([
        AIMessage(content="", tool_calls=[{
            "name": "propose_calendar_create", "id": "calendar-create",
            "args": {"title": "复习", "start": EVENT_START, "end": EVENT_END}}]),
        AIMessage(content="已经完成"),
    ], settings)
    user = SimpleNamespace(id="clock-owner", telegram_id="clock-user")
    thread_id, context = _synthetic_telegram_identity(user)
    _, initial = await _invoke_at_evaluation_time(
        run_agent(graph, user.id, user.telegram_id, context["chat_id"],
                  "明天创建复习日程", thread_id=thread_id,
                  origin=context["origin"], return_state=True),
        now=BEFORE, zone_name="Asia/Shanghai", settings=settings)
    assert initial.get("__interrupt__") and not executed
    config = {"configurable": {"thread_id": official_thread_id(thread_id)}}
    paused = await graph.aget_state(config)
    pending = pending_official_approvals(paused, user_id=user.id)
    assert len(pending) == 1
    resume = official_decision_for_action(
        paused, user_id=user.id, action_id=pending[0].proposal.action_id,
        decision="approve", checkpoint_id=paused.config["configurable"]["checkpoint_id"])
    await _invoke_at_evaluation_time(
        graph.ainvoke(Command(resume=resume), config=config, context=context),
        now=BEFORE + timedelta(days=2) if expired_on_resume else BEFORE,
        zone_name="Asia/Shanghai", settings=settings)
    finished = await graph.aget_state(config)
    assert len(executed) == (0 if expired_on_resume else 1)
    assert availability_times == [BEFORE.astimezone(timezone.utc)] * (
        1 if expired_on_resume else 2)
    paired = [item for item in finished.values["messages"]
              if isinstance(item, ToolMessage) and item.tool_call_id == "calendar-create"]
    assert len(paired) == 1
    assert paired[0].status == ("error" if expired_on_resume else "success")
    assert len(model._received) == 2


@pytest.mark.asyncio
async def test_concurrent_toolnode_calls_keep_independent_clock(monkeypatch):
    settings = _settings()
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    seen = []

    class Calendar:
        async def availability(self, start, *_):
            await asyncio.sleep(0)
            seen.append((start, business_now(timezone.utc)))
            return {"calendars": {"primary": {"busy": []}}}

    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: Calendar())

    async def one(now, label):
        day = 10 if label == "first" else 12
        calls = [{"name": "calendar_availability", "id": f"{label}-{index}",
                  "args": {"start": f"2026-11-{day + index}T13:00:00+08:00",
                           "end": f"2026-11-{day + index}T14:00:00+08:00"}}
                 for index in range(2)]
        graph, _ = _graph([AIMessage(content="", tool_calls=calls),
                           AIMessage(content="查询完成")], settings)
        return await _invoke_at_evaluation_time(
            run_agent(graph, label, "clock-user", "clock-user", "查日历",
                      thread_id=f"clock-{label}", origin="telegram"),
            now=now, zone_name="Asia/Shanghai", settings=settings)

    second = BEFORE + timedelta(days=2)
    assert await asyncio.gather(one(BEFORE, "first"), one(second, "second")) == [
        "查询完成", "查询完成"]
    assert len(seen) == 4
    for start, instant in seen:
        expected = BEFORE if int(start[8:10]) in {10, 11} else second
        assert instant == expected.astimezone(timezone.utc)
