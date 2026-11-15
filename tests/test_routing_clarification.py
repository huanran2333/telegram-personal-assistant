"""Clarification is separate from authorization and exact source resolution."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app.agent.planner import (CoreIntent, RouteModelError, RoutedRequest,
                               classify_core_intent, analyze_route, route_request,
                               selected_calendar_route_hint, recent_todo_route_hint)


NOW = datetime(2026, 10, 7, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
RECENT_EVENT = {"id": "synthetic-event-1", "title": "合成项目同步会",
                "start": "2026-10-08T14:00:00+08:00", "end": "2026-10-08T15:00:00+08:00"}


@pytest.mark.asyncio
async def test_core_route_provider_failure_is_not_misreported_as_user_ambiguity():
    class BrokenModel:
        def with_structured_output(self, schema):
            assert schema is CoreIntent
            return self

        async def ainvoke(self, messages):
            raise ConnectionError("synthetic provider outage")

    with pytest.raises(RouteModelError, match="model could not safely classify") as caught:
        await classify_core_intent("设置合成提醒", BrokenModel())
    assert isinstance(caught.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_legacy_todo_graph_reports_model_failure_without_retrying_user_text(monkeypatch):
    from langchain_core.messages import HumanMessage
    from langgraph.store.memory import InMemoryStore
    from app.agent import nodes
    from app.agent import planner
    from app.localization import tr

    class DummyModel:
        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *args, **kwargs: DummyModel())
    async def unavailable(*_args, **_kwargs):
        raise RouteModelError("synthetic model failure")

    monkeypatch.setattr(planner, "classify_core_intent", unavailable)
    classify, ask = nodes.make_nodes([], llm_tools=[])[:2]
    state = {"messages": [HumanMessage(content="PRIVATE_TEST_TEXT")], "user_id": "synthetic-user"}
    classification = await classify(state, InMemoryStore())
    assert classification["classification_failure_kind"] == "model"
    result = await ask({**state, **classification})
    assert result["messages"][0].content == tr("model_unavailable")
    assert "PRIVATE_TEST_TEXT" not in result["messages"][0].content


class FixedRouteModel:
    def __init__(self, route: RoutedRequest):
        self.route = route
        self.calls = 0

    def with_structured_output(self, schema):
        assert schema is RoutedRequest
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.route.model_dump()


@pytest.mark.parametrize(("prompt", "route"), [
    ("把明天下午那个安排处理一下。", RoutedRequest(
        intent="calendar_delete", title="明天下午那个安排", query="明天下午那个安排",
        date_reference="tomorrow", day_part="afternoon", confidence=.95)),
    ("跟进一下 Alice 那件事。", RoutedRequest(intent="chat", confidence=.95)),
    ("帮我处理一下。", RoutedRequest(intent="chat", confidence=.95)),
    ("明天下午的安排怎么办？", RoutedRequest(intent="calendar_read", date_reference="tomorrow", confidence=.95)),
    ("把会议删了。", RoutedRequest(intent="calendar_delete", title="把会议删了", query="把会议删了", confidence=.95)),
    ("把任务改一下。", RoutedRequest(intent="todo", todo_operation="update", title="把任务改一下", confidence=.95)),
    ("这个不要了。", RoutedRequest(intent="todo", todo_operation="delete", title="这个不要了", confidence=.95)),
    ("取消一下。", RoutedRequest(intent="chat", confidence=.95)),
    ("跟进那个事情。", RoutedRequest(intent="chat", confidence=.95)),
    ("请再说清楚。", RoutedRequest(intent="clarify", clarification_kind="intent")),
])
@pytest.mark.asyncio
async def test_ambiguous_request_rejected_even_when_model_is_confident(prompt, route):
    model = FixedRouteModel(route)
    with pytest.raises(ValueError, match="safely understand"):
        await route_request(prompt, "Asia/Shanghai", llm=model, now=NOW)
    assert model.calls == 1


@pytest.mark.parametrize(("prompt", "route", "expected_mode"), [
    ("删除名为合成项目面试笔记的待办。", RoutedRequest(
        intent="todo", todo_operation="delete", title="合成项目面试笔记"), "todo_delete"),
    ("删除明天下午两点的合成项目同步会。", RoutedRequest(
        intent="calendar_delete", query="合成项目同步会", date_reference="tomorrow", hour=14), "calendar_delete"),
    ("取消合成项目同步会。", RoutedRequest(
        intent="calendar_delete", query="合成项目同步会"), "calendar_delete"),
    ("删除明天下午两点的会议。", RoutedRequest(
        intent="calendar_delete", query="会议", date_reference="tomorrow", hour=14), "calendar_delete"),
])
@pytest.mark.asyncio
async def test_explicit_action_keeps_source_resolution_and_hitl(prompt, route, expected_mode):
    model = FixedRouteModel(route)
    selected = await route_request(prompt, "Asia/Shanghai", llm=model, now=NOW)
    analysis = analyze_route(selected, "Asia/Shanghai", now=NOW, request=prompt)
    assert analysis.mode == expected_mode
    assert analysis.actions[-1].condition in {"exactly_one_match", "approved_id_set"}
    assert model.calls == 1


@pytest.mark.asyncio
async def test_referential_calendar_requires_real_saved_target():
    request = "把刚才那个会议改到明天下午三点。"
    model = FixedRouteModel(RoutedRequest(
        intent="calendar_update", query="刚才那个会议", relative_event=True,
        date_reference="tomorrow", hour=15))
    selected = await route_request(request, "Asia/Shanghai", llm=model, now=NOW)
    with pytest.raises(ValueError, match="identify which Calendar event"):
        analyze_route(selected, "Asia/Shanghai", now=NOW, request=request)
    analysis = analyze_route(selected, "Asia/Shanghai", now=NOW,
                             recent_event=RECENT_EVENT, request=request)
    assert analysis.mode == "calendar_update"
    assert analysis.actions[-1].args["_recent_event_id"] == RECENT_EVENT["id"]


@pytest.mark.asyncio
async def test_valid_todo_follow_up_remains_in_scoped_todo_path():
    request = "把刚才那个待办标记完成。"
    model = FixedRouteModel(RoutedRequest(
        intent="todo", todo_operation="update", title="刚才那个待办"))
    selected = await route_request(request, "Asia/Shanghai", llm=model, now=NOW)
    assert selected.intent == "todo"
    assert selected.todo_operation == "update"
    assert model.calls == 1


def test_selected_calendar_hint_requires_latest_completed_source():
    workflow = SimpleNamespace(
        values={"recent_event": RECENT_EVENT, "route": {"intent": "calendar_create"}},
        next=(), created_at="2026-10-08T08:00:00+00:00")
    older_todo = SimpleNamespace(values={"intent": "create"}, next=(),
                                 created_at="2026-10-08T07:00:00+00:00")
    newer_todo = SimpleNamespace(values={"intent": "update"}, next=(),
                                 created_at="2026-10-08T09:00:00+00:00")
    assert selected_calendar_route_hint(workflow, older_todo)
    assert not selected_calendar_route_hint(workflow, newer_todo)
    assert recent_todo_route_hint(workflow, newer_todo)
    assert not recent_todo_route_hint(workflow, older_todo)
    assert not selected_calendar_route_hint(
        SimpleNamespace(**{**vars(workflow), "next": ("approval",)}), older_todo)
    assert not selected_calendar_route_hint(SimpleNamespace(
        values={"recent_event": RECENT_EVENT, "route": {"intent": "email_search"}},
        next=(), created_at=workflow.created_at), older_todo)


@pytest.mark.asyncio
async def test_selected_calendar_pronoun_shift_uses_literal_relative_hour_not_model_clock():
    prompt = "把它推迟一小时。"
    model = FixedRouteModel(RoutedRequest(
        intent="calendar_update", query="把它推迟一小时", relative_event=True,
        hour=13))  # model-invented 13:00 must not become the new time
    selected = await route_request(prompt, "Asia/Shanghai", llm=model, now=NOW,
                                   selected_calendar_event=True)
    with pytest.raises(ValueError, match="identify which Calendar event"):
        analyze_route(selected, "Asia/Shanghai", now=NOW, request=prompt)
    analysis = analyze_route(selected, "Asia/Shanghai", now=NOW,
                             recent_event=RECENT_EVENT, request=prompt)
    assert analysis.actions[0].args["query"] == RECENT_EVENT["title"]
    assert analysis.actions[-1].args["start"] == "2026-10-08T15:00:00+08:00"
    assert analysis.actions[-1].args["end"] == "2026-10-08T16:00:00+08:00"
    assert analysis.actions[-1].args["_recent_event_id"] == RECENT_EVENT["id"]


def test_saved_pronoun_referent_keeps_exact_id_when_model_omits_relative_flag():
    route = RoutedRequest(intent="calendar_update", query="它", hour=13,
                          relative_event=False)
    plan = analyze_route(route, "Asia/Shanghai", now=NOW,
                         recent_event=RECENT_EVENT, request="把它推迟一小时。")
    assert plan.actions[0].args["query"] == RECENT_EVENT["title"]
    assert plan.actions[-1].args["_recent_event_id"] == RECENT_EVENT["id"]
    assert plan.actions[-1].args["start"] == "2026-10-08T15:00:00+08:00"
