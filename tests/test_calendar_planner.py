"""Deterministic planner boundaries; no external services."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.agent.actions import ActionProposal, RiskLevel
from app.agent.planner import (CalendarProposalEdit, RoutedRequest, analyze_route,
                               apply_edit, normalize_weekday_phrases, proposal_from_plan,
                               preserve_explicit_calendar_date, validate_email_action_intent,
                               resolve_window, route_request)

NOW = datetime(2026, 9, 30, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.mark.parametrize(("reference", "weekday", "expected"), [
    ("today", None, "2026-09-30"),
    ("tomorrow", None, "2026-10-01"),
    ("day_after_tomorrow", None, "2026-10-02"),
    ("weekday", 4, "2026-10-02"),
    ("next_weekday", 4, "2026-10-09"),
])
def test_relative_date_normalization(reference, weekday, expected):
    route = RoutedRequest(intent="calendar_read", date_reference=reference, weekday=weekday)
    start, end = resolve_window(route, "Asia/Shanghai", NOW)
    assert start.date().isoformat() == expected
    assert start.tzinfo == ZoneInfo("Asia/Shanghai")
    assert end - start == timedelta(days=1)


def test_tomorrow_afternoon_range():
    route = RoutedRequest(intent="calendar_read", date_reference="tomorrow", day_part="afternoon")
    start, end = resolve_window(route, "Asia/Shanghai", NOW)
    assert (start.hour, end.hour) == (12, 18)
    assert start.date().isoformat() == "2026-10-01"


def test_thursday_evening_eight_local_time():
    route = RoutedRequest(intent="calendar_read", date_reference="weekday", weekday=3,
                          day_part="evening", hour=20)
    start, end = resolve_window(route, "Asia/Shanghai", NOW)
    assert start.isoformat() == "2026-10-01T20:00:00+08:00"
    assert end - start == timedelta(hours=1)


def test_literal_chinese_weekdays_override_model_one_based_numbers():
    route = RoutedRequest(
        intent="cross_domain", date_reference="weekday", weekday=4,
        todo_due_date_reference="weekday", todo_due_weekday=5,
        reminder_date_reference="weekday", reminder_weekday=4,
    )
    corrected = normalize_weekday_phrases(
        route, "周五论文截止，看看周四下午有没有空，安排两小时修改论文，周四晚上八点提醒我"
    )
    assert (corrected.weekday, corrected.todo_due_weekday, corrected.reminder_weekday) == (3, 4, 3)


def test_literal_chinese_weekday_text_takes_precedence():
    route = RoutedRequest(intent="cross_domain", date_reference="weekday", weekday=4,
                          calendar_date_text="周四")
    assert normalize_weekday_phrases(route, "看看周四下午有没有空").weekday == 3


def test_explicit_create_title_duration_and_policy():
    route = RoutedRequest(intent="calendar_create", title="测试会议", date_reference="tomorrow",
                          day_part="afternoon", hour=16, duration_minutes=30)
    plan = analyze_route(route, "Asia/Shanghai", NOW)
    assert [step.operation for step in plan.actions] == ["availability", "create_event"]
    create = plan.actions[-1]
    assert create.args["title"] == "测试会议"
    assert datetime.fromisoformat(create.args["start"]).hour == 16
    assert datetime.fromisoformat(create.args["end"]) - datetime.fromisoformat(create.args["start"]) == timedelta(minutes=30)
    proposal = proposal_from_plan(plan, {"availability": {"calendars": {"primary": {"busy": []}}}}, "Asia/Shanghai")
    assert proposal.risk_level.value == "external_write"


def test_recent_event_resolution_and_ambiguous_clarification():
    recent = {"id": "recent-1", "title": "测试会议",
              "start": "2026-10-01T16:00:00+08:00", "end": "2026-10-01T16:30:00+08:00"}
    route = RoutedRequest(intent="calendar_update", query="测试会议", relative_event=True, hour=18)
    plan = analyze_route(route, "Asia/Shanghai", NOW, recent)
    assert plan.actions[-1].args["start"] == "2026-10-01T18:00:00+08:00"
    assert plan.actions[-1].args["end"] == "2026-10-01T18:30:00+08:00"
    matches = [{"id": "recent-1", "summary": "测试会议"}, {"id": "other", "summary": "测试会议"}]
    proposal = proposal_from_plan(plan, {"events": matches}, "Asia/Shanghai", recent)
    assert proposal.args["eventId"] == "recent-1"
    assert "_recent_event_id" not in proposal.args
    imperfect_route = RoutedRequest(intent="calendar_update", query="刚才那个测试会议",
                                    date_reference="today", hour=18)
    inferred = analyze_route(imperfect_route, "Asia/Shanghai", NOW, recent,
                             request="把刚才那个测试会议改到下午 6 点。")
    assert inferred.actions[0].args["query"] == "测试会议"
    assert inferred.actions[-1].args["start"] == "2026-10-01T18:00:00+08:00"
    without_context = analyze_route(RoutedRequest(intent="calendar_delete", query="测试会议"), "Asia/Shanghai", NOW)
    assert proposal_from_plan(without_context, {"events": matches}, "Asia/Shanghai") is None


def test_email_calendar_mutation_intent_and_explicit_date_survive_routing():
    request = ("根据主题包含 EVAL-SUBJECT-20261003 的邮件，把讨论安排在 "
               "2026年10月9日15:00–15:30（Asia/Shanghai），先给我提案，不要创建")
    model_output = RoutedRequest(intent="email_cross_domain", email_calendar_check=True,
                                 email_query='subject:"EVAL-SUBJECT-20261003"',
                                 date_reference="2026-10-10", hour=15)
    route = preserve_explicit_calendar_date(validate_email_action_intent(model_output, request), request)
    assert route.intent == "email_to_calendar"
    assert route.date_reference == "2026-10-09"
    assert [step.operation for step in analyze_route(route, "Asia/Shanghai", NOW).actions] == [
        "search", "create_event"]


@pytest.mark.asyncio
async def test_structured_route_cannot_silently_downgrade_email_calendar_mutation():
    class MisclassifiedLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            return RoutedRequest(intent="email_cross_domain", email_calendar_check=True,
                                 email_query='subject:"EVAL-SUBJECT-20261003"',
                                 date_reference="2026-10-10", hour=15).model_dump()

    request = ("根据主题包含 EVAL-SUBJECT-20261003 的邮件，把讨论安排在 "
               "2026年10月9日15:00–15:30（Asia/Shanghai），先给我提案，不要创建")
    route = await route_request(request, "Asia/Shanghai", llm=MisclassifiedLLM(), now=NOW)
    assert route.intent == "email_to_calendar"
    assert route.date_reference == "2026-10-09"


def test_deictic_calendar_delete_without_referent_requires_clarification():
    with pytest.raises(ValueError, match="identify which Calendar event"):
        analyze_route(RoutedRequest(intent="calendar_delete", query="那个会议"),
                      "Asia/Shanghai", NOW, request="删除那个会议")
    dated = analyze_route(RoutedRequest(intent="calendar_delete", query="会议",
                                        date_reference="tomorrow"), "Asia/Shanghai", NOW,
                          request="删除明天下午那个会议")
    assert proposal_from_plan(dated, {"events": [
        {"id": "a", "summary": "会议"}, {"id": "b", "summary": "会议"}]}, "Asia/Shanghai") is None


def test_update_delete_plan_generation_and_fail_closed():
    deletion = analyze_route(RoutedRequest(intent="calendar_delete", query="测试会议",
                                           date_reference="tomorrow"), "Asia/Shanghai", NOW)
    assert [step.operation for step in deletion.actions] == ["search_events", "delete_event"]
    assert deletion.actions[-1].mutation
    create = analyze_route(RoutedRequest(intent="calendar_create", title="测试会议",
                                         date_reference="tomorrow", hour=16), "Asia/Shanghai", NOW)
    assert proposal_from_plan(create, {"error": "OAuth failed"}, "Asia/Shanghai") is None
    assert proposal_from_plan(create, {"availability": {"calendars": {"primary": {"busy": [{"start": "x"}]}}}}, "Asia/Shanghai") is None


def test_cross_domain_finds_a_free_two_hour_slot_within_afternoon():
    route = RoutedRequest(intent="cross_domain", title="修改论文", date_reference="tomorrow",
                          day_part="afternoon", duration_minutes=120)
    plan = analyze_route(route, "UTC", NOW)
    freebusy = {"availability": {"calendars": {"primary": {"busy": [
        {"start": "2026-10-01T12:30:00+00:00", "end": "2026-10-01T14:00:00+00:00"},
    ]}}}}
    proposal = proposal_from_plan(plan, freebusy, "UTC")
    assert proposal.args["start"] == "2026-10-01T14:00:00+00:00"
    assert proposal.args["end"] == "2026-10-01T16:00:00+00:00"
    assert "_window_end" not in proposal.args


def test_cross_domain_child_todo_uses_optional_deadline_contract():
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="Synthetic planning meeting", todo_title="Write follow-up",
                          date_reference="tomorrow", hour=14)
    request = "明天下午两点安排 Synthetic planning meeting，并创建 Write follow-up 待办。"
    plan = analyze_route(route, "Asia/Shanghai", NOW, request=request)
    assert [step.operation for step in plan.actions] == ["availability", "create_event", "create_todo"]
    assert plan.actions[-1].args == {"title": "Write follow-up"}


def test_calendar_read_then_todo_cannot_become_unrequested_calendar_write():
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="Synthetic meeting", todo_title="Follow up",
                          date_reference="tomorrow", day_part="afternoon", read_kind="events")
    with pytest.raises(ValueError, match="no Calendar write was requested"):
        analyze_route(route, "Asia/Shanghai", NOW,
                      request="查看明天下午的会议，然后创建一个跟进合成项目的待办。")
    with pytest.raises(ValueError, match="no Calendar write was requested"):
        analyze_route(route, "Asia/Shanghai", NOW,
                      request="查看明天的安排有没有会议，然后创建跟进待办。")


def test_explicit_calendar_write_after_read_is_not_overblocked():
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="合成项目同步会", todo_title="写准备材料",
                          date_reference="tomorrow", day_part="afternoon")
    plan = analyze_route(route, "Asia/Shanghai", NOW,
                         request="查看明天下午日历空档，再安排合成项目同步会并创建写准备材料的待办。")
    assert [step.operation for step in plan.actions] == ["availability", "create_event", "create_todo"]


@pytest.mark.asyncio
async def test_structured_routing_schema_without_network():
    class Structured:
        async def ainvoke(self, messages):
            assert messages[-1].content == "明天下午 4 点安排测试会议"
            return {"intent": "calendar_create", "title": "测试会议", "date_reference": "tomorrow",
                    "day_part": "afternoon", "hour": 16, "duration_minutes": 30}

    class LLM:
        def with_structured_output(self, schema):
            assert schema is RoutedRequest
            return Structured()

    route = await route_request("明天下午 4 点安排测试会议", "Asia/Shanghai", LLM(), NOW)
    assert route.intent == "todo" and route.todo_operation == "create"
    assert route.title == "测试会议"
    assert route.planned_start_str and route.planned_end_str
    assert (datetime.fromisoformat(route.planned_end_str) -
            datetime.fromisoformat(route.planned_start_str)).total_seconds() == 1800


@pytest.mark.asyncio
async def test_explicit_standalone_google_event_keeps_approved_workflow():
    class Structured:
        async def ainvoke(self, messages):
            return {"intent": "calendar_create", "title": "测试会议", "date_reference": "tomorrow",
                    "hour": 16, "duration_minutes": 30}

    class LLM:
        def with_structured_output(self, schema):
            assert schema is RoutedRequest
            return Structured()

    route = await route_request("明天 16 点在 Google Calendar 创建独立事件：测试会议",
                                "Asia/Shanghai", LLM(), NOW)
    assert route.intent == "calendar_create"


def test_second_classifier_only_normalizes_same_timed_create():
    from app.agent.planner import CoreIntent, core_personal_time_block

    outer = {"intent": "todo", "todo_operation": "create", "title": "测试会议",
             "planned_start_str": "2026-10-15T14:00:00+08:00"}
    second = CoreIntent(intent="calendar_create", title="测试会议")
    result = core_personal_time_block(second, "明天14点开测试会议", outer)
    assert (result.intent, result.todo_operation, result.planned_start_str) == (
        "todo", "create", outer["planned_start_str"])
    assert core_personal_time_block(
        CoreIntent(intent="calendar_create"),
        "在 Google Calendar 创建独立事件", outer).intent == "calendar_create"
    assert core_personal_time_block(
        CoreIntent(intent="todo", todo_operation="delete"),
        "删除测试会议", outer).todo_operation == "delete"


@pytest.mark.asyncio
async def test_explicit_chinese_clock_recovers_empty_edit_extraction():
    class EmptyStructured:
        async def ainvoke(self, messages):
            return {}

    class LLM:
        def with_structured_output(self, schema):
            assert schema is CalendarProposalEdit
            return EmptyStructured()

    original = ActionProposal(domain="calendar", operation="create_event",
                              args={"title": "会议", "start": "2026-10-01T17:00:00+00:00",
                                    "end": "2026-10-01T17:30:00+00:00", "timeZone": "UTC"},
                              human_readable_preview="Create meeting", reason="test",
                              risk_level=RiskLevel.EXTERNAL_WRITE)
    revised = await apply_edit(original, "时间改为下午五点半开始", "UTC", LLM(), NOW)
    assert revised.args["start"] == "2026-10-01T17:30:00+00:00"
    assert revised.args["end"] == "2026-10-01T18:00:00+00:00"
    assert revised.action_id != original.action_id
    assert original.args["start"] == "2026-10-01T17:00:00+00:00"
