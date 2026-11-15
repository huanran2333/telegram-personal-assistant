from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.calendar import CalendarError
from app.agent.actions import ActionResult, ActionStatus
from app.agent.executor import ActionExecutor
from app.agent.planner import RoutedRequest, analyze_route, normalize_weekday_phrases, route_request
from app.agent.scheduling import SchedulingConstraints, find_candidate_slots
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow


class FakeLedger:
    def __init__(self):
        self.results = {}

    async def begin(self, proposal):
        if proposal.action_id in self.results:
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.EXECUTED,
                                message="Already processed", result=self.results[proposal.action_id])
        return None

    async def complete(self, proposal, result):
        self.results[proposal.action_id] = result

    async def fail(self, proposal, message):
        pass


NOW = datetime(2026, 10, 1, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


class SchedulingCalendar:
    def __init__(self):
        self.busy = [{"start": "2026-10-01T14:00:00+08:00",
                      "end": "2026-10-01T15:00:00+08:00"}]
        self.calls = []
        self.created = []
        self.fail = False

    async def availability(self, start, end, timezone):
        self.calls.append((start, end, timezone))
        if self.fail:
            raise CalendarError("OAuth unavailable")
        return {"calendars": {"primary": {"busy": self.busy}}}

    async def create_event(self, args):
        self.created.append(args)
        return {"id": f"event-{len(self.created)}"}


def test_constraint_extraction_and_weekday_exclusion():
    route = RoutedRequest(intent="smart_schedule", title="改论文", duration_minutes=120,
                          schedule_range="this_week", preferred_day_parts=["afternoon"])
    normalized = normalize_weekday_phrases(route, "这周找两小时改论文，最好下午，不要周五")
    plan = analyze_route(normalized, "Asia/Shanghai", NOW)
    assert [step.operation for step in plan.actions] == ["availability", "choose_slot", "create_event"]
    assert plan.actions[1].args["excluded_weekdays"] == [4]
    assert plan.actions[1].args["range_end"].startswith("2026-10-05")


@pytest.mark.asyncio
async def test_structured_llm_scheduling_route_with_realistic_constraints():
    class LLM:
        def with_structured_output(self, schema):
            assert schema is RoutedRequest
            return self

        async def ainvoke(self, messages):
            assert "这周找一个两小时" in messages[-1].content
            return {"intent": "smart_schedule", "title": "改论文", "duration_minutes": 120,
                    "schedule_range": "this_week", "preferred_day_parts": ["afternoon"],
                    "excluded_weekdays": [4]}

    route = await route_request("这周找一个两小时让我改论文，最好下午，不要周五", "Asia/Shanghai", LLM(), NOW)
    assert route.intent == "smart_schedule"
    assert route.excluded_weekdays == [4]


def test_slot_enumeration_is_deterministic_and_conflict_free():
    plan = analyze_route(RoutedRequest(intent="smart_schedule", title="改论文", duration_minutes=120,
                                       schedule_range="this_week", preferred_day_parts=["afternoon"],
                                       excluded_weekdays=[4]), "Asia/Shanghai", NOW)
    constraints = SchedulingConstraints.model_validate(plan.actions[1].args)
    busy = {"calendars": {"primary": {"busy": [
        {"start": "2026-10-01T14:00:00+08:00", "end": "2026-10-01T15:00:00+08:00"}
    ]}}}
    slots = find_candidate_slots(constraints, busy, now=NOW)
    assert 2 <= len(slots) <= 3
    assert all(datetime.fromisoformat(slot["start"]).weekday() != 4 for slot in slots)
    assert all(not (datetime.fromisoformat(slot["start"]).hour == 14) for slot in slots)
    assert len({datetime.fromisoformat(slot["start"]).date() for slot in slots}) == len(slots)


def test_afternoon_preference_does_not_rank_noon_as_afternoon():
    constraints = SchedulingConstraints(
        title="Review", duration_minutes=60,
        range_start=datetime(2026, 10, 5, 12, tzinfo=ZoneInfo("Asia/Shanghai")),
        range_end=datetime(2026, 10, 5, 17, tzinfo=ZoneInfo("Asia/Shanghai")),
        timezone="Asia/Shanghai", preferred_day_parts=["afternoon"],
    )
    freebusy = {"calendars": {"primary": {"busy": []}}}
    slots = find_candidate_slots(constraints, freebusy, now=NOW)
    assert [datetime.fromisoformat(slot["start"]).strftime("%H:%M") for slot in slots] == [
        "13:00", "14:00", "15:00",
    ]


def test_calendar_error_is_not_a_free_slot():
    constraints = SchedulingConstraints(title="Write", duration_minutes=60,
                                        range_start=NOW, range_end=NOW.replace(day=2),
                                        timezone="Asia/Shanghai")
    with pytest.raises(CalendarError):
        find_candidate_slots(constraints, {})
    with pytest.raises(CalendarError):
        find_candidate_slots(constraints, {"calendars": {"primary": {"busy": [{"start": "bad"}]}}})
    with pytest.raises(CalendarError):
        find_candidate_slots(constraints, {"calendars": {"primary": {"busy": [
            {"start": "2026-10-01T11:00:00", "end": "2026-10-01T12:00:00"}
        ]}}})


def test_scheduling_rejects_naive_range_and_invalid_weekday():
    with pytest.raises(ValueError):
        SchedulingConstraints(title="Write", duration_minutes=60,
                              range_start=NOW.replace(tzinfo=None), range_end=NOW,
                              timezone="Asia/Shanghai")
    with pytest.raises(ValueError):
        SchedulingConstraints(title="Write", duration_minutes=60,
                              range_start=NOW, range_end=NOW + timedelta(days=1),
                              timezone="Asia/Shanghai", excluded_weekdays=[7])


@pytest.mark.asyncio
async def test_malformed_smart_schedule_analysis_does_not_guess_free_time():
    class MalformedLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            return {"intent": "smart_schedule", "duration_minutes": "not-a-duration"}

    with pytest.raises(ValueError, match="could not safely understand"):
        await route_request("帮我找一段时间", "Asia/Shanghai", MalformedLLM(), NOW)


@pytest.mark.asyncio
async def test_choose_then_approve_uses_existing_hitl_and_fresh_freebusy():
    calendar = SchedulingCalendar()
    graph = build_productivity_graph(calendar, ActionExecutor(calendar, FakeLedger(), None), InMemorySaver())
    tomorrow = datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    route = RoutedRequest(intent="smart_schedule", title="修改论文", duration_minutes=120,
                          range_start_date=tomorrow.isoformat(),
                          range_end_date=(tomorrow + timedelta(days=2)).isoformat(),
                          preferred_day_parts=["afternoon"])
    first = await start_workflow(graph, "u1", "这周找两小时改论文，最好下午，不要周五", "Asia/Shanghai",
                                 "smart-1", route=route)
    assert first["__interrupt__"][0].value["kind"] == "slot_selection"
    assert calendar.created == []
    chosen = await resume_workflow(graph, "smart-1", {"selection_index": 0})
    assert chosen["__interrupt__"][0].value["kind"] == "action_proposal"
    assert calendar.created == []
    assert len(calendar.calls) == 1
    await resume_workflow(graph, "smart-1", {"decision": "approve"})
    assert len(calendar.calls) == 2
    assert len(calendar.created) == 1
    assert calendar.created[0]["title"] == "修改论文"


@pytest.mark.asyncio
async def test_fresh_conflict_or_failure_prevents_write():
    calendar = SchedulingCalendar()
    graph = build_productivity_graph(calendar, ActionExecutor(calendar, FakeLedger(), None), InMemorySaver())
    route = RoutedRequest(intent="smart_schedule", title="Review", duration_minutes=60,
                          schedule_range="tomorrow")
    await start_workflow(graph, "u1", "明天找一小时复习", "Asia/Shanghai", "smart-conflict", route=route)
    await resume_workflow(graph, "smart-conflict", {"selection_index": 0})
    calendar.fail = True
    result = await resume_workflow(graph, "smart-conflict", {"decision": "approve"})
    assert "OAuth unavailable" in result["response"]
    assert calendar.created == []


@pytest.mark.asyncio
async def test_invalid_slot_choice_does_not_propose_or_write():
    calendar = SchedulingCalendar()
    graph = build_productivity_graph(calendar, ActionExecutor(calendar, FakeLedger(), None), InMemorySaver())
    route = RoutedRequest(intent="smart_schedule", title="Review", duration_minutes=60,
                          schedule_range="tomorrow")
    await start_workflow(graph, "u1", "明天找一小时复习", "Asia/Shanghai", "smart-invalid", route=route)
    result = await resume_workflow(graph, "smart-invalid", {"selection_index": 99})
    assert "invalid" in result["response"]
    assert calendar.created == []


@pytest.mark.asyncio
async def test_calendar_oauth_owner_isolated_from_other_telegram_user():
    calendar = SchedulingCalendar()
    graph = build_productivity_graph(calendar, ActionExecutor(calendar, FakeLedger(), None),
                                     InMemorySaver(), calendar_owner_telegram_id="owner-telegram")
    route = RoutedRequest(intent="smart_schedule", title="Private meeting", duration_minutes=60,
                          schedule_range="tomorrow")
    result = await start_workflow(graph, "other-db-user", "明天找时间开会", "Asia/Shanghai",
                                  "other-thread", route=route, telegram_id="other-telegram")
    assert "not connected for this user" in result["response"]
    assert calendar.calls == []
    assert calendar.created == []
