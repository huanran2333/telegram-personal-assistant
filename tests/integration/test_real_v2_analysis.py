"""Opt-in provider checks for V2.1 typed extraction; no Google writes."""
import pytest

from app.agent.planner import analyze_route, route_request
from app.config import get_settings

pytestmark = [pytest.mark.llm_integration, pytest.mark.asyncio]


async def test_real_llm_extracts_smart_scheduling_constraints():
    timezone = get_settings().default_timezone
    route = await route_request("这周找一个两小时让我改论文，最好下午，不要周五。", timezone)
    assert route.intent == "smart_schedule"
    assert route.duration_minutes == 120
    assert "afternoon" in route.preferred_day_parts
    assert 4 in route.excluded_weekdays
    plan = analyze_route(route, timezone)
    assert [step.operation for step in plan.actions] == ["availability", "choose_slot", "create_event"]


async def test_real_llm_routes_meeting_prep_to_read_only_plan():
    timezone = get_settings().default_timezone
    route = await route_request("帮我准备明天下午三点和导师的讨论。", timezone)
    assert route.intent == "meeting_prep"
    plan = analyze_route(route, timezone)
    assert all(not step.mutation for step in plan.actions)
    assert [step.operation for step in plan.actions] == ["search_events", "list_related_todos"]
