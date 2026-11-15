"""Opt-in real provider extraction; no Calendar or Telegram side effect."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agent.planner import analyze_route, route_request
from app.config import get_settings


@pytest.mark.llm_integration
@pytest.mark.asyncio
async def test_real_llm_extracts_calendar_todo_reminder_triplet():
    route = await route_request(
        "周五论文截止，看看周四下午有没有空，安排两小时让我修改论文，并在周四晚上八点提醒我。",
        get_settings().default_timezone,
    )
    assert route.intent == "cross_domain"
    assert route.todo_operation == "create"
    assert route.title
    assert route.date_reference
    assert route.day_part == "afternoon"
    assert route.duration_minutes == 120
    assert route.todo_due_date_reference
    assert route.reminder_requested
    assert route.reminder_date_reference
    assert route.reminder_hour == 20
    assert (route.weekday, route.todo_due_weekday, route.reminder_weekday) == (3, 4, 3)
    plan = analyze_route(route, get_settings().default_timezone)
    assert [step.operation for step in plan.actions] == ["availability", "create_event", "create_todo"]
    todo = plan.actions[-1]
    assert todo.args["due_date"] != todo.args["reminder_at"]
    reminder_local = datetime.fromisoformat(todo.args["reminder_at"]).astimezone(
        ZoneInfo(get_settings().default_timezone)
    )
    assert (reminder_local.hour, reminder_local.minute) == (20, 0)


@pytest.mark.llm_integration
@pytest.mark.asyncio
async def test_real_llm_routes_todo_delete_to_safe_action():
    route = await route_request("删除任务：Telegram验收论文", get_settings().default_timezone)
    assert route.intent == "todo"
    assert route.todo_operation == "delete"
    assert route.title == "Telegram验收论文" or route.query == "Telegram验收论文"
    plan = analyze_route(route, get_settings().default_timezone)
    assert [step.operation for step in plan.actions] == ["find_todo", "delete_todo"]


@pytest.mark.llm_integration
@pytest.mark.asyncio
async def test_real_llm_routes_bulk_tag_delete_to_safe_action():
    route = await route_request("删除所有标签 #work 的待办任务", get_settings().default_timezone)
    assert route.intent == "todo" and route.todo_operation == "delete"
    assert route.todo_tag == "work"
    plan = analyze_route(route, get_settings().default_timezone)
    assert [step.operation for step in plan.actions] == ["find_todos_by_tag", "delete_todos_by_tag"]
