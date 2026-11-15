"""Opt-in real-provider Gmail routing checks; no Gmail API access."""
import pytest

from app.agent.planner import analyze_route, route_request
from app.config import get_settings


pytestmark = [pytest.mark.llm_integration, pytest.mark.asyncio]


async def test_real_llm_routes_gmail_search_and_draft():
    timezone = get_settings().default_timezone
    search = await route_request("搜索最近关于论文的邮件。", timezone)
    assert search.intent in {"email_search", "email_unread"}
    assert search.email_query
    assert analyze_route(search, timezone).actions[0].domain == "email"

    draft = await route_request("查找最近关于论文的邮件，给我起草一份回复，不要发送。", timezone)
    assert draft.intent in {"email_draft", "email_cross_domain"}
    assert draft.email_query
    assert draft.intent == "email_draft" or draft.email_draft_requested


async def test_real_llm_plans_gmail_calendar_read_without_mutation():
    timezone = get_settings().default_timezone
    route = await route_request("查找最近关于论文的邮件，看看周五下午有没有空，并给我起草回复，不要发送。", timezone)
    assert route.intent == "email_cross_domain"
    assert route.email_query and route.email_calendar_check and route.email_draft_requested
    plan = analyze_route(route, timezone)
    assert [(step.domain, step.operation) for step in plan.actions] == [
        ("email", "search"), ("calendar", "availability")]
    assert all(not step.mutation for step in plan.actions)


async def test_real_llm_routes_email_derived_todo_and_calendar_proposals():
    timezone = get_settings().default_timezone
    todo = await route_request("把主题为 PPA V2 Gmail E2E 的邮件里要求的审阅工作建成待办。", timezone)
    assert todo.intent == "email_to_todo"
    assert todo.email_query
    todo_plan = analyze_route(todo, timezone)
    assert [(step.domain, step.operation) for step in todo_plan.actions] == [
        ("email", "search"), ("todo", "create_todo_from_email")]

    calendar = await route_request("把主题为 PPA V2 Gmail Incoming 的邮件里约定的会议加入日历。", timezone)
    assert calendar.intent == "email_to_calendar"
    assert calendar.email_query
    calendar_plan = analyze_route(calendar, timezone)
    assert [(step.domain, step.operation) for step in calendar_plan.actions] == [
        ("email", "search"), ("calendar", "create_event")]
