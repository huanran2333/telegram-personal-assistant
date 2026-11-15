"""Opt-in Calendar scope/exact-version read; does not send event data to an LLM."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.agent.calendar import MCPCalendarAdapter, discover_calendar_tools
from app.agent.calendar_conditional import GoogleCalendarConditionalTransport, McpOAuthTokenSource
from app.config import get_settings

pytestmark = [pytest.mark.calendar_integration, pytest.mark.asyncio]


async def test_readable_scope_and_exact_get_preserve_event_identity():
    settings = get_settings()
    calendar = MCPCalendarAdapter(
        await discover_calendar_tools(settings),
        conditional=GoogleCalendarConditionalTransport(
            McpOAuthTokenSource(settings.google_credentials_file)))
    account, calendar_id = await calendar.resolve_read_scope("primary")
    assert account and calendar_id and calendar_id != "primary"
    start = datetime.now(ZoneInfo(settings.default_timezone)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    candidates = await calendar.search_events("", start.isoformat(),
                                               (start + timedelta(days=30)).isoformat())
    if not candidates:
        pytest.skip("No existing Calendar event in the bounded read window")
    first = next((item for item in candidates[:50] if item.get("id")), None)
    if first is None:
        pytest.skip("Calendar returned no event with an exact identity")
    snapshot = await calendar.get_event(calendar_id, first["id"])
    assert snapshot.id == first["id"] and snapshot.etag and snapshot.status


async def test_readable_calendar_scope_is_resolved_without_write():
    settings = get_settings()
    calendar = MCPCalendarAdapter(
        [], conditional=GoogleCalendarConditionalTransport(
            McpOAuthTokenSource(settings.google_credentials_file)))
    account, calendar_id = await calendar.resolve_read_scope("primary")
    assert account and calendar_id and calendar_id != "primary"
