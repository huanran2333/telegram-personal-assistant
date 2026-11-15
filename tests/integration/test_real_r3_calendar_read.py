"""R3 opt-in Calendar MCP read regression; no external writes or event titles."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.agent.calendar import MCPCalendarAdapter, discover_calendar_tools
from app.config import get_settings

pytestmark = [pytest.mark.calendar_integration, pytest.mark.asyncio]


async def test_real_calendar_primary_freebusy_read_only():
    settings = get_settings()
    if not settings.google_calendar_enabled or not settings.google_calendar_owner_telegram_id:
        pytest.skip("Calendar owner mapping is not configured")
    calendar = MCPCalendarAdapter(await discover_calendar_tools(settings))
    zone = ZoneInfo(settings.default_timezone)
    start = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    result = await calendar.availability(start.isoformat(), end.isoformat(),
                                         settings.default_timezone)
    assert isinstance(result["calendars"]["primary"]["busy"], list)
