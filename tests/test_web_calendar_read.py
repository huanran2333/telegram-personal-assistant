"""The Web's real FullCalendar timestamps must satisfy the pinned MCP schema."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from app.web import calendar as calendar_api


def test_month_calendar_window_accepts_six_weeks_but_not_unbounded_read():
    start, end = calendar_api._window("2026-10-25T00:00:00Z", "2026-12-06T00:00:00Z")
    assert (end - start).days == 42
    with pytest.raises(HTTPException) as error:
        calendar_api._window("2026-10-25T00:00:00Z", "2026-12-08T00:00:00Z")
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_bounded_calendar_read_normalizes_fullcalendar_milliseconds(monkeypatch):
    calls = []

    class ReadOnlyCalendar:
        async def resolve_read_scope(self, identity):
            assert identity == "primary"
            return "test-account", "test-calendar"

        async def search_events(self, query, start, end):
            calls.append((query, start, end))
            return [{"id": "synthetic-event", "summary": "synthetic test"}]

    async def calendar_for(_request):
        return ReadOnlyCalendar()

    monkeypatch.setattr(calendar_api, "_calendar_for", calendar_for)
    query = b"from=2026-10-09T00%3A00%3A00.000Z"
    request = Request({"type": "http", "method": "GET", "path": "/api/v3/calendar/events",
                       "query_string": query, "headers": []})
    result = await calendar_api.calendar_events(
        request, Response(), to="2026-10-16T00:00:00.000Z")
    assert result["events"][0]["event_id"] == "synthetic-event"
    assert calls == [("", "2026-10-09T00:00:00+00:00", "2026-10-16T00:00:00+00:00")]
