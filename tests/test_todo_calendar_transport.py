"""No network: exact Google REST contract for opt-in Todo projections."""

import httpx
import pytest

from app.agent.calendar_conditional import (
    CalendarProviderError, GoogleCalendarConditionalTransport,
)


@pytest.mark.asyncio
async def test_time_block_create_has_stable_id_and_disables_google_alerts():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        assert request.method == "POST"
        assert request.url.path == "/calendar/v3/calendars/owner@example.test/events"
        assert "owner%40example.test" in str(request.url)
        assert request.url.params["sendUpdates"] == "none"
        return httpx.Response(200, json={
            "id": "pstableid", "etag": '"v1"', "status": "confirmed",
            "summary": "synthetic", "start": {"dateTime": "2026-10-15T14:00:00+08:00"},
            "end": {"dateTime": "2026-10-15T15:00:00+08:00"},
            "extendedProperties": {"private": {"pa_todo_id": "synthetic-todo"}},
        })

    transport = GoogleCalendarConditionalTransport(
        lambda: token(),
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    body = {"id": "pstableid", "summary": "synthetic",
            "start": {"dateTime": "2026-10-15T14:00:00+08:00"},
            "end": {"dateTime": "2026-10-15T15:00:00+08:00"},
            "extendedProperties": {"private": {"pa_todo_id": "synthetic-todo"}},
            "reminders": {"useDefault": False, "overrides": []}}
    result = await transport.create_time_block("owner@example.test", "pstableid", body)
    assert result.id == "pstableid" and result.etag == '"v1"'
    assert len(seen) == 1 and seen[0].headers["Authorization"] == "Bearer synthetic-token"
    assert b'"useDefault":false' in seen[0].content


async def token():
    return "synthetic-token"


@pytest.mark.asyncio
async def test_time_block_create_conflict_is_not_reported_as_success():
    transport = GoogleCalendarConditionalTransport(
        token,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(409))))
    with pytest.raises(CalendarProviderError) as exc:
        await transport.create_time_block("synthetic-calendar", "pstableid", {"id": "pstableid"})
    assert exc.value.category == "conflict"
