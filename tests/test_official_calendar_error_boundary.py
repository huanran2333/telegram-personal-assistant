"""OA11's handled exception is deliberately narrower than tool execution failure."""
from types import SimpleNamespace

import httpx
import pytest
from langchain.agents.middleware import ToolErrorMiddleware

from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError, GmailUnavailable
from app.agent.official_graph import _calendar_read_error, _read_tool_error
from app.agent.official_read_tools import _read_with_retry


def test_calendar_read_error_is_sanitized_and_other_errors_propagate():
    message = _calendar_read_error(
        CalendarError("private OAuth refresh token or stack trace"), None)
    assert "未获得可靠结果" in message
    assert "OAuth" not in message
    assert _calendar_read_error(PermissionError("owner mismatch"), None) is None
    assert _calendar_read_error(RuntimeError("write outcome unknown"), None) is None


@pytest.mark.asyncio
async def test_official_error_middleware_does_not_convert_write_or_permission_failure():
    middleware = ToolErrorMiddleware(
        on_error=_calendar_read_error,
        tools=["calendar_events", "calendar_availability",
               "calendar_candidate_slots", "calendar_event"])

    async def raises_calendar(_request):
        raise CalendarError("remote write outcome unknown")

    async def raises_permission(_request):
        raise PermissionError("owner mismatch")

    with pytest.raises(CalendarError, match="remote write outcome unknown"):
        await middleware.awrap_tool_call(
            SimpleNamespace(tool=None, tool_call={
                "name": "propose_calendar_create", "id": "write-call"}),
            raises_calendar)
    with pytest.raises(PermissionError, match="owner mismatch"):
        await middleware.awrap_tool_call(
            SimpleNamespace(tool=None, tool_call={
                "name": "calendar_events", "id": "read-call"}),
            raises_permission)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["calendar", "gmail"])
async def test_http_403_read_is_not_retried_and_explains_permission(domain):
    tool_name = "calendar_events" if domain == "calendar" else "gmail_unread"
    error_type = CalendarError if domain == "calendar" else GmailError
    middleware = ToolErrorMiddleware(on_error=_read_tool_error, tools=[tool_name])
    request = SimpleNamespace(tool=None, tool_call={"name": tool_name, "id": "read-403"})
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if domain == "calendar":
            raise CalendarError("private token=SECRET internal URL") from CalendarProviderError(
                "provider_unavailable", 403)
        provider_request = httpx.Request("GET", "https://internal.example/private")
        response = httpx.Response(403, request=provider_request)
        raise GmailUnavailable("private token=SECRET") from httpx.HTTPStatusError(
            "private Authorization header", request=provider_request, response=response)

    async def handler(_):
        return await _read_with_retry(operation, error_type)

    result = await middleware.awrap_tool_call(request, handler)
    assert calls == 1
    assert result.tool_call_id == "read-403" and result.status == "error"
    assert tool_name in result.content and "permission_denied" in result.content
    assert "权限" in result.content and "未获得可靠结果" in result.content
    assert "SECRET" not in result.content and "internal.example" not in result.content
    assert "Authorization" not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["calendar", "gmail"])
async def test_http_503_read_retries_then_explains_unavailable(domain):
    tool_name = "calendar_events" if domain == "calendar" else "gmail_unread"
    error_type = CalendarError if domain == "calendar" else GmailError
    middleware = ToolErrorMiddleware(on_error=_read_tool_error, tools=[tool_name])
    request = SimpleNamespace(tool=None, tool_call={"name": tool_name, "id": "read-503"})
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if domain == "calendar":
            raise CalendarError("private token=SECRET") from CalendarProviderError(
                "provider_unavailable", 503, "0")
        provider_request = httpx.Request("GET", "https://internal.example/private")
        response = httpx.Response(503, request=provider_request, headers={"Retry-After": "0"})
        raise GmailUnavailable("private token=SECRET") from httpx.HTTPStatusError(
            "private Authorization header", request=provider_request, response=response)

    async def handler(_):
        return await _read_with_retry(operation, error_type)

    result = await middleware.awrap_tool_call(request, handler)
    assert calls == 2
    assert result.tool_call_id == "read-503" and result.status == "error"
    assert tool_name in result.content and "service_unavailable" in result.content
    assert "暂时不可用" in result.content and "未获得可靠结果" in result.content
    assert "SECRET" not in result.content and "internal.example" not in result.content
    assert "Authorization" not in result.content


def test_not_found_is_not_claimed_as_proven_absence_and_unknown_text_is_hidden():
    request = SimpleNamespace(tool_call={"name": "calendar_event", "id": "event-read"})
    try:
        raise CalendarError("private event URL") from CalendarProviderError(
            "not_found_or_inaccessible", 404)
    except CalendarError as exc:
        content = _read_tool_error(exc, request)
    assert "resource_unavailable" in content and "或当前账号无权查看" in content
    assert "private" not in content

    content = _read_tool_error(CalendarError("token=SECRET"), request)
    assert "read_failed" in content and "SECRET" not in content


@pytest.mark.asyncio
async def test_production_read_error_handler_does_not_capture_write_or_unknown_failure():
    middleware = ToolErrorMiddleware(on_error=_read_tool_error,
                                     tools=["calendar_events", "gmail_unread"])

    async def calendar_failure(_):
        raise CalendarError("remote write outcome unknown")

    async def programming_failure(_):
        raise RuntimeError("internal invariant failed")

    async def permission_failure(_):
        raise PermissionError("owner mismatch")

    with pytest.raises(CalendarError, match="remote write outcome unknown"):
        await middleware.awrap_tool_call(
            SimpleNamespace(tool=None, tool_call={
                "name": "propose_calendar_create", "id": "write-call"}),
            calendar_failure)
    request = SimpleNamespace(tool=None, tool_call={
        "name": "calendar_events", "id": "read-call"})
    with pytest.raises(RuntimeError, match="internal invariant failed"):
        await middleware.awrap_tool_call(request, programming_failure)
    with pytest.raises(PermissionError, match="owner mismatch"):
        await middleware.awrap_tool_call(request, permission_failure)
