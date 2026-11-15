"""Only known transient read failures receive a bounded local retry."""

import asyncio

import httpx
import pytest

from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError, GmailUnavailable
from app.agent.official_read_tools import _read_with_retry


@pytest.mark.asyncio
async def test_transient_calendar_read_retries_once():
    calls = 0

    async def read():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CalendarError("safe category") from httpx.ReadTimeout("synthetic")
        return ["verified event"]

    assert await _read_with_retry(read, CalendarError) == ["verified event"]
    assert calls == 2


@pytest.mark.asyncio
async def test_permanent_calendar_failure_and_permission_do_not_retry():
    calls = 0

    async def permanent():
        nonlocal calls
        calls += 1
        raise CalendarError("InvalidCalendarInterval")

    with pytest.raises(CalendarError):
        await _read_with_retry(permanent, CalendarError)
    assert calls == 1

    async def denied():
        raise PermissionError("WrongOwner")

    with pytest.raises(PermissionError):
        await _read_with_retry(denied, CalendarError)


@pytest.mark.asyncio
async def test_retry_after_exceeding_deadline_stops_without_second_call():
    calls = 0

    async def limited():
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "https://example.test/read")
        response = httpx.Response(429, request=request, headers={"Retry-After": "60"})
        raise GmailUnavailable("Gmail API returned HTTP 429") from httpx.HTTPStatusError(
            "rate limited", request=request, response=response)

    with pytest.raises(GmailError):
        await _read_with_retry(limited, GmailError)
    assert calls == 1


@pytest.mark.asyncio
async def test_calendar_provider_retry_after_survives_adapter_error_translation():
    calls = 0

    async def limited():
        nonlocal calls
        calls += 1
        raise CalendarError("rate limited") from CalendarProviderError(
            "provider_unavailable", 429, "60")

    with pytest.raises(CalendarError):
        await _read_with_retry(limited, CalendarError)
    assert calls == 1


@pytest.mark.asyncio
async def test_calendar_http_403_does_not_retry_even_with_transient_category():
    calls = 0

    async def denied():
        nonlocal calls
        calls += 1
        raise CalendarError("unavailable") from CalendarProviderError(
            "provider_unavailable", 403)

    with pytest.raises(CalendarError):
        await _read_with_retry(denied, CalendarError)
    assert calls == 1


@pytest.mark.asyncio
async def test_http_503_gmail_read_retries_at_most_twice(monkeypatch):
    calls = 0
    sleeps = []
    original_sleep = asyncio.sleep

    async def record_sleep(delay):
        sleeps.append(delay)
        await original_sleep(0)

    monkeypatch.setattr("app.agent.official_read_tools.asyncio.sleep", record_sleep)

    async def unavailable():
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "https://example.test/read")
        response = httpx.Response(503, request=request, headers={"Retry-After": "1"})
        raise GmailUnavailable("Gmail API returned HTTP 503") from httpx.HTTPStatusError(
            "unavailable", request=request, response=response)

    with pytest.raises(GmailError):
        await _read_with_retry(unavailable, GmailError)
    assert calls == 2
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_auth_failure_is_not_treated_as_transient():
    calls = 0

    async def auth_failed():
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "https://example.test/read")
        response = httpx.Response(403, request=request)
        raise GmailUnavailable("authorization failed") from httpx.HTTPStatusError(
            "forbidden", request=request, response=response)

    with pytest.raises(GmailError):
        await _read_with_retry(auth_failed, GmailError)
    assert calls == 1


@pytest.mark.asyncio
async def test_hung_read_is_bounded_by_attempt_and_total_deadline(monkeypatch):
    monkeypatch.setattr("app.agent.official_read_tools._READ_ATTEMPT_SECONDS", 0.01)
    monkeypatch.setattr("app.agent.official_read_tools._READ_DEADLINE_SECONDS", 0.03)
    calls = 0

    async def hung_read():
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    with pytest.raises(CalendarError, match="ReadTimedOut"):
        await _read_with_retry(hung_read, CalendarError)
    assert 1 <= calls <= 2
