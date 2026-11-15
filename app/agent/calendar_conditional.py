"""Narrow Google Calendar resource/version transport for R5.

MCP remains the Calendar workflow adapter. Its pinned 2.7.0 structured event
omits etag and its update/delete tools cannot forward If-Match. This transport
only supplies exact resource reads and conditional writes through that adapter.
It does not decide approval, retries, or reconciliation.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from pydantic import BaseModel, ConfigDict, Field


class CalendarProviderError(RuntimeError):
    """Safe category only; never include provider response or OAuth content."""

    def __init__(self, category: str, status_code: int | None = None,
                 retry_after: str | None = None):
        super().__init__(category)
        self.category = category
        self.status_code = status_code
        self.retry_after = retry_after


class CalendarEventSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    etag: str
    status: str
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    start: dict[str, Any] = Field(default_factory=dict)
    end: dict[str, Any] = Field(default_factory=dict)
    extended_properties: dict[str, Any] = Field(default_factory=dict, alias="extendedProperties")
    recurrence: list[str] | None = None
    recurring_event_id: str | None = Field(default=None, alias="recurringEventId")
    original_start_time: dict[str, Any] | None = Field(default=None, alias="originalStartTime")
    attendees: list[dict[str, Any]] = Field(default_factory=list)
    updated: str | None = None
    html_link: str | None = Field(default=None, alias="htmlLink")


class McpOAuthTokenSource:
    """Read MCP's OAuth account without taking ownership of its token file.

    MCP alone writes/refreshed-token persistence. A short-lived credential copy
    may refresh in memory; its token is not logged or written by this class.
    """

    def __init__(self, credentials_file: str, token_file: str | None = None,
                 account_mode: str | None = None):
        self.credentials_file = Path(credentials_file)
        self.token_file = Path(token_file or os.environ.get("GOOGLE_CALENDAR_MCP_TOKEN_PATH") or
                               Path.home() / ".config" / "google-calendar-mcp" / "tokens.json")
        self.account_mode = account_mode or os.environ.get("GOOGLE_ACCOUNT_MODE", "normal")

    def _load_token(self) -> str:
        try:
            keys = json.loads(self.credentials_file.read_text(encoding="utf-8"))
            client = keys.get("installed") or keys.get("web") or keys
            tokens = json.loads(self.token_file.read_text(encoding="utf-8"))
            account = tokens.get(self.account_mode, tokens) if isinstance(tokens, dict) else None
            if not isinstance(account, dict) or not account.get("refresh_token"):
                raise ValueError("Missing MCP OAuth account")
            expiry_ms = account.get("expiry_date")
            expiry = (datetime.fromtimestamp(expiry_ms / 1000, timezone.utc).replace(tzinfo=None)
                      if isinstance(expiry_ms, (int, float)) else None)
            credential = Credentials(
                token=account.get("access_token"), refresh_token=account["refresh_token"],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client["client_id"], client_secret=client["client_secret"],
                expiry=expiry,
            )
            if not credential.valid:
                credential.refresh(GoogleAuthRequest())
            if not credential.token:
                raise ValueError("OAuth access token unavailable")
            return credential.token
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CalendarProviderError("oauth_unavailable") from exc
        except Exception as exc:
            raise CalendarProviderError("oauth_refresh_failed") from exc

    async def __call__(self) -> str:
        return await asyncio.to_thread(self._load_token)


class GoogleCalendarConditionalTransport:
    """Exact GET plus real If-Match PATCH/DELETE, no independent business policy."""

    def __init__(self, token_source: Callable[[], Awaitable[str]],
                 *, client_factory: Callable[[], httpx.AsyncClient] | None = None):
        self.token_source = token_source
        self.client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=20.0))

    async def _resolve_calendar(self, calendar_id: str, *, roles: set[str]) -> str:
        """Resolve an alias to its stable ID under the same OAuth account.

        An exact event 404 is not proof of absence unless this separate scope
        check succeeds with write access. Never infer account identity from a
        mutable MCP auto-selection decision.
        """
        if not calendar_id:
            raise CalendarProviderError("invalid_identity")
        token = await self.token_source()
        url = ("https://www.googleapis.com/calendar/v3/users/me/calendarList/" +
               quote(calendar_id, safe=""))
        try:
            async with self.client_factory() as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise CalendarProviderError("provider_unavailable") from exc
        if response.status_code >= 400:
            raise CalendarProviderError(
                "calendar_scope_unavailable", response.status_code,
                response.headers.get("Retry-After"))
        try:
            data = response.json()
        except ValueError as exc:
            raise CalendarProviderError("calendar_scope_unavailable") from exc
        if (not isinstance(data, dict) or not data.get("id") or
                data.get("accessRole") not in roles):
            raise CalendarProviderError("calendar_scope_unavailable")
        return data["id"]

    async def resolve_writable_calendar(self, calendar_id: str) -> str:
        """R5 mutation scope: a reader cannot authorize writes."""
        return await self._resolve_calendar(calendar_id, roles={"owner", "writer"})

    async def resolve_readable_calendar(self, calendar_id: str) -> str:
        """R6 exact-event scope: `freeBusyReader` cannot read event details."""
        return await self._resolve_calendar(calendar_id, roles={"owner", "writer", "reader"})

    async def _request(self, method: str, calendar_id: str, event_id: str,
                       *, approved_etag: str | None = None,
                       body: dict[str, Any] | None = None) -> dict[str, Any]:
        if not calendar_id or not event_id:
            raise CalendarProviderError("invalid_identity")
        if method in {"PATCH", "DELETE"} and not approved_etag:
            raise CalendarProviderError("approved_etag_required")
        token = await self.token_source()
        headers = {"Authorization": f"Bearer {token}"}
        if approved_etag is not None:
            headers["If-Match"] = approved_etag
        url = ("https://www.googleapis.com/calendar/v3/calendars/" +
               quote(calendar_id, safe="") + "/events/" + quote(event_id, safe=""))
        try:
            async with self.client_factory() as client:
                response = await client.request(method, url, headers=headers, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise CalendarProviderError("provider_unavailable" if method == "GET"
                                        else "remote_outcome_uncertain") from exc
        if response.status_code >= 400:
            category = ({404: "not_found_or_inaccessible", 409: "conflict",
                         410: "gone", 412: "etag_conflict"}.get(response.status_code) or
                        (("provider_unavailable" if method == "GET" else
                          "remote_outcome_uncertain") if response.status_code >= 500 else
                         "provider_rejected"))
            raise CalendarProviderError(category, response.status_code,
                                        response.headers.get("Retry-After"))
        if method == "DELETE":
            return {"eventId": event_id, "accepted": True}
        try:
            data = response.json()
        except ValueError as exc:
            raise CalendarProviderError("missing_resource_version" if method == "GET"
                                        else "remote_outcome_uncertain") from exc
        if not isinstance(data, dict):
            raise CalendarProviderError("missing_resource_version" if method == "GET"
                                        else "remote_outcome_uncertain")
        return data

    async def get_event(self, calendar_id: str, event_id: str) -> CalendarEventSnapshot:
        data = await self._request("GET", calendar_id, event_id)
        try:
            snapshot = CalendarEventSnapshot.model_validate(data)
        except ValueError as exc:
            raise CalendarProviderError("missing_resource_version") from exc
        if snapshot.id != event_id or not snapshot.etag:
            raise CalendarProviderError("missing_resource_version")
        return snapshot

    async def create_time_block(self, calendar_id: str, event_id: str,
                                body: dict[str, Any]) -> CalendarEventSnapshot:
        """Create a Todo projection with a stable ID and no Google alert.

        This is deliberately narrower than the approved standalone-event MCP
        workflow: no attendees, no default reminders, no model-supplied fields.
        """
        if not calendar_id or not event_id or body.get("id") != event_id:
            raise CalendarProviderError("invalid_identity")
        token = await self.token_source()
        url = ("https://www.googleapis.com/calendar/v3/calendars/" +
               quote(calendar_id, safe="") + "/events?sendUpdates=none")
        try:
            async with self.client_factory() as client:
                response = await client.post(url, headers={"Authorization": f"Bearer {token}"},
                                             json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise CalendarProviderError("remote_outcome_uncertain") from exc
        if response.status_code >= 400:
            category = ("conflict" if response.status_code == 409 else
                        "remote_outcome_uncertain" if response.status_code >= 500 else
                        "provider_rejected")
            raise CalendarProviderError(category, response.status_code)
        try:
            snapshot = CalendarEventSnapshot.model_validate(response.json())
        except ValueError as exc:
            raise CalendarProviderError("remote_outcome_uncertain") from exc
        if snapshot.id != event_id or not snapshot.etag:
            raise CalendarProviderError("remote_outcome_uncertain")
        return snapshot

    async def update_event(self, calendar_id: str, event_id: str,
                           approved_etag: str, patch: dict[str, Any]) -> CalendarEventSnapshot:
        data = await self._request("PATCH", calendar_id, event_id,
                                   approved_etag=approved_etag, body=patch)
        try:
            snapshot = CalendarEventSnapshot.model_validate(data)
        except ValueError as exc:
            raise CalendarProviderError("missing_resource_version") from exc
        if snapshot.id != event_id or not snapshot.etag:
            raise CalendarProviderError("missing_resource_version")
        return snapshot

    async def delete_event(self, calendar_id: str, event_id: str,
                           approved_etag: str) -> dict[str, Any]:
        return await self._request("DELETE", calendar_id, event_id,
                                   approved_etag=approved_etag)
