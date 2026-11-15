"""Calendar boundary.  MCP calls live here, never in Telegram handlers."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from app.agent.calendar_conditional import (CalendarEventSnapshot, CalendarProviderError,
                                            GoogleCalendarConditionalTransport)
from app.agent.business_clock import business_now

logger = logging.getLogger(__name__)

READ_TOOL_NAMES = {
    "list-calendars", "list-events", "search-events", "get-event",
    "get-freebusy", "get-current-time",
}
WRITE_TOOL_NAMES = {"create-event", "update-event", "delete-event"}


class CalendarError(RuntimeError):
    pass


class CalendarUnavailable(CalendarError):
    pass


class CalendarAdapter(Protocol):
    async def availability(self, start: str, end: str, timezone: str) -> dict[str, Any]: ...
    async def search_events(self, query: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]: ...
    async def create_event(self, args: dict[str, Any]) -> dict[str, Any]: ...
    async def update_event(self, args: dict[str, Any]) -> dict[str, Any]: ...
    async def delete_event(self, args: dict[str, Any]) -> dict[str, Any]: ...
    async def get_event(self, calendar_id: str, event_id: str) -> CalendarEventSnapshot: ...
    async def resolve_create_scope(self, calendar_id: str) -> tuple[str, str]: ...
    async def resolve_read_scope(self, calendar_id: str) -> tuple[str, str]: ...
    async def create_event_prepared(self, planned_event_id: str,
                                    immutable_body: dict[str, Any],
                                    private_action_marker: str) -> dict[str, Any]: ...
    async def create_time_block(self, calendar_id: str, event_id: str,
                                body: dict[str, Any]) -> CalendarEventSnapshot: ...
    async def update_event_if_match(self, calendar_id: str, event_id: str,
                                    approved_etag: str,
                                    patch: dict[str, Any]) -> CalendarEventSnapshot: ...
    async def delete_event_if_match(self, calendar_id: str, event_id: str,
                                    approved_etag: str) -> dict[str, Any]: ...


class UnavailableCalendarAdapter:
    def __init__(self, reason: str = "Calendar is not configured"):
        self.reason = reason

    async def _raise(self, *args, **kwargs):
        raise CalendarUnavailable(self.reason)

    availability = _raise
    search_events = _raise
    create_event = _raise
    update_event = _raise
    delete_event = _raise
    get_event = _raise
    resolve_create_scope = _raise
    resolve_read_scope = _raise
    create_event_prepared = _raise
    create_time_block = _raise
    update_event_if_match = _raise
    delete_event_if_match = _raise


class MCPCalendarAdapter:
    """Small adapter over discovered LangChain MCP tools.

    The server owns the exact argument schema.  We select only known names and
    turn transport/JSON failures into explicit CalendarError values.
    """
    def __init__(self, tools: list,
                 conditional: GoogleCalendarConditionalTransport | None = None):
        self.tools = {tool.name: tool for tool in tools}
        self.conditional = conditional

    def _conditional(self) -> GoogleCalendarConditionalTransport:
        if self.conditional is None:
            raise CalendarUnavailable("Calendar conditional provider is not configured")
        return self.conditional

    async def get_event(self, calendar_id: str, event_id: str) -> CalendarEventSnapshot:
        try:
            return await self._conditional().get_event(calendar_id, event_id)
        except CalendarProviderError as exc:
            raise CalendarError(exc.category) from exc

    async def resolve_create_scope(self, calendar_id: str) -> tuple[str, str]:
        try:
            resolved = await self._conditional().resolve_writable_calendar(calendar_id)
        except CalendarProviderError as exc:
            raise CalendarError(exc.category) from exc
        source = self._conditional().token_source
        mode = getattr(source, "account_mode", None)
        if not mode:
            raise CalendarUnavailable("Calendar OAuth account identity is unavailable")
        return mode, resolved

    async def resolve_read_scope(self, calendar_id: str) -> tuple[str, str]:
        try:
            resolved = await self._conditional().resolve_readable_calendar(calendar_id)
        except CalendarProviderError as exc:
            raise CalendarError(exc.category) from exc
        mode = getattr(self._conditional().token_source, "account_mode", None)
        if not mode:
            raise CalendarUnavailable("Calendar OAuth account identity is unavailable")
        return mode, resolved

    async def create_event_prepared(self, planned_event_id: str,
                                    immutable_body: dict[str, Any],
                                    private_action_marker: str) -> dict[str, Any]:
        """R5.2 create transport under the single Calendar domain adapter."""
        from app.config import get_settings
        account = immutable_body.get("account")
        if not account:
            raise CalendarUnavailable("Calendar create account identity is missing")
        try:
            result = await self._call("create-event", {
                "account": account,
                "calendarId": immutable_body["calendarId"],
                "eventId": planned_event_id,
                "summary": immutable_body.get("summary") or immutable_body["title"],
                "start": immutable_body["start"], "end": immutable_body["end"],
                "timeZone": immutable_body.get("timeZone", get_settings().default_timezone),
                "extendedProperties": {"private": {"assistant_action": private_action_marker}},
            })
        except CalendarError as exc:
            detail = str(exc)
            # Pinned 2.7.0 CreateEventHandler.runTool performs this check before
            # createEvent/events.insert. Its other errors may follow dispatch.
            if "Duplicate event detected (" in detail and "To create anyway" in detail:
                raise CalendarProviderError("similarity_precheck") from exc
            if "Event ID '" in detail and "already exists" in detail:
                raise CalendarProviderError("resource_conflict", 409) from exc
            raise CalendarProviderError("remote_outcome_uncertain") from exc
        event = result.get("event") if isinstance(result, dict) else None
        if not isinstance(event, dict) or event.get("id") != planned_event_id:
            raise CalendarProviderError("remote_outcome_uncertain")
        return {**result, "id": event["id"]}

    async def create_time_block(self, calendar_id: str, event_id: str,
                                body: dict[str, Any]) -> CalendarEventSnapshot:
        try:
            return await self._conditional().create_time_block(calendar_id, event_id, body)
        except CalendarProviderError as exc:
            raise CalendarError(exc.category) from exc

    async def update_event_if_match(self, calendar_id: str, event_id: str,
                                    approved_etag: str,
                                    patch: dict[str, Any]) -> CalendarEventSnapshot:
        try:
            return await self._conditional().update_event(calendar_id, event_id,
                                                          approved_etag, patch)
        except CalendarProviderError as exc:
            raise CalendarError(exc.category) from exc

    async def delete_event_if_match(self, calendar_id: str, event_id: str,
                                    approved_etag: str) -> dict[str, Any]:
        try:
            return await self._conditional().delete_event(calendar_id, event_id,
                                                          approved_etag)
        except CalendarProviderError as exc:
            raise CalendarError(exc.category) from exc

    @property
    def read_tools(self) -> set[str]:
        return set(self.tools) & READ_TOOL_NAMES

    @property
    def write_tools(self) -> set[str]:
        return set(self.tools) & WRITE_TOOL_NAMES

    async def _call(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise CalendarUnavailable(f"Calendar MCP does not provide {name}")
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:
            raise CalendarError(f"Calendar MCP {name} failed: {exc}") from exc
        if isinstance(result, list):
            text = next((item.get("text") for item in result if isinstance(item, dict) and item.get("type") == "text"), None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    raise CalendarError(f"Calendar MCP {name} returned a non-JSON response: {text[:200]}")
        if not isinstance(result, (dict, list)):
            raise CalendarError(f"Calendar MCP {name} returned an unexpected response type")
        return result

    async def availability(self, start: str, end: str, timezone: str) -> dict[str, Any]:
        result = await self._call("get-freebusy", {
            "calendars": [{"id": "primary"}], "timeMin": start,
            "timeMax": end, "timeZone": timezone,
        })
        primary = result.get("calendars", {}).get("primary") if isinstance(result, dict) else None
        if not isinstance(primary, dict) or primary.get("errors") or not isinstance(primary.get("busy"), list):
            raise CalendarError("Calendar MCP did not return valid primary calendar availability")
        return result

    async def search_events(self, query: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
        from app.config import get_settings
        timezone = get_settings().default_timezone
        today = business_now(ZoneInfo(timezone)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        payload = {
            "calendarId": "primary",
            "timeMin": start or today.isoformat(),
            "timeMax": end or (today + timedelta(days=365)).isoformat(),
            "timeZone": timezone,
        }
        # The discovered MCP schema requires a query for search-events, while
        # list-events is the correct read for an unfiltered date range.
        if query:
            payload["query"] = query
        data = await self._call("search-events" if query else "list-events", payload)
        events = data if isinstance(data, list) else data.get("events")
        if not isinstance(events, list):
            raise CalendarError("Calendar MCP event query returned no events list")
        # Google text search can miss a newly visible event even when the same
        # MCP server lists it. Only for a bounded, explicit interval, verify an
        # empty search against event titles in that exact owner-scoped window.
        if query and not events and start and end and "list-events" in self.tools:
            try:
                interval = datetime.fromisoformat(end) - datetime.fromisoformat(start)
            except (TypeError, ValueError):
                interval = timedelta(0)
            if timedelta(0) < interval <= timedelta(days=35):
                listed = await self._call("list-events", {key: value for key, value in payload.items()
                                                           if key != "query"})
                listed_events = listed if isinstance(listed, list) else listed.get("events")
                if not isinstance(listed_events, list):
                    raise CalendarError("Calendar MCP fallback returned no events list")
                needle = query.casefold().strip()
                return [item for item in listed_events if isinstance(item, dict)
                        and item.get("status") != "cancelled"
                        and needle in str(item.get("summary") or "").casefold()]
        return events

    async def create_event(self, args: dict[str, Any]) -> dict[str, Any]:
        from app.config import get_settings
        result = await self._call("create-event", {
            "calendarId": args.get("calendarId", "primary"),
            "summary": args.get("summary") or args["title"],
            "start": args["start"], "end": args["end"],
            "timeZone": args.get("timeZone", get_settings().default_timezone),
        })
        # The current MCP server wraps the created Google event in {"event": ...}.
        event = result.get("event") if isinstance(result, dict) else None
        if not isinstance(event, dict) or not event.get("id"):
            raise CalendarError("Calendar MCP create-event returned no event id")
        return {**result, "id": event["id"]}

    async def update_event(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = {"calendarId": args.get("calendarId", "primary"), "eventId": args["eventId"]}
        for field in ("summary", "start", "end", "timeZone", "description", "location"):
            if field in args:
                payload[field] = args[field]
        if "title" in args:
            payload["summary"] = args["title"]
        result = await self._call("update-event", payload)
        event = result.get("event") if isinstance(result, dict) else None
        if not isinstance(event, dict) or not event.get("id"):
            raise CalendarError("Calendar MCP update-event returned no event id")
        return {**result, "id": event["id"]}

    async def delete_event(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self._call("delete-event", {
            "calendarId": args.get("calendarId", "primary"), "eventId": args["eventId"],
        })


async def discover_calendar_tools(settings) -> list:
    """Discover the server's actual tool names once at startup.

    Discovery is optional: lack of OAuth/npm must leave Todo/chat operational.
    Write tools are returned only to the controlled workflow, never to the
    legacy free tool loop.
    """
    if not settings.google_calendar_enabled:
        raise CalendarUnavailable("Calendar is disabled by GOOGLE_CALENDAR_ENABLED")
    from langchain_mcp_adapters.client import MultiServerMCPClient
    try:
        client = MultiServerMCPClient({
            "google-calendar": {
                "command": "npx", "args": ["-y", "@cocal/google-calendar-mcp@2.7.0"],
                "transport": "stdio",
                "env": {"GOOGLE_OAUTH_CREDENTIALS": settings.google_credentials_file},
            }
        })
        tools = await client.get_tools()
    except Exception as exc:
        raise CalendarUnavailable(f"Calendar MCP discovery failed: {exc}") from exc
    by_name = {tool.name: tool for tool in tools}
    required = {"create-event", "get-event", "update-event", "delete-event"}
    if not required.issubset(by_name):
        raise CalendarUnavailable("Pinned Calendar MCP is missing required tools")
    def fields(tool):
        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            return set(args)
        schema = getattr(tool, "args_schema", None)
        return set(schema.model_fields) if schema is not None and hasattr(schema, "model_fields") else set()
    if not {"eventId", "extendedProperties"}.issubset(fields(by_name["create-event"])):
        raise CalendarUnavailable("Pinned Calendar MCP create-event schema changed")
    if not {"calendarId", "eventId"}.issubset(fields(by_name["get-event"])):
        raise CalendarUnavailable("Pinned Calendar MCP get-event schema changed")
    logger.info("Calendar MCP pinned_version=2.7.0 discovered tool_count=%s", len(tools))
    return tools
