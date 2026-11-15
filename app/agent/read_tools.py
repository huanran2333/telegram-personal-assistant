"""Owner-scoped, read-only capabilities for the ordinary tool-calling graph.

The model sees no identity or OAuth arguments.  The executor supplies the
server-validated route and user through a task-local context at invocation.
These tools cannot create an ActionProposal or call a write provider.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from langchain_core.tools import tool

from app.agent.planner import RoutedRequest, resolve_window


@dataclass
class ReadScope:
    route: RoutedRequest
    request: str
    user_id: str
    telegram_id: str
    recent_gmail_thread_id: str | None = None
    selected_gmail_thread_id: str | None = None


_scope: ContextVar[ReadScope | None] = ContextVar("assistant_read_scope", default=None)


@contextmanager
def read_scope(*, route: dict, request: str, user_id: str,
               telegram_id: str, recent_gmail_thread_id: str | None = None) -> Iterator[ReadScope]:
    value = ReadScope(RoutedRequest.model_validate(route), request, user_id,
                      telegram_id, recent_gmail_thread_id)
    token = _scope.set(value)
    try:
        yield value
    finally:
        _scope.reset(token)


def _current() -> ReadScope:
    value = _scope.get()
    if value is None or not value.user_id or not value.telegram_id:
        raise PermissionError("ReadToolScopeMissing")
    return value


def _calendar_scope(scope: ReadScope) -> tuple[str, str, str]:
    from app.config import get_settings
    settings = get_settings()
    if (scope.route.intent not in {"calendar_read", "read_cross_domain", "email_cross_domain"} or not settings.google_calendar_enabled
            or not settings.google_calendar_owner_telegram_id
            or scope.telegram_id != settings.google_calendar_owner_telegram_id):
        raise PermissionError("CalendarReadUnavailable")
    start, end = resolve_window(scope.route, settings.default_timezone)
    if (end - start).total_seconds() > 35 * 86400:
        raise ValueError("CalendarReadWindowTooWide")
    return start.isoformat(), end.isoformat(), settings.default_timezone


@tool
async def read_calendar_events() -> str:
    """Read the authorized user's events for the date in the current request. Never write events."""
    scope = _current()
    start, end, timezone = _calendar_scope(scope)
    from app.agent.workflow_runtime import get_calendar_adapter
    from app.agent.workflow import format_calendar_events
    events = await get_calendar_adapter().search_events(scope.route.query or "", start, end)
    return format_calendar_events(events, timezone)


@tool
async def read_calendar_availability() -> str:
    """Read verified free/busy for the authorized user's requested date. Never reserve time."""
    scope = _current()
    start, end, timezone = _calendar_scope(scope)
    from app.agent.workflow_runtime import get_calendar_adapter
    from app.localization import tr
    result = await get_calendar_adapter().availability(start, end, timezone)
    primary = result.get("calendars", {}).get("primary", {})
    busy = primary.get("busy")
    if not isinstance(busy, list) or primary.get("errors"):
        raise ValueError("CalendarAvailabilityUnverified")
    return tr("calendar_busy") if busy else tr("calendar_free")


@tool
async def read_gmail_evidence() -> str:
    """Read authorized Gmail evidence and return a source-checked summary or unsent draft preview."""
    scope = _current()
    from app.config import get_settings
    from app.agent.email_workflow import select_action_message, synthesize_email
    from app.agent.nodes import _build_chat_llm
    from app.agent.workflow_runtime import get_gmail_source
    settings = get_settings()
    route = scope.route
    if (route.intent not in {"email_search", "email_unread", "email_thread", "email_draft",
                             "email_cross_domain", "read_cross_domain"}
            or not settings.gmail_enabled or not settings.gmail_owner_telegram_id
            or scope.telegram_id != settings.gmail_owner_telegram_id):
        raise PermissionError("GmailReadUnavailable")
    gmail = get_gmail_source()
    if gmail is None:
        raise ValueError("GmailReadUnavailable")
    if route.email_thread_id:
        messages = await gmail.thread(route.email_thread_id)
    elif route.intent == "email_thread":
        if not scope.recent_gmail_thread_id:
            raise ValueError("GmailThreadRequired")
        messages = await gmail.thread(scope.recent_gmail_thread_id)
    elif route.intent == "email_unread":
        messages = await gmail.unread(route.email_query or "", limit=5)
    elif route.email_query:
        messages = await gmail.search(route.email_query, limit=5)
    elif route.intent == "email_draft" and scope.recent_gmail_thread_id:
        messages = await gmail.thread(scope.recent_gmail_thread_id)
    else:
        raise ValueError("GmailQueryRequired")
    draft = route.intent == "email_draft" or (
        route.intent == "email_cross_domain" and route.email_draft_requested)
    if draft and messages:
        source = select_action_message(messages, route.email_select_latest,
                                       allow_single_thread_latest=True)
        messages = await gmail.thread(source.thread_id)
    thread_ids = {item.thread_id for item in messages}
    scope.selected_gmail_thread_id = next(iter(thread_ids)) if len(thread_ids) == 1 else None
    account = await gmail.account_address() if draft and messages else None
    llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    return await synthesize_email(messages, scope.request, llm,
                                  draft=draft, account_address=account)


READ_TOOLS = (read_calendar_events, read_calendar_availability, read_gmail_evidence)
