"""Application-lifetime holder for the interrupt/resume productivity graph."""
from __future__ import annotations

from app.agent.calendar import MCPCalendarAdapter, UnavailableCalendarAdapter, discover_calendar_tools
from app.agent.calendar_conditional import (GoogleCalendarConditionalTransport,
                                            McpOAuthTokenSource)
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.calendar_modify import CalendarModifyCoordinator
from app.agent.calendar_continuation import CalendarContinuationRuntime
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.workflow import build_productivity_graph
from app.agent.gmail import OAuthGmailSource
from app.config import get_settings
from pathlib import Path

_productivity_graph = None
_calendar_adapter = None
_calendar_create_coordinator = None
_calendar_modify_coordinator = None
_calendar_continuation_runtime = None
_gmail_source = None
_action_executor = None


def clear_productivity_runtime() -> None:
    """Prevent a failed later startup from exposing a stale graph/adapter."""
    global _productivity_graph, _calendar_adapter, _calendar_create_coordinator, _calendar_modify_coordinator, _calendar_continuation_runtime, _gmail_source, _action_executor
    _productivity_graph = None
    _calendar_adapter = None
    _calendar_create_coordinator = None
    _calendar_modify_coordinator = None
    _calendar_continuation_runtime = None
    _gmail_source = None
    _action_executor = None


async def build_productivity_runtime(db_session_factory, checkpointer, store=None,
                                     calendar_owner_telegram_id: str = "",
                                     memory_orchestrator=None):
    global _productivity_graph, _calendar_adapter, _calendar_create_coordinator, _calendar_modify_coordinator, _calendar_continuation_runtime, _gmail_source, _action_executor
    clear_productivity_runtime()
    settings = get_settings()
    try:
        calendar = MCPCalendarAdapter(
            await discover_calendar_tools(settings),
            conditional=GoogleCalendarConditionalTransport(
                McpOAuthTokenSource(settings.google_credentials_file)),
        )
    except Exception as exc:
        calendar = UnavailableCalendarAdapter(str(exc))
    _calendar_adapter = calendar
    # Keep evidence readable/recoverable even if Calendar discovery is down on
    # this restart. UnavailableCalendarAdapter fails provider calls closed;
    # it must not route an already-armed action back to the legacy executor.
    _calendar_create_coordinator = CalendarCreateCoordinator(calendar, db_session_factory)
    _calendar_modify_coordinator = CalendarModifyCoordinator(calendar, db_session_factory)
    executor = ActionExecutor(
        calendar, SqlAlchemyActionLedger(db_session_factory), db_session_factory,
        calendar_create_coordinator=_calendar_create_coordinator,
        calendar_modify_coordinator=_calendar_modify_coordinator)
    _action_executor = executor
    gmail = (OAuthGmailSource(Path(settings.gmail_token_file))
             if settings.gmail_enabled and settings.gmail_token_file else None)
    _productivity_graph = build_productivity_graph(
        calendar, executor, checkpointer, store=store,
        calendar_owner_telegram_id=calendar_owner_telegram_id,
        gmail=gmail, gmail_owner_telegram_id=settings.gmail_owner_telegram_id,
        memory_orchestrator=memory_orchestrator,
        memory_token_budget=settings.memory_context_token_budget)
    _gmail_source = gmail
    _calendar_continuation_runtime = CalendarContinuationRuntime(
        _productivity_graph, db_session_factory)
    return _productivity_graph


async def get_productivity_graph():
    if _productivity_graph is None:
        raise RuntimeError("Productivity workflow has not been built")
    return _productivity_graph


def get_calendar_adapter():
    return _calendar_adapter or UnavailableCalendarAdapter("Calendar runtime not initialized")


def get_gmail_source():
    return _gmail_source


def get_action_executor():
    if _action_executor is None:
        raise RuntimeError("Action executor has not been built")
    return _action_executor


def get_calendar_create_coordinator():
    return _calendar_create_coordinator


def get_calendar_modify_coordinator():
    return _calendar_modify_coordinator


def get_calendar_continuation_runtime():
    return _calendar_continuation_runtime
