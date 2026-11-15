"""High-level Agent capabilities that hand off to the existing durable graph.

These are *not* Calendar/To-do mutation tools.  A model can request that the
server prepare an action, but cannot approve it or call the executor directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from langchain_core.tools import tool

from app.agent.capability_policy import capability_for_route
from app.agent.planner import RoutedRequest


@dataclass(frozen=True)
class CapabilityScope:
    route: RoutedRequest
    request: str
    user_id: str
    telegram_id: str
    turn_id: str
    origin: str
    memory_history_threads: tuple[str, ...]
    recent_context: tuple[dict[str, str], ...]

    @property
    def workflow_thread_id(self) -> str:
        return (f"workflow_user_{self.telegram_id}" if self.origin == "telegram"
                else f"web:workflow:{self.user_id}")


class CapabilityRejected(ValueError):
    """A deterministic pre-execution authorization or pending-workflow denial."""


_scope: ContextVar[CapabilityScope | None] = ContextVar("agent_capability_scope", default=None)


@contextmanager
def capability_scope(*, route: dict, request: str, user_id: str, telegram_id: str,
                     turn_id: str, origin: str,
                     memory_history_threads: tuple[str, ...] = (),
                     recent_context: tuple[dict[str, str], ...] = ()) -> Iterator[CapabilityScope]:
    if origin not in {"telegram", "web"} or not user_id or not telegram_id or not turn_id:
        raise CapabilityRejected("CapabilityIdentityMissing")
    value = CapabilityScope(RoutedRequest.model_validate(route), request, user_id,
                            telegram_id, turn_id, origin, memory_history_threads,
                            recent_context)
    token = _scope.set(value)
    try:
        yield value
    finally:
        _scope.reset(token)


async def _handoff(expected_capability: str) -> dict[str, str | bool]:
    scope = _scope.get()
    if scope is None or capability_for_route(scope.route) != expected_capability:
        raise CapabilityRejected("CapabilityRouteMismatch")

    from app.config import get_settings
    from app.agent.workflow import start_workflow
    from app.agent.workflow_runtime import get_productivity_graph

    settings = get_settings()
    needs_calendar = scope.route.intent in {
        "calendar_create", "calendar_update", "calendar_delete", "cross_domain",
        "smart_schedule", "meeting_prep", "email_to_calendar",
    }
    needs_gmail = scope.route.intent in {"email_to_todo", "email_to_calendar"}
    if needs_calendar and (not settings.google_calendar_enabled or
                           scope.telegram_id != settings.google_calendar_owner_telegram_id):
        raise CapabilityRejected("CalendarCapabilityUnavailable")
    if needs_gmail and (not settings.gmail_enabled or
                        scope.telegram_id != settings.gmail_owner_telegram_id):
        raise CapabilityRejected("GmailCapabilityUnavailable")

    graph = await get_productivity_graph()
    thread_id = scope.workflow_thread_id
    prior = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if prior and prior.values and prior.values.get("user_id") not in {None, scope.user_id}:
        raise CapabilityRejected("WorkflowOwnerMismatch")
    if prior and prior.next and prior.values.get("transcript_turn_id") != scope.turn_id:
        raise CapabilityRejected("PendingWorkflowRequiresDecision")
    recent_context = list(scope.recent_context) if expected_capability == "prepare_meeting_brief" else []
    if recent_context and prior and prior.values:
        previous_request = prior.values.get("request")
        if isinstance(previous_request, str) and previous_request != scope.request:
            recent_context.append({"source": "previous workflow request", "text": previous_request})
    await start_workflow(
        graph, scope.user_id, scope.request, settings.default_timezone, thread_id,
        route=scope.route, telegram_id=scope.telegram_id, turn_id=scope.turn_id,
        memory_history_threads=scope.memory_history_threads,
        recent_context=recent_context,
    )
    return {"handoff": True, "thread_id": thread_id, "turn_id": scope.turn_id,
            "response": "The request was handed to the durable action or evidence workflow."}


@tool
async def prepare_action_request() -> dict[str, str | bool]:
    """Prepare a requested Todo deletion, Calendar write, or email-derived action for policy review. Does not approve or directly execute it."""
    return await _handoff("prepare_action_request")


@tool
async def prepare_meeting_brief() -> dict[str, str | bool]:
    """Prepare a read-only meeting brief using exact Calendar, Gmail, Todo, and memory evidence; never modify them."""
    return await _handoff("prepare_meeting_brief")


CAPABILITY_TOOLS = (prepare_action_request, prepare_meeting_brief)
