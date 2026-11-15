"""Capability policy for the ordinary Agent's per-turn tool selection.

All new natural-language turns enter the ordinary Agent.  This policy only
selects candidate capabilities; approval and resource authorization remain
server-side and are checked again when a capability is invoked.
"""
from __future__ import annotations

from app.agent.planner import RoutedRequest


AGENT_READ_INTENTS = frozenset({
    "calendar_read", "email_search", "email_unread", "email_thread", "email_draft",
    "email_cross_domain", "read_cross_domain",
})

ACTION_INTENTS = frozenset({
    "calendar_create", "calendar_update", "calendar_delete", "cross_domain",
    "smart_schedule", "email_to_todo", "email_to_calendar",
})


def capability_for_route(route: RoutedRequest) -> str | None:
    """Return the high-level Agent capability, never a raw mutation tool."""
    if route.intent in ACTION_INTENTS or (route.intent == "todo" and route.todo_operation == "delete"):
        return "prepare_action_request"
    if route.intent == "meeting_prep":
        return "prepare_meeting_brief"
    return None


def required_read_sources(route: RoutedRequest) -> frozenset[str]:
    """Evidence families that must be observed before a read answer is final."""
    if route.intent == "calendar_read":
        return frozenset({"calendar"})
    if route.intent in {"email_search", "email_unread", "email_thread", "email_draft"}:
        return frozenset({"gmail"})
    if route.intent == "email_cross_domain":
        return frozenset({"gmail", "calendar"} if route.email_calendar_check else {"gmail"})
    if route.intent == "read_cross_domain":
        sources = set()
        if route.date_reference:
            sources.add("calendar")
        if route.email_query or route.email_thread_id:
            sources.add("gmail")
        if route.todo_operation in {"list", "search"}:
            sources.add("todo")
        return frozenset(sources)
    return frozenset()
