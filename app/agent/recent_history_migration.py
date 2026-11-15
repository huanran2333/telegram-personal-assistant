"""Owner-checked migration of settled conversation history to the official graph.

This does not write either Saver, resolve old approvals, or replay tool calls.
The caller must first cancel and notify old pending approvals through an audited
cutover procedure, then initialize a fresh checkpoint namespace with the result.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.context_protocol import message_groups


@dataclass(frozen=True)
class RecentHistoryMigration:
    messages: tuple
    retained_turns: int
    dropped_turns: int


def select_recent_history(snapshot, *, user_id: str, thread_id: str,
                          telegram_id: str | None = None,
                          suppressed_message_ids: frozenset[str],
                          suppressed_terms: tuple[str, ...],
                          max_messages: int = 16) -> RecentHistoryMigration:
    """Select only completed, owner-checked turns from an old Saver snapshot.

    The caller must obtain both suppression inputs from the verified owner's
    memory service; an unavailable suppression query blocks migration. A
    missing owner, a running/interrupted old graph, malformed tool envelopes,
    or a suppressed source cannot become new model context.  Suppressed turns
    are excluded as a whole to avoid leaking the same value via tool arguments
    or a cached assistant reply.  Tool calls that survive are *history*, never
    invocations: the new graph is invoked with a fresh human message after them.
    """
    if max_messages < 1:
        raise ValueError("max_messages must be positive")
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict) or not values:
        return RecentHistoryMigration((), 0, 0)
    if str(values.get("user_id", "")) != user_id:
        raise PermissionError("CheckpointOwnerMismatch")
    if str(values.get("conversation_thread_id", "")) != thread_id:
        raise PermissionError("CheckpointThreadMismatch")
    if telegram_id is not None and str(values.get("telegram_id", "")) != telegram_id:
        raise PermissionError("CheckpointTelegramIdentityMismatch")
    if getattr(snapshot, "next", ()) or getattr(snapshot, "interrupts", ()):
        raise RuntimeError("OldCheckpointNotSettled")
    history = values.get("messages")
    if not isinstance(history, list):
        raise ValueError("CheckpointMessagesInvalid")

    safe_groups = []
    dropped = 0
    seen_ids = set()
    forbidden = tuple(term.casefold() for term in suppressed_terms if term)
    for group in message_groups(history):
        if not group.complete or not group.finalized:
            dropped += 1
            continue
        ids = [str(message.id) for message in group.messages if getattr(message, "id", None)]
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            dropped += 1
            continue
        seen_ids.update(ids)
        if any(identifier in suppressed_message_ids for identifier in ids):
            dropped += 1
            continue
        # Include call arguments as well as ToolMessage and assistant content.
        # A tombstoned value must not reappear via another message in its turn.
        if forbidden and any(any(term in str(message.model_dump()).casefold() for term in forbidden)
                             for message in group.messages):
            dropped += 1
            continue
        safe_groups.append(group)

    selected = []
    for group in reversed(safe_groups):
        if selected and len(selected) + len(group.messages) > max_messages:
            break
        selected[:0] = group.messages
        if len(selected) >= max_messages:
            break
    if any(not isinstance(message, (HumanMessage, AIMessage, ToolMessage)) for message in selected):
        raise ValueError("UnsupportedCheckpointMessage")
    if selected and not isinstance(selected[0], HumanMessage):
        raise AssertionError("MigratedHistoryMustBeginWithHuman")
    return RecentHistoryMigration(tuple(deepcopy(selected)),
                                  len(message_groups(selected)), dropped)
