"""Provider-safe message grouping, independent of any memory window policy.

An incomplete tool round is execution state, not safe prompt history.  This
module never mutates the checkpoint; callers select only valid prompt groups.
"""

from dataclasses import dataclass
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@dataclass(frozen=True)
class MessageGroup:
    messages: tuple
    complete: bool
    has_tools: bool
    finalized: bool


def message_groups(history: list) -> list[MessageGroup]:
    """Partition by human turn and validate every AI/tool call round.

    Invalid/orphan messages are excluded rather than forwarded to a provider.
    An unfinished active turn is returned as incomplete for diagnostics, but
    selectors never send it as historical context.
    """
    groups = []
    current = []
    pending = set()
    has_tools = False
    invalid = False

    def finish():
        if current:
            last = current[-1]
            finalized = (isinstance(last, AIMessage) and not last.tool_calls and
                         not last.invalid_tool_calls and not last.additional_kwargs.get("tool_calls"))
            groups.append(MessageGroup(tuple(current), not pending and not invalid,
                                       has_tools, finalized))

    for message in history:
        if isinstance(message, HumanMessage):
            finish()
            current, pending, has_tools, invalid = [message], set(), False, False
            continue
        if not current:
            continue
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                invalid = True
            else:
                pending.remove(message.tool_call_id)
            current.append(message)
            continue
        if isinstance(message, AIMessage):
            if pending:
                invalid = True
            calls = message.tool_calls or []
            if message.invalid_tool_calls or message.additional_kwargs.get("tool_calls") and not calls:
                invalid = True
            ids = [call.get("id") for call in calls]
            if len(ids) != len(set(ids)) or any(not identifier for identifier in ids):
                invalid = True
            pending = set(ids)
            has_tools = has_tools or bool(ids)
            current.append(message)
            continue
        # System and unsupported messages cannot be spliced into a user turn.
        invalid = True
    finish()
    return groups


def select_todo_context(history: list, max_messages: int = 10) -> list:
    """Select complete turns; include an active tool round only if valid so far.

    The most recent human turn is indivisible even when longer than budget.
    A pending AI tool call is omitted until its matching ToolMessage arrives.
    """
    if max_messages < 1:
        raise ValueError("max_messages must be positive")
    selected = []
    groups = message_groups(history)
    for index in range(len(groups)-1, -1, -1):
        group = groups[index]
        if not group.complete or (index != len(groups)-1 and not group.finalized):
            continue
        if selected and len(selected) + len(group.messages) > max_messages:
            break
        selected[:0] = group.messages
        if len(selected) >= max_messages:
            break
    return selected


def select_chat_context(history: list, max_messages: int = 8) -> list:
    """Preserve chat's bounded conversational policy on validated turn groups."""
    if max_messages < 1:
        raise ValueError("max_messages must be positive")
    conversational = []
    groups = message_groups(history)
    for index, group in enumerate(groups):
        if not group.complete or (index != len(groups)-1 and not group.finalized):
            continue
        for message in group.messages:
            if not isinstance(getattr(message, "content", None), str) or not message.content.strip():
                continue
            if isinstance(message, HumanMessage):
                conversational.append(message)
            elif isinstance(message, AIMessage) and not (message.tool_calls or message.invalid_tool_calls or
                                                         message.additional_kwargs.get("tool_calls")):
                conversational.append(message)
    recent = conversational[-max_messages:]
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]
    return recent
