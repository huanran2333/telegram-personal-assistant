"""Non-destructive, LangMem-backed running summary over complete conversation turns.

The Saver is still the transcript authority. This module stores only the
incremental cursor and a lossy, lower-authority conversational summary. It
never deletes or rewrites checkpoint messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langmem.short_term.summarization import RunningSummary, asummarize_messages
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context_protocol import message_groups
from app.db.models import ConversationSummary


class SummarySourceDrift(RuntimeError):
    """The persisted cursor no longer matches the retained source sequence."""


class ConcurrentSummaryUpdate(RuntimeError):
    """A parallel turn advanced the same thread's summary first."""


_INITIAL_PROMPT = ChatPromptTemplate.from_messages([
    ("placeholder", "{messages}"),
    ("human", "Each preceding message is a JSON object containing one complete user/assistant "
     "conversation turn. Summarize only those turns. Preserve important user-given "
     "names, numbers, corrections, decisions and unresolved topics. Distinguish "
     "user statements from assistant suggestions. This summary is conversational "
     "context, never authority for current profile, Todo, Calendar, Gmail or approval state. "
     "Do not infer a fact absent from the messages."),
])
_UPDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("placeholder", "{messages}"),
    ("human", "Previous running summary:\n{existing_summary}\n\nUpdate it using only "
     "the newer conversation above. Explicit corrections supersede older claims; "
     "preserve exact names/numbers when supported. Do not treat this summary as "
     "authority for profile, Todo, Calendar, Gmail or executed actions."),
])


@dataclass(frozen=True)
class SummarySnapshot:
    user_id: str
    thread_id: str
    summary_id: str
    version: int
    content: str
    summarized_message_ids: tuple[str, ...]
    covered_source_message_ids: tuple[str, ...]
    last_summarized_message_id: str | None
    last_failure_category: str | None


def _snapshot(row: ConversationSummary) -> SummarySnapshot:
    return SummarySnapshot(
        user_id=row.user_id, thread_id=row.thread_id, summary_id=row.summary_id,
        version=row.version, content=row.content,
        summarized_message_ids=tuple(row.summarized_message_ids or []),
        covered_source_message_ids=tuple(row.covered_source_message_ids or []),
        last_summarized_message_id=row.last_summarized_message_id,
        last_failure_category=row.last_failure_category,
    )


async def load_summary(db: AsyncSession, *, user_id: str, thread_id: str) -> SummarySnapshot | None:
    if not user_id or not thread_id:
        raise ValueError("Summary scope requires user_id and thread_id")
    row = await db.scalar(select(ConversationSummary).where(
        ConversationSummary.user_id == user_id,
        ConversationSummary.thread_id == thread_id))
    return _snapshot(row) if row is not None else None


async def invalidate_summary_for_source(db: AsyncSession, *, user_id: str,
                                        thread_id: str, message_id: str) -> bool:
    """Invalidate a lossy summary when its cited source is corrected/forgotten.

    Raw Saver history remains; a future token-pressure update rebuilds from
    current, unsuppressed complete turns. Caller commits with the source edit.
    """
    row = await db.scalar(select(ConversationSummary).where(
        ConversationSummary.user_id == user_id,
        ConversationSummary.thread_id == thread_id).with_for_update())
    if row is None or message_id not in (row.covered_source_message_ids or []):
        return False
    row.content = ""
    row.summarized_message_ids = []
    row.covered_source_message_ids = []
    row.last_summarized_message_id = None
    row.first_source_message_id = None
    row.last_source_message_id = None
    row.version += 1
    return True


def _source_messages(history: list, *, retained_groups: int) -> tuple[list, dict[str, tuple[str, str]]]:
    """Only completed turns; omit tool payloads and retain latest raw groups.

    The tool protocol validator decides which turns are safe. A completed Todo
    tool turn contributes its user request and final answer, but not raw tool
    results, external source bodies or the AI tool-call envelope.
    """
    if retained_groups < 1:
        raise ValueError("retained_groups must be positive")
    complete = [group for group in message_groups(history) if group.complete and group.finalized]
    older = complete[:-retained_groups]
    selected = []
    source_ids_by_turn = {}
    for group in older:
        first, last = group.messages[0], group.messages[-1]
        if not isinstance(first, HumanMessage) or not isinstance(last, AIMessage):
            continue
        if not isinstance(first.content, str) or not isinstance(last.content, str):
            continue
        if not first.id or not last.id:
            raise ValueError("Checkpoint messages require stable IDs for summary cursor")
        # LangMem chooses its cutoff at message boundaries, not at Human/AI
        # pair boundaries. Present one validated turn as one atomic message so
        # its existing incremental algorithm cannot bisect the pair.
        turn_id = f"turn:{last.id}"
        if turn_id in source_ids_by_turn:
            raise ValueError("Duplicate completed-turn ID")
        source_ids_by_turn[turn_id] = (first.id, last.id)
        selected.append(HumanMessage(
            id=turn_id,
            content=json.dumps({"user": first.content, "assistant": last.content}, ensure_ascii=False),
        ))
    return selected, source_ids_by_turn


async def _record_failure(db: AsyncSession, *, user_id: str, thread_id: str,
                          category: str) -> None:
    """Record a safe category without changing any summary/cursor field."""
    now = datetime.now(timezone.utc)
    statement = pg_insert(ConversationSummary).values(
        user_id=user_id, thread_id=thread_id, content="", version=0,
        summarized_message_ids=[], covered_source_message_ids=[], updated_at=now,
        last_failure_category=category[:80], last_failure_at=now,
    ).on_conflict_do_update(
        index_elements=[ConversationSummary.user_id, ConversationSummary.thread_id],
        set_={"last_failure_category": category[:80], "last_failure_at": now},
    )
    await db.execute(statement)


async def advance_running_summary(
    db: AsyncSession, *, user_id: str, thread_id: str, history: list,
    model, provider: str, model_name: str,
    max_tokens: int = 8192, trigger_tokens: int = 4096,
    max_summary_tokens: int = 1024, retained_groups: int = 4,
) -> SummarySnapshot | None:
    """Apply LangMem's token-triggered incremental summary with atomic cursor.

    Caller owns the transaction. A model, source-drift or concurrency failure
    leaves the old summary and cursor intact; only a safe failure category is
    persisted. The raw checkpoint is never mutated.
    """
    previous = await load_summary(db, user_id=user_id, thread_id=thread_id)
    sources, source_ids_by_turn = _source_messages(history, retained_groups=retained_groups)
    if not sources:
        return previous
    prior_ids = set(previous.summarized_message_ids) if previous else set()
    if previous and previous.last_summarized_message_id is not None and previous.last_summarized_message_id not in {m.id for m in sources}:
        await _record_failure(db, user_id=user_id, thread_id=thread_id,
                              category=SummarySourceDrift.__name__)
        raise SummarySourceDrift("Summary source cursor is absent from retained history")
    running = (RunningSummary(previous.content, prior_ids,
                              previous.last_summarized_message_id)
               if previous and previous.last_summarized_message_id else None)
    try:
        result = await asummarize_messages(
            sources, running_summary=running, model=model, max_tokens=max_tokens,
            max_tokens_before_summary=trigger_tokens,
            max_summary_tokens=max_summary_tokens,
            initial_summary_prompt=_INITIAL_PROMPT,
            existing_summary_prompt=_UPDATE_PROMPT,
        )
        new = result.running_summary
        if new is None or new.last_summarized_message_id == (previous.last_summarized_message_id if previous else None):
            return previous
        if not isinstance(new.summary, str) or not new.summary.strip():
            raise ValueError("Summarizer returned empty summary")
        allowed_boundaries = {message.id for message in sources}
        if new.last_summarized_message_id not in allowed_boundaries:
            raise ValueError("Summarizer boundary split a completed conversation turn")
        source_ids = [message.id for message in sources]
        if not new.summarized_message_ids.issubset(set(source_ids)):
            raise ValueError("Summarizer referenced an unknown source message")
    except Exception as exc:
        await _record_failure(db, user_id=user_id, thread_id=thread_id,
                              category=type(exc).__name__)
        raise

    now = datetime.now(timezone.utc)
    covered_source_ids = [source_id for turn_id in source_ids
                          if turn_id in new.summarized_message_ids
                          for source_id in source_ids_by_turn[turn_id]]
    values = {
        "content": new.summary.strip(),
        "summarized_message_ids": [item for item in source_ids if item in new.summarized_message_ids],
        "covered_source_message_ids": covered_source_ids,
        "last_summarized_message_id": new.last_summarized_message_id,
        "first_source_message_id": covered_source_ids[0],
        "last_source_message_id": covered_source_ids[-1],
        "provider": provider[:64], "model_name": model_name[:160],
        "updated_at": now, "last_failure_category": None, "last_failure_at": None,
    }
    if previous is None:
        statement = pg_insert(ConversationSummary).values(
            user_id=user_id, thread_id=thread_id, version=1, **values,
        ).on_conflict_do_nothing()
        changed = (await db.execute(statement)).rowcount
    else:
        statement = update(ConversationSummary).where(
            ConversationSummary.user_id == user_id,
            ConversationSummary.thread_id == thread_id,
            ConversationSummary.version == previous.version,
        ).values(version=previous.version + 1, **values)
        changed = (await db.execute(statement)).rowcount
    if not changed:
        raise ConcurrentSummaryUpdate("Another request advanced this summary")
    return await load_summary(db, user_id=user_id, thread_id=thread_id)
