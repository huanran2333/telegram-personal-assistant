"""Derived, scoped transcript search projection.

Only checkpointed user/assistant conversational text is indexed. Tool payloads,
system prompts and external source bodies are deliberately excluded. A damaged
index can be rebuilt from the current Saver state; it is never execution authority.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import case, delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TranscriptEmbedding, TranscriptProjection, TranscriptSuppression

MAX_INDEXED_CHARS = 8192


@dataclass(frozen=True)
class HistoryHit:
    user_id: str
    thread_id: str
    message_id: str
    group_id: str
    role: str
    occurred_at: datetime
    content: str
    source_ref: str
    supersedes_message_id: str | None
    stale: bool
    retrieval_method: str
    score: float | None
    supersedes_thread_id: str | None = None


class HistorySearchBackend(Protocol):
    """Algorithm boundary for R3 lexical/semantic/hybrid comparison."""

    async def search(self, db: AsyncSession, *, user_id: str, thread_id: str,
                     query: str, top_k: int, include_stale: bool) -> list[HistoryHit]: ...


def _projection_rows(user_id: str, thread_id: str, messages: Iterable) -> list[dict]:
    rows = []
    group_id = ""
    for position, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage) and not (message.tool_calls or message.invalid_tool_calls):
            role = "assistant"
        else:
            continue
        if not isinstance(message.content, str) or not message.content.strip():
            continue
        content = re.sub(r"\s+", " ", message.content).strip()[:MAX_INDEXED_CHARS]
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        identifier = str(message.id or f"position-{position}-{digest[:16]}")
        if role == "user":
            group_id = identifier
        if not group_id:
            continue
        source_time = message.additional_kwargs.get("occurred_at_utc") if role == "user" else None
        try:
            occurred_at = datetime.fromisoformat(source_time) if source_time else datetime.now(timezone.utc)
            if occurred_at.tzinfo is None:
                raise ValueError("source timestamp needs timezone")
        except (TypeError, ValueError):
            occurred_at = datetime.now(timezone.utc)
        supersedes = message.additional_kwargs.get("supersedes_message_id") if role == "user" else None
        supersedes_thread = (message.additional_kwargs.get("supersedes_thread_id")
                             if supersedes else None)
        if (not isinstance(supersedes_thread, str) or
                not 1 <= len(supersedes_thread) <= 160):
            supersedes_thread = None
        if (not isinstance(supersedes, str) or not 1 <= len(supersedes) <= 160 or
                (supersedes == identifier and supersedes_thread in {None, thread_id})):
            supersedes = supersedes_thread = None
        rows.append({
            "user_id": user_id, "thread_id": thread_id, "message_id": identifier,
            "group_id": group_id, "role": role,
            "occurred_at": occurred_at,
            "search_text": content, "source_ref": f"checkpoint:{thread_id}:{identifier}",
            "content_sha256": digest,
            "supersedes_message_id": supersedes,
            "supersedes_thread_id": supersedes_thread,
        })
    return rows


async def sync_checkpoint_messages(db: AsyncSession, *, user_id: str, thread_id: str,
                                   messages: Iterable) -> int:
    """Idempotent projection of a checkpoint snapshot; caller commits separately."""
    rows = _projection_rows(user_id, thread_id, messages)
    suppressed = set((await db.scalars(select(TranscriptSuppression.message_id).where(
        TranscriptSuppression.user_id == user_id,
        TranscriptSuppression.thread_id == thread_id))).all())
    suppressed_groups = {row["group_id"] for row in rows if row["message_id"] in suppressed}
    for row in rows:
        if row["message_id"] in suppressed or row["group_id"] in suppressed_groups:
            continue
        statement = pg_insert(TranscriptProjection).values(**row).on_conflict_do_nothing(
            constraint="uq_transcript_source")
        await db.execute(statement)
    return len(rows)


async def sync_from_saver(db: AsyncSession, *, graph, user_id: str, thread_id: str) -> int:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        return 0
    if snapshot.values.get("user_id") != user_id:
        raise PermissionError("Checkpoint owner does not match transcript scope")
    return await sync_checkpoint_messages(db, user_id=user_id, thread_id=thread_id,
                                          messages=snapshot.values.get("messages", []))


async def rebuild_from_saver(db: AsyncSession, *, graph, user_id: str, thread_id: str) -> int:
    """Replace only one user's thread projection in one transaction.

    The raw checkpoint is read before deletion; transaction rollback preserves
    the old index on failure. Search never scans checkpoint revisions.
    """
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot or not snapshot.values:
        raise ValueError("Cannot rebuild without a checkpoint snapshot")
    if snapshot.values.get("user_id") != user_id:
        raise PermissionError("Checkpoint owner does not match transcript scope")
    messages = snapshot.values.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Checkpoint has no message history")
    await db.execute(delete(TranscriptProjection).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id))
    await db.execute(delete(TranscriptEmbedding).where(
        TranscriptEmbedding.user_id == user_id,
        TranscriptEmbedding.thread_id == thread_id))
    return await sync_checkpoint_messages(db, user_id=user_id, thread_id=thread_id, messages=messages)


async def sync_workflow_turn(db: AsyncSession, *, user_id: str, thread_id: str,
                             turn_id: str, request: str, response: str = "",
                             occurred_at: str | None = None) -> int:
    """Project typed workflow's source-backed request and final response only."""
    if not turn_id or not request:
        return 0
    messages = [HumanMessage(id=f"{turn_id}:user", content=request,
                             additional_kwargs={"occurred_at_utc": occurred_at} if occurred_at else {})]
    if response:
        messages.append(AIMessage(id=f"{turn_id}:assistant", content=response))
    return await sync_checkpoint_messages(db, user_id=user_id,
                                          thread_id=thread_id, messages=messages)


async def load_projected_conversation(db: AsyncSession, *, user_id: str,
                                      thread_id: str, max_messages: int = 5000) -> list:
    """Read a scoped derived conversation view for typed workflow context.

    Ordinary Chat/Todo use Saver messages directly. Typed workflow revisions
    hold one request at a time, so its separately rebuildable projection is the
    indexed conversation view. This function never treats it as execution
    authority and surfaces an oversized history rather than silently clipping.
    """
    if not user_id or not thread_id or not 1 <= max_messages <= 10000:
        raise ValueError("Bounded user/thread history scope is required")
    rows = (await db.scalars(select(TranscriptProjection).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id,
        TranscriptProjection.deleted_at.is_(None),
        ~exists(select(TranscriptSuppression.message_id).where(
            TranscriptSuppression.user_id == TranscriptProjection.user_id,
            TranscriptSuppression.thread_id == TranscriptProjection.thread_id,
            TranscriptSuppression.message_id == TranscriptProjection.message_id)),
    ).order_by(TranscriptProjection.occurred_at, TranscriptProjection.message_id)
      .limit(max_messages + 1))).all()
    if len(rows) > max_messages:
        raise RuntimeError("Projected conversation exceeds the measured context history capacity")
    result = []
    for row in rows:
        if row.role == "user":
            result.append(HumanMessage(
                id=row.message_id, content=row.search_text,
                additional_kwargs={"occurred_at_utc": row.occurred_at.isoformat()}))
        elif row.role == "assistant":
            result.append(AIMessage(id=row.message_id, content=row.search_text))
    return result


async def rebuild_workflow_from_saver(db: AsyncSession, *, graph, user_id: str,
                                      thread_id: str) -> int:
    """Rare repair path: scan revisions, never used by search or request routing."""
    by_turn = {}
    async for snapshot in graph.aget_state_history({"configurable": {"thread_id": thread_id}}):
        state = snapshot.values or {}
        if state.get("user_id") != user_id or not state.get("transcript_turn_id"):
            continue
        turn_id = state["transcript_turn_id"]
        # History is newest first. Preserve newest nonempty response, while
        # retaining the request even when the workflow is interrupted.
        if turn_id not in by_turn:
            by_turn[turn_id] = (state.get("request", ""), state.get("response", ""),
                                state.get("transcript_occurred_at"))
    await db.execute(delete(TranscriptProjection).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id))
    await db.execute(delete(TranscriptEmbedding).where(
        TranscriptEmbedding.user_id == user_id,
        TranscriptEmbedding.thread_id == thread_id))
    count = 0
    for turn_id, (request, response, occurred_at) in reversed(list(by_turn.items())):
        count += await sync_workflow_turn(db, user_id=user_id, thread_id=thread_id,
                                          turn_id=turn_id, request=request, response=response,
                                          occurred_at=occurred_at)
    return count


async def tombstone_message(db: AsyncSession, *, user_id: str, thread_id: str,
                            message_id: str) -> bool:
    """Persist privacy suppression so rebuild does not resurrect the hit."""
    supersedes = await db.execute(select(
        TranscriptProjection.supersedes_message_id,
        TranscriptProjection.supersedes_thread_id).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id,
        TranscriptProjection.message_id == message_id))
    supersedes_id, supersedes_thread = supersedes.one_or_none() or (None, None)
    await db.execute(pg_insert(TranscriptSuppression).values(
        user_id=user_id, thread_id=thread_id, message_id=message_id,
        supersedes_message_id=supersedes_id,
        supersedes_thread_id=supersedes_thread,
        suppressed_at=datetime.now(timezone.utc)).on_conflict_do_nothing())
    result = await db.execute(update(TranscriptProjection).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id,
        TranscriptProjection.message_id == message_id,
        TranscriptProjection.deleted_at.is_(None)).values(deleted_at=datetime.now(timezone.utc)))
    await db.execute(delete(TranscriptEmbedding).where(
        TranscriptEmbedding.user_id == user_id,
        TranscriptEmbedding.thread_id == thread_id,
        TranscriptEmbedding.message_id == message_id))
    # A lossy summary cannot be surgically edited. Invalidate its cursor so
    # the next token-pressure update rebuilds from unsuppressed raw Saver turns.
    from app.agent.memory_summary import invalidate_summary_for_source
    await invalidate_summary_for_source(db, user_id=user_id,
                                        thread_id=thread_id, message_id=message_id)
    return bool(result.rowcount)


async def search_history(db: AsyncSession, *, user_id: str, thread_id: str,
                         query: str, top_k: int = 5, include_stale: bool = False,
                         backend: HistorySearchBackend | None = None) -> list[HistoryHit]:
    """Internal scoped API; the retrieval algorithm is replaceable in R3."""
    if not user_id or not thread_id or not query.strip():
        raise ValueError("user_id, thread_id and query are required")
    if not 1 <= top_k <= 24:
        raise ValueError("top_k must be between 1 and 24")
    engine = backend or PostgresLexicalSearch()
    return await engine.search(db, user_id=user_id, thread_id=thread_id, query=query,
                               top_k=top_k, include_stale=include_stale)


class PostgresLexicalSearch:
    async def search(self, db: AsyncSession, *, user_id: str, thread_id: str,
                     query: str, top_k: int, include_stale: bool) -> list[HistoryHit]:
        return await _lexical_search_history(db, user_id=user_id, thread_id=thread_id,
                                             query=query, top_k=top_k, include_stale=include_stale)


async def _lexical_search_history(db: AsyncSession, *, user_id: str, thread_id: str,
                                  query: str, top_k: int, include_stale: bool) -> list[HistoryHit]:
    ts_query = func.websearch_to_tsquery("simple", query)
    document = func.to_tsvector("simple", TranscriptProjection.search_text)
    rank = func.ts_rank_cd(document, ts_query)
    newer = TranscriptProjection.__table__.alias("newer")
    suppressed_newer = TranscriptSuppression.__table__.alias("suppressed_newer")
    stale_exists = or_(exists(select(newer.c.id).where(
        newer.c.user_id == TranscriptProjection.user_id,
        func.coalesce(newer.c.supersedes_thread_id, newer.c.thread_id) == TranscriptProjection.thread_id,
        newer.c.supersedes_message_id == TranscriptProjection.message_id)),
        exists(select(suppressed_newer.c.message_id).where(
            suppressed_newer.c.user_id == TranscriptProjection.user_id,
            func.coalesce(suppressed_newer.c.supersedes_thread_id,
                          suppressed_newer.c.thread_id) == TranscriptProjection.thread_id,
            suppressed_newer.c.supersedes_message_id == TranscriptProjection.message_id)))
    # PostgreSQL default FTS does not segment Chinese. Bounded substring
    # fallback is explicit, measurable and *not* semantic recall.
    has_cjk = bool(re.search(r"[\u3400-\u9fff]", query))
    if has_cjk:
        segments = re.findall(r"[\u3400-\u9fff]+|[A-Za-z]+-[0-9]+", query[:200])
        parts = []
        for segment in segments:
            if re.search(r"[\u3400-\u9fff]", segment) and len(segment) > 2:
                parts.extend(segment[index:index+2] for index in range(len(segment)-1))
            else:
                parts.append(segment)
        parts = list(dict.fromkeys(parts))[:16]
        predicates = [TranscriptProjection.search_text.ilike(f"%{part}%") for part in parts]
        match = or_(*predicates)
        rank = sum((case((predicate, 1), else_=0) for predicate in predicates), start=0)
    else:
        match = document.op("@@")(ts_query)
    statement = select(TranscriptProjection, rank.label("score"), stale_exists.label("stale")).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id,
        TranscriptProjection.deleted_at.is_(None),
        ~exists(select(TranscriptSuppression.message_id).where(
            TranscriptSuppression.user_id == TranscriptProjection.user_id,
            TranscriptSuppression.thread_id == TranscriptProjection.thread_id,
            TranscriptSuppression.message_id == TranscriptProjection.message_id)),
        match)
    if not include_stale:
        statement = statement.where(~stale_exists)
    statement = statement.order_by(rank.desc(), TranscriptProjection.occurred_at.desc()).limit(top_k)
    result = await db.execute(statement)
    return [HistoryHit(row.user_id, row.thread_id, row.message_id, row.group_id,
                       row.role, row.occurred_at, row.search_text, row.source_ref,
                       row.supersedes_message_id, bool(stale), "postgres_lexical", float(score or 0),
                       row.supersedes_thread_id)
            for row, score, stale in result.all()]
