"""Production lifetime and bounded background maintenance for mature Memory.

The response path reads selected evidence. Index generation and LangMem
consolidation run outside user-response latency, using durable source cursors.
Saver, profile Store and domain databases retain their independent authority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, exists, func, or_, select

from app.agent.memory_consolidation import consolidate_pending
from app.agent.memory_context import ContextOrchestrator
from app.agent.memory_embedding import (
    EmbeddingEgressPolicy, OpenAICompatibleEmbeddingProvider, SemanticIndexIncomplete,
    index_transcript_embeddings,
)
from app.agent.structured_search import index_structured_memories
from app.agent.memory_pgvector import (
    ensure_pgvector_projection, pgvector_projection_complete,
    pending_pgvector_scopes, pending_pgvector_structured_users,
    sync_structured_halfvec,
    sync_transcript_halfvec,
)
from app.db.models import (
    ConversationSummary, MemoryConsolidationCursor, StructuredMemory,
    TranscriptEmbedding, TranscriptProjection,
)
from app.trace import emit_trace

logger = logging.getLogger(__name__)


@dataclass
class MemoryRuntime:
    orchestrator: ContextOrchestrator
    provider: OpenAICompatibleEmbeddingProvider
    egress: EmbeddingEgressPolicy
    session_factory: object
    extraction_model: object
    vector_backend: str = "pgvector_halfvec_exact"

    async def preflight_storage(self) -> None:
        """Verify required SQL substrate and private-data egress policy."""
        if self.vector_backend == "transcript_lexical":
            from app.db.models import TranscriptSuppression
            async with self.session_factory() as db:
                for table in (TranscriptProjection, TranscriptSuppression):
                    await db.execute(select(1).select_from(table).limit(1))
            return
        self.egress.assert_allowed(private_data=True)
        async with self.session_factory() as db:
            for table in (ConversationSummary, StructuredMemory, TranscriptProjection,
                          TranscriptEmbedding, MemoryConsolidationCursor):
                await db.execute(select(1).select_from(table).limit(1))

    async def preflight_embedding(self) -> None:
        """Probe the optional semantic backend with synthetic text only."""
        if self.vector_backend == "transcript_lexical":
            return
        if self.vector_backend == "pgvector_halfvec_exact":
            async with self.session_factory() as db:
                await ensure_pgvector_projection(db)
                await db.commit()
                if not await pgvector_projection_complete(
                        db, model_id=self.provider.model_id,
                        dimension=self.provider.dimension):
                    raise SemanticIndexIncomplete("Pgvector projection is stale")
        vectors = await self.provider.embed(["R3 synthetic readiness probe"])
        if len(vectors) != 1:
            raise RuntimeError("Embedding preflight returned an incomplete result")

    async def maintain_thread(self, *, user_id: str, thread_id: str,
                              consolidate: bool = True) -> dict[str, int]:
        """Idempotent derived-index update and one transactional source batch.

        If embedding fails, the source/cursor stays intact and a later tick can
        retry. A failed consolidation does not advance its cursor. No user text
        or model result is logged.
        """
        if self.vector_backend == "transcript_lexical":
            return {"transcript_embeddings": 0, "memory_changes": 0,
                    "structured_embeddings": 0, "rejected_candidates": 0}
        counts = {"transcript_embeddings": 0, "memory_changes": 0,
                  "structured_embeddings": 0, "rejected_candidates": 0}
        async with self.session_factory() as db:
            counts["transcript_embeddings"] = await index_transcript_embeddings(
                db, user_id=user_id, thread_id=thread_id,
                provider=self.provider, egress=self.egress)
            if self.vector_backend == "pgvector_halfvec_exact":
                await sync_transcript_halfvec(
                    db, user_id=user_id, thread_id=thread_id,
                    model_id=self.provider.model_id,
                    dimension=self.provider.dimension)
            await db.commit()
        if consolidate:
            result = await consolidate_pending(
                self.session_factory, user_id=user_id, thread_id=thread_id,
                model=self.extraction_model)
            counts["memory_changes"] = len(result.changed_memory_ids)
            counts["rejected_candidates"] = result.rejected_candidates
            emit_trace("memory_maintenance", operation="consolidation_batch", success=True,
                       model_calls=result.model_calls,
                       token_usage=(result.input_tokens + result.output_tokens
                                    if result.input_tokens is not None and
                                    result.output_tokens is not None else None),
                       latency_ms=result.model_latency_ms,
                       source_count=len(result.processed_source_ids))
            if result.rejected_candidates:
                emit_trace("memory_maintenance", operation="candidate_rejected",
                           success=True, source_count=result.rejected_candidates)
        async with self.session_factory() as db:
            counts["structured_embeddings"] = await index_structured_memories(
                db, user_id=user_id, provider=self.provider, egress=self.egress)
            if self.vector_backend == "pgvector_halfvec_exact":
                await sync_structured_halfvec(
                    db, user_id=user_id, model_id=self.provider.model_id,
                    dimension=self.provider.dimension)
            await db.commit()
        return counts

    async def maintenance_tick(self, *, max_threads: int = 20,
                               idle_seconds: int = 90) -> dict[str, int]:
        """Process idle source ranges and stale embedding projections.

        Source selection is DB-backed and restart-safe. A cursor's ordering,
        not an in-memory timer, decides what is already consolidated. We select
        index gaps independently so a previously failed embedding can retry
        even after the consolidation cursor advanced.
        """
        if self.vector_backend == "transcript_lexical":
            return {"processed_threads": 0, "failed_threads": 0}
        if not 1 <= max_threads <= 100 or idle_seconds < 0:
            raise ValueError("Bounded maintenance settings are required")
        now = datetime.now(timezone.utc)
        missing_embedding = ~exists(select(TranscriptEmbedding.id).where(and_(
            TranscriptEmbedding.user_id == TranscriptProjection.user_id,
            TranscriptEmbedding.thread_id == TranscriptProjection.thread_id,
            TranscriptEmbedding.message_id == TranscriptProjection.message_id,
            TranscriptEmbedding.model_id == self.provider.model_id,
            TranscriptEmbedding.content_sha256 == TranscriptProjection.content_sha256,
        )))
        async with self.session_factory() as db:
            pending = (await db.execute(select(
                TranscriptProjection.user_id, TranscriptProjection.thread_id,
                func.max(TranscriptProjection.occurred_at).label("last_seen"),
            ).outerjoin(MemoryConsolidationCursor, and_(
                MemoryConsolidationCursor.user_id == TranscriptProjection.user_id,
                MemoryConsolidationCursor.thread_id == TranscriptProjection.thread_id,
            )).where(
                TranscriptProjection.deleted_at.is_(None),
                or_(
                    missing_embedding,
                    and_(
                        TranscriptProjection.role == "user",
                        TranscriptProjection.occurred_at <= now - timedelta(seconds=idle_seconds),
                        or_(
                            MemoryConsolidationCursor.last_source_occurred_at.is_(None),
                            TranscriptProjection.occurred_at >
                            MemoryConsolidationCursor.last_source_occurred_at,
                        ),
                    ),
                ),
            ).group_by(TranscriptProjection.user_id, TranscriptProjection.thread_id)
             .order_by(func.max(TranscriptProjection.occurred_at)).limit(max_threads))).all()
            structured_users: list[str] = []
            if self.vector_backend == "pgvector_halfvec_exact":
                # The ordinary query only sees missing JSON embeddings. A
                # backend migration can leave those complete while the new
                # derived halfvec sidecar is absent; select that gap too.
                known = {(user_id, thread_id) for user_id, thread_id, _ in pending}
                for user_id, thread_id in await pending_pgvector_scopes(
                        db, model_id=self.provider.model_id,
                        dimension=self.provider.dimension, limit=max_threads):
                    if len(pending) >= max_threads:
                        break
                    if (user_id, thread_id) not in known:
                        pending.append((user_id, thread_id, None))
                        known.add((user_id, thread_id))
                structured_users = await pending_pgvector_structured_users(
                    db, model_id=self.provider.model_id,
                    dimension=self.provider.dimension, limit=max_threads)
        done, failed = 0, 0
        refreshed_users: set[str] = set()
        for user_id, thread_id, last_seen in pending:
            try:
                await self.maintain_thread(
                    user_id=user_id, thread_id=thread_id,
                    consolidate=bool(last_seen and last_seen <=
                                     now - timedelta(seconds=idle_seconds)))
                done += 1
                refreshed_users.add(user_id)
                emit_trace("memory_maintenance", operation="thread_batch", success=True)
            except Exception as exc:
                failed += 1
                logger.warning("memory_maintenance stage=thread_batch category=%s",
                               type(exc).__name__)
                emit_trace("memory_maintenance", operation="thread_batch", success=False,
                           error_category=type(exc).__name__)
        structured_done = 0
        if self.vector_backend == "pgvector_halfvec_exact":
            for user_id in structured_users:
                if user_id in refreshed_users:
                    continue
                try:
                    async with self.session_factory() as db:
                        await index_structured_memories(
                            db, user_id=user_id, provider=self.provider,
                            egress=self.egress)
                        await sync_structured_halfvec(
                            db, user_id=user_id, model_id=self.provider.model_id,
                            dimension=self.provider.dimension)
                        await db.commit()
                    structured_done += 1
                    emit_trace("memory_maintenance", operation="structured_backfill",
                               success=True)
                except Exception as exc:
                    failed += 1
                    logger.warning("memory_maintenance stage=structured_backfill category=%s",
                                   type(exc).__name__)
                    emit_trace("memory_maintenance", operation="structured_backfill",
                               success=False, error_category=type(exc).__name__)
        result = {"processed_threads": done, "failed_threads": failed}
        if self.vector_backend == "pgvector_halfvec_exact":
            result["processed_structured_users"] = structured_done
        return result

    async def aclose(self) -> None:
        if self.provider is not None:
            await self.provider.aclose()


def create_memory_runtime(*, session_factory, settings) -> MemoryRuntime:
    """Construct from explicit config; never inherit chat provider credentials."""
    if settings.memory_mode == "transcript_lexical":
        return MemoryRuntime(
            orchestrator=ContextOrchestrator(
                session_factory=session_factory, summary_model=None,
                summary_provider="", summary_model_name="",
                embedding_provider=None, embedding_egress=None,
                vector_backend="transcript_lexical"),
            provider=None, egress=None, session_factory=session_factory,
            extraction_model=None, vector_backend="transcript_lexical")
    from app.agent.nodes import _build_chat_llm

    egress = EmbeddingEgressPolicy(
        base_url=settings.memory_embedding_base_url,
        local_self_hosted=settings.memory_embedding_local_self_hosted,
        no_unknown_proxy=settings.memory_embedding_no_unknown_proxy,
        body_logging_disabled=settings.memory_embedding_body_logging_disabled,
    )
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=settings.memory_embedding_base_url,
        model_id=settings.memory_embedding_model,
        dimension=settings.memory_embedding_dimension,
        api_key=settings.memory_embedding_api_key,
        egress=egress, private_data=True,
    )
    model = _build_chat_llm(settings.llm_provider, settings.llm_model)
    return MemoryRuntime(
        orchestrator=ContextOrchestrator(
            session_factory=session_factory, summary_model=model,
            summary_provider=settings.llm_provider,
            summary_model_name=settings.llm_model,
            embedding_provider=provider, embedding_egress=egress,
            vector_backend=settings.memory_vector_backend,
        ),
        provider=provider, egress=egress,
        session_factory=session_factory, extraction_model=model,
        vector_backend=settings.memory_vector_backend,
    )
