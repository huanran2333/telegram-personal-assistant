"""Rebuildable scoped transcript embeddings with a private-data egress boundary.

The default pgvector sidecar is the indexed retrieval path. JSON/Python cosine
remains an explicitly selected, bounded experimental fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Protocol

from openai import AsyncOpenAI
import httpx2

from sqlalchemy import and_, exists, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.transcript import HistoryHit
from app.db.models import TranscriptEmbedding, TranscriptProjection, TranscriptSuppression


class EmbeddingProvider(Protocol):
    model_id: str
    provider_name: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SemanticIndexIncomplete(RuntimeError):
    """A scoped transcript has not been fully indexed for this model."""


class OpenAICompatibleEmbeddingProvider:
    """Thin provider adapter; egress is enforced here and by index/search.

    The selected embedding endpoint remains independent of the chat model.
    It deliberately does not inherit the chat-model endpoint or key.
    """

    def __init__(self, *, base_url: str, model_id: str, dimension: int,
                 egress: EmbeddingEgressPolicy, private_data: bool,
                 api_key: str = "not-required", timeout_seconds: float = 30):
        if not model_id or dimension < 1:
            raise ValueError("Embedding model and dimension are required")
        if egress.base_url != base_url:
            raise ValueError("Embedding policy URL must match provider URL")
        self.model_id = model_id
        self.provider_name = urlparse(base_url).hostname or "unknown"
        self.dimension = dimension
        self._egress = egress
        self._private_data = private_data
        self.last_usage_tokens: int | None = None
        self.total_usage_tokens: int | None = 0
        # A self-hosted deployment must not inherit an ambient HTTP proxy.
        # The deployment operator still owns DNS/routing and server log policy.
        self._http_client = (httpx2.AsyncClient(trust_env=False)
                             if egress.local_self_hosted else None)
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds,
            max_retries=0, http_client=self._http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._egress.assert_allowed(private_data=self._private_data)
        if not texts or len(texts) > 128 or any(not item.strip() for item in texts):
            raise ValueError("A nonempty bounded embedding batch is required")
        # Qwen3-Embedding-8B defaults to 4096 dimensions. Ask Ollama for the
        # configured 2560-dimensional Matryoshka projection so the same
        # dimension is used for indexing and querying in pgvector.
        result = await self._client.embeddings.create(
            model=self.model_id, input=texts, dimensions=self.dimension)
        usage = getattr(result, "usage", None)
        self.last_usage_tokens = getattr(usage, "total_tokens", None)
        if self.last_usage_tokens is None:
            self.total_usage_tokens = None
        elif self.total_usage_tokens is not None:
            self.total_usage_tokens += self.last_usage_tokens
        ordered = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("Embedding provider returned incomplete/duplicate indices")
        return [_validated_vector(item.embedding, self.dimension) for item in ordered]


@dataclass(frozen=True)
class EmbeddingEgressPolicy:
    base_url: str
    private_data_approved: bool = False
    local_self_hosted: bool = False
    no_unknown_proxy: bool = False
    body_logging_disabled: bool = False

    def assert_allowed(self, *, private_data: bool) -> None:
        if not private_data:
            return
        parsed = urlparse(self.base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        approved_remote = parsed.scheme == "https" and self.private_data_approved
        approved_self_hosted = (self.local_self_hosted and self.no_unknown_proxy and
                                self.body_logging_disabled and parsed.scheme in {"http", "https"})
        if not (local or approved_remote or approved_self_hosted):
            raise PermissionError("Private transcript embedding requires a verified provider boundary")


def _validated_vector(vector: list[float], dimension: int) -> list[float]:
    if len(vector) != dimension or dimension < 1:
        raise ValueError("Embedding dimension mismatch")
    values = [float(item) for item in vector]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("Embedding contains non-finite values")
    if math.sqrt(sum(item * item for item in values)) == 0:
        raise ValueError("Zero embedding is invalid")
    return values


async def index_transcript_embeddings(
    db: AsyncSession, *, user_id: str, thread_id: str,
    provider: EmbeddingProvider, egress: EmbeddingEgressPolicy,
    private_data: bool = True, batch_size: int = 32,
) -> int:
    """Idempotently index new/changed nondeleted messages in one scoped thread.

    Caller commits. A failed provider call cannot partially commit a batch.
    The provider is never called before the private-data policy check.
    """
    if not user_id or not thread_id or not 1 <= batch_size <= 128:
        raise ValueError("Bounded embedding scope and batch are required")
    egress.assert_allowed(private_data=private_data)
    rows = (await db.scalars(select(TranscriptProjection).where(
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id,
        TranscriptProjection.deleted_at.is_(None),
        ~exists(select(TranscriptSuppression.message_id).where(and_(
            TranscriptSuppression.user_id == TranscriptProjection.user_id,
            TranscriptSuppression.thread_id == TranscriptProjection.thread_id,
            TranscriptSuppression.message_id == TranscriptProjection.message_id,
        ))),
    ).order_by(TranscriptProjection.occurred_at, TranscriptProjection.message_id))).all()
    existing = (await db.execute(select(
        TranscriptEmbedding.message_id, TranscriptEmbedding.content_sha256,
    ).where(TranscriptEmbedding.user_id == user_id,
            TranscriptEmbedding.thread_id == thread_id,
            TranscriptEmbedding.model_id == provider.model_id))).all()
    hashes = dict(existing)
    pending = [row for row in rows if hashes.get(row.message_id) != row.content_sha256]
    changed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = await provider.embed([row.search_text for row in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned an incomplete batch")
        checked = [_validated_vector(vector, provider.dimension) for vector in vectors]
        for row, vector in zip(batch, checked):
            statement = pg_insert(TranscriptEmbedding).values(
                user_id=user_id, thread_id=thread_id, message_id=row.message_id,
                model_id=provider.model_id, provider=provider.provider_name,
                dimension=provider.dimension, vector=vector,
                content_sha256=row.content_sha256,
            ).on_conflict_do_update(
                constraint="uq_transcript_embedding_source_model",
                set_={"vector": vector, "dimension": provider.dimension,
                      "content_sha256": row.content_sha256,
                      "provider": provider.provider_name},
            )
            await db.execute(statement)
            changed += 1
    return changed


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    magnitude = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / magnitude if magnitude else 0.0


async def semantic_history_candidates(
    db: AsyncSession, *, user_id: str, thread_id: str, query: str,
    provider: EmbeddingProvider, egress: EmbeddingEgressPolicy,
    private_data: bool = True, top_k: int = 24, include_stale: bool = False,
    max_scanned: int = 200,
) -> list[HistoryHit]:
    """Bounded OpenClaw-style in-process cosine fallback for a scoped thread.

    A missing/stale index is an explicit error, not an empty-memory claim.
    This is only a bounded fallback candidate generator, not an indexed
    production backend. The 200-row default is based on the isolated 100/200/
    1k/10k measurements; callers may override it for rollback-only benchmarks.
    """
    if not user_id or not thread_id or not query.strip() or not 1 <= top_k <= 100:
        raise ValueError("Bounded semantic search scope is required")
    egress.assert_allowed(private_data=private_data)
    newer = TranscriptProjection.__table__.alias("newer")
    suppressed_newer = TranscriptSuppression.__table__.alias("suppressed_newer")
    stale = exists(select(newer.c.id).where(
        newer.c.user_id == TranscriptProjection.user_id,
        func.coalesce(newer.c.supersedes_thread_id, newer.c.thread_id) == TranscriptProjection.thread_id,
        newer.c.supersedes_message_id == TranscriptProjection.message_id,
    )) | exists(select(suppressed_newer.c.message_id).where(
        suppressed_newer.c.user_id == TranscriptProjection.user_id,
        func.coalesce(suppressed_newer.c.supersedes_thread_id,
                      suppressed_newer.c.thread_id) == TranscriptProjection.thread_id,
        suppressed_newer.c.supersedes_message_id == TranscriptProjection.message_id,
    ))
    scope_predicates = (
        TranscriptProjection.user_id == user_id,
        TranscriptProjection.thread_id == thread_id,
        TranscriptProjection.deleted_at.is_(None),
        ~exists(select(TranscriptSuppression.message_id).where(and_(
            TranscriptSuppression.user_id == TranscriptProjection.user_id,
            TranscriptSuppression.thread_id == TranscriptProjection.thread_id,
            TranscriptSuppression.message_id == TranscriptProjection.message_id,
        ))),
    )
    statement = select(TranscriptProjection, TranscriptEmbedding, stale.label("stale")).join(
        TranscriptEmbedding, and_(
            TranscriptEmbedding.user_id == TranscriptProjection.user_id,
            TranscriptEmbedding.thread_id == TranscriptProjection.thread_id,
            TranscriptEmbedding.message_id == TranscriptProjection.message_id,
            TranscriptEmbedding.content_sha256 == TranscriptProjection.content_sha256,
            TranscriptEmbedding.model_id == provider.model_id,
        )
    ).where(*scope_predicates)
    count_statement = select(func.count()).select_from(TranscriptProjection).where(*scope_predicates)
    if not include_stale:
        statement = statement.where(~stale)
        count_statement = count_statement.where(~stale)
    expected_count = await db.scalar(count_statement)
    if expected_count > max_scanned:
        raise RuntimeError("Scoped semantic fallback exceeds measured row cap")
    query_vectors = await provider.embed([query])
    if len(query_vectors) != 1:
        raise ValueError("Embedding provider returned no query vector")
    query_vector = _validated_vector(query_vectors[0], provider.dimension)
    rows = (await db.execute(statement.limit(max_scanned + 1))).all()
    if len(rows) != expected_count:
        raise SemanticIndexIncomplete("Scoped transcript embedding index is incomplete")
    result = []
    for projection, embedding, is_stale in rows:
        vector = _validated_vector(embedding.vector, provider.dimension)
        result.append(HistoryHit(
            user_id, thread_id, projection.message_id, projection.group_id,
            projection.role, projection.occurred_at, projection.search_text,
            projection.source_ref, projection.supersedes_message_id,
            bool(is_stale), "semantic_cosine", _cosine(query_vector, vector),
            projection.supersedes_thread_id,
        ))
    return sorted(result, key=lambda hit: (-(hit.score or 0), hit.message_id))[:top_k]
