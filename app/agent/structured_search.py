"""Namespace-scoped semantic selection over typed durable knowledge.

LangGraph Store's namespace/search contract is adapted to the relational
authority needed for correction/tombstone transactions. Embeddings are a
rebuildable projection; canonical StructuredMemory rows remain authoritative.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory_embedding import (
    EmbeddingEgressPolicy, EmbeddingProvider, SemanticIndexIncomplete,
    _cosine, _validated_vector,
)
from app.db.models import StructuredMemory, StructuredMemoryEmbedding


@dataclass(frozen=True)
class StructuredHit:
    memory_id: str
    user_id: str
    kind: str
    payload: dict
    canonical_text: str
    source_refs: tuple[dict, ...]
    updated_at: datetime
    score: float
    retrieval_method: str


def _current_scope(user_id: str):
    return (
        StructuredMemory.user_id == user_id,
        StructuredMemory.status == "current",
        (StructuredMemory.expires_at.is_(None) |
         (StructuredMemory.expires_at > datetime.now(timezone.utc))),
    )


async def index_structured_memories(
    db: AsyncSession, *, user_id: str,
    provider: EmbeddingProvider, egress: EmbeddingEgressPolicy,
    private_data: bool = True, batch_size: int = 32,
) -> int:
    """Idempotently index current facts; caller owns the transaction."""
    if not user_id or not 1 <= batch_size <= 128:
        raise ValueError("Bounded user-scoped indexing is required")
    egress.assert_allowed(private_data=private_data)
    rows = (await db.scalars(select(StructuredMemory).where(*_current_scope(user_id)).order_by(
        StructuredMemory.id))).all()
    existing = (await db.execute(select(
        StructuredMemoryEmbedding.memory_id, StructuredMemoryEmbedding.content_sha256,
    ).where(StructuredMemoryEmbedding.user_id == user_id,
            StructuredMemoryEmbedding.model_id == provider.model_id))).all()
    hashes = dict(existing)
    pending = [row for row in rows if hashes.get(row.id) != hashlib.sha256(
        row.canonical_text.encode("utf-8")).hexdigest()]
    changed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = await provider.embed([row.canonical_text for row in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned an incomplete batch")
        checked = [_validated_vector(vector, provider.dimension) for vector in vectors]
        for row, vector in zip(batch, checked):
            digest = hashlib.sha256(row.canonical_text.encode("utf-8")).hexdigest()
            await db.execute(pg_insert(StructuredMemoryEmbedding).values(
                user_id=user_id, memory_id=row.id, model_id=provider.model_id,
                provider=provider.provider_name, dimension=provider.dimension,
                vector=vector, content_sha256=digest,
            ).on_conflict_do_update(
                constraint="uq_structured_embedding_source_model",
                set_={"vector": vector, "dimension": provider.dimension,
                      "content_sha256": digest, "provider": provider.provider_name},
            ))
            changed += 1
    return changed


async def search_structured_memories(
    db: AsyncSession, *, user_id: str, query: str,
    provider: EmbeddingProvider, egress: EmbeddingEgressPolicy,
    private_data: bool = True, top_k: int = 6,
    max_scanned: int = 200,
) -> list[StructuredHit]:
    """Return only current, unexpired, model-indexed facts with provenance.

    An incomplete index is an unavailable capability, not an empty result.
    This candidate path is lower authority than Profile/DB/Calendar/Gmail.
    """
    if not user_id or not query.strip() or not 1 <= top_k <= 20:
        raise ValueError("Bounded user-scoped structured search is required")
    egress.assert_allowed(private_data=private_data)
    count = await db.scalar(select(func.count()).select_from(StructuredMemory).where(*_current_scope(user_id)))
    if count > max_scanned:
        raise RuntimeError("Structured semantic fallback exceeds measured row cap")
    query_vectors = await provider.embed([query])
    if len(query_vectors) != 1:
        raise ValueError("Embedding provider returned no query vector")
    query_vector = _validated_vector(query_vectors[0], provider.dimension)
    rows = (await db.execute(select(StructuredMemory, StructuredMemoryEmbedding).join(
        StructuredMemoryEmbedding, and_(
            StructuredMemoryEmbedding.user_id == StructuredMemory.user_id,
            StructuredMemoryEmbedding.memory_id == StructuredMemory.id,
            StructuredMemoryEmbedding.model_id == provider.model_id,
        )
    ).where(*_current_scope(user_id)).limit(max_scanned + 1))).all()
    if len(rows) != count or any(
        vector.content_sha256 != hashlib.sha256(row.canonical_text.encode("utf-8")).hexdigest()
        for row, vector in rows
    ):
        raise SemanticIndexIncomplete("Structured memory embedding index is incomplete")
    scored = []
    for row, vector in rows:
        embedding = _validated_vector(vector.vector, provider.dimension)
        scored.append(StructuredHit(
            memory_id=row.id, user_id=user_id, kind=row.kind, payload=row.payload,
            canonical_text=row.canonical_text, source_refs=tuple(row.source_refs),
            updated_at=row.updated_at, score=_cosine(query_vector, embedding),
            retrieval_method="structured_semantic",
        ))
    return sorted(scored, key=lambda item: (-item.score, item.memory_id))[:top_k]
