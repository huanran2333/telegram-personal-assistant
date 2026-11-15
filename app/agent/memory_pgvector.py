"""PostgreSQL/pgvector exact halfvec candidate index for R3 Memory.

This is a derived projection of the existing source-backed embedding rows.
It uses pgvector's exact distance operator after mandatory user/thread/model
scoping. The isolated ANN experiment lost recall under tenant filtering, so
an approximate HNSW query is deliberately not used here.
"""

from __future__ import annotations

import json
from hashlib import sha256

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.memory_embedding import (
    EmbeddingEgressPolicy, EmbeddingProvider, SemanticIndexIncomplete,
    _validated_vector,
)
from app.agent.structured_search import StructuredHit
from app.agent.transcript import HistoryHit


class PgvectorUnavailable(RuntimeError):
    """The configured PostgreSQL database lacks the required vector extension."""


async def ensure_pgvector_projection(db: AsyncSession) -> None:
    """Create only rebuildable vector sidecars; never mutate source tables.

    The extension must be provisioned by the database operator. A missing
    extension is an explicit capability failure, not a fallback to JSON scan.
    """
    installed = await db.scalar(text(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
    if not installed:
        raise PgvectorUnavailable("pgvector extension is not installed")
    await db.execute(text("""
        CREATE TABLE IF NOT EXISTS transcript_embedding_halfvec (
            user_id varchar(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            thread_id varchar(160) NOT NULL,
            message_id varchar(160) NOT NULL,
            model_id varchar(200) NOT NULL,
            dimension integer NOT NULL,
            content_sha256 varchar(64) NOT NULL,
            embedding halfvec NOT NULL,
            PRIMARY KEY (user_id, thread_id, message_id, model_id)
        )
    """))
    await db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_transcript_halfvec_scope
        ON transcript_embedding_halfvec (user_id, thread_id, model_id)
    """))
    await db.execute(text("""
        CREATE TABLE IF NOT EXISTS structured_embedding_halfvec (
            user_id varchar(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            memory_id varchar(36) NOT NULL,
            model_id varchar(200) NOT NULL,
            dimension integer NOT NULL,
            content_sha256 varchar(64) NOT NULL,
            source_md5 varchar(32) NOT NULL,
            embedding halfvec NOT NULL,
            PRIMARY KEY (user_id, memory_id, model_id)
        )
    """))
    await db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_structured_halfvec_scope
        ON structured_embedding_halfvec (user_id, model_id)
    """))


async def pgvector_projection_complete(
    db: AsyncSession, *, model_id: str, dimension: int,
) -> bool:
    """Check startup freshness without reading private text or vector payloads."""
    if not model_id or dimension < 1:
        raise ValueError("Model and dimension are required")
    params = {"model_id": model_id, "dimension": dimension}
    transcript_gap = await db.scalar(text("""
        SELECT EXISTS (
            SELECT 1 FROM transcript_projection p
            WHERE p.deleted_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM transcript_suppressions s
                  WHERE s.user_id=p.user_id AND s.thread_id=p.thread_id
                    AND s.message_id=p.message_id)
              AND NOT EXISTS (SELECT 1 FROM transcript_projection newer
                  WHERE newer.user_id=p.user_id
                    AND COALESCE(newer.supersedes_thread_id,newer.thread_id)=p.thread_id
                    AND newer.supersedes_message_id=p.message_id)
              AND NOT EXISTS (SELECT 1 FROM transcript_suppressions newer_s
                  WHERE newer_s.user_id=p.user_id
                    AND COALESCE(newer_s.supersedes_thread_id,newer_s.thread_id)=p.thread_id
                    AND newer_s.supersedes_message_id=p.message_id)
              AND NOT EXISTS (
                  SELECT 1 FROM transcript_embedding_halfvec h
                  WHERE h.user_id=p.user_id AND h.thread_id=p.thread_id
                    AND h.message_id=p.message_id
                    AND h.model_id=:model_id AND h.dimension=:dimension
                    AND h.content_sha256=p.content_sha256)
        )
    """), params)
    structured_gap = await db.scalar(text("""
        SELECT EXISTS (
            SELECT 1 FROM structured_memories m
            WHERE m.status='current'
              AND (m.expires_at IS NULL OR m.expires_at > now())
              AND NOT EXISTS (
                  SELECT 1 FROM structured_embedding_halfvec h
                  WHERE h.user_id=m.user_id AND h.memory_id=m.id
                    AND h.model_id=:model_id AND h.dimension=:dimension
                    AND h.source_md5=md5(m.canonical_text))
        )
    """), params)
    return not (transcript_gap or structured_gap)


async def pending_pgvector_scopes(
    db: AsyncSession, *, model_id: str, dimension: int, limit: int,
) -> list[tuple[str, str]]:
    """Find rebuildable transcript index gaps after a backend migration.

    The JSON source embedding is already present, so the ordinary missing-JSON
    maintenance query would never select these threads. No message text leaves
    PostgreSQL while discovering the gap.
    """
    if not model_id or dimension < 1 or not 1 <= limit <= 100:
        raise ValueError("Bounded pgvector maintenance scope is required")
    rows = (await db.execute(text("""
        SELECT p.user_id, p.thread_id
        FROM transcript_projection p
        JOIN transcript_embeddings e ON e.user_id=p.user_id
             AND e.thread_id=p.thread_id AND e.message_id=p.message_id
             AND e.content_sha256=p.content_sha256
        WHERE p.deleted_at IS NULL AND e.model_id=:model_id
          AND e.dimension=:dimension
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions s
              WHERE s.user_id=p.user_id AND s.thread_id=p.thread_id
                AND s.message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_embedding_halfvec h
              WHERE h.user_id=p.user_id AND h.thread_id=p.thread_id
                AND h.message_id=p.message_id AND h.model_id=:model_id
                AND h.dimension=:dimension AND h.content_sha256=p.content_sha256)
        GROUP BY p.user_id, p.thread_id
        ORDER BY p.user_id, p.thread_id LIMIT :limit
    """), {"model_id": model_id, "dimension": dimension, "limit": limit})).all()
    return [(user_id, thread_id) for user_id, thread_id in rows]


async def pending_pgvector_structured_users(
    db: AsyncSession, *, model_id: str, dimension: int, limit: int,
) -> list[str]:
    """Select users whose current typed-memory sidecar needs rebuilding."""
    if not model_id or dimension < 1 or not 1 <= limit <= 100:
        raise ValueError("Bounded pgvector maintenance scope is required")
    rows = (await db.execute(text("""
        SELECT m.user_id FROM structured_memories m
        WHERE m.status='current'
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND NOT EXISTS (SELECT 1 FROM structured_embedding_halfvec h
              WHERE h.user_id=m.user_id AND h.memory_id=m.id
                AND h.model_id=:model_id AND h.dimension=:dimension
                AND h.source_md5=md5(m.canonical_text))
        GROUP BY m.user_id ORDER BY m.user_id LIMIT :limit
    """), {"model_id": model_id, "dimension": dimension, "limit": limit})).scalars().all()
    return list(rows)


async def sync_transcript_halfvec(
    db: AsyncSession, *, user_id: str, thread_id: str, model_id: str,
    dimension: int,
) -> int:
    """Idempotently rebuild one scoped sidecar from JSON embeddings in SQL."""
    if not user_id or not thread_id or not model_id or dimension < 1:
        raise ValueError("Scoped model and dimension are required")
    params = {"user_id": user_id, "thread_id": thread_id,
              "model_id": model_id, "dimension": dimension}
    await db.execute(text("""
        INSERT INTO transcript_embedding_halfvec
            (user_id, thread_id, message_id, model_id, dimension,
             content_sha256, embedding)
        SELECT e.user_id, e.thread_id, e.message_id, e.model_id,
               e.dimension, e.content_sha256, CAST(e.vector::text AS halfvec)
        FROM transcript_embeddings e
        JOIN transcript_projection p ON p.user_id = e.user_id
             AND p.thread_id = e.thread_id AND p.message_id = e.message_id
             AND p.content_sha256 = e.content_sha256
        WHERE e.user_id = :user_id AND e.thread_id = :thread_id
          AND e.model_id = :model_id AND e.dimension = :dimension
          AND p.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM transcript_suppressions s
              WHERE s.user_id = p.user_id AND s.thread_id = p.thread_id
                AND s.message_id = p.message_id)
        ON CONFLICT (user_id, thread_id, message_id, model_id) DO UPDATE
        SET dimension = EXCLUDED.dimension,
            content_sha256 = EXCLUDED.content_sha256,
            embedding = EXCLUDED.embedding
        WHERE transcript_embedding_halfvec.content_sha256
              IS DISTINCT FROM EXCLUDED.content_sha256
           OR transcript_embedding_halfvec.dimension
              IS DISTINCT FROM EXCLUDED.dimension
    """), params)
    await db.execute(text("""
        DELETE FROM transcript_embedding_halfvec h
        WHERE h.user_id = :user_id AND h.thread_id = :thread_id
          AND h.model_id = :model_id
          AND NOT EXISTS (
              SELECT 1 FROM transcript_projection p
              JOIN transcript_embeddings e ON e.user_id = p.user_id
                   AND e.thread_id = p.thread_id AND e.message_id = p.message_id
                   AND e.content_sha256 = p.content_sha256
              WHERE p.user_id = h.user_id AND p.thread_id = h.thread_id
                AND p.message_id = h.message_id AND p.deleted_at IS NULL
                AND e.model_id = h.model_id AND e.dimension = :dimension
                AND NOT EXISTS (
                    SELECT 1 FROM transcript_suppressions s
                    WHERE s.user_id = p.user_id AND s.thread_id = p.thread_id
                      AND s.message_id = p.message_id))
    """), params)
    count = await db.scalar(text("""
        SELECT count(*) FROM transcript_embedding_halfvec
        WHERE user_id=:user_id AND thread_id=:thread_id AND model_id=:model_id
    """), params)
    return int(count)


async def search_transcript_halfvec(
    db: AsyncSession, *, user_id: str, thread_id: str, query: str,
    provider: EmbeddingProvider, egress: EmbeddingEgressPolicy,
    private_data: bool = True, top_k: int = 24,
) -> list[HistoryHit]:
    """Exact scoped cosine candidate search with source/currentness checks."""
    if not user_id or not thread_id or not query.strip() or not 1 <= top_k <= 100:
        raise ValueError("Bounded semantic search scope is required")
    egress.assert_allowed(private_data=private_data)
    params = {"user_id": user_id, "thread_id": thread_id,
              "model_id": provider.model_id, "dimension": provider.dimension}
    expected = await db.scalar(text("""
        SELECT count(*) FROM transcript_projection p
        WHERE p.user_id=:user_id AND p.thread_id=:thread_id
          AND p.deleted_at IS NULL
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions s
              WHERE s.user_id=p.user_id AND s.thread_id=p.thread_id
                AND s.message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_projection newer
              WHERE newer.user_id=p.user_id
                AND COALESCE(newer.supersedes_thread_id,newer.thread_id)=p.thread_id
                AND newer.supersedes_message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions newer_s
              WHERE newer_s.user_id=p.user_id
                AND COALESCE(newer_s.supersedes_thread_id,newer_s.thread_id)=p.thread_id
                AND newer_s.supersedes_message_id=p.message_id)
    """), params)
    indexed = await db.scalar(text("""
        SELECT count(*) FROM transcript_projection p
        JOIN transcript_embedding_halfvec h ON h.user_id=p.user_id
             AND h.thread_id=p.thread_id AND h.message_id=p.message_id
             AND h.content_sha256=p.content_sha256
        WHERE p.user_id=:user_id AND p.thread_id=:thread_id
          AND h.model_id=:model_id AND h.dimension=:dimension
          AND p.deleted_at IS NULL
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions s
              WHERE s.user_id=p.user_id AND s.thread_id=p.thread_id
                AND s.message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_projection newer
              WHERE newer.user_id=p.user_id
                AND COALESCE(newer.supersedes_thread_id,newer.thread_id)=p.thread_id
                AND newer.supersedes_message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions newer_s
              WHERE newer_s.user_id=p.user_id
                AND COALESCE(newer_s.supersedes_thread_id,newer_s.thread_id)=p.thread_id
                AND newer_s.supersedes_message_id=p.message_id)
    """), params)
    if indexed != expected:
        raise SemanticIndexIncomplete("Scoped pgvector transcript index is incomplete")
    vectors = await provider.embed([query])
    if len(vectors) != 1:
        raise ValueError("Embedding provider returned no query vector")
    params["embedding"] = json.dumps(
        _validated_vector(vectors[0], provider.dimension), separators=(",", ":"))
    params["top_k"] = top_k
    rows = (await db.execute(text("""
        SELECT p.user_id, p.thread_id, p.message_id, p.group_id, p.role,
               p.occurred_at, p.search_text, p.source_ref,
               p.supersedes_message_id, p.supersedes_thread_id,
               h.embedding <=> CAST(:embedding AS halfvec) AS distance
        FROM transcript_projection p
        JOIN transcript_embedding_halfvec h ON h.user_id=p.user_id
             AND h.thread_id=p.thread_id AND h.message_id=p.message_id
             AND h.content_sha256=p.content_sha256
        WHERE p.user_id=:user_id AND p.thread_id=:thread_id
          AND h.model_id=:model_id AND h.dimension=:dimension
          AND p.deleted_at IS NULL
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions s
              WHERE s.user_id=p.user_id AND s.thread_id=p.thread_id
                AND s.message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_projection newer
              WHERE newer.user_id=p.user_id
                AND COALESCE(newer.supersedes_thread_id,newer.thread_id)=p.thread_id
                AND newer.supersedes_message_id=p.message_id)
          AND NOT EXISTS (SELECT 1 FROM transcript_suppressions newer_s
              WHERE newer_s.user_id=p.user_id
                AND COALESCE(newer_s.supersedes_thread_id,newer_s.thread_id)=p.thread_id
                AND newer_s.supersedes_message_id=p.message_id)
        ORDER BY distance, p.message_id LIMIT :top_k
    """), params)).mappings().all()
    return [HistoryHit(
        row["user_id"], row["thread_id"], row["message_id"],
        row["group_id"], row["role"], row["occurred_at"],
        row["search_text"], row["source_ref"],
        row["supersedes_message_id"], False, "pgvector_halfvec_exact",
        1.0 - float(row["distance"]), row["supersedes_thread_id"],
    ) for row in rows]


async def sync_structured_halfvec(
    db: AsyncSession, *, user_id: str, model_id: str, dimension: int,
) -> int:
    """Idempotently rebuild current typed-memory vectors from their source index."""
    if not user_id or not model_id or dimension < 1:
        raise ValueError("Scoped model and dimension are required")
    params = {"user_id": user_id, "model_id": model_id,
              "dimension": dimension}
    # The canonical row may have changed before its JSON embedding was
    # regenerated. Never re-label an old vector as fresh merely by copying
    # the canonical row's new md5 into the halfvec sidecar.
    source_hashes = (await db.execute(text("""
        SELECT m.canonical_text, e.content_sha256
        FROM structured_memories m
        JOIN structured_memory_embeddings e ON e.memory_id=m.id
             AND e.user_id=m.user_id
        WHERE m.user_id=:user_id AND m.status='current'
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND e.model_id=:model_id AND e.dimension=:dimension
    """), params)).all()
    if any(sha256(content.encode("utf-8")).hexdigest() != digest
           for content, digest in source_hashes):
        raise SemanticIndexIncomplete("Structured source embedding is stale")
    await db.execute(text("""
        INSERT INTO structured_embedding_halfvec
            (user_id, memory_id, model_id, dimension, content_sha256,
             source_md5, embedding)
        SELECT e.user_id, e.memory_id, e.model_id, e.dimension,
               e.content_sha256, md5(m.canonical_text),
               CAST(e.vector::text AS halfvec)
        FROM structured_memory_embeddings e
        JOIN structured_memories m ON m.id=e.memory_id AND m.user_id=e.user_id
        WHERE e.user_id=:user_id AND e.model_id=:model_id
          AND e.dimension=:dimension AND m.status='current'
          AND (m.expires_at IS NULL OR m.expires_at > now())
        ON CONFLICT (user_id, memory_id, model_id) DO UPDATE
        SET dimension=EXCLUDED.dimension,
            content_sha256=EXCLUDED.content_sha256,
            source_md5=EXCLUDED.source_md5,
            embedding=EXCLUDED.embedding
        WHERE structured_embedding_halfvec.content_sha256
              IS DISTINCT FROM EXCLUDED.content_sha256
           OR structured_embedding_halfvec.source_md5
              IS DISTINCT FROM EXCLUDED.source_md5
           OR structured_embedding_halfvec.dimension
              IS DISTINCT FROM EXCLUDED.dimension
    """), params)
    await db.execute(text("""
        DELETE FROM structured_embedding_halfvec h
        WHERE h.user_id=:user_id AND h.model_id=:model_id
          AND NOT EXISTS (
              SELECT 1 FROM structured_memories m
              JOIN structured_memory_embeddings e ON e.memory_id=m.id
                   AND e.user_id=m.user_id
              WHERE m.id=h.memory_id AND m.user_id=h.user_id
                AND m.status='current'
                AND (m.expires_at IS NULL OR m.expires_at > now())
                AND e.model_id=h.model_id AND e.dimension=:dimension
                AND e.content_sha256=h.content_sha256
                AND md5(m.canonical_text)=h.source_md5)
    """), params)
    count = await db.scalar(text("""
        SELECT count(*) FROM structured_embedding_halfvec
        WHERE user_id=:user_id AND model_id=:model_id
    """), params)
    return int(count)


async def search_structured_halfvec(
    db: AsyncSession, *, user_id: str, query: str,
    provider: EmbeddingProvider, egress: EmbeddingEgressPolicy,
    private_data: bool = True, top_k: int = 6,
) -> list[StructuredHit]:
    """Exact user-scoped typed-memory candidates with stale-source denial."""
    if not user_id or not query.strip() or not 1 <= top_k <= 20:
        raise ValueError("Bounded structured search scope is required")
    egress.assert_allowed(private_data=private_data)
    params = {"user_id": user_id, "model_id": provider.model_id,
              "dimension": provider.dimension}
    expected = await db.scalar(text("""
        SELECT count(*) FROM structured_memories m
        WHERE m.user_id=:user_id AND m.status='current'
          AND (m.expires_at IS NULL OR m.expires_at > now())
    """), params)
    indexed = await db.scalar(text("""
        SELECT count(*) FROM structured_memories m
        JOIN structured_embedding_halfvec h ON h.user_id=m.user_id
             AND h.memory_id=m.id AND h.source_md5=md5(m.canonical_text)
        WHERE m.user_id=:user_id AND m.status='current'
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND h.model_id=:model_id AND h.dimension=:dimension
    """), params)
    if indexed != expected:
        raise SemanticIndexIncomplete("Scoped pgvector structured index is incomplete")
    vectors = await provider.embed([query])
    if len(vectors) != 1:
        raise ValueError("Embedding provider returned no query vector")
    params["embedding"] = json.dumps(
        _validated_vector(vectors[0], provider.dimension), separators=(",", ":"))
    params["top_k"] = top_k
    rows = (await db.execute(text("""
        SELECT m.id, m.user_id, m.kind, m.payload, m.canonical_text,
               m.source_refs, m.updated_at,
               h.embedding <=> CAST(:embedding AS halfvec) AS distance
        FROM structured_memories m
        JOIN structured_embedding_halfvec h ON h.user_id=m.user_id
             AND h.memory_id=m.id AND h.source_md5=md5(m.canonical_text)
        WHERE m.user_id=:user_id AND m.status='current'
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND h.model_id=:model_id AND h.dimension=:dimension
        ORDER BY distance, m.id LIMIT :top_k
    """), params)).mappings().all()
    return [StructuredHit(
        memory_id=row["id"], user_id=row["user_id"], kind=row["kind"],
        payload=row["payload"], canonical_text=row["canonical_text"],
        source_refs=tuple(row["source_refs"]), updated_at=row["updated_at"],
        score=1.0 - float(row["distance"]),
        retrieval_method="pgvector_halfvec_exact",
    ) for row in rows]
