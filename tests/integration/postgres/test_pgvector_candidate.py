"""Isolated PG18/pgvector exact-sidecar contract; synthetic text only.

Uses a generated, disposable schema in the active test database; never
modifies real user rows or asserts a production Memory acceptance result.
"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.memory_embedding import EmbeddingEgressPolicy, SemanticIndexIncomplete
from app.agent.memory_context import ContextOrchestrator, ContextRequest
from app.agent.memory_runtime import MemoryRuntime
from app.agent.memory_pgvector import (
    ensure_pgvector_projection, search_structured_halfvec,
    search_transcript_halfvec, sync_structured_halfvec, sync_transcript_halfvec,
    pending_pgvector_scopes, pending_pgvector_structured_users,
    pgvector_projection_complete,
)
from app.db.models import (
    Base, StructuredMemory, StructuredMemoryEmbedding, TranscriptEmbedding,
    TranscriptProjection, TranscriptSuppression, User,
)

pytestmark = pytest.mark.pgvector_integration


class SyntheticProvider:
    model_id = "synthetic-pgvector-2d"
    provider_name = "synthetic-fixture"
    dimension = 2

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "Blue" in item else [0.0, 1.0]
                for item in texts]


class NeverSummarize:
    async def ainvoke(self, *_args, **_kwargs):
        raise AssertionError("Short selected context must not summarize")


@pytest.mark.asyncio
async def test_exact_sidecar_scope_currentness_rebuild_and_restart(isolated_pg18_target):
    # The shared fixture validates major version and exact data directory;
    # no experiment port or alternate cluster selector lives in this test.
    schema = f"r3_pgvector_{uuid4().hex}"
    admin = create_async_engine(isolated_pg18_target)
    async with admin.begin() as connection:
        assert await connection.scalar(text(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')"))
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(
        isolated_pg18_target,
        connect_args={"server_settings": {"search_path": f"{schema},public"}},
    )
    provider = SyntheticProvider()
    egress = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(lambda sync: Base.metadata.create_all(sync, checkfirst=False))
        factory = async_sessionmaker(engine, expire_on_commit=False)
        runtime = MemoryRuntime(None, provider, egress, factory, None,
                                vector_backend="pgvector_halfvec_exact")
        async with factory() as db:
            await ensure_pgvector_projection(db)
            assert await db.scalar(text("SELECT to_regclass(:name)"),
                                   {"name": f"{schema}.users"}) is not None
            assert await db.scalar(text("SELECT to_regclass(:name)"),
                                   {"name": f"{schema}.transcript_embedding_halfvec"}) is not None
            owner = User(telegram_id=f"synthetic-{uuid4().hex}")
            other = User(telegram_id=f"synthetic-{uuid4().hex}")
            db.add_all([owner, other])
            await db.flush()
            now = datetime.now(timezone.utc)
            for user, identifier, vector in (
                (owner, "old", [1.0, 0.0]),
                (owner, "new", [0.0, 1.0]),
                (other, "private", [1.0, 0.0]),
            ):
                db.add(TranscriptProjection(
                    user_id=user.id, thread_id="synthetic-thread",
                    message_id=identifier, group_id=identifier, role="user",
                    occurred_at=now + timedelta(minutes=int(identifier == "new")),
                    search_text=f"{identifier} Blue fact", source_ref=f"checkpoint:{identifier}",
                    content_sha256=f"digest-{identifier}",
                    supersedes_message_id="old" if identifier == "new" else None,
                ))
                db.add(TranscriptEmbedding(
                    user_id=user.id, thread_id="synthetic-thread",
                    message_id=identifier, model_id=provider.model_id,
                    provider=provider.provider_name, dimension=provider.dimension,
                    vector=vector, content_sha256=f"digest-{identifier}",
                ))
            await db.commit()
            owner_id, other_id = owner.id, other.id

        async with factory() as db:
            assert not await pgvector_projection_complete(
                db, model_id=provider.model_id, dimension=provider.dimension)
            with pytest.raises(SemanticIndexIncomplete):
                await runtime.preflight_embedding()
            assert (owner_id, "synthetic-thread") in await pending_pgvector_scopes(
                db, model_id=provider.model_id, dimension=provider.dimension, limit=10)
            with pytest.raises(SemanticIndexIncomplete):
                await search_transcript_halfvec(
                    db, user_id=owner_id, thread_id="synthetic-thread",
                    query="Blue", provider=provider, egress=egress)
            assert provider.calls == 0
            assert await sync_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                model_id=provider.model_id, dimension=provider.dimension) == 2
            assert await sync_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                model_id=provider.model_id, dimension=provider.dimension) == 2
            await db.commit()
            assert (owner_id, "synthetic-thread") not in await pending_pgvector_scopes(
                db, model_id=provider.model_id, dimension=provider.dimension, limit=10)
            hits = await search_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                query="Blue", provider=provider, egress=egress)
            assert [hit.message_id for hit in hits] == ["new"]
            assert hits[0].retrieval_method == "pgvector_halfvec_exact"
            with pytest.raises(SemanticIndexIncomplete):
                await search_transcript_halfvec(
                    db, user_id=other_id, thread_id="synthetic-thread",
                    query="Blue", provider=provider, egress=egress)
            assert await sync_transcript_halfvec(
                db, user_id=other_id, thread_id="synthetic-thread",
                model_id=provider.model_id, dimension=provider.dimension) == 1
            assert [hit.message_id for hit in await search_transcript_halfvec(
                db, user_id=other_id, thread_id="synthetic-thread",
                query="Blue", provider=provider, egress=egress)] == ["private"]
            memory = StructuredMemory(
                user_id=owner_id, memory_key="synthetic-project-budget",
                kind="project", payload={"value": "47"},
                canonical_text="Blue project budget is 47",
                source_refs=[{"source_type": "checkpoint", "source_id": "old"}],
                status="current",
            )
            db.add(memory)
            await db.flush()
            memory_id = memory.id
            db.add(StructuredMemoryEmbedding(
                user_id=owner_id, memory_id=memory_id,
                model_id=provider.model_id, provider=provider.provider_name,
                dimension=provider.dimension, vector=[1.0, 0.0],
                content_sha256=sha256(memory.canonical_text.encode()).hexdigest(),
            ))
            await db.commit()
            assert owner_id in await pending_pgvector_structured_users(
                db, model_id=provider.model_id, dimension=provider.dimension, limit=10)
            with pytest.raises(SemanticIndexIncomplete):
                await search_structured_halfvec(
                    db, user_id=owner_id, query="Blue budget",
                    provider=provider, egress=egress)
            assert await sync_structured_halfvec(
                db, user_id=owner_id, model_id=provider.model_id,
                dimension=provider.dimension) == 1
            assert await sync_structured_halfvec(
                db, user_id=owner_id, model_id=provider.model_id,
                dimension=provider.dimension) == 1
            await db.commit()
            assert owner_id not in await pending_pgvector_structured_users(
                db, model_id=provider.model_id, dimension=provider.dimension, limit=10)
            assert await pgvector_projection_complete(
                db, model_id=provider.model_id, dimension=provider.dimension)
            await runtime.preflight_storage()
            await runtime.preflight_embedding()
            assert [hit.memory_id for hit in await search_structured_halfvec(
                db, user_id=owner_id, query="Blue budget",
                provider=provider, egress=egress)] == [memory_id]
            assert await search_structured_halfvec(
                db, user_id=other_id, query="Blue budget",
                provider=provider, egress=egress) == []
            orchestrator = ContextOrchestrator(
                session_factory=factory, summary_model=NeverSummarize(),
                summary_provider="synthetic", summary_model_name="unused",
                embedding_provider=provider, embedding_egress=egress,
                vector_backend="pgvector_halfvec_exact",
            )
            bundle = await orchestrator.assemble(ContextRequest(
                user_id=owner_id, thread_id="synthetic-thread",
                request="What was the Blue project budget?", route="chat",
                messages=(HumanMessage(id="current", content="What was the Blue project budget?"),),
                token_budget=4096, search_query="Blue project budget",
                search_purpose="exact", allow_summary_update=False,
            ))
            assert "semantic_search_unavailable" not in bundle.degraded
            assert "structured_search_unavailable" not in bundle.degraded
            assert all(item.message_id not in {"old", "private"} for item in bundle.history)
            assert any(item.memory_id == memory_id for item in bundle.structured)

        await engine.dispose()
        engine = create_async_engine(
            isolated_pg18_target,
            connect_args={"server_settings": {"search_path": f"{schema},public"}},
        )
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as db:
            assert [hit.message_id for hit in await search_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                query="Blue", provider=provider, egress=egress)] == ["new"]
            assert [hit.memory_id for hit in await search_structured_halfvec(
                db, user_id=owner_id, query="Blue budget",
                provider=provider, egress=egress)] == [memory_id]
            memory = await db.get(StructuredMemory, memory_id)
            memory.canonical_text = "Blue project budget is now 52"
            await db.flush()
            with pytest.raises(SemanticIndexIncomplete):
                await search_structured_halfvec(
                    db, user_id=owner_id, query="Blue budget",
                    provider=provider, egress=egress)
            with pytest.raises(SemanticIndexIncomplete, match="source embedding is stale"):
                await sync_structured_halfvec(
                    db, user_id=owner_id, model_id=provider.model_id,
                    dimension=provider.dimension)
            memory.status = "tombstoned"
            db.add(TranscriptSuppression(
                user_id=owner_id, thread_id="synthetic-thread", message_id="new"))
            await db.commit()
            await sync_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                model_id=provider.model_id, dimension=provider.dimension)
            assert await sync_structured_halfvec(
                db, user_id=owner_id, model_id=provider.model_id,
                dimension=provider.dimension) == 0
            await db.commit()
            assert await search_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                query="Blue", provider=provider, egress=egress) == []
            # A lost derived sidecar can be rebuilt from intact source rows.
            await db.execute(text("DELETE FROM transcript_embedding_halfvec"))
            await db.execute(text("DELETE FROM structured_embedding_halfvec"))
            await sync_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                model_id=provider.model_id, dimension=provider.dimension)
            await sync_structured_halfvec(
                db, user_id=owner_id, model_id=provider.model_id,
                dimension=provider.dimension)
            await db.commit()
            assert await search_transcript_halfvec(
                db, user_id=owner_id, thread_id="synthetic-thread",
                query="Blue", provider=provider, egress=egress) == []
            assert await search_structured_halfvec(
                db, user_id=owner_id, query="Blue budget",
                provider=provider, egress=egress) == []
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
