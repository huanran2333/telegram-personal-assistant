"""Restart-safe background index selection on the real isolated PostgreSQL."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.memory_embedding import EmbeddingEgressPolicy, SemanticIndexIncomplete
from app.agent.memory_runtime import MemoryRuntime
from app.agent.memory_pgvector import (
    ensure_pgvector_projection, search_structured_halfvec,
    search_transcript_halfvec,
)
from app.agent.transcript import sync_checkpoint_messages
from app.db.models import Base, StructuredMemory, TranscriptEmbedding, User
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


class SyntheticProvider:
    model_id = "synthetic-runtime-2d"
    provider_name = "test"
    dimension = 2

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_memory_maintenance_indexes_new_projection_and_restart_is_idempotent(
    isolated_pg18_target,
):
    # maintenance_tick intentionally scans *all* pending threads. Give it an
    # isolated schema so this test cannot index another test or real user's
    # transcript with its synthetic provider, and remains repeatable.
    url = isolated_pg18_target
    schema = f"r3_maintenance_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(
        url, connect_args={"server_settings": {"search_path": f"{schema},public"}})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _assert_maintenance_in_isolated_schema(factory, engine, schema)
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def _assert_maintenance_in_isolated_schema(factory, engine, schema):
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: create_current_tables(sync, checkfirst=False))
    async with factory() as db:
        user = User(telegram_id=f"itest-{uuid4().hex}")
        db.add(user)
        await db.commit()
    thread = "memory-maintenance-test"
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=user.id, thread_id=thread,
                                       messages=[HumanMessage(
                                           id="runtime-user-message", content="Synthetic project code ZX-41",
                                           additional_kwargs={"occurred_at_utc":
                                                              datetime.now(timezone.utc).isoformat()})])
        await db.commit()

    provider = SyntheticProvider()
    policy = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
    # This first phase intentionally exercises the pre-pgvector JSON index.
    # The production default is halfvec; using it here would resolve the
    # public sidecar before this isolated schema creates its own FK-bound copy.
    runtime = MemoryRuntime(None, provider, policy, factory, None,
                            vector_backend="json_bounded")
    first = await runtime.maintenance_tick(max_threads=5, idle_seconds=86400)
    assert first == {"processed_threads": 1, "failed_threads": 0}
    assert provider.calls == ["Synthetic project code ZX-41"]
    async with factory() as db:
        assert await db.scalar(select(func.count()).select_from(TranscriptEmbedding).where(
            TranscriptEmbedding.user_id == user.id,
            TranscriptEmbedding.thread_id == thread)) == 1

    restarted = MemoryRuntime(None, SyntheticProvider(), policy, factory, None,
                              vector_backend="json_bounded")
    second = await restarted.maintenance_tick(max_threads=5, idle_seconds=86400)
    assert second == {"processed_threads": 0, "failed_threads": 0}

    # Migrating an existing JSON-complete index must still backfill the new
    # halfvec sidecar, even though the ordinary missing-embedding query is empty.
    async with factory() as db:
        await ensure_pgvector_projection(db)
        assert await db.scalar(text("""
            SELECT count(*) = 1 FROM pg_constraint
            WHERE conrelid = to_regclass(:sidecar)
              AND confrelid = to_regclass(:owner) AND contype = 'f'
        """), {"sidecar": f"{schema}.transcript_embedding_halfvec",
               "owner": f"{schema}.users"})
        await db.commit()
    indexed = MemoryRuntime(None, SyntheticProvider(), policy, factory, None,
                            vector_backend="pgvector_halfvec_exact")
    # A source row without its current-model sidecar must make semantic
    # readiness fail closed; the maintenance tick below repairs that gap.
    with pytest.raises(SemanticIndexIncomplete):
        await indexed.preflight_embedding()
    migrated = await indexed.maintenance_tick(max_threads=5, idle_seconds=86400)
    assert migrated == {"processed_threads": 1, "failed_threads": 0,
                        "processed_structured_users": 0}
    async with factory() as db:
        assert await db.scalar(text(
            "SELECT count(*) FROM transcript_embedding_halfvec WHERE user_id=:user_id"),
            {"user_id": user.id}) == 1
        hits = await search_transcript_halfvec(
            db, user_id=user.id, thread_id=thread, query="ZX-41",
            provider=indexed.provider, egress=policy)
        assert [hit.message_id for hit in hits] == ["runtime-user-message"]
    await indexed.preflight_embedding()
    assert await indexed.maintenance_tick(max_threads=5, idle_seconds=86400) == {
        "processed_threads": 0, "failed_threads": 0,
        "processed_structured_users": 0}
    async with factory() as db:
        db.add(StructuredMemory(
            user_id=user.id, memory_key="synthetic-project-code",
            kind="project", payload={"code": "ZX-41"},
            canonical_text="Synthetic project code ZX-41",
            source_refs=[{"source_type": "checkpoint", "source_id": "runtime-user-message"}],
            status="current"))
        await db.commit()
    with pytest.raises(SemanticIndexIncomplete):
        await indexed.preflight_embedding()
    structured_backfill = await indexed.maintenance_tick(max_threads=5, idle_seconds=86400)
    assert structured_backfill == {"processed_threads": 0, "failed_threads": 0,
                                   "processed_structured_users": 1}
    async with factory() as db:
        assert await db.scalar(text(
            "SELECT count(*) FROM structured_embedding_halfvec WHERE user_id=:user_id"),
            {"user_id": user.id}) == 1
        hits = await search_structured_halfvec(
            db, user_id=user.id, query="ZX-41",
            provider=indexed.provider, egress=policy)
        assert len(hits) == 1 and hits[0].user_id == user.id
    await indexed.preflight_embedding()
