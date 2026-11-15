"""Opt-in synthetic-only check of the configured local Ollama -> pgvector path."""

import os
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.memory_embedding import (
    EmbeddingEgressPolicy, OpenAICompatibleEmbeddingProvider,
    SemanticIndexIncomplete,
)
from app.agent.memory_pgvector import (
    ensure_pgvector_projection, search_transcript_halfvec,
)
from app.agent.memory_runtime import MemoryRuntime
from app.config import Settings
from app.db.models import Base, TranscriptProjection, User


pytestmark = pytest.mark.pgvector_integration


@pytest.mark.asyncio
async def test_local_qwen_embedding_round_trip(isolated_pg18_target):
    if os.environ.get("PPA_RUN_LOCAL_OLLAMA_EMBEDDING_TEST") != "1":
        pytest.skip("Explicit local Ollama opt-in is required")

    # Never read a live user's transcript, even when the selected test DB is isolated.
    settings = Settings(_env_file=None, llm_provider="openai",
                        llm_model="synthetic-test-model",
                        openai_api_key="synthetic-key", memory_enabled=True)
    assert settings.memory_vector_backend == "pgvector_halfvec_exact"
    assert settings.memory_embedding_base_url == "http://127.0.0.1:11434/v1"
    assert settings.memory_embedding_model == "qwen3-embedding:8b"
    assert settings.memory_embedding_dimension == 2560

    schema = f"r3_ollama_{uuid4().hex}"
    admin = create_async_engine(isolated_pg18_target)
    engine = None
    policy = EmbeddingEgressPolicy(settings.memory_embedding_base_url,
                                   local_self_hosted=True)
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=policy.base_url, model_id=settings.memory_embedding_model,
        dimension=settings.memory_embedding_dimension,
        egress=policy, private_data=False)
    try:
        async with admin.begin() as connection:
            assert await connection.scalar(text(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')"))
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(
            isolated_pg18_target,
            connect_args={"server_settings": {"search_path": f"{schema},public"}},
        )
        async with engine.begin() as connection:
            await connection.run_sync(lambda sync: Base.metadata.create_all(
                sync, checkfirst=False))
        factory = async_sessionmaker(engine, expire_on_commit=False)
        source_text = "合成测试：明天下午整理资料。"
        async with factory() as db:
            await ensure_pgvector_projection(db)
            user = User(telegram_id=f"synthetic-ollama-{uuid4().hex}")
            db.add(user)
            await db.flush()
            user_id = user.id
            db.add(TranscriptProjection(
                user_id=user_id, thread_id="synthetic-thread",
                message_id="synthetic-message", group_id="synthetic-message",
                role="user", occurred_at=datetime.now(timezone.utc),
                search_text=source_text, source_ref="synthetic:test",
                content_sha256=sha256(source_text.encode()).hexdigest(),
            ))
            await db.commit()

        runtime = MemoryRuntime(None, provider, policy, factory, None,
                                vector_backend="pgvector_halfvec_exact")
        with pytest.raises(SemanticIndexIncomplete):
            await runtime.preflight_embedding()
        # Exercise the production maintenance selector and transactional
        # JSON/halfvec indexing path with an actual local embedding model.
        assert await runtime.maintenance_tick(max_threads=5, idle_seconds=86400) == {
            "processed_threads": 1, "failed_threads": 0,
            "processed_structured_users": 0,
        }
        await runtime.preflight_embedding()
        async with factory() as db:
            dimension = await db.scalar(text("""
                SELECT vector_dims(embedding) FROM transcript_embedding_halfvec
                WHERE user_id=:user_id AND message_id='synthetic-message'
            """), {"user_id": user_id})
            assert dimension == 2560
            hits = await search_transcript_halfvec(
                db, user_id=user_id, thread_id="synthetic-thread",
                query=source_text, provider=provider, egress=policy,
                private_data=False, top_k=1)
            assert [hit.message_id for hit in hits] == ["synthetic-message"]
            assert hits[0].retrieval_method == "pgvector_halfvec_exact"
    finally:
        await provider.aclose()
        if engine is not None:
            await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()
