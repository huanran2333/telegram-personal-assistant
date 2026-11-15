"""Scoped, rebuildable semantic candidate index on synthetic PostgreSQL data."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.agent.memory_embedding import (
    EmbeddingEgressPolicy, SemanticIndexIncomplete,
    index_transcript_embeddings, semantic_history_candidates,
)
from app.agent.transcript import sync_checkpoint_messages, tombstone_message
from app.db.models import Base, TranscriptEmbedding
from tests.integration.postgres.current_schema import create_current_tables
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.integration


class SyntheticProvider:
    model_id = "synthetic-test-2d"
    provider_name = "synthetic-fixture"
    dimension = 2

    def __init__(self):
        self.inputs = []

    async def embed(self, texts):
        self.inputs.extend(texts)
        return [[1.0, 0.0] if "蓝鲸" in text or "whale" in text.lower()
                else [0.0, 1.0] for text in texts]


@pytest.mark.asyncio
async def test_index_restart_scope_tombstone_and_idempotency(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    owner, stranger = await new_user(), await new_user()
    provider = SyntheticProvider()
    egress = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
    now = datetime.now(timezone.utc).isoformat()
    async with factory() as db:
        for uid in (owner.id, stranger.id):
            await sync_checkpoint_messages(db, user_id=uid, thread_id="thread-a", messages=[
                HumanMessage(id=f"m-{uid}-whale", content="蓝鲸项目代号",
                             additional_kwargs={"occurred_at_utc": now}),
                HumanMessage(id=f"m-{uid}-other", content="普通会议",
                             additional_kwargs={"occurred_at_utc": now}),
            ])
        await db.commit()

    async with factory() as db:
        assert await index_transcript_embeddings(db, user_id=owner.id, thread_id="thread-a",
                                                 provider=provider, egress=egress) == 2
        assert await index_transcript_embeddings(db, user_id=owner.id, thread_id="thread-a",
                                                 provider=provider, egress=egress) == 0
        await db.commit()
    assert len(provider.inputs) == 2

    async with factory() as restarted:
        calls_before_cap = len(provider.inputs)
        with pytest.raises(RuntimeError, match="fallback exceeds measured row cap"):
            await semantic_history_candidates(
                restarted, user_id=owner.id, thread_id="thread-a", query="蓝鲸",
                provider=provider, egress=egress, top_k=1, max_scanned=1)
        assert len(provider.inputs) == calls_before_cap
        hits = await semantic_history_candidates(
            restarted, user_id=owner.id, thread_id="thread-a", query="蓝鲸",
            provider=provider, egress=egress, top_k=2)
        assert hits[0].message_id == f"m-{owner.id}-whale"
        assert all(hit.user_id == owner.id for hit in hits)
        assert not await semantic_history_candidates(
            restarted, user_id=owner.id, thread_id="thread-b", query="蓝鲸",
            provider=provider, egress=egress)
        with pytest.raises(SemanticIndexIncomplete):
            await semantic_history_candidates(
                restarted, user_id=stranger.id, thread_id="thread-a", query="蓝鲸",
                provider=provider, egress=egress)
        await tombstone_message(restarted, user_id=owner.id, thread_id="thread-a",
                                message_id=f"m-{owner.id}-whale")
        await restarted.commit()

    async with factory() as db:
        hits = await semantic_history_candidates(
            db, user_id=owner.id, thread_id="thread-a", query="蓝鲸",
            provider=provider, egress=egress)
        assert all(hit.message_id != f"m-{owner.id}-whale" for hit in hits)
        rows = (await db.scalars(select(TranscriptEmbedding).where(
            TranscriptEmbedding.user_id == owner.id))).all()
        assert len(rows) == 1  # privacy tombstone purges the derived vector
