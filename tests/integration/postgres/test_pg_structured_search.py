"""Real PostgreSQL derived-index lifecycle for typed memory (synthetic only)."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.agent.memory_embedding import EmbeddingEgressPolicy, SemanticIndexIncomplete
from app.agent.structured_memory import EvidenceRef, ProjectFact, forget_memory, upsert_memory
from app.agent.structured_search import index_structured_memories, search_structured_memories
from app.db.models import Base, StructuredMemoryEmbedding
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


class SyntheticProvider:
    model_id = "synthetic-structured-2d"
    provider_name = "synthetic-fixture"
    dimension = 2

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "Aurora" in text or "极光" in text
                else [0.0, 1.0] for text in texts]


def _evidence(source, quote):
    return [EvidenceRef(source_type="checkpoint", source_id=source,
                        occurred_at=datetime.now(timezone.utc), evidence_quote=quote)], {
        ("checkpoint", source): f"Synthetic user said {quote}."
    }


@pytest.mark.asyncio
async def test_structured_search_correction_forget_and_user_isolation(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    owner, stranger = await new_user(), await new_user()
    provider = SyntheticProvider()
    egress = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
    key = f"Aurora-{uuid4().hex}"
    first_fact = ProjectFact(project_key=key, attribute="page_count", value="47")
    async with factory() as db:
        refs, catalog = _evidence("source-old", "47")
        first = await upsert_memory(db, user_id=owner.id, payload=first_fact,
                                    refs=refs, source_catalog=catalog)
        first_id = first.id
        await db.commit()

    async with factory() as db:
        with pytest.raises(SemanticIndexIncomplete):
            await search_structured_memories(
                db, user_id=owner.id, query="Aurora pages", provider=provider, egress=egress)
        assert await index_structured_memories(db, user_id=owner.id,
                                               provider=provider, egress=egress) == 1
        assert await index_structured_memories(db, user_id=owner.id,
                                               provider=provider, egress=egress) == 0
        await db.commit()

    async with factory() as restarted:
        calls_before_cap = provider.calls
        with pytest.raises(RuntimeError, match="fallback exceeds measured row cap"):
            await search_structured_memories(
                restarted, user_id=owner.id, query="Aurora pages",
                provider=provider, egress=egress, max_scanned=0)
        assert provider.calls == calls_before_cap
        hits = await search_structured_memories(
            restarted, user_id=owner.id, query="Aurora pages",
            provider=provider, egress=egress)
        assert [hit.memory_id for hit in hits] == [first_id]
        assert hits[0].source_refs[0]["source_id"] == "source-old"
        assert await search_structured_memories(
            restarted, user_id=stranger.id, query="Aurora pages",
            provider=provider, egress=egress) == []
        refs, catalog = _evidence("source-new", "52")
        corrected = await upsert_memory(
            restarted, user_id=owner.id,
            payload=first_fact.model_copy(update={"value": "52"}),
            refs=refs, source_catalog=catalog)
        corrected_id = corrected.id
        await restarted.commit()

    async with factory() as db:
        assert (await db.scalars(select(StructuredMemoryEmbedding).where(
            StructuredMemoryEmbedding.user_id == owner.id))).all() == []
        with pytest.raises(SemanticIndexIncomplete):
            await search_structured_memories(
                db, user_id=owner.id, query="Aurora pages",
                provider=provider, egress=egress)
        assert await index_structured_memories(db, user_id=owner.id,
                                               provider=provider, egress=egress) == 1
        hits = await search_structured_memories(
            db, user_id=owner.id, query="Aurora pages", provider=provider, egress=egress)
        assert [hit.memory_id for hit in hits] == [corrected_id]
        assert hits[0].payload["value"] == "52"
        assert await forget_memory(db, user_id=owner.id, memory_id=corrected_id)
        await db.commit()

    async with factory() as db:
        assert await search_structured_memories(
            db, user_id=owner.id, query="Aurora pages", provider=provider, egress=egress) == []
        assert (await db.scalars(select(StructuredMemoryEmbedding).where(
            StructuredMemoryEmbedding.user_id == owner.id))).all() == []
