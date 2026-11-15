"""Real PostgreSQL tests for correction, reinforcement and forget lifecycle."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.agent.structured_memory import (
    EvidenceRef, MemorySuppressedError, ProjectFact, current_memories,
    forget_memory, remove_source, upsert_memory,
)
from app.db.models import Base, MemorySuppression, StructuredMemory
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


def _source(identifier: str, quote: str):
    ref = EvidenceRef(source_type="checkpoint", source_id=identifier,
                      occurred_at=datetime.now(timezone.utc), evidence_quote=quote)
    return [ref], {("checkpoint", identifier): f"The user said {quote} in a synthetic test."}


@pytest.mark.asyncio
async def test_create_reinforce_correct_forget_restart_isolated(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    owner, stranger = await new_user(), await new_user()
    old = ProjectFact(project_key=f"Aurora-{uuid4().hex}", attribute="page_count", value="47")
    new = old.model_copy(update={"value": "52"})

    async with factory() as db:
        refs, catalog = _source("m-old", "47")
        first = await upsert_memory(db, user_id=owner.id, payload=old,
                                    refs=refs, source_catalog=catalog)
        await db.commit()
        first_id = first.id
        assert first.lineage_version == 1

    async with factory() as db:
        refs, catalog = _source("m-reinforce", "47")
        reinforced = await upsert_memory(db, user_id=owner.id, payload=old,
                                          refs=refs, source_catalog=catalog)
        await db.commit()
        assert reinforced.id == first_id
        assert len(reinforced.source_refs) == 2
        assert reinforced.row_version == 2

    async with factory() as db:
        refs, catalog = _source("m-correct", "52")
        corrected = await upsert_memory(db, user_id=owner.id, payload=new,
                                         refs=refs, source_catalog=catalog)
        await db.commit()
        corrected_id = corrected.id
        assert corrected.supersedes_id == first_id
        assert corrected.lineage_version == 2

    async with factory() as restarted:
        current = await current_memories(restarted, user_id=owner.id)
        assert [item.id for item in current] == [corrected_id]
        assert not await current_memories(restarted, user_id=stranger.id)
        old_row = await restarted.get(StructuredMemory, first_id)
        assert old_row.status == "superseded"
        assert not await forget_memory(restarted, user_id=stranger.id, memory_id=corrected_id)
        assert await forget_memory(restarted, user_id=owner.id, memory_id=corrected_id)
        await restarted.commit()

    async with factory() as db:
        assert not await current_memories(db, user_id=owner.id)
        suppression = await db.scalar(select(MemorySuppression).where(
            MemorySuppression.user_id == owner.id))
        assert suppression is not None and suppression.source_memory_id == corrected_id
        refs, catalog = _source("m-old-replayed", "47")
        with pytest.raises(MemorySuppressedError):
            await upsert_memory(db, user_id=owner.id, payload=old,
                                refs=refs, source_catalog=catalog)
        refs, catalog = _source("m-explicit-new", "52")
        restored = await upsert_memory(db, user_id=owner.id, payload=new,
                                       refs=refs, source_catalog=catalog,
                                       explicit_restore=True)
        await db.commit()
        assert restored.lineage_version == 3


@pytest.mark.asyncio
async def test_source_removal_requires_remaining_provenance_or_tombstones(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    user = await new_user()
    payload = ProjectFact(project_key=f"Beta-{uuid4().hex}", attribute="status", value="active")
    async with factory() as db:
        refs1, catalog1 = _source("m1", "active")
        refs2, catalog2 = _source("m2", "active")
        row = await upsert_memory(db, user_id=user.id, payload=payload,
                                  refs=refs1 + refs2, source_catalog=catalog1 | catalog2)
        await db.commit()
        identifier = row.id
    async with factory() as db:
        assert await remove_source(db, user_id=user.id,
                                   source_type="checkpoint", source_id="m1") == 1
        await db.commit()
        remaining = (await current_memories(db, user_id=user.id))[0]
        assert remaining.id == identifier and len(remaining.source_refs) == 1
        assert remaining.source_refs[0]["source_id"] == "m2"
        assert await remove_source(db, user_id=user.id,
                                   source_type="checkpoint", source_id="m2") == 1
        await db.commit()
        assert not await current_memories(db, user_id=user.id)
