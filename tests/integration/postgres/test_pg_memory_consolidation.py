"""Transactional consolidation cursor and provenance, with model output mocked.

The PostgreSQL transitions are real; these cases are not LLM-quality evidence.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import select, text

from app.agent import memory_consolidation
from app.agent.memory_consolidation import ExtractionCandidate, consolidate_pending
from app.agent.structured_memory import ProjectFact, current_memories, forget_memory
from app.agent.transcript import sync_checkpoint_messages
from app.db.models import Base, MemoryConsolidationCursor
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


class _FakeManager:
    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = 0

    async def ainvoke(self, input, config=None):
        self.calls += 1
        return [SimpleNamespace(id=f"extracted-{self.calls}", content=self.candidate)]


@pytest.mark.asyncio
async def test_consolidation_corrects_forgets_and_does_not_advance_on_bad_evidence(pg_harness, monkeypatch):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
        await connection.execute(text(
            "ALTER TABLE memory_consolidation_cursors ADD COLUMN IF NOT EXISTS "
            "last_source_occurred_at TIMESTAMPTZ"))
    user, other = await new_user(), await new_user()
    thread = f"r3-consolidate-{uuid4().hex}"
    project = f"Aurora-{uuid4().hex}"
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)

    async def project_message(identifier, value, index):
        async with factory() as db:
            await sync_checkpoint_messages(db, user_id=user.id, thread_id=thread,
                                           messages=[HumanMessage(
                                               id=identifier,
                                               content=f"The {project} report has {value} pages.",
                                               additional_kwargs={"occurred_at_utc": (
                                                   start + timedelta(minutes=index)).isoformat()},
                                           )])
            await db.commit()

    def candidate(identifier, value, quote):
        return ExtractionCandidate(
            payload=ProjectFact(project_key=project, attribute="page_count", value=value),
            source_id=identifier, evidence_quote=quote)

    await project_message("m1", "52", 1)
    manager = _FakeManager(candidate("m1", "52", "52 pages"))
    monkeypatch.setattr(memory_consolidation, "create_memory_manager", lambda *_args, **_kw: manager)
    first = await consolidate_pending(factory, user_id=user.id, thread_id=thread, model=object())
    assert first.processed_source_ids == ("m1",) and len(first.changed_memory_ids) == 1
    assert manager.calls == 1
    assert not (await consolidate_pending(factory, user_id=user.id, thread_id=thread,
                                          model=object())).model_called
    assert manager.calls == 1

    await project_message("m2", "58", 2)
    manager.candidate = candidate("m2", "58", "58 pages")
    second = await consolidate_pending(factory, user_id=user.id, thread_id=thread, model=object())
    assert second.processed_source_ids == ("m2",)
    async with factory() as db:
        current = await current_memories(db, user_id=user.id)
        assert len(current) == 1 and current[0].payload["value"] == "58"
        assert current[0].lineage_version == 2
        assert not await current_memories(db, user_id=other.id)
        assert await forget_memory(db, user_id=user.id, memory_id=current[0].id)
        await db.commit()

    await project_message("m3", "58", 3)
    manager.candidate = candidate("m3", "58", "58 pages")
    suppressed = await consolidate_pending(factory, user_id=user.id, thread_id=thread, model=object())
    assert suppressed.suppressed_candidates == 1 and suppressed.changed_memory_ids == ()
    async with factory() as db:
        assert not await current_memories(db, user_id=user.id)

    await project_message("m4", "60", 4)
    manager.candidate = candidate("m4", "60", "NOT IN SOURCE")
    with pytest.raises(ValueError):
        await consolidate_pending(factory, user_id=user.id, thread_id=thread, model=object())
    async with factory() as db:
        cursor = await db.scalar(select(MemoryConsolidationCursor).where(
            MemoryConsolidationCursor.user_id == user.id,
            MemoryConsolidationCursor.thread_id == thread))
        assert cursor.version == 3
        assert cursor.last_source_message_id == "m3"
        assert cursor.last_failure_category == "MemoryEvidenceError"
