"""R6 provenance on the isolated PG18 R3 substrate; no external provider writes."""
from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.meeting_prep_full import collect_meeting_evidence, revalidate_todos
from app.agent.transcript import search_history, sync_checkpoint_messages, tombstone_message
from app.db import crud
from app.db.models import StructuredMemory, TranscriptProjection

pytestmark = pytest.mark.integration


class ReadOnlyCalendar:
    async def resolve_read_scope(self, calendar_id):
        return "test-account", "test-calendar"

    async def get_event(self, calendar_id, event_id):
        from app.agent.calendar_conditional import CalendarEventSnapshot
        return CalendarEventSnapshot.model_validate({
            "id": event_id, "etag": '"e1"', "status": "confirmed", "summary": "Project Sync",
            "start": {"dateTime": "2026-10-08T16:00:00+08:00"},
            "end": {"dateTime": "2026-10-08T17:00:00+08:00"},
            "attendees": [{"email": "alice@example.test"}],
        })


@pytest.mark.asyncio
async def test_r3_ids_currentness_restart_and_isolation(pg_harness, isolated_pg18_target):
    factory, new_user, _ = pg_harness
    owner, other = await new_user(), await new_user()
    thread = "r6-source-thread"
    source_id = "r6-source-message"
    async with factory() as db:
        await crud.create_todo(db, owner.id, "Project Sync benchmark")
        await crud.create_todo(db, other.id, "Project Sync other user's secret")
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread, messages=[
            HumanMessage(id=source_id, content="Project Sync decided to use plan B",
                         additional_kwargs={"occurred_at_utc": datetime.now(timezone.utc).isoformat()})])
        memory = StructuredMemory(
            user_id=owner.id, memory_key="r6-project-decision", kind="decision",
            payload={"decision": "plan B"}, canonical_text="Project Sync uses plan B",
            source_refs=[{"source_type": "checkpoint", "source_id": source_id}], status="current")
        db.add(memory)
        await db.commit()
        memory_id = memory.id
    async with factory() as db:
        hits = await search_history(db, user_id=owner.id, thread_id=thread,
                                    query="Project Sync", top_k=3)
        unrelated = await search_history(db, user_id=other.id, thread_id=thread,
                                         query="Project Sync", top_k=3)
    assert len(hits) == 1 and hits[0].message_id == source_id and not unrelated
    items = [
        {"user_id": owner.id, "source_type": "transcript", "source_id": hits[0].message_id,
         "thread_id": hits[0].thread_id, "occurred_at": hits[0].occurred_at.isoformat(),
         "currentness": "historical", "retrieval_method": "historical",
         "source_ref": hits[0].source_ref, "text": hits[0].content},
        {"user_id": owner.id, "source_type": "structured_memory", "source_id": memory_id,
         "kind": "decision", "currentness": "current", "text": "Project Sync uses plan B",
         "provenance_refs": [{"source_type": "checkpoint", "source_id": source_id}]},
        {"user_id": other.id, "source_type": "structured_memory", "source_id": "other-secret",
         "kind": "decision", "currentness": "current", "text": "Secret"},
    ]
    collector_args = dict(
        user_id=owner.id, telegram_id="r6-owner", timezone_name="Asia/Shanghai",
        candidate={"id": "event-1"}, calendar=ReadOnlyCalendar(), gmail=None,
        gmail_owner_telegram_id=None, memory_items=items, memory_degraded=[],
    )
    first = await collect_meeting_evidence(session_factory=factory, **collector_args)
    ids = {item.source_id for item in first.evidence}
    assert source_id in ids and memory_id in ids and "other-secret" not in ids
    assert first.dropped_counts["memory"] == 1
    assert all(item.owner_scope == f"user:{owner.id}" for item in first.evidence
               if item.source_type in {"todo", "transcript", "structured_memory"})
    assert await revalidate_todos(factory, first)

    # A new engine/runtime reads the same R3 rows and reconstructs R6 evidence.
    restarted_engine = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
    restarted_factory = async_sessionmaker(restarted_engine, expire_on_commit=False)
    try:
        restored = await collect_meeting_evidence(session_factory=restarted_factory, **collector_args)
        assert {item.source_id for item in restored.evidence} == ids
        assert await revalidate_todos(restarted_factory, restored)
        async with restarted_factory() as db:
            assert await db.scalar(select(func.count()).select_from(TranscriptProjection).where(
                TranscriptProjection.user_id == owner.id,
                TranscriptProjection.message_id == source_id)) == 1
            await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread, messages=[
                HumanMessage(id="r6-correction", content="Project Sync decided to use plan C",
                             additional_kwargs={"occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                                                "supersedes_message_id": source_id})])
            await db.commit()
        assert not await revalidate_todos(restarted_factory, restored)
        async with restarted_factory() as db:
            await db.execute(delete(TranscriptProjection).where(
                TranscriptProjection.user_id == owner.id,
                TranscriptProjection.message_id == "r6-correction"))
            await tombstone_message(db, user_id=owner.id, thread_id=thread, message_id=source_id)
            stored = await db.get(StructuredMemory, memory_id)
            stored.status = "tombstoned"
            await db.commit()
        assert not await revalidate_todos(restarted_factory, restored)
    finally:
        await restarted_engine.dispose()
