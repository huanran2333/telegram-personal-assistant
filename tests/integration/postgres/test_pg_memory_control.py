"""Explicit corrections and forgets keep lineage, source and search in sync."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.agent.memory_control import apply_memory_control
from app.agent.memory_embedding import EmbeddingEgressPolicy, index_transcript_embeddings
from app.agent.structured_memory import EvidenceRef, ProjectFact, current_memories, upsert_memory
from app.agent.transcript import rebuild_from_saver, search_history, sync_checkpoint_messages
from app.db.models import Base, MemorySuppression, StructuredMemory, TranscriptProjection
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


class SyntheticProvider:
    model_id = "synthetic-control-2d"
    provider_name = "synthetic"
    dimension = 2

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class ChoiceModel:
    def __init__(self, choice):
        self.choice = choice

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        return self.choice


@pytest.mark.asyncio
async def test_correction_then_forget_tombstones_derived_evidence(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    user = await new_user()
    thread = "r3-memory-control"
    old = HumanMessage(id="r3-old-day", content="Atlas meeting day is Friday.")
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=user.id, thread_id=thread,
                                       messages=[old, AIMessage(id="r3-old-answer", content="Noted.")])
        row = await upsert_memory(
            db, user_id=user.id,
            payload=ProjectFact(project_key="Atlas", attribute="meeting day", value="Friday"),
            refs=[EvidenceRef(source_type="checkpoint", source_id=old.id,
                              thread_id=thread, occurred_at=datetime.now(timezone.utc),
                              evidence_quote="Friday")],
            source_catalog={("checkpoint", old.id): old.content})
        await db.commit()
        old_memory_id = row.id

    policy = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
    provider = SyntheticProvider()
    correction = ChoiceModel({
        "target_type": "structured", "target_id": old_memory_id,
        "target_quote": "Friday", "confidence": 0.99,
        "replacement_payload": ProjectFact(
            project_key="Atlas", attribute="meeting day", value="Monday").model_dump(),
        "replacement_quote": "周一",
    })
    result = await apply_memory_control(
        session_factory=factory, provider=provider, egress=policy, model=correction,
        user_id=user.id, thread_id=thread, operation="correct",
        target_query="Atlas meeting day", request="不是周五，是周一。",
        current_message_id="r3-new-day", history=[old])
    assert result.changed and result.supersedes_message_id == old.id
    async with factory() as db:
        rows = await current_memories(db, user_id=user.id)
        assert len(rows) == 1 and rows[0].payload["value"] == "Monday"
        assert rows[0].supersedes_id == old_memory_id
        new_memory_id = rows[0].id
        await sync_checkpoint_messages(db, user_id=user.id, thread_id=thread,
                                       messages=[HumanMessage(
                                           id="r3-new-day", content="不是周五，是周一。",
                                           additional_kwargs={"supersedes_message_id": old.id}),
                                                 AIMessage(id="r3-new-answer", content="已更正。")])
        await db.commit()
    async with factory() as db:
        old_hits = await search_history(db, user_id=user.id, thread_id=thread, query="Friday")
        assert all(hit.message_id != old.id for hit in old_hits)

    forgetting = ChoiceModel({
        "target_type": "structured", "target_id": new_memory_id,
        "target_quote": "Monday", "confidence": 0.99,
    })
    forgot = await apply_memory_control(
        session_factory=factory, provider=provider, egress=policy, model=forgetting,
        user_id=user.id, thread_id=thread, operation="forget",
        target_query="Atlas meeting day", request="别记 Atlas 的会议日期了",
        current_message_id="r3-forget-request", history=[old])
    assert forgot.changed
    async with factory() as db:
        assert await current_memories(db, user_id=user.id) == []
        current = await db.scalar(select(StructuredMemory).where(
            StructuredMemory.id == new_memory_id))
        assert current.status == "tombstoned"
        suppression = await db.scalar(select(MemorySuppression).where(
            MemorySuppression.user_id == user.id))
        assert suppression is not None
        projected = await db.scalar(select(TranscriptProjection).where(
            TranscriptProjection.user_id == user.id,
            TranscriptProjection.thread_id == thread,
            TranscriptProjection.message_id == "r3-new-day"))
        assert projected.deleted_at is not None

    class RestoredSaverSnapshot:
        async def aget_state(self, _config):
            return SimpleNamespace(values={
                "user_id": user.id,
                "messages": [old, AIMessage(id="r3-old-answer", content="Noted."),
                             HumanMessage(id="r3-new-day", content="不是周五，是周一。",
                                          additional_kwargs={"supersedes_message_id": old.id}),
                             AIMessage(id="r3-new-answer", content="已更正。")],
            })

    async with factory() as db:
        await rebuild_from_saver(db, graph=RestoredSaverSnapshot(),
                                 user_id=user.id, thread_id=thread)
        await index_transcript_embeddings(db, user_id=user.id, thread_id=thread,
                                          provider=provider, egress=policy)
        await db.commit()
        assert not await search_history(db, user_id=user.id, thread_id=thread,
                                        query="Friday")
        assert not await search_history(db, user_id=user.id, thread_id=thread,
                                        query="周一")
        assert await current_memories(db, user_id=user.id) == []
