"""Real LLM + Qwen + isolated PostgreSQL Todo alias resolution, synthetic user.

This is opt-in and mutates only fixture-owned rows. It is not Telegram E2E.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.agent import graph as graph_module
from app.agent.memory_embedding import index_transcript_embeddings
from app.agent.memory_runtime import create_memory_runtime
from app.agent.memory_pgvector import (
    ensure_pgvector_projection, sync_structured_halfvec,
    sync_transcript_halfvec,
)
from app.agent.structured_memory import Alias, EvidenceRef, upsert_memory
from app.agent.structured_search import index_structured_memories
from app.agent.store import generate_store
from app.agent.transcript import sync_checkpoint_messages
from app.config import get_settings
from app.db import crud
from app.db.models import Todo

pytestmark = [pytest.mark.llm_integration]


@pytest.mark.asyncio
async def test_old_alias_reaches_current_todo_row_not_memory_authority(pg_harness, monkeypatch):
    settings = get_settings()
    if not (settings.memory_embedding_local_self_hosted and
            settings.memory_embedding_no_unknown_proxy and
            settings.memory_embedding_body_logging_disabled):
        pytest.skip("Self-hosted embedding egress attestations are not configured")
    factory, new_user, _engine = pg_harness
    user = await new_user()
    telegram_id = user.telegram_id
    thread = f"user_{telegram_id}"
    current_title = f"准备十月底 Agent 实习申请 {uuid4().hex[:6]}"
    alias = "十月底找 Agent 实习那个任务"
    old_source = HumanMessage(
        id=f"old-alias-{uuid4().hex}",
        content=f"我把待办「{current_title}」简称为「{alias}」，当时原定 2026-10-20 截止。")
    turns = [old_source, AIMessage(id=f"ack-{uuid4().hex}", content="记下了。")]
    for index in range(12):
        turns.extend([HumanMessage(id=f"noise-u-{uuid4().hex}",
                                   content=f"无关的日常讨论 {index}。"),
                      AIMessage(id=f"noise-a-{uuid4().hex}", content="好的。")])
    async with factory() as db:
        # Conversation history is useful for the alias, but deliberately has
        # a stale deadline. Only the current Todo row may determine the edit.
        todo = await crud.create_todo(db, user.id, current_title,
                                     due_date=datetime(2026, 10, 30, 23, 59, tzinfo=timezone.utc))
        todo_id = todo.id
        await sync_checkpoint_messages(db, user_id=user.id, thread_id=thread,
                                       messages=turns)
        await upsert_memory(
            db, user_id=user.id,
            payload=Alias(entity_kind="todo", entity_key=current_title,
                          label="旧称", alias=alias),
            refs=[EvidenceRef(source_type="checkpoint", source_id=old_source.id,
                              thread_id=thread, occurred_at=datetime.now(timezone.utc),
                              evidence_quote=alias)],
            source_catalog={("checkpoint", old_source.id): old_source.content})
        await db.commit()
    runtime = create_memory_runtime(session_factory=factory, settings=settings)
    async def no_calendar_discovery(_settings):
        return []
    monkeypatch.setattr(graph_module, "discover_calendar_tools", no_calendar_discovery)
    try:
        async with factory() as db:
            if settings.memory_vector_backend == "pgvector_halfvec_exact":
                await ensure_pgvector_projection(db)
            await index_transcript_embeddings(
                db, user_id=user.id, thread_id=thread,
                provider=runtime.provider, egress=runtime.egress)
            await index_structured_memories(
                db, user_id=user.id, provider=runtime.provider,
                egress=runtime.egress)
            if settings.memory_vector_backend == "pgvector_halfvec_exact":
                await sync_transcript_halfvec(
                    db, user_id=user.id, thread_id=thread,
                    model_id=runtime.provider.model_id,
                    dimension=runtime.provider.dimension)
                await sync_structured_halfvec(
                    db, user_id=user.id, model_id=runtime.provider.model_id,
                    dimension=runtime.provider.dimension)
            await db.commit()
        async with generate_store() as store:
            graph = await graph_module.build_graph(factory, store=store,
                                                   memory_runtime=runtime)
            config = {"configurable": {"thread_id": thread}}
            try:
                await graph.aupdate_state(config, {
                    "user_id": user.id, "telegram_id": telegram_id,
                    "chat_id": "synthetic", "messages": turns,
                }, as_node="chat_response_node")
                reply = await graph_module.run_agent(
                    graph, user.id, telegram_id, "synthetic",
                    "把之前说的十月底找 Agent 实习那个任务提前三天。",
                    projection_session_factory=factory)
                async with factory() as db:
                    actual = await db.scalar(select(Todo).where(Todo.id == todo_id,
                                                                Todo.user_id == user.id))
                    all_rows = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
                assert len(all_rows) == 1, reply
                assert actual.due_date.date().isoformat() == "2026-10-27", reply
            finally:
                await graph.checkpointer.adelete_thread(thread)
                await graph.checkpointer.conn.close()
    finally:
        await runtime.aclose()
