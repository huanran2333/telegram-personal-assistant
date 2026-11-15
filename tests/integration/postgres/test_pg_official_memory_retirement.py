"""Retired conversational memory edits must not weaken source-backed reads."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy import select

from app.agent.memory_runtime import create_memory_runtime
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import official_thread_id, run_agent
from app.agent.transcript import search_history, sync_checkpoint_messages, tombstone_message
from app.config import get_settings
from app.db.models import TranscriptProjection, TranscriptSuppression
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_retired_control_keeps_existing_suppression_and_search(pg_harness):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    source_thread = f"web:memory-source:{uuid4().hex}"
    query_thread = f"web:memory-query:{uuid4().hex}"
    old_suppressed_id = f"old-suppressed-{uuid4().hex}"
    old_visible_id = f"old-visible-{uuid4().hex}"
    async with factory() as db:
        await sync_checkpoint_messages(
            db, user_id=owner.id, thread_id=source_thread,
            messages=[HumanMessage(id=old_suppressed_id, content="旧代号 ZX-701。"),
                      HumanMessage(id=old_visible_id, content="现行代号 ZX-702。")])
        await tombstone_message(db, user_id=owner.id, thread_id=source_thread,
                                message_id=old_suppressed_id)
        await db.commit()

    settings = get_settings().model_copy(update={"memory_mode": "transcript_lexical"})
    memory = create_memory_runtime(session_factory=factory, settings=settings)
    model = ScriptedModel([AIMessage(content="请说明具体目标。")])
    graph, tools = build_official_agent(
        model=model, session_factory=factory, settings=settings,
        checkpointer=InMemorySaver(), memory_orchestrator=memory.orchestrator)
    assert "search_memory" in {item.name for item in tools}
    assert "control_memory" not in {item.name for item in tools}

    request_id = f"web-request:{uuid4()}"
    reply, _ = await run_agent(
        graph, owner.id, owner.telegram_id, "", "请忘掉 ZX-702。",
        projection_session_factory=factory, thread_id=query_thread,
        turn_id=request_id, origin="web", memory_history_threads=(source_thread,),
        return_state=True)
    assert reply == "请说明具体目标。"
    assert len(model._received) == 1

    async with factory() as db:
        current_thread = official_thread_id(query_thread)
        assert (await db.scalar(select(TranscriptSuppression.message_id).where(
            TranscriptSuppression.user_id == owner.id,
            TranscriptSuppression.thread_id == current_thread,
            TranscriptSuppression.message_id == request_id))) is None
        assert not await search_history(db, user_id=owner.id, thread_id=source_thread,
                                        query="ZX-701")
        assert await search_history(db, user_id=owner.id, thread_id=source_thread,
                                    query="ZX-702")


@pytest.mark.asyncio
async def test_retired_control_preserves_profile_and_cross_thread_search(pg_harness):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    source_thread = f"web:memory-source:{uuid4().hex}"
    query_thread = f"web:memory-read:{uuid4().hex}"
    async with factory() as db:
        await sync_checkpoint_messages(
            db, user_id=owner.id, thread_id=source_thread,
            messages=[HumanMessage(id=f"source-{uuid4().hex}",
                                   content="合成项目现行代号 ZX-702。")])
        await db.commit()

    settings = get_settings().model_copy(update={"memory_mode": "transcript_lexical"})
    memory = create_memory_runtime(session_factory=factory, settings=settings)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "search_memory", "id": "memory-read-1",
            "args": {"query": "ZX-702", "purpose": "exact"}}]),
        AIMessage(content="已读取有来源的历史。"),
    ])
    async with AsyncPostgresStore.from_conn_string(settings.database_url_sync) as store:
        await store.setup()
        await store.aput(("profile", owner.id), "user_profile",
                         {"preferences": ["回答尽量简短"]})
        graph, tools = build_official_agent(
            model=model, session_factory=factory, settings=settings,
            checkpointer=InMemorySaver(), store=store,
            memory_orchestrator=memory.orchestrator)
        assert "search_memory" in {item.name for item in tools}
        assert "control_memory" not in {item.name for item in tools}
        reply, state = await run_agent(
            graph, owner.id, owner.telegram_id, "", "查找历史合成项目代号。",
            projection_session_factory=factory, thread_id=query_thread,
            turn_id=f"web-request:{uuid4()}", origin="web",
            memory_history_threads=(source_thread,), return_state=True)
        assert reply == "已读取有来源的历史。"
        evidence = [item for item in state["messages"] if isinstance(item, ToolMessage)
                    and item.tool_call_id == "memory-read-1"]
        assert len(evidence) == 1 and evidence[0].status == "success"
        assert "ZX-702" in str(evidence[0].content)
        assert any("回答尽量简短" in str(message.content)
                   for message in model._received[0])
        await store.adelete(("profile", owner.id), "user_profile")
