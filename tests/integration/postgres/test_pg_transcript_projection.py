"""R2: derived index survives restart and can be rebuilt from real Saver state."""
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from sqlalchemy import delete, func, select, text

from app.agent.state import AgentState
from app.agent.graph import run_agent
from app.agent.transcript import (rebuild_from_saver, search_history,
                                  sync_from_saver, sync_workflow_turn,
                                  rebuild_workflow_from_saver, tombstone_message)
from app.agent.workflow import ProductivityState
from app.config import get_settings
from app.db.models import Base, TranscriptProjection
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


def _graph(saver):
    builder = StateGraph(AgentState)
    builder.add_node("reply", lambda state: {})
    builder.set_entry_point("reply")
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_projection_restart_scope_idempotency_correction_rebuild(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
        await connection.execute(text("ALTER TABLE transcript_suppressions ADD COLUMN IF NOT EXISTS supersedes_message_id VARCHAR(160)"))
        await connection.execute(text("CREATE INDEX IF NOT EXISTS ix_transcript_fts ON transcript_projection USING GIN (to_tsvector('simple', search_text))"))
    user, other = await new_user(), await new_user()
    thread = f"r2-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread}}
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        graph = _graph(saver)
        await graph.aupdate_state(config, {"user_id": user.id, "messages": [
            HumanMessage(id="old-fact", content="Aurora report has 47 pages"),
            AIMessage(id="answer-old", content="Noted")]
        }, as_node="reply")
        await graph.aupdate_state(config, {"messages": [
            HumanMessage(id="new-fact", content="Correction: Aurora report has 52 pages",
                         additional_kwargs={"supersedes_message_id": "old-fact"}),
            AIMessage(id="answer-new", content="Updated")]
        }, as_node="reply")
        async with factory() as db:
            with pytest.raises(PermissionError):
                await sync_from_saver(db, graph=graph, user_id=other.id, thread_id=thread)
            assert await sync_from_saver(db, graph=graph, user_id=user.id, thread_id=thread) == 4
            await sync_from_saver(db, graph=graph, user_id=user.id, thread_id=thread)
            await db.commit()
        # A separate tenant and thread must never see the same fact.
        async with factory() as db:
            assert not await search_history(db, user_id=other.id, thread_id=thread, query="Aurora")
            assert not await search_history(db, user_id=user.id, thread_id="other", query="Aurora")
            hits = await search_history(db, user_id=user.id, thread_id=thread,
                                        query="Aurora", include_stale=True)
            assert {h.message_id for h in hits} >= {"old-fact", "new-fact"}
            assert any(h.message_id == "old-fact" and h.stale for h in hits)
            current = await search_history(db, user_id=user.id, thread_id=thread, query="Aurora")
            assert all(h.message_id != "old-fact" for h in current)
            assert await db.scalar(select(func.count()).select_from(TranscriptProjection).where(
                TranscriptProjection.user_id == user.id)) == 4

    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as restarted:
        graph = _graph(restarted)
        async with factory() as db:
            assert await search_history(db, user_id=user.id, thread_id=thread, query="Aurora")
            await db.execute(delete(TranscriptProjection).where(
                TranscriptProjection.user_id == user.id, TranscriptProjection.thread_id == thread))
            assert not await search_history(db, user_id=user.id, thread_id=thread, query="Aurora")
            assert await rebuild_from_saver(db, graph=graph, user_id=user.id, thread_id=thread) == 4
            await db.commit()
            rebuilt = await search_history(db, user_id=user.id, thread_id=thread, query="Aurora")
            assert rebuilt and all(hit.message_id != "old-fact" for hit in rebuilt)
            assert await tombstone_message(db, user_id=user.id, thread_id=thread,
                                           message_id="new-fact")
            await db.commit()
            assert all(hit.message_id != "new-fact" for hit in await search_history(
                db, user_id=user.id, thread_id=thread, query="Aurora"))
            await rebuild_from_saver(db, graph=graph, user_id=user.id, thread_id=thread)
            await db.commit()
            assert all(hit.message_id != "new-fact" for hit in await search_history(
                db, user_id=user.id, thread_id=thread, query="Aurora", include_stale=True))
            assert all(hit.message_id != "old-fact" for hit in await search_history(
                db, user_id=user.id, thread_id=thread, query="Aurora"))
        await restarted.adelete_thread(thread)


@pytest.mark.asyncio
async def test_workflow_projection_turn_identity_is_idempotent(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    user = await new_user()
    thread = f"r2-workflow-{uuid4().hex}"
    async with factory() as db:
        await sync_workflow_turn(db, user_id=user.id, thread_id=thread,
                                 turn_id="stable-turn", request="检查周五日历")
        await sync_workflow_turn(db, user_id=user.id, thread_id=thread,
                                 turn_id="stable-turn", request="检查周五日历", response="找到日程")
        await sync_workflow_turn(db, user_id=user.id, thread_id=thread,
                                 turn_id="stable-turn", request="检查周五日历", response="找到日程")
        await db.commit()
        rows = (await db.scalars(select(TranscriptProjection).where(
            TranscriptProjection.user_id == user.id, TranscriptProjection.thread_id == thread))).all()
        assert len(rows) == 2
        assert {r.role for r in rows} == {"user", "assistant"}


@pytest.mark.asyncio
async def test_workflow_backfill_reads_real_saver_revisions(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    user = await new_user()
    thread = f"r2-workflow-rebuild-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread}}
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        builder = StateGraph(ProductivityState)
        builder.add_node("reply", lambda state: {})
        builder.set_entry_point("reply")
        graph = builder.compile(checkpointer=saver)
        await graph.aupdate_state(config, {"user_id": user.id, "transcript_turn_id": "t1",
                                          "request": "查找蓝鲸项目", "response": ""}, as_node="reply")
        await graph.aupdate_state(config, {"response": "找到了蓝鲸项目"}, as_node="reply")
        await graph.aupdate_state(config, {"user_id": user.id, "transcript_turn_id": "t2",
                                          "request": "查找海风七号", "response": ""}, as_node="reply")
        await graph.aupdate_state(config, {"response": "找到了海风七号"}, as_node="reply")
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as restarted:
        graph = builder.compile(checkpointer=restarted)
        async with factory() as db:
            assert await rebuild_workflow_from_saver(db, graph=graph,
                                                      user_id=user.id, thread_id=thread) == 4
            await db.commit()
            rows = (await db.scalars(select(TranscriptProjection).where(
                TranscriptProjection.user_id == user.id,
                TranscriptProjection.thread_id == thread))).all()
            assert {row.message_id for row in rows} == {"t1:user", "t1:assistant", "t2:user", "t2:assistant"}
        await restarted.adelete_thread(thread)


@pytest.mark.asyncio
async def test_workflow_delivery_projects_checkpointed_response(pg_harness, monkeypatch):
    from app.telegram import webhook
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    user = await new_user()
    thread = f"r2-workflow-delivery-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread}}
    delivered = []
    async def sender(_chat_id, response):
        delivered.append(response)
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        builder = StateGraph(ProductivityState)
        builder.add_node("reply", lambda state: {})
        builder.set_entry_point("reply")
        graph = builder.compile(checkpointer=saver)
        await graph.aupdate_state(config, {"user_id": user.id, "transcript_turn_id": "delivery-t1",
                                          "request": "synthetic request", "response": "synthetic response"}, as_node="reply")
        async def provide_graph():
            return graph
        monkeypatch.setattr(webhook, "get_productivity_graph", provide_graph)
        monkeypatch.setattr(webhook, "send_message", sender)
        monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
        await webhook._send_workflow_result(1, thread, {"response": "synthetic response"})
        assert delivered == ["synthetic response"]
        assert (await graph.aget_state(config)).values["response"] == "synthetic response"
        async with factory() as db:
            hits = await search_history(db, user_id=user.id, thread_id=thread, query="synthetic")
            assert {hit.message_id for hit in hits} == {"delivery-t1:user", "delivery-t1:assistant"}
        await saver.adelete_thread(thread)


@pytest.mark.asyncio
async def test_production_ordinary_sync_hook_after_real_checkpoint(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
    user = await new_user()
    telegram_id = f"r2-ordinary-{uuid4().hex}"
    thread = f"user_{telegram_id}"
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        builder = StateGraph(AgentState)
        builder.add_node("reply", lambda state: {"messages": [AIMessage(content="synthetic response")]})
        builder.set_entry_point("reply")
        graph = builder.compile(checkpointer=saver)
        assert await run_agent(graph, user.id, telegram_id, "synthetic-chat", "synthetic request",
                               projection_session_factory=factory) == "synthetic response"
        assert (await graph.aget_state({"configurable": {"thread_id": thread}})).values["messages"]
        async with factory() as db:
            hits = await search_history(db, user_id=user.id, thread_id=thread, query="synthetic")
            assert {hit.role for hit in hits} == {"user", "assistant"}
            assert all(hit.source_ref.startswith(f"checkpoint:{thread}:") for hit in hits)
        await saver.adelete_thread(thread)
