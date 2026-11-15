"""Current unified graph with source-only memory on the guarded PG18 clone."""

from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy import select

from app.db.models import TranscriptProjection, TranscriptSuppression

pytestmark = pytest.mark.integration


class SourceModel:
    def __init__(self):
        self.inputs = []

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.inputs.append(messages)
        last_user = next((index for index in range(len(messages) - 1, -1, -1)
                          if isinstance(messages[index], HumanMessage)), 0)
        user = str(messages[last_user].content)
        tools = [item for item in messages[last_user + 1:] if isinstance(item, ToolMessage)]
        if "忘掉" in user and not tools:
            return AIMessage(content="", tool_calls=[{
                "name": "control_memory", "args": {"operation": "forget", "target_query": "RJ-385"},
                "id": "forget-source", "type": "tool_call"}])
        if "更正" in user and not tools:
            return AIMessage(content="", tool_calls=[{
                "name": "control_memory", "args": {"operation": "correct", "target_query": "V1-111"},
                "id": "correct-source", "type": "tool_call"}])
        if "查找历史" in user and not tools:
            query = "V2-222" if "版本" in user else "RJ-385"
            return AIMessage(content="", tool_calls=[{
                "name": "search_memory", "args": {"query": query, "purpose": "exact"},
                "id": "search-source", "type": "tool_call"}])
        return AIMessage(content="没有可用来源" if not tools else str(tools[-1].content))


@pytest.mark.asyncio
async def test_unified_graph_source_lookup_correction_forget_and_isolation(
        pg_harness, monkeypatch):
    from app.agent import graph, nodes, unified_loop
    from app.agent.graph import run_agent
    from app.agent.memory_runtime import create_memory_runtime
    from app.config import get_settings

    factory, new_user, _ = pg_harness
    owner, stranger = await new_user(), await new_user()
    source = f"web:simple-memory:{uuid4().hex}"
    query = f"web:simple-memory-query:{uuid4().hex}"
    other = f"web:simple-memory-other:{uuid4().hex}"
    settings = get_settings().model_copy(update={"memory_mode": "transcript_lexical"})
    runtime = create_memory_runtime(session_factory=factory, settings=settings)
    assert runtime.provider is None and runtime.extraction_model is None
    await runtime.preflight_storage()
    model = SourceModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    async with AsyncPostgresStore.from_conn_string(settings.database_url_sync) as store:
        await store.setup()
        compiled = await graph.build_graph(factory, store=store, memory_runtime=runtime,
                                           durable=True)
        async def turn(user, thread, text, *, scopes=()):
            return await run_agent(
                compiled, user.id, user.telegram_id, "", text,
                projection_session_factory=factory, thread_id=thread,
                turn_id=f"web-request:{uuid4()}", origin="web",
                memory_history_threads=scopes, return_state=True)
        try:
            await turn(owner, source, "我偏好简短回答。")
            await turn(owner, source, "青亭项目旧联络码是 RJ-385。")
            await turn(owner, source, "项目版本是 V1-111。")
            profile = await store.aget(("profile", owner.id), "user_profile")
            assert profile and "简短" in str(profile.value)
            response, state = await turn(owner, query, "查找历史青亭项目联络码。", scopes=(source,))
            assert "RJ-385" in response
            assert any("简短" in str(item.content) for item in model.inputs[-1]
                       if item.type == "system")
            outsider, _ = await turn(stranger, other, "查找历史青亭项目联络码。",
                                     scopes=(source,))
            assert "RJ-385" not in outsider

            await turn(owner, source, "更正：项目版本 V1-111 改成 V2-222。")
            changed, _ = await turn(owner, query, "查找历史项目版本。", scopes=(source,))
            assert "V2-222" in changed
            async with factory() as db:
                old = (await db.scalars(select(TranscriptProjection).where(
                    TranscriptProjection.user_id == owner.id,
                    TranscriptProjection.search_text.contains("项目版本是 V1-111")))).one()
                assert old.deleted_at is not None

            forgotten, state = await turn(owner, source, "请忘掉 RJ-385，不要再从旧对话告诉我。")
            assert "RJ-385" not in forgotten
            assert any(isinstance(item, ToolMessage) and item.tool_call_id == "forget-source" and
                       item.status == "success" for item in state["messages"])
            after, _ = await turn(owner, query, "查找历史青亭项目联络码。", scopes=(source,))
            assert "RJ-385" not in after
            async with factory() as db:
                suppressed = (await db.scalars(select(TranscriptSuppression).where(
                    TranscriptSuppression.user_id == owner.id,
                    TranscriptSuppression.thread_id == source))).all()
                assert len(suppressed) >= 2
                visible = (await db.scalars(select(TranscriptProjection).where(
                    TranscriptProjection.user_id == owner.id,
                    TranscriptProjection.thread_id == source,
                    TranscriptProjection.deleted_at.is_(None),
                    TranscriptProjection.search_text.contains("RJ-385")))).all()
                assert not visible
            assert all("RJ-385" not in str(item.content) for item in model.inputs[-1])
        finally:
            for thread in (source, query, other):
                await compiled.checkpointer.adelete_thread(thread)
            await compiled.checkpointer.conn.close()
            await store.adelete(("profile", owner.id), "user_profile")


@pytest.mark.asyncio
async def test_ambiguous_forget_suppresses_request_but_not_arbitrary_source(pg_harness):
    from app.agent.memory_control import apply_source_control
    from app.agent.transcript import sync_checkpoint_messages

    factory, new_user, _ = pg_harness
    owner = await new_user()
    thread = f"web:simple-ambiguous:{uuid4().hex}"
    ids = [f"source-{uuid4().hex}" for _ in range(2)]
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread,
                                       messages=[HumanMessage(id=ids[0], content="青亭项目有旧资料 A。"),
                                                 HumanMessage(id=ids[1], content="青亭项目有旧资料 B。")])
        await db.commit()
    request_id = f"request-{uuid4().hex}"
    result = await apply_source_control(
        session_factory=factory, user_id=owner.id, thread_id=thread,
        operation="forget", target_query="青亭项目", request="请忘掉青亭项目旧资料。",
        current_message_id=request_id, history=[])
    assert not result.changed
    async with factory() as db:
        # run_agent projects the current request after graph completion; the
        # precommitted suppression must prevent that fresh copy from indexing.
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread,
                                       messages=[HumanMessage(id=request_id,
                                           content="请忘掉青亭项目旧资料。")])
        await db.commit()
        assert (await db.scalars(select(TranscriptSuppression.message_id).where(
            TranscriptSuppression.user_id == owner.id,
            TranscriptSuppression.thread_id == thread))).all() == [request_id]
        sources = (await db.scalars(select(TranscriptProjection).where(
            TranscriptProjection.user_id == owner.id,
            TranscriptProjection.thread_id == thread))).all()
        assert len(sources) == 2 and all(row.deleted_at is None for row in sources)


@pytest.mark.asyncio
async def test_model_expanded_target_with_explicit_code_anchor_can_forget_unique_source(pg_harness):
    from app.agent.memory_control import apply_source_control
    from app.agent.transcript import sync_checkpoint_messages

    factory, new_user, _ = pg_harness
    owner = await new_user()
    thread = f"web:simple-expanded-target:{uuid4().hex}"
    source_id = f"source-{uuid4().hex}"
    request_id = f"request-{uuid4().hex}"
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread,
                                       messages=[HumanMessage(id=source_id,
                                           content="青砚项目的旧联络码是 RQ-908。")])
        await db.commit()
    # The model can include words from the original source in target_query.
    # Only the distinctive code appears verbatim in the new user request.
    result = await apply_source_control(
        session_factory=factory, user_id=owner.id, thread_id=thread,
        operation="forget", target_query="青砚项目的旧联络码是 RQ-908",
        request="请忘掉 RQ-908 这个旧联络码，不要再使用它。",
        current_message_id=request_id, history=[])
    assert result.changed
    async with factory() as db:
        source = (await db.scalars(select(TranscriptProjection).where(
            TranscriptProjection.user_id == owner.id,
            TranscriptProjection.thread_id == thread,
            TranscriptProjection.message_id == source_id))).one()
        assert source.deleted_at is not None
        suppressed = (await db.scalars(select(TranscriptSuppression.message_id).where(
            TranscriptSuppression.user_id == owner.id,
            TranscriptSuppression.thread_id == thread))).all()
        assert set(suppressed) == {source_id, request_id}


@pytest.mark.asyncio
async def test_failed_forget_does_not_reintroduce_secret_in_same_graph_turn(
        pg_harness, monkeypatch):
    from app.agent import graph, nodes, unified_loop
    from app.agent.graph import run_agent
    from app.agent.memory_runtime import create_memory_runtime
    from app.config import get_settings

    factory, new_user, _ = pg_harness
    owner = await new_user()
    thread = f"web:simple-failed-forget:{uuid4().hex}"
    runtime = create_memory_runtime(session_factory=factory, settings=get_settings().model_copy(
        update={"memory_mode": "transcript_lexical"}))
    model = SourceModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    compiled = await graph.build_graph(factory, memory_runtime=runtime, durable=True)
    async def turn(text):
        return await run_agent(compiled, owner.id, owner.telegram_id, "", text,
                               projection_session_factory=factory, thread_id=thread,
                               turn_id=f"web-request:{uuid4()}", origin="web",
                               return_state=True)
    try:
        await turn("旧资料甲的代码 RJ-385。")
        await turn("旧资料乙也写了 RJ-385。")
        reply, state = await turn("请忘掉 RJ-385，不要再从旧对话告诉我。")
        results = [item for item in state["messages"] if isinstance(item, ToolMessage)
                   and item.tool_call_id == "forget-source"]
        assert results and results[-1].status == "error"
        assert "RJ-385" not in reply
        assert all("RJ-385" not in str(item) for item in model.inputs[-1])
        assert not any(isinstance(item, ToolMessage) and item.tool_call_id == "search-source"
                       for item in state["messages"])
        async with factory() as db:
            source_rows = (await db.scalars(select(TranscriptProjection).where(
                TranscriptProjection.user_id == owner.id,
                TranscriptProjection.thread_id == thread,
                TranscriptProjection.deleted_at.is_(None),
                TranscriptProjection.search_text.contains("RJ-385")))).all()
            assert len(source_rows) == 2  # ambiguous originals remain; request is suppressed
    finally:
        await compiled.checkpointer.adelete_thread(thread)
        await compiled.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_profile_bearing_source_is_not_falsely_reported_forgotten(pg_harness):
    from app.agent.memory_control import apply_source_control
    from app.agent.transcript import sync_checkpoint_messages

    factory, new_user, _ = pg_harness
    owner = await new_user()
    thread = f"web:simple-profile-boundary:{uuid4().hex}"
    source_id = f"source-{uuid4().hex}"
    request_id = f"request-{uuid4().hex}"
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread,
                                       messages=[HumanMessage(id=source_id,
                                           content="我偏好简短回答。")])
        await db.commit()
    result = await apply_source_control(
        session_factory=factory, user_id=owner.id, thread_id=thread,
        operation="forget", target_query="简短回答", request="请忘掉简短回答偏好。",
        current_message_id=request_id, history=[])
    assert not result.changed
    async with factory() as db:
        source = (await db.scalars(select(TranscriptProjection).where(
            TranscriptProjection.user_id == owner.id,
            TranscriptProjection.thread_id == thread,
            TranscriptProjection.message_id == source_id))).one()
        assert source.deleted_at is None
        suppressed = (await db.scalars(select(TranscriptSuppression.message_id).where(
            TranscriptSuppression.user_id == owner.id,
            TranscriptSuppression.thread_id == thread))).all()
        assert request_id in suppressed and source_id not in suppressed
