"""Current unified graph regressions on the guarded disposable PG18 target."""

from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy import select

from app.db.models import Todo

pytestmark = pytest.mark.integration


class CapturingModel:
    def __init__(self, *, propose_create=False):
        self.inputs = []
        self.propose_create = propose_create

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.inputs.append(messages)
        if self.propose_create and len(self.inputs) == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_todo", "args": {"title": "整理厨房"},
                "id": "attempt-create", "type": "tool_call"}])
        return AIMessage(content="收到")


async def _graph(factory, monkeypatch, model, *, store=None):
    from app.agent import graph, nodes, unified_loop
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    return await graph.build_graph(factory, store=store, durable=True)


@pytest.mark.asyncio
async def test_profile_store_is_injected_in_full_graph_and_is_user_scoped(pg_harness, monkeypatch):
    from app.agent.graph import run_agent
    from app.config import get_settings

    factory, new_user, _ = pg_harness
    owner, other = await new_user(), await new_user()
    model = CapturingModel()
    threads = [f"web:r3-profile:{uuid4().hex}" for _ in range(3)]
    async with AsyncPostgresStore.from_conn_string(get_settings().database_url_sync) as store:
        await store.setup()
        compiled = await _graph(factory, monkeypatch, model, store=store)
        try:
            await run_agent(compiled, owner.id, owner.telegram_id, "", "我偏好简短回答。",
                            thread_id=threads[0], turn_id=f"web-request:{uuid4()}", origin="web")
            saved = await store.aget(("profile", owner.id), "user_profile")
            assert saved and "简短回答" in str(saved.value)
            assert "简短回答" in str(model.inputs[-1][0].content)
            await run_agent(compiled, owner.id, owner.telegram_id, "", "你好",
                            thread_id=threads[1], turn_id=f"web-request:{uuid4()}", origin="web")
            assert "简短回答" in str(model.inputs[-1][0].content)
            await run_agent(compiled, other.id, other.telegram_id, "", "你好",
                            thread_id=threads[2], turn_id=f"web-request:{uuid4()}", origin="web")
            assert "简短回答" not in str(model.inputs[-1][0].content)
            assert await store.aget(("profile", other.id), "user_profile") is None
        finally:
            for thread in threads:
                await compiled.checkpointer.adelete_thread(thread)
            await compiled.checkpointer.conn.close()
            await store.adelete(("profile", owner.id), "user_profile")


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance,should_create", [
    ("周二我准备整理厨房。", False),
    ("我打算周末读一本新书。", False),
    ("帮我创建一个整理厨房的待办。", True),
])
async def test_unified_graph_rejects_unrequested_todo_but_keeps_explicit_create(
        pg_harness, monkeypatch, utterance, should_create):
    from app.agent.graph import run_agent
    factory, new_user, _ = pg_harness
    user = await new_user()
    model = CapturingModel(propose_create=True)
    compiled = await _graph(factory, monkeypatch, model)
    thread = f"web:r3-todo-authorization:{uuid4().hex}"
    try:
        await run_agent(compiled, user.id, user.telegram_id, "", utterance,
                        thread_id=thread, turn_id=f"web-request:{uuid4()}", origin="web")
        state = await compiled.aget_state({"configurable": {"thread_id": thread}})
        tool_results = [message for message in state.values["messages"]
                        if isinstance(message, ToolMessage)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_call_id == "attempt-create"
        assert (tool_results[0].status == "success") is should_create
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
        assert bool(rows) is should_create
    finally:
        await compiled.checkpointer.adelete_thread(thread)
        await compiled.checkpointer.conn.close()


@pytest.mark.asyncio
async def test_assistant_inference_is_not_promoted_to_cross_thread_fact(pg_harness):
    from langchain_core.messages import HumanMessage
    from app.agent.memory_context import ContextOrchestrator, ContextRequest, memory_evidence_message
    from app.agent.memory_embedding import EmbeddingEgressPolicy
    from app.agent.transcript import search_history, sync_checkpoint_messages

    class SyntheticEmbedding:
        model_id = "r3-source-authority-2d"
        provider_name = "local-synthetic"
        dimension = 2

        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class NoSummary:
        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("This short source cannot trigger summary generation")

    factory, new_user, _ = pg_harness
    owner = await new_user()
    source = f"r3-authority:{uuid4().hex}"
    query = f"r3-authority-query:{uuid4().hex}"
    prior = [
        HumanMessage(id=f"user-{uuid4().hex}", content="鹭原项目汇报改为周五，周三作废。"),
        AIMessage(id=f"assistant-{uuid4().hex}",
                  content="鹭原项目汇报时间可能是 2026-10-23，请用户确认。"),
    ]
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=source, messages=prior)
        await db.commit()
        raw = await search_history(db, user_id=owner.id, thread_id=source,
                                   query="鹭原项目汇报", top_k=8)
    assert {hit.role for hit in raw} == {"user", "assistant"}
    orchestrator = ContextOrchestrator(
        session_factory=factory, summary_model=NoSummary(),
        summary_provider="synthetic", summary_model_name="unused",
        embedding_provider=SyntheticEmbedding(),
        embedding_egress=EmbeddingEgressPolicy("http://127.0.0.1:11434/v1"))
    bundle = await orchestrator.assemble(ContextRequest(
        user_id=owner.id, thread_id=query, request="鹭原项目汇报现在何时？",
        route="chat", messages=(HumanMessage(content="鹭原项目汇报现在何时？"),),
        search_query="鹭原项目汇报", search_purpose="exact", token_budget=4096,
        additional_history_threads=(source,), allow_summary_update=False))
    assert bundle.history and all(hit.role == "user" for hit in bundle.history)
    evidence = memory_evidence_message(bundle)
    assert evidence and "周五" in evidence.content
    assert "2026-10-23" not in evidence.content


@pytest.mark.asyncio
async def test_forget_commits_source_and_request_suppression_before_new_thread_search(
        pg_harness, monkeypatch):
    import json
    from langchain_core.messages import HumanMessage
    from sqlalchemy import text
    from app.agent.graph import run_agent
    from app.agent.memory_context import ContextOrchestrator
    from app.agent.memory_embedding import EmbeddingEgressPolicy
    from app.agent.memory_pgvector import ensure_pgvector_projection
    from app.agent.memory_runtime import MemoryRuntime
    from app.agent.transcript import search_history
    from app.db.models import TranscriptProjection, TranscriptSuppression

    class SyntheticEmbedding:
        model_id = "r3-forget-2d"
        provider_name = "local-synthetic"
        dimension = 2

        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class SelectSource:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            catalog = json.loads(messages[-1].content)["candidates"]
            key = next(key for key, value in catalog.items()
                       if key.startswith("conversation:") and "HN-2674" in value["text"])
            return {"target_type": "conversation", "target_id": key.split(":", 1)[1],
                    "target_quote": "HN-2674", "confidence": 0.99}

    class ForgetModel(CapturingModel):
        async def ainvoke(self, messages):
            self.inputs.append(messages)
            latest = next((item for item in reversed(messages)
                           if isinstance(item, HumanMessage)), None)
            tools = [item for item in messages if isinstance(item, ToolMessage)]
            if latest and "请忘掉" in str(latest.content) and not tools:
                return AIMessage(content="", tool_calls=[{
                    "name": "control_memory", "args": {
                        "operation": "forget", "target_query": "HN-2674"},
                    "id": "forget-request", "type": "tool_call"}])
            if latest and "你还记得" in str(latest.content) and not tools:
                return AIMessage(content="", tool_calls=[{
                    "name": "search_memory", "args": {"query": "HN-2674", "purpose": "exact"},
                    "id": "search-after-forget", "type": "tool_call"}])
            if latest and "内部口令是" in str(latest.content):
                return AIMessage(content="已记住 HN-2674")
            return AIMessage(content="没有可用的旧口令证据")

    factory, new_user, _ = pg_harness
    owner = await new_user()
    source_thread = f"web:r3-forget:{uuid4().hex}"
    query_thread = f"web:r3-forget-query:{uuid4().hex}"
    egress = EmbeddingEgressPolicy("http://127.0.0.1:11434/v1")
    provider = SyntheticEmbedding()

    class NoSummary:
        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("Short source cannot require summary generation")

    orchestrator = ContextOrchestrator(
        session_factory=factory, summary_model=NoSummary(),
        summary_provider="synthetic", summary_model_name="unused",
        embedding_provider=provider, embedding_egress=egress)
    runtime = MemoryRuntime(orchestrator=orchestrator, provider=provider, egress=egress,
                            session_factory=factory, extraction_model=SelectSource())
    model = ForgetModel()
    from app.agent import graph, nodes, unified_loop
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    async with factory() as db:
        await ensure_pgvector_projection(db)
        await db.commit()
    compiled = await graph.build_graph(factory, memory_runtime=runtime, durable=True)
    try:
        source_turn = f"web-request:{uuid4()}"
        await run_agent(compiled, owner.id, owner.telegram_id, "",
                        "晴禾项目内部口令是 HN-2674。", projection_session_factory=factory,
                        thread_id=source_thread, turn_id=source_turn, origin="web")
        async with factory() as db:
            source_rows = (await db.scalars(select(TranscriptProjection).where(
                TranscriptProjection.user_id == owner.id,
                TranscriptProjection.thread_id == source_thread,
                TranscriptProjection.search_text.contains("HN-2674")))).all()
        assert len(source_rows) == 2
        before = await runtime.maintain_thread(user_id=owner.id, thread_id=source_thread,
                                               consolidate=False)
        assert before["transcript_embeddings"] > 0
        forget_turn = f"web-request:{uuid4()}"
        reply, finished = await run_agent(
            compiled, owner.id, owner.telegram_id, "",
            "请忘掉晴禾项目内部口令 HN-2674，不要再从旧对话里告诉我。",
            projection_session_factory=factory, thread_id=source_thread,
            turn_id=forget_turn, origin="web", return_state=True)
        forget_results = [item for item in finished["messages"] if isinstance(item, ToolMessage)
                          and item.tool_call_id == "forget-request"]
        assert len(forget_results) == 1 and forget_results[0].status == "success"
        async with factory() as db:
            suppressed = set((await db.scalars(select(TranscriptSuppression.message_id).where(
                TranscriptSuppression.user_id == owner.id,
                TranscriptSuppression.thread_id == source_thread))).all())
            visible = (await db.scalars(select(TranscriptProjection).where(
                TranscriptProjection.user_id == owner.id,
                TranscriptProjection.thread_id == source_thread,
                TranscriptProjection.deleted_at.is_(None),
                TranscriptProjection.search_text.contains("HN-2674")))).all()
        assert {row.message_id for row in source_rows} <= suppressed
        assert forget_turn in suppressed
        assert not visible
        assert all("HN-2674" not in str(item.content) for item in model.inputs[-1])
        await runtime.maintain_thread(user_id=owner.id, thread_id=source_thread,
                                      consolidate=False)
        async with factory() as db:
            for row in source_rows:
                remaining_sidecars = await db.scalar(text(
                    "SELECT count(*) FROM transcript_embedding_halfvec "
                    "WHERE user_id=:owner AND thread_id=:thread "
                    "AND message_id=:message"),
                    {"owner": owner.id, "thread": source_thread,
                     "message": row.message_id})
                assert remaining_sidecars == 0
        await run_agent(compiled, owner.id, owner.telegram_id, "", "你还记得晴禾旧口令吗？",
                        projection_session_factory=factory, thread_id=query_thread,
                        turn_id=f"web-request:{uuid4()}", origin="web",
                        memory_history_threads=(source_thread,))
        state = await compiled.aget_state({"configurable": {"thread_id": query_thread}})
        search = [item for item in state.values["messages"] if isinstance(item, ToolMessage)
                  and item.tool_call_id == "search-after-forget"]
        assert len(search) == 1 and "HN-2674" not in search[0].content
        async with factory() as db:
            assert not await search_history(db, user_id=owner.id, thread_id=source_thread,
                                            query="HN-2674", top_k=8)
    finally:
        for thread in (source_thread, query_thread):
            await compiled.checkpointer.adelete_thread(thread)
        await compiled.checkpointer.conn.close()
