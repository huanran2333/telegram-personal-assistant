"""Context selection uses scoped evidence, not all-memory concatenation."""

from datetime import datetime, timezone, timedelta

import pytest
pytest.skip("Legacy R3 semantic context is not the current transcript-lexical path",
            allow_module_level=True)
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.memory_context import ContextOrchestrator, ContextRequest, memory_evidence_message
from app.agent.memory_embedding import EmbeddingEgressPolicy, index_transcript_embeddings
from app.agent.structured_memory import EvidenceRef, ProjectFact, upsert_memory
from app.agent.structured_search import index_structured_memories
from app.agent.transcript import sync_checkpoint_messages
from app.agent import nodes
from app.agent.graph import todo_llm_tool_binding
from app.agent.tools import make_todo_tools
from app.db.models import Todo
from sqlalchemy import select
from app.db.models import Base

pytestmark = pytest.mark.integration


class SyntheticProvider:
    model_id = "synthetic-context-2d"
    provider_name = "synthetic-fixture"
    dimension = 2

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.extend(texts)
        return [[1.0, 0.0] if "Aurora" in text else [0.0, 1.0] for text in texts]


class NeverSummarize:
    async def ainvoke(self, *_args, **_kwargs):
        raise AssertionError("Short corpus must not trigger a summarizer model call")


@pytest.mark.asyncio
async def test_exact_recall_selects_structured_and_dedupes_history(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    owner, stranger = await new_user(), await new_user()
    thread_id = "r3-context-test"
    provider = SyntheticProvider()
    egress = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    history = []
    for index in range(10):
        content = "Aurora project code is AX-4731; budget is exactly 813 dollars" if index == 0 else f"Synthetic noise {index}"
        history += [HumanMessage(id=f"u-{index}", content=content,
                                 additional_kwargs={"occurred_at_utc":
                                                    (start + timedelta(minutes=index)).isoformat()}),
                    AIMessage(id=f"a-{index}", content="Acknowledged")]
    history.append(HumanMessage(id="u-current", content="What was Aurora project code?"))
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread_id,
                                       messages=history[:-1])
        await sync_checkpoint_messages(db, user_id=stranger.id, thread_id=thread_id,
                                       messages=[HumanMessage(id="other-secret", content="Aurora ZZ-9999")])
        fact = ProjectFact(project_key="Aurora", attribute="code", value="AX-4731")
        ref = EvidenceRef(source_type="checkpoint", source_id="u-0",
                          occurred_at=start, evidence_quote="AX-4731")
        await upsert_memory(db, user_id=owner.id, payload=fact, refs=[ref],
                            source_catalog={("checkpoint", "u-0"): history[0].content})
        await db.commit()

    async with factory() as db:
        await index_transcript_embeddings(db, user_id=owner.id, thread_id=thread_id,
                                          provider=provider, egress=egress)
        await index_structured_memories(db, user_id=owner.id, provider=provider, egress=egress)
        await db.commit()
    provider.calls.clear()
    # A checkpoint/projection can advance between periodic maintenance ticks.
    # An explicit history search catches up this derived index in-scope.
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=thread_id,
                                       messages=[HumanMessage(id="u-late", content="Aurora update")])
        await db.commit()
    orchestrator = ContextOrchestrator(
        session_factory=factory, summary_model=NeverSummarize(),
        summary_provider="synthetic", summary_model_name="unused",
        embedding_provider=provider, embedding_egress=egress)
    bundle = await orchestrator.assemble(ContextRequest(
        user_id=owner.id, thread_id=thread_id, request="What was Aurora project code?",
        route="chat", messages=tuple(history), search_query="Aurora project code",
        search_purpose="exact", token_budget=4096))
    assert bundle.structured and bundle.structured[0].payload["value"] == "AX-4731"
    assert any(hit.message_id == "u-0" and "813" in hit.content for hit in bundle.history)
    assert all("ZZ-9999" not in hit.content for hit in bundle.history)
    assert bundle.estimated_tokens <= bundle.token_budget
    assert memory_evidence_message(bundle) is not None
    assert "summary_unavailable" not in bundle.degraded
    assert "semantic_search_unavailable" not in bundle.degraded
    assert "Aurora update" in provider.calls

    provider.calls.clear()
    greeting = await orchestrator.assemble(ContextRequest(
        user_id=owner.id, thread_id=thread_id, request="你好", route="chat",
        messages=tuple(history[:-1] + [HumanMessage(id="u-greet", content="你好")]),
        token_budget=4096))
    assert not greeting.history and not greeting.structured
    assert provider.calls == []  # no unconditional embedding/search on greeting


@pytest.mark.asyncio
async def test_agent_on_demand_search_recovers_old_fact_without_cross_user_leak(pg_harness, monkeypatch):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    owner, stranger = await new_user(), await new_user()
    thread_id = "r3-on-demand-evidence"
    source = HumanMessage(id="old-evidence", content="Aurora project review code is AX-4731")
    async with factory() as db:
        await sync_checkpoint_messages(
            db, user_id=owner.id, thread_id=thread_id, messages=[source])
        await sync_checkpoint_messages(
            db, user_id=stranger.id, thread_id=thread_id,
            messages=[HumanMessage(id="other-evidence", content="Aurora review code is ZZ-9999")])
        await db.commit()
    provider = SyntheticProvider()
    orchestrator = ContextOrchestrator(
        session_factory=factory, summary_model=NeverSummarize(),
        summary_provider="synthetic", summary_model_name="unused",
        embedding_provider=provider,
        embedding_egress=EmbeddingEgressPolicy("http://127.0.0.1:8000/v1"))
    current = HumanMessage(id="current", content="Create a note from the old Aurora review code")
    baseline = await orchestrator.assemble(ContextRequest(
        user_id=owner.id, thread_id=thread_id, request=current.content,
        route="todo", messages=(current,), token_budget=4096,
        allow_summary_update=False))
    assert not baseline.history and not baseline.structured
    assert provider.calls == []  # no embedding expense on a request without historical search

    class NoModel:
        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: NoModel())
    todo_tools = make_todo_tools(factory)
    execute = nodes.make_nodes(
        todo_tools, llm_tools=todo_llm_tool_binding(todo_tools, [], True),
        memory_orchestrator=orchestrator)[3]
    result = await execute({
        "user_id": owner.id, "telegram_id": "synthetic-owner", "intent": "search",
        "conversation_thread_id": thread_id,
        "messages": [current, AIMessage(content="", tool_calls=[{
            "name": "search_memory", "args": {"query": "Aurora review code", "purpose": "exact"},
            "id": "search-scoped"}])],
        "memory_search_rounds": 0,
    })
    evidence = result["messages"][0].content
    assert "AX-4731" in evidence
    assert "ZZ-9999" not in evidence
    assert result["memory_search_rounds"] == 1
    assert provider.calls  # explicit search may pay for a scoped semantic lookup

    # A following business tool call uses the authenticated owner, never the
    # owner guessed by the model. This is a synthetic Todo, not a live Bot run.
    business = await execute({
        "user_id": owner.id, "telegram_id": "synthetic-owner", "intent": "create",
        "conversation_thread_id": thread_id,
        "messages": [current, result["messages"][0], AIMessage(content="", tool_calls=[{
            "name": "create_todo", "args": {
                "title": "Review AX-4731", "user_id": stranger.id},
            "id": "create-scoped"}])],
        "memory_search_rounds": 1,
    })
    assert "error" not in business["messages"][0].content.lower()
    async with factory() as db:
        row = await db.scalar(select(Todo).where(Todo.title == "Review AX-4731"))
        assert row is not None and row.user_id == owner.id
