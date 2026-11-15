"""Synthetic long history through real Saver, Qwen, LangMem and production graph.

These are opt-in real-provider/real-PostgreSQL integration tests, not Telegram
E2E and not evidence from private user transcripts.
"""

from datetime import datetime, timedelta, timezone
from math import ceil
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.agent import graph as graph_module
from app.agent import planner as planner_module
from app.agent.memory_consolidation import consolidate_pending
from app.agent.memory_context import ContextRequest
from app.agent.memory_embedding import index_transcript_embeddings
from app.agent.memory_runtime import create_memory_runtime
from app.agent.memory_pgvector import (
    ensure_pgvector_projection, sync_structured_halfvec,
    sync_transcript_halfvec,
)
from app.agent.structured_search import index_structured_memories
from app.agent.store import generate_store
from app.agent.transcript import sync_from_saver
from app.config import get_settings
from app.db.models import ConversationSummary, MemoryConsolidationCursor, StructuredMemory
from evals.r2_memory_benchmark import corpus

pytestmark = pytest.mark.llm_integration


@pytest.mark.parametrize("turn_count", [50, 100, 160])
@pytest.mark.asyncio
async def test_long_thread_memory_lifecycle_and_restart(pg_harness, monkeypatch, turn_count):
    settings = get_settings()
    if not (settings.memory_embedding_local_self_hosted and
            settings.memory_embedding_no_unknown_proxy and
            settings.memory_embedding_body_logging_disabled):
        pytest.skip("Self-hosted embedding egress attestations are not configured")
    factory, new_user, _engine = pg_harness
    user = await new_user()
    thread = f"user_{user.telegram_id}"
    config = {"configurable": {"thread_id": thread}}
    selected = [turn for turn in corpus()[0] if turn.user_id == "synthetic-a" and
                turn.thread_id == "synthetic-main" and turn.turn <= turn_count]
    stamp = datetime(2026, 10, 4, tzinfo=timezone.utc)
    messages = []
    for turn in selected:
        # Noise creates real token pressure without additional external writes.
        body = (turn.content + " " + "routine planning noise " * 35
                if turn.content.startswith("Routine discussion") else turn.content)
        messages.extend([
            HumanMessage(id=turn.message_id, content=body,
                         additional_kwargs={
                             "occurred_at_utc": (stamp + timedelta(minutes=turn.turn)).isoformat(),
                             **({"supersedes_message_id": turn.supersedes}
                                if turn.supersedes else {}),
                         }),
            AIMessage(id=f"reply-{turn.message_id}", content="Acknowledged."),
        ])

    async def no_calendar_discovery(_settings):
        return []
    monkeypatch.setattr(graph_module, "discover_calendar_tools", no_calendar_discovery)
    runtime = create_memory_runtime(session_factory=factory, settings=settings)
    try:
        if settings.memory_vector_backend == "pgvector_halfvec_exact":
            async with factory() as db:
                await ensure_pgvector_projection(db)
                await db.commit()
        async with generate_store() as store:
            graph = await graph_module.build_graph(factory, store=store,
                                                   memory_runtime=runtime)
            try:
                for offset in range(0, len(messages), 20):
                    update = {"messages": messages[offset:offset + 20]}
                    if offset == 0:
                        update.update({"user_id": user.id, "telegram_id": user.telegram_id,
                                       "chat_id": "synthetic"})
                    await graph.aupdate_state(config, update, as_node="chat_response_node")
                async with factory() as db:
                    assert await sync_from_saver(db, graph=graph,
                                                 user_id=user.id, thread_id=thread) == len(messages)
                    await index_transcript_embeddings(
                        db, user_id=user.id, thread_id=thread,
                        provider=runtime.provider, egress=runtime.egress)
                    if settings.memory_vector_backend == "pgvector_halfvec_exact":
                        await sync_transcript_halfvec(
                            db, user_id=user.id, thread_id=thread,
                            model_id=runtime.provider.model_id,
                            dimension=runtime.provider.dimension)
                    await db.commit()
                for _ in range(ceil(turn_count / 12) + 1):
                    result = await consolidate_pending(
                        factory, user_id=user.id, thread_id=thread,
                        model=runtime.extraction_model)
                    if not result.model_called:
                        break
                else:
                    raise AssertionError("Consolidation cursor did not finish the synthetic thread")
                async with factory() as db:
                    await index_structured_memories(
                        db, user_id=user.id, provider=runtime.provider,
                        egress=runtime.egress)
                    if settings.memory_vector_backend == "pgvector_halfvec_exact":
                        await sync_structured_halfvec(
                            db, user_id=user.id,
                            model_id=runtime.provider.model_id,
                            dimension=runtime.provider.dimension)
                    await db.commit()
                    cursor = await db.scalar(select(MemoryConsolidationCursor).where(
                        MemoryConsolidationCursor.user_id == user.id,
                        MemoryConsolidationCursor.thread_id == thread))
                    structured_count = len((await db.scalars(select(StructuredMemory).where(
                        StructuredMemory.user_id == user.id,
                        StructuredMemory.status == "current"))).all())
                assert cursor is not None and cursor.version >= ceil(turn_count / 12)
                assert structured_count >= 1

                question = "What was the Blue Heron review code I mentioned earlier?"
                for attempt in range(12):
                    bundle = await runtime.orchestrator.assemble(ContextRequest(
                        user_id=user.id, thread_id=thread, request=question, route="chat",
                        messages=tuple(messages + [HumanMessage(
                            id=f"summary-probe-{attempt}", content=question)]),
                        token_budget=4096, search_query="Blue Heron review code",
                        search_purpose="exact"))
                    if not bundle.summary_updated:
                        break
                async with factory() as db:
                    summary = await db.scalar(select(ConversationSummary).where(
                        ConversationSummary.user_id == user.id,
                        ConversationSummary.thread_id == thread))
                assert summary is not None and summary.version >= 1
                initial_route = []
                initial_context = []
                original_classify = planner_module.classify_core_intent
                original_assemble = runtime.orchestrator.assemble
                async def record_initial_classify(message, model):
                    route = await original_classify(message, model)
                    initial_route.append({"intent": route.intent,
                                          "memory_query": route.memory_search_query,
                                          "purpose": route.memory_search_purpose})
                    return route
                async def record_initial_assemble(request):
                    selected = await original_assemble(request)
                    initial_context.append({
                        "search_query": request.search_query,
                        "has_review_code": any(
                            "AX-4731" in item.content for item in selected.history) or any(
                            "AX-4731" in item.canonical_text for item in selected.structured) or
                            "AX-4731" in (selected.summary or ""),
                        "degraded": selected.degraded,
                    })
                    return selected
                monkeypatch.setattr(planner_module, "classify_core_intent",
                                    record_initial_classify)
                monkeypatch.setattr(runtime.orchestrator, "assemble",
                                    record_initial_assemble)
                answer = await graph_module.run_agent(
                    graph, user.id, user.telegram_id, "synthetic", question,
                    projection_session_factory=factory)
                assert "AX-4731" in answer, {
                    "answer": answer, "observed_route": initial_route,
                    "observed_context": initial_context,
                }
                monkeypatch.setattr(planner_module, "classify_core_intent", original_classify)
                assert (await graph.aget_state(config)).values["messages"]
            finally:
                await graph.checkpointer.conn.close()

        # New Store, model, embedding client and Saver connection: all memory
        # layers must still be addressable by the same user/thread scope.
        await runtime.aclose()
        runtime = create_memory_runtime(session_factory=factory, settings=settings)
        async with generate_store() as restarted_store:
            restarted = await graph_module.build_graph(factory, store=restarted_store,
                                                       memory_runtime=runtime)
            try:
                restored = await restarted.aget_state(config)
                assert restored.values["user_id"] == user.id
                assert len(restored.values["messages"]) >= len(messages)
                recall_query = "Cedar pilot discussion with Mira changed to what weekday"
                recall_bundle = await runtime.orchestrator.assemble(ContextRequest(
                    user_id=user.id, thread_id=thread, request=recall_query, route="chat",
                    messages=tuple(restored.values["messages"]), token_budget=4096,
                    search_query=recall_query, search_purpose="exact"))
                assert any("Monday" in item.content for item in recall_bundle.history) or any(
                    "Monday" in item.canonical_text for item in recall_bundle.structured) or (
                    "Monday" in (recall_bundle.summary or "")), (
                        "Historical correction evidence absent from selected context; "
                        f"history={len(recall_bundle.history)}, "
                        f"structured={len(recall_bundle.structured)}, "
                        f"degraded={recall_bundle.degraded}")
                observed_context = []
                observed_route = []
                original_assemble = runtime.orchestrator.assemble
                original_classify = planner_module.classify_core_intent
                async def record_classify(message, model):
                    route = await original_classify(message, model)
                    observed_route.append({"intent": route.intent,
                                           "memory_query": route.memory_search_query,
                                           "purpose": route.memory_search_purpose})
                    return route
                monkeypatch.setattr(planner_module, "classify_core_intent", record_classify)
                async def record_assemble(request):
                    selected = await original_assemble(request)
                    observed_context.append({
                        "search_query": request.search_query,
                        "has_cedar_monday": any(
                            "Cedar" in item.content and "Monday" in item.content
                            for item in selected.history) or any(
                            "Cedar" in item.canonical_text and "Monday" in item.canonical_text
                            for item in selected.structured) or (
                            "Cedar" in (selected.summary or "") and
                            "Monday" in (selected.summary or "")),
                        "degraded": selected.degraded,
                    })
                    return selected
                monkeypatch.setattr(runtime.orchestrator, "assemble", record_assemble)
                answer = await graph_module.run_agent(
                    restarted, user.id, user.telegram_id, "synthetic",
                    "In our earlier conversation, what weekday did I say the Cedar "
                    "pilot discussion with Mira changed to? I am asking what I said, "
                    "not for the current Calendar event.",
                    projection_session_factory=factory)
                after_answer = await restarted.aget_state(config)
                assert "Monday" in answer, {
                    "answer": answer,
                    "intent": after_answer.values.get("intent"),
                    "memory_query": (after_answer.values.get("entities") or {}).get(
                        "memory_search_query"),
                    "observed_route": observed_route,
                    "observed_context": observed_context,
                }
            finally:
                await restarted.checkpointer.adelete_thread(thread)
                await restarted.checkpointer.conn.close()
    finally:
        await runtime.aclose()
