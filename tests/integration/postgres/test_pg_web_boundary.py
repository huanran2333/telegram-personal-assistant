"""V3.1 E4: isolated PG18 custody, sessions, Todo replay and real Saver."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from fastapi import FastAPI
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.agent.graph import run_agent
from app.agent.memory_control import apply_memory_control
from app.agent.memory_embedding import EmbeddingEgressPolicy, index_transcript_embeddings
from app.agent.memory_pgvector import (ensure_pgvector_projection,
                                       search_transcript_halfvec, sync_transcript_halfvec)
from app.agent.structured_memory import ProjectFact
from app.agent.state import AgentState
from app.config import get_settings
from app.db import crud
from app.db.models import (ActionExecution, Todo, TranscriptProjection,
                           WebCredential, WebRequest, WebSession)
from app.db.migrations.v3_1_web_boundary import apply as apply_web_migration
from app.db.migrations.v3_1_cross_thread_lineage import apply as apply_lineage_migration
from app.telegram.durable import DurableSafetyError, commit_todo_tool_operation
from app.agent.transcript import (rebuild_from_saver, search_history,
                                  sync_checkpoint_messages)
from app.telegram.processing_context import tool_call_scope
from app.web.auth import bind_existing_user, login, principal
from app.web.processing_context import web_run_scope
from app.web.runs import WebRunWorker, claim_one, finish_run, submit_run

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_web_credentials_sessions_and_owner_isolation(pg_harness):
    factory, new_user, _ = pg_harness
    a, b = await new_user(), await new_user()
    async with factory() as db:
        await bind_existing_user(db, a.id, "unique password for user A")
        await bind_existing_user(db, b.id, "unique password for user B")
        assert await db.get(WebCredential, a.id)
    run_a, _ = await submit_run(factory, user_id=a.id, operation="chat",
                                idempotency_key=uuid4().hex, body={"text": "private A"})
    async with factory() as db:
        assert await db.scalar(select(WebRequest).where(
            WebRequest.id == run_a.id, WebRequest.user_id == b.id)) is None
    async with factory() as db:
        token, session = await login(
            db, SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
            SimpleNamespace(secret_key="test-secret", web_session_hours=24),
            "unique password for user A")
        current, _ = await principal(db, token)
        assert current.id == a.id and session.user_id == a.id
    async with factory() as db:
        credential = await db.get(WebCredential, a.id)
        original_version = credential.version
        await bind_existing_user(db, a.id, "reset password for user A", reset=True)
        assert (await db.get(WebCredential, a.id)).version == original_version + 1
        with pytest.raises(HTTPException):
            await principal(db, token)
    async with factory() as db:
        fresh_token, fresh_session = await login(
            db, SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
            SimpleNamespace(secret_key="test-secret", web_session_hours=24),
            "reset password for user A")
        stored = await db.get(WebSession, fresh_session.token_hash, with_for_update=True)
        stored.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    async with factory() as db:
        with pytest.raises(HTTPException) as expired:
            await principal(db, fresh_token)
        assert expired.value.status_code == 401


@pytest.mark.asyncio
async def test_web_login_rate_limit_is_postgres_shared(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    async with factory() as db:
        await bind_existing_user(db, user.id, "correct test password")
    settings = SimpleNamespace(secret_key="test-secret-rate", web_session_hours=24)
    # The shared throttle is intentionally durable; isolate repeat test runs
    # with a distinct synthetic peer identity.
    request = SimpleNamespace(client=SimpleNamespace(host=f"synthetic-{uuid4().hex}"))
    for _ in range(5):
        async with factory() as db:
            with pytest.raises(HTTPException) as failure:
                await login(db, request, settings, "wrong test password")
            assert failure.value.status_code == 401
    async with factory() as db:
        with pytest.raises(HTTPException) as blocked:
            await login(db, request, settings, "correct test password")
        assert blocked.value.status_code == 429


@pytest.mark.asyncio
async def test_web_migration_fresh_and_existing_user_data_preserved(pg_harness):
    factory, new_user, engine = pg_harness
    user = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title="preserve existing Todo")
        await db.commit()
        todo_id, telegram_id = todo.id, user.telegram_id
    async with engine.begin() as connection:
        await connection.run_sync(apply_web_migration)
        await connection.run_sync(apply_web_migration)
        await connection.run_sync(apply_lineage_migration)
        await connection.run_sync(apply_lineage_migration)
    async with factory() as db:
        existing = await db.get(Todo, todo_id)
        assert existing.title == "preserve existing Todo"
        assert (await db.get(type(user), user.id)).telegram_id == telegram_id
    # Fresh V3.1 DDL in a transactionally rolled-back isolated schema.
    schema = "v31_" + uuid4().hex[:16]
    async with engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            await connection.execute(text(
                "CREATE TABLE users (id VARCHAR(36) PRIMARY KEY, telegram_id VARCHAR(64) UNIQUE NOT NULL)"))
            await connection.execute(text(
                "CREATE TABLE app_schema_migrations (revision VARCHAR(128) PRIMARY KEY)"))
            await connection.execute(text(
                "CREATE TABLE transcript_projection (user_id VARCHAR(36), "
                "thread_id VARCHAR(160), message_id VARCHAR(160), "
                "supersedes_message_id VARCHAR(160))"))
            await connection.execute(text(
                "CREATE TABLE transcript_suppressions (user_id VARCHAR(36), "
                "thread_id VARCHAR(160), message_id VARCHAR(160), "
                "supersedes_message_id VARCHAR(160))"))
            await connection.run_sync(apply_web_migration)
            await connection.run_sync(apply_lineage_migration)
            await connection.run_sync(apply_lineage_migration)
            count = await connection.scalar(text(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema=:schema "
                "AND table_name LIKE 'web_%'"), {"schema": schema})
            assert count == 5
            lineage_columns = await connection.scalar(text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_schema=:schema AND column_name='supersedes_thread_id' "
                "AND table_name IN ('transcript_projection','transcript_suppressions')"),
                {"schema": schema})
            assert lineage_columns == 2
        finally:
            await transaction.rollback()


@pytest.mark.asyncio
async def test_two_workers_same_user_serialize_but_distinct_users_progress(pg_harness):
    factory, new_user, _ = pg_harness
    a, b = await new_user(), await new_user()
    first, _ = await submit_run(factory, user_id=a.id, operation="chat",
                                idempotency_key=uuid4().hex, body={"text": "first"})
    second, _ = await submit_run(factory, user_id=a.id, operation="chat",
                                 idempotency_key=uuid4().hex, body={"text": "second"})
    other, _ = await submit_run(factory, user_id=b.id, operation="chat",
                                idempotency_key=uuid4().hex, body={"text": "other user"})
    claims = await asyncio.gather(claim_one(factory, owner="a", lease_seconds=120),
                                  claim_one(factory, owner="b", lease_seconds=120))
    assert {item[0] for item in claims if item} == {first.id, other.id}
    assert second.id not in {item[0] for item in claims if item}


@pytest.mark.asyncio
async def test_web_request_idempotency_claim_fence_and_restart(pg_harness, isolated_pg18_target):
    factory, new_user, _ = pg_harness
    user = await new_user()
    key = uuid4().hex
    first, created = await submit_run(factory, user_id=user.id, operation="chat",
                                      idempotency_key=key, body={"text": "one turn"})
    second, created_again = await submit_run(factory, user_id=user.id, operation="chat",
                                             idempotency_key=key, body={"text": "one turn"})
    assert created and not created_again and first.id == second.id
    with pytest.raises(ValueError, match="IdempotencyConflict"):
        await submit_run(factory, user_id=user.id, operation="chat",
                         idempotency_key=key, body={"text": "changed"})
    claim = await claim_one(factory, owner="worker-a", lease_seconds=120)
    assert claim and claim[0] == first.id
    assert not await finish_run(factory, first.id, "stale-fence", state="COMPLETED",
                                result={"response": "wrong"})
    # Browser disconnect does not affect committed custody; a fresh engine
    # observes the same request and can recover the expired lease.
    fresh_engine = create_async_engine(isolated_pg18_target)
    fresh_factory = async_sessionmaker(fresh_engine, expire_on_commit=False)
    try:
        async with fresh_factory() as db:
            row = await db.get(WebRequest, first.id, with_for_update=True)
            row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        recovered = await claim_one(fresh_factory, owner="worker-b", lease_seconds=120)
        assert recovered and recovered[0] == first.id and recovered[1] != claim[1]
        assert not await finish_run(factory, first.id, claim[1], state="COMPLETED")
        assert await finish_run(fresh_factory, first.id, recovered[1],
                                state="COMPLETED", result={"response": "done"})
    finally:
        await fresh_engine.dispose()


@pytest.mark.asyncio
async def test_retired_web_agent_row_never_calls_model(pg_harness, monkeypatch):
    from app.agent import official_runtime

    factory, new_user, _ = pg_harness
    user = await new_user()
    request, _ = await submit_run(factory, user_id=user.id, operation="chat",
                                  idempotency_key=uuid4().hex,
                                  body={"text": "synthetic no-side-effect chat"})

    class EmptyGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={}, next=())

    async def graph():
        return EmptyGraph()

    async def outage(*_args, **_kwargs):
        try:
            raise ConnectionError("synthetic provider unavailable")
        except ConnectionError as cause:
            raise ValueError("wrapped route failure") from cause

    monkeypatch.setattr(official_runtime, "get_graph", graph)
    monkeypatch.setattr(official_runtime, "run_agent", outage)
    claimed = await claim_one(factory, owner="outage-worker", lease_seconds=120)
    assert claimed and claimed[0] == request.id
    await WebRunWorker(factory, get_settings().database_url).process(*claimed)
    async with factory() as db:
        stored = await db.get(WebRequest, request.id)
        assert stored.state == "FAILED"
        assert stored.safe_error == "WebRequestRejected"
        assert stored.route_payload is None
        assert not (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()


@pytest.mark.asyncio
async def test_web_decision_and_parent_status_commit_atomically(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    parent, _ = await submit_run(factory, user_id=user.id, operation="chat",
                                 idempotency_key=uuid4().hex,
                                 body={"text": "synthetic approved proposal"})
    async with factory() as db:
        row = await db.get(WebRequest, parent.id, with_for_update=True)
        row.state = "WAITING_APPROVAL"
        await db.commit()
    decision, _ = await submit_run(
        factory, user_id=user.id, operation="approval",
        idempotency_key=uuid4().hex,
        body={"parent_run_id": parent.id, "action_id": "synthetic-action",
              "checkpoint_version": "synthetic-checkpoint", "decision": "approve"})
    claim = await claim_one(factory, owner="worker-a", lease_seconds=120)
    assert claim and claim[0] == decision.id
    assert not await finish_run(factory, decision.id, "stale-fence", state="COMPLETED")
    async with factory() as db:
        assert (await db.get(WebRequest, parent.id)).state == "WAITING_APPROVAL"
    assert await finish_run(factory, decision.id, claim[1], state="COMPLETED",
                            result={"response": "approved synthetic result"})
    async with factory() as db:
        final_decision = await db.get(WebRequest, decision.id)
        final_parent = await db.get(WebRequest, parent.id)
        assert final_decision.state == final_parent.state == "COMPLETED"
        assert final_decision.result == final_parent.result


@pytest.mark.asyncio
async def test_web_todo_commit_and_action_proof_are_atomic_on_replay(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    run, _ = await submit_run(factory, user_id=user.id, operation="chat",
                              idempotency_key=uuid4().hex, body={"text": "create one Todo"})
    claimed = await claim_one(factory, owner="worker", lease_seconds=120)
    assert claimed and claimed[0] == run.id

    async def mutate():
        async with factory() as db:
            async def perform():
                todo = await crud.create_todo(db, user_id=user.id, title="one synthetic Todo")
                await db.flush()
                return todo.id
            return await commit_todo_tool_operation(
                db, name="create_todo", args={"user_id": user.id, "title": "one synthetic Todo"},
                perform=perform)

    with web_run_scope(run.id, user.id, claimed[1]), tool_call_scope("stable-tool-call"):
        first = await mutate()
        # Crash after Todo+ledger COMMIT but before the graph checkpoint:
        # replaying the same tool ID returns the prior proof, not a new Todo.
        second = await mutate()
    assert first == second
    async with factory() as db:
        todos = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
        proofs = (await db.scalars(select(ActionExecution).where(
            ActionExecution.user_id == user.id, ActionExecution.domain == "todo"))).all()
    assert len(todos) == len(proofs) == 1 and proofs[0].status == "executed"
    async with factory() as db:
        async def perform_changed():
            return "should not run"
        with web_run_scope(run.id, user.id, claimed[1]), tool_call_scope("stable-tool-call"):
            with pytest.raises(DurableSafetyError, match="WebTodoReplayProofMismatch"):
                await commit_todo_tool_operation(
                    db, name="create_todo", args={"title": "changed"},
                    perform=perform_changed)
    with web_run_scope(run.id, user.id, "stale-fence"), tool_call_scope("new-call"):
        with pytest.raises(DurableSafetyError, match="LostWebRequestFence"):
            await mutate()


@pytest.mark.asyncio
async def test_web_saver_thread_recreate_does_not_touch_telegram_checkpoint(pg_harness):
    _, new_user, _ = pg_harness
    user = await new_user()
    telegram = user.telegram_id

    async def answer(state):
        return {"messages": [AIMessage(content="safe synthetic answer")]}

    builder = StateGraph(AgentState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    dsn = get_settings().database_url_sync
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
        graph = builder.compile(checkpointer=saver)
        answer_text = await run_agent(graph, user.id, telegram, "", "synthetic Web question",
                                      thread_id=f"web:ordinary:{user.id}",
                                      turn_id=f"web-request:{uuid4()}", origin="web")
        assert answer_text == "safe synthetic answer"
        telegram_state = await graph.aget_state({"configurable": {"thread_id": f"user_{telegram}"}})
        assert not telegram_state.values
    async with AsyncPostgresSaver.from_conn_string(dsn) as restarted:
        graph = builder.compile(checkpointer=restarted)
        snapshot = await graph.aget_state({"configurable": {"thread_id": f"web:ordinary:{user.id}"}})
        assert snapshot.values["user_id"] == user.id
        assert sum(isinstance(message, HumanMessage) for message in snapshot.values["messages"]) == 1


@pytest.mark.asyncio
async def test_web_forget_same_user_telegram_history_preserves_other_user(pg_harness):
    factory, new_user, _ = pg_harness
    owner, stranger = await new_user(), await new_user()
    legacy = f"user_{owner.telegram_id}"
    web = f"web:ordinary:{owner.id}"
    source = HumanMessage(id="old-telegram-fact", content="Atlas project code is AX-47.")
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=owner.id, thread_id=legacy,
                                       messages=[source, AIMessage(id="old-answer", content="Noted.")])
        await sync_checkpoint_messages(db, user_id=stranger.id,
                                       thread_id=f"user_{stranger.telegram_id}",
                                       messages=[HumanMessage(id="stranger-fact", content=source.content),
                                                 AIMessage(id="stranger-answer", content="Noted.")])
        await db.commit()

    class Provider:
        model_id, provider_name, dimension = "synthetic-web-scope", "synthetic", 2
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Choice:
        def with_structured_output(self, schema):
            return self
        async def ainvoke(self, messages):
            return {"target_type": "conversation", "target_id": source.id,
                    "target_quote": "AX-47", "confidence": 0.99}

    result = await apply_memory_control(
        session_factory=factory, provider=Provider(),
        egress=EmbeddingEgressPolicy("http://127.0.0.1:8000/v1"), model=Choice(),
        user_id=owner.id, thread_id=web, operation="forget",
        target_query="Atlas AX-47", request="Forget the old Atlas code",
        current_message_id="web-forget-turn", history=[],
        additional_history_threads=(legacy,))
    assert result.changed
    async with factory() as db:
        old = await db.scalar(select(TranscriptProjection).where(
            TranscriptProjection.user_id == owner.id,
            TranscriptProjection.thread_id == legacy,
            TranscriptProjection.message_id == source.id))
        assert old.deleted_at is not None
        assert not await search_history(db, user_id=owner.id, thread_id=legacy, query="AX-47")
        assert await search_history(db, user_id=stranger.id,
                                    thread_id=f"user_{stranger.telegram_id}", query="AX-47")
        class Restored:
            async def aget_state(self, config):
                return SimpleNamespace(values={"user_id": owner.id,
                                               "messages": [source, AIMessage(
                                                   id="old-answer", content="Noted.")]})
        await rebuild_from_saver(db, graph=Restored(), user_id=owner.id, thread_id=legacy)
        await db.commit()
        assert not await search_history(db, user_id=owner.id, thread_id=legacy,
                                        query="AX-47")


@pytest.mark.asyncio
async def test_web_correction_supersedes_same_user_telegram_fact(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    legacy, web = f"user_{user.telegram_id}", f"web:ordinary:{user.id}"
    old = HumanMessage(id="prior-project-code", content="Atlas project code is AX-47.")
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=user.id, thread_id=legacy,
                                       messages=[old, AIMessage(id="prior-answer", content="Noted.")])
        await db.commit()

    class Provider:
        model_id, provider_name, dimension = "synthetic-web-correction", "synthetic", 2
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Choice:
        def with_structured_output(self, schema):
            return self
        async def ainvoke(self, messages):
            return {"target_type": "conversation", "target_id": old.id,
                    "target_quote": "AX-47", "confidence": 0.99,
                    "replacement_payload": ProjectFact(
                        project_key="Atlas", attribute="project code", value="AX-52").model_dump(),
                    "replacement_quote": "AX-52"}

    corrected = await apply_memory_control(
        session_factory=factory, provider=Provider(),
        egress=EmbeddingEgressPolicy("http://127.0.0.1:8000/v1"), model=Choice(),
        user_id=user.id, thread_id=web, operation="correct",
        target_query="Atlas AX-47", request="Atlas project code is AX-52, not AX-47.",
        current_message_id="web-correction-turn", history=[],
        additional_history_threads=(legacy,))
    assert (corrected.changed and corrected.supersedes_message_id == old.id
            and corrected.supersedes_thread_id == legacy)
    async with factory() as db:
        await sync_checkpoint_messages(db, user_id=user.id, thread_id=web,
                                       messages=[HumanMessage(
                                           id="web-correction-turn",
                                           content="Atlas project code is AX-52, not AX-47.",
                                           additional_kwargs={"supersedes_message_id": old.id,
                                                              "supersedes_thread_id": legacy}),
                                                 AIMessage(id="web-correction-answer", content="Corrected.")])
        await db.commit()
    async with factory() as db:
        assert not await search_history(db, user_id=user.id, thread_id=legacy,
                                        query="AX-47")
        assert await search_history(db, user_id=user.id, thread_id=web,
                                    query="AX-52")
        class Restored:
            async def aget_state(self, config):
                return SimpleNamespace(values={"user_id": user.id, "messages": [
                    HumanMessage(id="web-correction-turn",
                                 content="Atlas project code is AX-52, not AX-47.",
                                 additional_kwargs={"supersedes_message_id": old.id,
                                                    "supersedes_thread_id": legacy}),
                    AIMessage(id="web-correction-answer", content="Corrected.")]})
        await rebuild_from_saver(db, graph=Restored(), user_id=user.id, thread_id=web)
        await db.commit()
        assert not await search_history(db, user_id=user.id, thread_id=legacy,
                                        query="AX-47")
        provider = Provider()
        egress = EmbeddingEgressPolicy("http://127.0.0.1:8000/v1")
        await ensure_pgvector_projection(db)
        for scope in (legacy, web):
            await index_transcript_embeddings(db, user_id=user.id, thread_id=scope,
                                              provider=provider, egress=egress)
            await sync_transcript_halfvec(db, user_id=user.id, thread_id=scope,
                                          model_id=provider.model_id,
                                          dimension=provider.dimension)
        await db.commit()
        legacy_hits = await search_transcript_halfvec(
            db, user_id=user.id, thread_id=legacy, query="AX-47",
            provider=provider, egress=egress)
        assert all(hit.message_id != old.id for hit in legacy_hits)
        assert await search_transcript_halfvec(
            db, user_id=user.id, thread_id=web, query="AX-52",
            provider=provider, egress=egress)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Retired Web Agent worker; only Telegram Receipt runs Agent")
async def test_cancelled_web_worker_leaves_recoverable_lease(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    row, _ = await submit_run(factory, user_id=user.id, operation="chat",
                              idempotency_key=uuid4().hex, body={"text": "slow synthetic"})
    claim = await claim_one(factory, owner="worker-a", lease_seconds=120)
    from app.web import runs
    entered = asyncio.Event()

    async def blocked_execute(*args):
        entered.set()
        await asyncio.Event().wait()

    class Guard:
        async def __aenter__(self):
            return None
        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(runs, "_execute", blocked_execute)
    lock_scopes = []
    def capture_guard(*args):
        lock_scopes.append(args[1:])
        return Guard()
    monkeypatch.setattr(runs, "thread_invocation_guard", capture_guard)
    worker = WebRunWorker(factory, get_settings().database_url, lease_seconds=120)
    task = asyncio.create_task(worker.process(*claim))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lock_scopes == [("workflow", f"web:workflow:{user.id}")]
    async with factory() as db:
        active = await db.get(WebRequest, row.id)
        assert active.state == "RUNNING" and active.fence_token == claim[1]
        active.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    recovered = await claim_one(factory, owner="worker-b", lease_seconds=120)
    assert recovered and recovered[0] == row.id and recovered[1] != claim[1]


@pytest.mark.asyncio
@pytest.mark.skip(reason="Retired Web approval API; Web auth has separate current tests")
async def test_web_api_login_csrf_sessions_and_cross_user_run_visibility(pg_harness, monkeypatch):
    from app.web import api
    from app.agent import official_runtime, workflow_runtime
    from app.agent.actions import ActionProposal, RiskLevel
    factory, new_user, _ = pg_harness
    a, b = await new_user(), await new_user()
    async with factory() as db:
        await bind_existing_user(db, a.id, "unique password for A")
        await bind_existing_user(db, b.id, "unique password for B")
    settings = SimpleNamespace(web_public_origin="https://workbench.test",
                               web_session_hours=24, secret_key="test-only-secret",
                               default_timezone="UTC", google_calendar_enabled=False,
                               google_calendar_owner_telegram_id="", gmail_enabled=False,
                               gmail_owner_telegram_id="", memory_enabled=False)
    monkeypatch.setattr(api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    proposal = ActionProposal(
        action_id="action-a", domain="todo", operation="delete_todo",
        args={"user_id": a.id, "todo_id": "synthetic-a",
              "approved_todo_version": 1},
        human_readable_preview="safe preview", reason="owner-bound fixture",
        risk_level=RiskLevel.DESTRUCTIVE)

    class FakeOfficialGraph:
        async def aget_state(self, config):
            if config["configurable"]["thread_id"] != f"official-v1:web:ordinary:{a.id}":
                return SimpleNamespace(values={}, next=(), interrupts=(), config={})
            return SimpleNamespace(
                values={"user_id": a.id, "prepared_actions": {
                    "delete-a": {"proposal": proposal.model_dump(mode="json")}},
                    "messages": []},
                next=("tools",),
                interrupts=(SimpleNamespace(id="interrupt-a", value={
                    "type": "tool_approval", "tool_call_id": "delete-a"}),),
                config={"configurable": {"checkpoint_id": "ck1"}})

    monkeypatch.setattr(official_runtime, "get_graph",
                        AsyncMock(return_value=FakeOfficialGraph()))
    monkeypatch.setattr(workflow_runtime, "get_productivity_graph",
                        AsyncMock(return_value=SimpleNamespace(
                            aget_state=AsyncMock(return_value=None))))
    application = FastAPI()
    application.include_router(api.router)
    application.state.web_ready = True
    application.state.web_worker = SimpleNamespace(wake=lambda: None)
    headers = {"Origin": "https://workbench.test"}
    transport = ASGITransport(app=application)

    async def signed_in(password):
        client = AsyncClient(transport=transport, base_url="https://workbench.test")
        bootstrap = await client.get("/api/v3/auth/bootstrap", headers=headers)
        assert bootstrap.status_code == 200
        logged = await client.post("/api/v3/auth/login", json={"password": password},
                                   headers={**headers, "X-CSRF-Token": bootstrap.json()["csrf_token"]})
        assert logged.status_code == 204
        me = await client.get("/api/v3/me")
        assert me.status_code == 200 and "user_id" not in me.json()
        return client, me.json()["csrf_token"]

    client_a, csrf_a = await signed_in("unique password for A")
    client_b, csrf_b = await signed_in("unique password for B")
    try:
        assert (await client_a.post("/api/v3/chat/runs", json={"text": "private A"},
                                    headers={**headers, "X-CSRF-Token": csrf_a})).status_code == 404
        assert (await client_a.get("/api/v3/chat/history")).status_code == 404
        # Historical Web-origin pending rows must not become approvable.
        row, _ = await submit_run(factory, user_id=a.id, operation="chat",
                                  idempotency_key=uuid4().hex, body={"text": "historical"})
        run_id = row.id
        async with factory() as db:
            row = await db.get(WebRequest, run_id, with_for_update=True)
            row.thread_id = f"official-v1:web:ordinary:{a.id}"
            row.state = "WAITING_APPROVAL"
            await db.commit()
        assert (await client_a.get("/api/v3/approvals")).json()["items"] == []
        assert (await client_b.get("/api/v3/approvals")).json()["items"] == []
        assert (await client_b.get(f"/api/v3/approval-runs/{run_id}")).status_code == 404
        assert (await client_a.get(f"/api/v3/approval-runs/{run_id}")).status_code == 404
        foreign_decision = await client_b.post(
            "/api/v3/approvals/action-a/decision",
            json={"run_id": run_id, "decision": "approve", "checkpoint_version": "ck1"},
            headers={**headers, "X-CSRF-Token": csrf_b,
                     "Idempotency-Key": uuid4().hex})
        assert foreign_decision.status_code == 404
        logout = await client_a.post("/api/v3/auth/logout", headers={**headers,
                                       "X-CSRF-Token": csrf_a})
        assert logout.status_code == 204
        assert (await client_a.get("/api/v3/me")).status_code == 401
        assert (await client_b.get("/api/v3/me")).status_code == 200
    finally:
        await client_a.aclose()
        await client_b.aclose()
