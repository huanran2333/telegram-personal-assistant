"""Real PostgreSQL ledger/Todo integration; Gmail is a boundary fake here."""
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.agent.actions import ActionStatus
from app.agent.email_workflow import EmailActionExtraction
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.gmail import GmailMessage
from app.agent.planner import RoutedRequest
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.db.models import ActionExecution, Todo


pytestmark = pytest.mark.integration


class Gmail:
    def __init__(self):
        self.message = GmailMessage(message_id="pg-message-1", thread_id="pg-thread-1",
                                    sender="Sender <sender@example.org>", subject="Prepare a paper",
                                    body="Prepare a paper by 2026-10-10.",
                                    timestamp=datetime(2026, 10, 2, tzinfo=timezone.utc))

    async def search(self, query, limit=5):
        return [self.message]


class LLM:
    def with_structured_output(self, schema):
        assert schema is EmailActionExtraction
        return self

    async def ainvoke(self, messages):
        return {"operation": "create_todo", "message_id": "pg-message-1",
                "thread_id": "pg-thread-1", "evidence_quote": "Prepare a paper",
                "title": "Prepare a paper", "due_date": "2026-10-10T20:00:00+08:00"}


class Calendar:
    async def availability(self, *args, **kwargs):
        raise AssertionError("Email→Todo must not call Calendar")


@pytest.mark.asyncio
async def test_pg_email_todo_requires_approval_and_ledger_deduplicates(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = Calendar()
    executor = ActionExecutor(calendar, SqlAlchemyActionLedger(factory), factory)
    graph = build_productivity_graph(calendar, executor, InMemorySaver(), gmail=Gmail(),
                                     gmail_owner_telegram_id="pg-owner", email_llm=LLM())
    route = RoutedRequest(intent="email_to_todo", email_query="paper")
    result = await start_workflow(graph, user.id, "Create a Todo from this email", "Asia/Shanghai",
                                  "pg-gmail-action", route=route, telegram_id="pg-owner")
    assert result["__interrupt__"][0].value["kind"] == "action_proposal"
    state = await graph.aget_state({"configurable": {"thread_id": "pg-gmail-action"}})
    proposal = state.values["proposal"]
    assert proposal["operation"] == "create_todo_from_email"
    async with factory() as db:
        assert (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all() == []
    approved = await resume_workflow(graph, "pg-gmail-action", {"decision": "approve"})
    assert approved["proposal"]["status"] == ActionStatus.EXECUTED.value
    async with factory() as db:
        todos = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
        ledger = await db.get(ActionExecution, proposal["action_id"])
        assert len(todos) == 1
        assert todos[0].description == "Gmail message pg-message-1; thread pg-thread-1"
        assert todos[0].due_date.date().isoformat() == "2026-10-10"
        assert ledger.status == "executed"
        assert ledger.result["source_message_id"] == "pg-message-1"
        assert ledger.result["source_thread_id"] == "pg-thread-1"
    # Duplicate approval-equivalent execution is prevented by the real unique ledger.
    from app.agent.actions import ActionProposal
    repeated = await executor.execute(ActionProposal.model_validate(proposal))
    assert repeated.status == ActionStatus.EXECUTED
    async with factory() as db:
        assert len((await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()) == 1


@pytest.mark.asyncio
async def test_email_todo_does_not_split_business_commit_from_ledger(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    from app.agent.actions import ActionProposal, RiskLevel
    proposal = ActionProposal(
        domain="todo", operation="create_todo_from_email",
        args={"user_id": user.id, "title": "itest atomic email todo",
              "source_message_id": "synthetic-message", "source_thread_id": "synthetic-thread"},
        human_readable_preview="Create test Todo", reason="fault injection",
        risk_level=RiskLevel.INTERNAL_LOW_RISK,
    )
    ledger = SqlAlchemyActionLedger(factory)

    async def fail_split_complete(*_args, **_kwargs):
        raise RuntimeError("injected split ledger failure")

    monkeypatch.setattr(ledger, "complete", fail_split_complete)
    result = await ActionExecutor(None, ledger, factory).execute(proposal)
    async with factory() as db:
        rows = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
        row = await db.get(ActionExecution, proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert len(rows) == 1
    assert row.status == "executed" and row.result["id"] == rows[0].id
    assert row.result["source_message_id"] == "synthetic-message"


@pytest.mark.asyncio
async def test_email_todo_crash_before_commit_rolls_back_and_retries(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    from app.agent.actions import ActionProposal, RiskLevel
    from app.db import crud
    proposal = ActionProposal(
        domain="todo", operation="create_todo_from_email",
        args={"user_id": user.id, "title": "itest rollback email todo"},
        human_readable_preview="Create test Todo", reason="fault injection",
        risk_level=RiskLevel.INTERNAL_LOW_RISK,
    )
    ledger = SqlAlchemyActionLedger(factory)
    original_create = crud.create_todo

    async def crash_after_flush(*args, **kwargs):
        await original_create(*args, **kwargs)
        raise RuntimeError("injected crash after Todo flush")

    monkeypatch.setattr(crud, "create_todo", crash_after_flush)
    with pytest.raises(RuntimeError, match="injected crash"):
        await ActionExecutor(None, ledger, factory).execute(proposal)
    async with factory() as db:
        assert (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all() == []
        assert await db.get(ActionExecution, proposal.action_id) is None
    monkeypatch.setattr(crud, "create_todo", original_create)
    result = await ActionExecutor(None, ledger, factory).execute(proposal)
    assert result.status == ActionStatus.EXECUTED
    replay = await ActionExecutor(None, ledger, factory).execute(proposal)
    assert replay.status == ActionStatus.EXECUTED and replay.result == result.result
    async with factory() as db:
        assert len((await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()) == 1


@pytest.mark.asyncio
async def test_email_todo_response_lost_after_commit_replays_terminal_proof(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    from app.agent.actions import ActionProposal, RiskLevel
    proposal = ActionProposal(
        domain="todo", operation="create_todo_from_email",
        args={"user_id": user.id, "title": "itest lost response email todo"},
        human_readable_preview="Create test Todo", reason="response loss",
        risk_level=RiskLevel.INTERNAL_LOW_RISK,
    )
    ledger = SqlAlchemyActionLedger(factory)
    original_execute = ledger.create_todo_atomically

    async def lose_response_after_commit(action):
        await original_execute(action)
        raise RuntimeError("injected response loss after commit")

    monkeypatch.setattr(ledger, "create_todo_atomically", lose_response_after_commit)
    executor = ActionExecutor(None, ledger, factory)
    with pytest.raises(RuntimeError, match="response loss"):
        await executor.execute(proposal)
    monkeypatch.setattr(ledger, "create_todo_atomically", original_execute)
    replay = await executor.execute(proposal)
    assert replay.status == ActionStatus.EXECUTED
    async with factory() as db:
        rows = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
        ledger_row = await db.get(ActionExecution, proposal.action_id)
    assert len(rows) == 1 and ledger_row.status == "executed"
    assert replay.result["id"] == rows[0].id == ledger_row.result["id"]
