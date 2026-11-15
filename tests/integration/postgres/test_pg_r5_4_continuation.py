"""R5.4: real PG18/Saver process-boundary continuation and local atomic child."""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar_continuation import CalendarContinuationRuntime
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.planner import RoutedRequest
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.config import get_settings
from app.db import crud
from app.db.models import (ActionExecution, CalendarDependentContinuation,
                           TelegramThreadLease)
from tests.test_r5_2_create_reconciliation import FakeCalendar

pytestmark = pytest.mark.integration


def _runtime(calendar, factory, saver):
    coordinator = CalendarCreateCoordinator(calendar, factory)
    executor = ActionExecutor(calendar, SqlAlchemyActionLedger(factory), factory,
                              calendar_create_coordinator=coordinator)
    graph = build_productivity_graph(calendar, executor, saver)
    return graph, coordinator


@pytest.mark.asyncio
async def test_late_parent_confirmation_survives_real_saver_and_runtime_restart(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    thread = f"r54-pg-{uuid4().hex}"
    calendar = FakeCalendar("accepted_then_timeout")
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="Synthetic review", todo_title="Write synthetic report",
                          date_reference="tomorrow", day_part="afternoon",
                          duration_minutes=60)
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        graph, coordinator = _runtime(calendar, factory, saver)
        first = await start_workflow(graph, user.id, "Schedule and create Todo", "UTC",
                                     thread, route=route)
        assert first["__interrupt__"][0].value["kind"] == "action_proposal"
        original_create = calendar.create_event_prepared

        async def accepted_then_scope_lost(*args):
            try:
                return await original_create(*args)
            finally:
                calendar.scope_available = False
                calendar.hidden_reads = 3

        calendar.create_event_prepared = accepted_then_scope_lost
        # Simulate a process dying after Saver persisted the wait interrupt
        # but before the normal caller could bind that newer checkpoint ID.
        approved = await graph.ainvoke(Command(resume={"decision": "approve"}),
                                      {"configurable": {"thread_id": thread}})
        assert approved["__interrupt__"][0].value["kind"] == "calendar_confirmation"
        action_id = approved["proposal"]["action_id"]
        async with factory() as db:
            locator = await db.get(CalendarDependentContinuation, action_id)
            assert locator and locator.checkpoint_id and locator.state == "WAITING"
            assert not await crud.list_todos(db, user.id)
        await start_workflow(graph, user.id, "List Todos while Calendar settles",
                             "UTC", thread,
                             route=RoutedRequest(intent="todo", todo_operation="list"))

    # Different Saver connection, graph object, coordinator and worker.
    calendar.scope_available = True
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        graph, coordinator = _runtime(calendar, factory, saver)
        assert (await coordinator.recover(action_id)).status == ActionStatus.EXECUTED
        # This is the real confirmed-before-wakeup crash boundary.
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        graph, _ = _runtime(calendar, factory, saver)
        crashed_worker = CalendarContinuationRuntime(graph, factory)
        await crashed_worker.preflight()
        # A callback receipt already owns this exact R4 graph/thread lease.
        # The continuation cannot invoke the same Saver thread concurrently.
        async with factory() as db:
            db.add(TelegramThreadLease(
                graph_kind="workflow", thread_id=thread,
                receipt_id=str(uuid4()), lease_owner="active-callback",
                fence_token=str(uuid4()),
                lease_until=datetime.now(timezone.utc) + timedelta(seconds=120)))
            await db.commit()
        assert not await crashed_worker.recover(action_id)
        async with factory() as db:
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": thread})
            await db.delete(lease)
            await db.commit()
        old_claim = await crashed_worker.store.claim(action_id, "crashed")
        assert old_claim is not None
        old_token = old_claim[0]
        async with factory() as db:
            locator = await db.get(CalendarDependentContinuation, action_id,
                                   with_for_update=True)
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": thread},
                with_for_update=True)
            locator.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        assert not await crashed_worker.store.finish(action_id, old_token, completed=True)
        # A second process reclaims; the first fence can no longer finalize.
        worker_a = CalendarContinuationRuntime(graph, factory)
        second_graph, _ = _runtime(calendar, factory, saver)
        worker_b = CalendarContinuationRuntime(second_graph, factory)
        outcomes = await asyncio.gather(worker_a.recover(action_id), worker_b.recover(action_id))
        assert outcomes.count(True) == 1
        assert not await crashed_worker.store.finish(action_id, old_token, completed=True)
        assert not await worker_a.recover(action_id)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert not snapshot.next and snapshot.values["proposal"]["action_id"] == action_id
        assert len(calendar.creates) == 1
        async with factory() as db:
            todos = await crud.list_todos(db, user.id)
            child = await db.get(ActionExecution, action_id + ":todo")
            locator = await db.get(CalendarDependentContinuation, action_id)
            assert len(todos) == 1 and child.status == "executed"
            assert todos[0].due_date is None
            assert child.external_id == todos[0].id
            assert locator.state == "COMPLETED"
        await saver.adelete_thread(thread)


@pytest.mark.asyncio
async def test_child_todo_and_ledger_commit_atomically_across_crash_and_race(pg_harness,
                                                                            monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    proposal = ActionProposal(
        action_id=f"{uuid4().hex}:todo", domain="todo", operation="create_todo",
        args={"user_id": user.id, "title": "R5.4 atomic child"},
        human_readable_preview="Synthetic", reason="E4",
        risk_level=RiskLevel.INTERNAL_LOW_RISK)
    ledger = SqlAlchemyActionLedger(factory)
    original_create = crud.create_todo

    async def crash_before_commit(*args, **kwargs):
        await original_create(*args, **kwargs)
        raise RuntimeError("synthetic crash before commit")

    monkeypatch.setattr(crud, "create_todo", crash_before_commit)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        await ledger.create_todo_atomically(proposal)
    async with factory() as db:
        assert await db.get(ActionExecution, proposal.action_id) is None
        assert not await crud.list_todos(db, user.id)
    monkeypatch.setattr(crud, "create_todo", original_create)
    results = await asyncio.gather(
        SqlAlchemyActionLedger(factory).create_todo_atomically(proposal),
        SqlAlchemyActionLedger(factory).create_todo_atomically(proposal))
    assert all(result.status == ActionStatus.EXECUTED for result in results)
    # Crash after commit but before graph checkpoint: replay reads the same proof.
    replay = await ledger.create_todo_atomically(proposal)
    assert replay.status == ActionStatus.EXECUTED
    async with factory() as db:
        todos = await crud.list_todos(db, user.id)
        row = await db.get(ActionExecution, proposal.action_id)
        assert len(todos) == 1 and row.status == "executed"
        assert row.external_id == todos[0].id == replay.result["id"]


@pytest.mark.asyncio
async def test_graph_crash_during_child_restarts_from_real_saver_without_duplicate(pg_harness,
                                                                                  monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    thread = f"r54-crash-{uuid4().hex}"
    calendar = FakeCalendar("accepted_then_timeout")
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="Synthetic review", todo_title="Crash-window Todo",
                          date_reference="tomorrow", day_part="afternoon",
                          duration_minutes=60,
                          todo_due_date_reference="day_after_tomorrow")
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        await saver.setup()
        graph, coordinator = _runtime(calendar, factory, saver)
        await start_workflow(graph, user.id, "Schedule and create Todo", "UTC",
                             thread, route=route)
        original_create = calendar.create_event_prepared

        async def accepted_then_scope_lost(*args):
            try:
                return await original_create(*args)
            finally:
                calendar.scope_available = False
                calendar.hidden_reads = 3

        calendar.create_event_prepared = accepted_then_scope_lost
        waiting = await resume_workflow(graph, thread, {"decision": "approve"})
        action_id = waiting["proposal"]["action_id"]

    calendar.scope_available = True
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        graph, coordinator = _runtime(calendar, factory, saver)
        assert (await coordinator.recover(action_id)).status == ActionStatus.EXECUTED
        original_todo = crud.create_todo

        async def crash_after_todo_flush(*args, **kwargs):
            await original_todo(*args, **kwargs)
            raise RuntimeError("synthetic child crash before atomic commit")

        monkeypatch.setattr(crud, "create_todo", crash_after_todo_flush)
        with pytest.raises(RuntimeError, match="synthetic child crash"):
            await CalendarContinuationRuntime(graph, factory).recover(action_id)
        monkeypatch.setattr(crud, "create_todo", original_todo)
        async with factory() as db:
            assert not await crud.list_todos(db, user.id)
            assert await db.get(ActionExecution, action_id + ":todo") is None
            locator = await db.get(CalendarDependentContinuation, action_id,
                                   with_for_update=True)
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": thread},
                with_for_update=True)
            locator.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        graph, _ = _runtime(calendar, factory, saver)
        original_execute_child = ActionExecutor.execute_child_todo

        async def crash_after_atomic_commit(self, proposal):
            await original_execute_child(self, proposal)
            raise RuntimeError("synthetic crash after atomic child commit")

        monkeypatch.setattr(ActionExecutor, "execute_child_todo", crash_after_atomic_commit)
        with pytest.raises(RuntimeError, match="after atomic child commit"):
            await CalendarContinuationRuntime(graph, factory).recover(action_id)
        monkeypatch.setattr(ActionExecutor, "execute_child_todo", original_execute_child)
        async with factory() as db:
            todos = await crud.list_todos(db, user.id)
            child = await db.get(ActionExecution, action_id + ":todo")
            assert len(todos) == 1 and child.status == "executed"
            locator = await db.get(CalendarDependentContinuation, action_id,
                                   with_for_update=True)
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "workflow", "thread_id": thread},
                with_for_update=True)
            locator.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
    async with AsyncPostgresSaver.from_conn_string(get_settings().database_url_sync) as saver:
        graph, _ = _runtime(calendar, factory, saver)
        assert await CalendarContinuationRuntime(graph, factory).recover(action_id)
        async with factory() as db:
            assert len(await crud.list_todos(db, user.id)) == 1
        assert len(calendar.creates) == 1
        await saver.adelete_thread(thread)
