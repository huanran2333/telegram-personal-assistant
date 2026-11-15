"""Executable R5.4 closure: late Calendar confirmation resumes approved Todo."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.actions import ActionStatus
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.calendar_continuation import CalendarContinuationRuntime
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.planner import RoutedRequest
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.db import crud
from app.db.models import ActionExecution, Base, CalendarActionEvidence
from tests.test_r5_2_create_reconciliation import FakeCalendar


@pytest.mark.asyncio
async def test_late_calendar_confirmation_resumes_approved_cross_domain_todo():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as db:
            user = await crud.get_or_create_user(db, "r54-acceptance-gap")
            await db.commit()
            user_id = user.id
        calendar = FakeCalendar("accepted_then_timeout")
        coordinator = CalendarCreateCoordinator(calendar, factory)
        executor = ActionExecutor(calendar, SqlAlchemyActionLedger(factory), factory,
                                  calendar_create_coordinator=coordinator)
        graph = build_productivity_graph(calendar, executor, InMemorySaver())
        route = RoutedRequest(
            intent="cross_domain", todo_operation="create", title="Synthetic review",
            todo_title="Write synthetic report", date_reference="tomorrow",
            day_part="afternoon", duration_minutes=60,
            todo_due_date_reference="day_after_tomorrow")
        first = await start_workflow(graph, user_id, "Schedule and create Todo", "UTC",
                                     "r54-cross-domain-gap", route=route)
        assert first["__interrupt__"][0].value["kind"] == "action_proposal"
        original_create = calendar.create_event_prepared

        async def accepted_then_scope_lost(*args):
            try:
                return await original_create(*args)
            finally:
                calendar.scope_available = False
                calendar.hidden_reads = 3

        calendar.create_event_prepared = accepted_then_scope_lost
        approved = await resume_workflow(graph, "r54-cross-domain-gap", {"decision": "approve"})
        action_id = approved["proposal"]["action_id"]
        assert approved["__interrupt__"][0].value["kind"] == "calendar_confirmation"
        async with factory() as db:
            evidence = await db.get(CalendarActionEvidence, action_id)
            ledger = await db.get(ActionExecution, action_id)
            assert evidence.outcome == "UNCERTAIN" and ledger.status == "executing"
            assert await crud.list_todos(db, user_id) == []
        # The user may start another request on the same graph thread while
        # remote Calendar confirmation is pending. The locator retains the
        # historical waiting checkpoint rather than treating latest as auth.
        await start_workflow(graph, user_id, "List my Todos", "UTC",
                             "r54-cross-domain-gap",
                             route=RoutedRequest(intent="todo", todo_operation="list"))
        calendar.scope_available = True
        assert (await coordinator.recover(action_id)).status == ActionStatus.EXECUTED
        assert await CalendarContinuationRuntime(graph, factory).recover_due() == 1
        assert len(calendar.creates) == 1
        async with factory() as db:
            todos = await crud.list_todos(db, user_id)
            child = await db.get(ActionExecution, action_id + ":todo")
            assert len(todos) == 1 and child.status == "executed"
            assert child.external_id == todos[0].id
    finally:
        await engine.dispose()
