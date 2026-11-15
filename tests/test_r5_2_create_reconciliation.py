"""R5.2 E3: no real Google/Telegram side effects in this module."""
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar import CalendarError, MCPCalendarAdapter, UnavailableCalendarAdapter
from app.agent.calendar_action_evidence import CalendarEvidenceIntent
from app.agent.calendar_conditional import CalendarEventSnapshot, CalendarProviderError
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.calendar_continuation import CalendarContinuationRuntime
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.planner import RoutedRequest
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.db import crud
from app.db.models import ActionExecution, CalendarActionEvidence, Todo


class FakeCalendar:
    def __init__(self, mode="success"):
        self.mode = mode
        self.events = {}
        self.creates = []
        self.reads = 0
        self.hidden_reads = 0
        self.scope_available = True

    async def resolve_create_scope(self, calendar_id):
        if not self.scope_available:
            raise CalendarError("calendar_scope_unavailable")
        return "normal", "owner@example.test"

    async def availability(self, start, end, timezone):
        return {"calendars": {"primary": {"busy": []}}}

    async def search_events(self, query, start=None, end=None):
        return []

    async def get_event(self, calendar_id, event_id):
        self.reads += 1
        if self.hidden_reads:
            self.hidden_reads -= 1
            raise CalendarError("not_found_or_inaccessible")
        if event_id not in self.events:
            raise CalendarError("not_found_or_inaccessible")
        return CalendarEventSnapshot.model_validate(self.events[event_id])

    async def create_event_prepared(self, event_id, body, marker):
        self.creates.append((event_id, body.copy(), marker))
        event = {
            "id": event_id, "etag": '"v1"', "status": "confirmed",
            "summary": body["summary"],
            "start": {"dateTime": body["start"]},
            "end": {"dateTime": body["end"]},
            "extendedProperties": {"private": {"assistant_action": marker}},
        }
        if self.mode == "similarity_precheck":
            raise CalendarProviderError("similarity_precheck")
        if self.mode == "conflict":
            if event_id not in self.events:
                event["summary"] = "Different resource"
                self.events[event_id] = event
            raise CalendarProviderError("resource_conflict", 409)
        self.events[event_id] = event
        if self.mode == "accepted_then_timeout":
            raise CalendarProviderError("remote_outcome_uncertain")
        return {"id": event_id, "event": event}


async def _proposal(factory, *, action_id="r5-create-1", title="Harmless synthetic"):
    async with factory() as db:
        user = await crud.get_or_create_user(db, "r5-e3-" + action_id)
        await db.commit()
        user_id = user.id
    return ActionProposal(
        action_id=action_id, domain="calendar", operation="create_event",
        args={"user_id": user_id, "calendarId": "primary", "title": title,
              "start": "2026-10-12T15:00:00+08:00",
              "end": "2026-10-12T15:30:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="Synthetic create", reason="test",
        risk_level=RiskLevel.EXTERNAL_WRITE)


async def _rows(factory, action_id):
    async with factory() as db:
        return await db.get(CalendarActionEvidence, action_id), await db.get(ActionExecution, action_id)


@pytest.mark.asyncio
async def test_create_confirms_atomically_and_duplicate_uses_same_resource(session_factory):
    calendar = FakeCalendar()
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory)
    first = await service.execute(proposal)
    second = await service.execute(proposal)
    assert first.status == second.status == ActionStatus.EXECUTED
    assert first.result["id"] == second.result["id"]
    assert len(calendar.creates) == 1
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert (evidence.outcome, ledger.status, ledger.external_id) == (
        "CONFIRMED", "executed", first.result["id"])
    assert evidence.immutable_body["calendarId"] == "owner@example.test"
    assert calendar.creates[0][2] == proposal.action_id


@pytest.mark.asyncio
async def test_accepted_timeout_and_409_match_reconcile_without_second_insert(session_factory):
    for index, mode in enumerate(("accepted_then_timeout", "conflict")):
        calendar = FakeCalendar(mode)
        service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
        proposal = await _proposal(session_factory, action_id=f"r5-timeout-{index}")
        if mode == "conflict":
            intent = CalendarEvidenceIntent.from_approved_proposal(
                proposal, "normal", "owner@example.test")
            calendar.events[intent.event_id] = {
                "id": intent.event_id, "etag": '"v1"', "status": "confirmed",
                "summary": proposal.args["title"],
                "start": {"dateTime": proposal.args["start"]},
                "end": {"dateTime": proposal.args["end"]},
                "extendedProperties": {"private": {"assistant_action": proposal.action_id}},
            }
        result = await service.execute(proposal)
        assert result.status == ActionStatus.EXECUTED
        assert len(calendar.creates) == 1
        evidence, ledger = await _rows(session_factory, proposal.action_id)
        assert evidence.confirmation_source == "exact_readback"
        assert ledger.status == "executed"


@pytest.mark.asyncio
async def test_409_mismatch_parks_and_never_recreates(session_factory):
    calendar = FakeCalendar("conflict")
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory, action_id="r5-conflict")
    result = await service.execute(proposal)
    assert result.status == ActionStatus.EXECUTING
    assert result.result["remote_state"] == "PARKED_REVIEW"
    assert len(calendar.creates) == 1
    again = await service.execute(proposal)
    assert again.status == ActionStatus.EXECUTING and len(calendar.creates) == 1
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert evidence.outcome == "PARKED_REVIEW" and ledger.status == "executing"


@pytest.mark.asyncio
async def test_preinsert_similarity_rejection_is_definite(session_factory):
    calendar = FakeCalendar("similarity_precheck")
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory, action_id="r5-precheck")
    result = await service.execute(proposal)
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.FAILED
    assert evidence.outcome == "FAILED_DEFINITE" and ledger.status == "failed"
    assert not calendar.events


@pytest.mark.asyncio
async def test_retry_similarity_precheck_cannot_erase_first_attempt_uncertainty(session_factory):
    class LateVisibleCalendar(FakeCalendar):
        async def create_event_prepared(self, event_id, body, marker):
            if self.creates:
                raise CalendarProviderError("similarity_precheck")
            return await super().create_event_prepared(event_id, body, marker)

    calendar = LateVisibleCalendar("accepted_then_timeout")
    calendar.hidden_reads = 3
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory, action_id="r5-retry-precheck")
    result = await service.execute(proposal)
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert evidence.outcome == "CONFIRMED" and ledger.status == "executed"
    assert evidence.attempt_no == 2 and len(calendar.creates) == 1


@pytest.mark.asyncio
async def test_late_visibility_and_inaccessible_calendar_do_not_assume_absence(session_factory):
    visible = FakeCalendar("accepted_then_timeout")
    visible.hidden_reads = 1
    first_service = CalendarCreateCoordinator(visible, session_factory, sleep=lambda _: _noop())
    first = await _proposal(session_factory, action_id="r5-late-visible")
    assert (await first_service.execute(first)).status == ActionStatus.EXECUTED
    assert visible.reads == 2 and len(visible.creates) == 1

    inaccessible = FakeCalendar("accepted_then_timeout")
    inaccessible.hidden_reads = 4
    second_service = CalendarCreateCoordinator(inaccessible, session_factory,
                                               sleep=lambda _: _noop())
    second = await _proposal(session_factory, action_id="r5-inaccessible")
    # Scope loss occurs after the create was accepted, before 404 validation.
    original = inaccessible.create_event_prepared

    async def create_then_lose_scope(*args):
        try:
            return await original(*args)
        finally:
            inaccessible.scope_available = False

    inaccessible.create_event_prepared = create_then_lose_scope
    result = await second_service.execute(second)
    evidence, ledger = await _rows(session_factory, second.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert evidence.outcome == "UNCERTAIN" and ledger.status == "executing"
    assert len(inaccessible.creates) == 1


@pytest.mark.asyncio
async def test_crash_after_prepared_and_after_armed_restarts_reconcile_first(session_factory):
    calendar = FakeCalendar()
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory, action_id="r5-crash")
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "owner@example.test")
    body = {"account": "normal", "calendarId": "owner@example.test",
            "summary": proposal.args["title"], "start": proposal.args["start"],
            "end": proposal.args["end"], "timeZone": proposal.args["timeZone"]}
    await service.store.prepare_create(intent, body, proposal.action_id)
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert evidence.outcome == "NOT_STARTED" and ledger.status == "executing"
    token = await service.store.arm(proposal.action_id, "crashed-worker")
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    restarted = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    # Armed, but no fake provider call occurred. Recovery checks exact ID first,
    # then retries the same immutable body/ID after bounded absent observations.
    result = await restarted.recover(proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert calendar.reads >= 3 and len(calendar.creates) == 1
    assert calendar.creates[0][0] == intent.event_id
    assert not await service.store.finalize_create(proposal.action_id, token, "CONFIRMED",
                                                    confirmation_source="provider_response")


@pytest.mark.asyncio
async def test_restart_after_not_started_commit_uses_prepared_identity(session_factory):
    calendar = FakeCalendar()
    proposal = await _proposal(session_factory, action_id="r5-prepared-only")
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "owner@example.test")
    body = {"account": "normal", "calendarId": "owner@example.test",
            "summary": proposal.args["title"], "start": proposal.args["start"],
            "end": proposal.args["end"], "timeZone": proposal.args["timeZone"]}
    first = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    await first.store.prepare_create(intent, body, proposal.action_id)
    assert calendar.creates == []
    fresh = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    result = await fresh.recover(proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert len(calendar.creates) == 1 and calendar.creates[0][0] == intent.event_id


@pytest.mark.asyncio
async def test_calendar_discovery_outage_cannot_downgrade_prepared_action(session_factory):
    proposal = await _proposal(session_factory, action_id="r5-discovery-outage")
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "owner@example.test")
    body = {"account": "normal", "calendarId": "owner@example.test",
            "summary": proposal.args["title"], "start": proposal.args["start"],
            "end": proposal.args["end"], "timeZone": proposal.args["timeZone"]}
    first = CalendarCreateCoordinator(FakeCalendar(), session_factory, sleep=lambda _: _noop())
    await first.store.prepare_create(intent, body, proposal.action_id)
    unavailable = CalendarCreateCoordinator(UnavailableCalendarAdapter("test outage"),
                                            session_factory, sleep=lambda _: _noop())
    result = await unavailable.execute(proposal)
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert evidence.outcome == "UNCERTAIN" and ledger.status == "executing"


@pytest.mark.asyncio
async def test_remote_success_then_local_crash_reconciles_without_second_create(session_factory):
    calendar = FakeCalendar()
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory, action_id="r5-local-crash")
    original_finalize = service.store.finalize_create

    async def crash_before_commit(*args, **kwargs):
        raise RuntimeError("synthetic crash before local terminal commit")

    service.store.finalize_create = crash_before_commit
    with pytest.raises(RuntimeError, match="synthetic crash"):
        await service.execute(proposal)
    evidence, ledger = await _rows(session_factory, proposal.action_id)
    assert evidence.outcome == "IN_FLIGHT" and ledger.status == "executing"
    assert len(calendar.creates) == 1
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    restarted = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    assert (await restarted.recover(proposal.action_id)).status == ActionStatus.EXECUTED
    assert len(calendar.creates) == 1
    assert restarted.store.finalize_create is not original_finalize


@pytest.mark.asyncio
async def test_changed_args_never_reuse_approved_action(session_factory):
    calendar = FakeCalendar()
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    proposal = await _proposal(session_factory, action_id="r5-changed")
    assert (await service.execute(proposal)).status == ActionStatus.EXECUTED
    changed = proposal.model_copy(deep=True)
    changed.args["title"] = "Unauthorized changed title"
    result = await service.execute(changed)
    assert result.status == ActionStatus.EXECUTING
    assert result.result["remote_state"] == "APPROVAL_MISMATCH"
    assert len(calendar.creates) == 1


@pytest.mark.asyncio
async def test_cross_domain_child_waits_for_calendar_confirmation_and_reply_is_honest(session_factory):
    calendar = FakeCalendar("accepted_then_timeout")
    coordinator = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    executor = ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory,
                              calendar_create_coordinator=coordinator)
    graph = build_productivity_graph(calendar, executor, InMemorySaver())
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "r5-cross-domain")
        await db.commit()
        user_id = user.id
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="Synthetic review", todo_title="Write synthetic report",
                          date_reference="tomorrow", day_part="afternoon",
                          duration_minutes=60,
                          todo_due_date_reference="day_after_tomorrow")
    thread = "r5-cross-domain-uncertain"
    first = await start_workflow(graph, user_id, "Check tomorrow and schedule report work",
                                 "UTC", thread, route=route)
    assert first["__interrupt__"][0].value["kind"] == "action_proposal"
    original = calendar.create_event_prepared

    async def accepted_then_scope_lost(*args):
        try:
            return await original(*args)
        finally:
            calendar.scope_available = False
            calendar.hidden_reads = 3

    calendar.create_event_prepared = accepted_then_scope_lost
    result = await resume_workflow(graph, thread, {"decision": "approve"})
    assert "not yet confirmed" in result["response"]
    async with session_factory() as db:
        assert await crud.list_todos(db, user_id) == []
        action_id = result["proposal"]["action_id"]
        evidence = await db.get(CalendarActionEvidence, action_id)
        ledger = await db.get(ActionExecution, action_id)
        assert evidence.outcome == "UNCERTAIN" and ledger.status == "executing"
        assert await db.get(ActionExecution, action_id + ":todo") is None


@pytest.mark.asyncio
async def test_parked_parent_and_duplicate_resume_cannot_authorize_child(session_factory):
    calendar = FakeCalendar("conflict")
    coordinator = CalendarCreateCoordinator(calendar, session_factory,
                                            sleep=lambda _: _noop())
    executor = ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory),
                              session_factory, calendar_create_coordinator=coordinator)
    graph = build_productivity_graph(calendar, executor, InMemorySaver())
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "r5-cross-domain-parked")
        await db.commit()
        user_id = user.id
    route = RoutedRequest(intent="cross_domain", todo_operation="create",
                          title="Synthetic review", todo_title="Write synthetic report",
                          date_reference="tomorrow", day_part="afternoon",
                          duration_minutes=60,
                          todo_due_date_reference="day_after_tomorrow")
    thread = "r5-cross-domain-parked"
    first = await start_workflow(graph, user_id, "Schedule and create Todo", "UTC",
                                 thread, route=route)
    assert first["__interrupt__"][0].value["kind"] == "action_proposal"
    waiting = await resume_workflow(graph, thread, {"decision": "approve"})
    action_id = waiting["proposal"]["action_id"]
    assert waiting["__interrupt__"][0].value["kind"] == "calendar_confirmation"
    assert "review" in waiting["__interrupt__"][0].value["message"].lower()
    duplicate_signal = await resume_workflow(graph, thread, {"calendar_confirmation": action_id})
    assert duplicate_signal["__interrupt__"][0].value["kind"] == "calendar_confirmation"
    assert await CalendarContinuationRuntime(graph, session_factory).recover_due() == 0
    async with session_factory() as db:
        assert not await crud.list_todos(db, user_id)
        assert await db.get(ActionExecution, action_id + ":todo") is None


@pytest.mark.asyncio
async def test_new_similar_approved_action_has_distinct_identity(session_factory):
    calendar = FakeCalendar()
    service = CalendarCreateCoordinator(calendar, session_factory, sleep=lambda _: _noop())
    first = await _proposal(session_factory, action_id="r5-independent-a")
    second = await _proposal(session_factory, action_id="r5-independent-b")
    assert (await service.execute(first)).status == ActionStatus.EXECUTED
    # A separate approval is not silently fuzzy-merged by title/time. A real
    # MCP may reject this as a duplicate precheck; identity remains distinct.
    assert (await service.execute(second)).status == ActionStatus.EXECUTED
    assert len(calendar.creates) == 2
    assert calendar.creates[0][0] != calendar.creates[1][0]


@pytest.mark.asyncio
async def test_pinned_mcp_precheck_and_409_are_distinct_and_do_not_override_policy():
    calls = []

    class Tool:
        name = "create-event"

        async def ainvoke(self, args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError("Duplicate event detected (98% similar). Event X already exists. "
                                   "To create anyway, set allowDuplicates to true.")
            raise RuntimeError("Event ID 'pstable' already exists. Please use a different ID.")

    adapter = MCPCalendarAdapter([Tool()])
    body = {"account": "normal", "calendarId": "owner@example.test",
            "summary": "Synthetic", "start": "2026-10-12T15:00:00+08:00",
            "end": "2026-10-12T15:30:00+08:00", "timeZone": "Asia/Shanghai"}
    with pytest.raises(CalendarProviderError) as precheck:
        await adapter.create_event_prepared("pstable", body, "action-1")
    assert precheck.value.category == "similarity_precheck"
    with pytest.raises(CalendarProviderError) as conflict:
        await adapter.create_event_prepared("pstable", body, "action-1")
    assert conflict.value.category == "resource_conflict" and conflict.value.status_code == 409
    assert all("allowDuplicates" not in args for args in calls)


async def _noop():
    return None
