"""R5.3 E3 conditional modification; all provider responses are synthetic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar import CalendarError, UnavailableCalendarAdapter
from app.agent.calendar_action_evidence import CalendarEvidenceIntent
from app.agent.calendar_conditional import CalendarEventSnapshot
from app.agent.calendar_modify import CalendarModifyCoordinator
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.workflow import bind_approved_calendar_version
from app.db import crud
from app.db.models import ActionExecution, CalendarActionEvidence


async def _noop(_delay=0):
    return None


class FakeConditionalCalendar:
    def __init__(self, *, mode="normal"):
        self.mode = mode
        self.scope_available = True
        self.get_error = None
        self.reads = 0
        self.updates = []
        self.deletes = []
        self.event = {
            "id": "synthetic-event", "etag": '"v1"', "status": "confirmed",
            "summary": "Before", "start": {"dateTime": "2026-10-15T15:00:00+08:00"},
            "end": {"dateTime": "2026-10-15T15:30:00+08:00"},
        }

    async def resolve_create_scope(self, calendar_id):
        if not self.scope_available:
            raise CalendarError("calendar_scope_unavailable")
        return "normal", "owner@example.test"

    async def get_event(self, calendar_id, event_id):
        self.reads += 1
        if self.get_error:
            raise CalendarError(self.get_error)
        if self.event is None:
            raise CalendarError("not_found_or_inaccessible")
        return CalendarEventSnapshot.model_validate(self.event)

    async def update_event_if_match(self, calendar_id, event_id, approved_etag, patch):
        self.updates.append((calendar_id, event_id, approved_etag, patch.copy()))
        if approved_etag != self.event["etag"]:
            raise CalendarError("etag_conflict")
        if self.mode == "timeout_before":
            raise CalendarError("remote_outcome_uncertain")
        self.event.update(patch)
        self.event["etag"] = '"v2"'
        if self.mode == "timeout_after":
            raise CalendarError("remote_outcome_uncertain")
        return CalendarEventSnapshot.model_validate(self.event)

    async def delete_event_if_match(self, calendar_id, event_id, approved_etag):
        self.deletes.append((calendar_id, event_id, approved_etag))
        if self.event is None or approved_etag != self.event["etag"]:
            raise CalendarError("etag_conflict")
        if self.mode == "timeout_before":
            raise CalendarError("remote_outcome_uncertain")
        if self.mode != "delete_noop":
            self.event["status"] = "cancelled"
            self.event["etag"] = '"v2"'
        if self.mode == "timeout_after":
            raise CalendarError("remote_outcome_uncertain")
        return {"eventId": event_id, "accepted": True}


async def _proposal(factory, calendar, operation="update_event", action_id="r53-a"):
    async with factory() as db:
        user = await crud.get_or_create_user(db, "r53-" + action_id)
        await db.commit()
        user_id = user.id
    args = {"user_id": user_id, "eventId": "synthetic-event"}
    if operation == "update_event":
        args["title"] = "After"
    proposal = ActionProposal(
        action_id=action_id, domain="calendar", operation=operation, args=args,
        human_readable_preview="Synthetic conditional action", reason="E3",
        risk_level=RiskLevel.DESTRUCTIVE if operation == "delete_event" else RiskLevel.EXTERNAL_WRITE)
    await bind_approved_calendar_version(calendar, proposal)
    return proposal


async def _rows(factory, action_id):
    async with factory() as db:
        return await db.get(CalendarActionEvidence, action_id), await db.get(ActionExecution, action_id)


@pytest.mark.asyncio
async def test_update_success_uses_approved_etag_and_atomic_ledger(session_factory):
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(session_factory, calendar)
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    result = await service.execute(proposal)
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert calendar.updates == [("owner@example.test", "synthetic-event", '"v1"',
                                 {"summary": "After"})]
    assert (row.outcome, ledger.status, row.confirmed_etag) == ("CONFIRMED", "executed", '"v2"')
    assert (await service.execute(proposal)).status == ActionStatus.EXECUTED
    assert len(calendar.updates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update_event", "delete_event"])
async def test_production_executor_routes_modification_without_legacy_write(
        session_factory, operation):
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(session_factory, calendar, operation,
                               "r53-executor-" + operation)
    coordinator = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    executor = ActionExecutor(
        calendar, SqlAlchemyActionLedger(session_factory), session_factory,
        calendar_modify_coordinator=coordinator)
    assert (await executor.execute(proposal)).status == ActionStatus.EXECUTED
    assert (await executor.execute(proposal)).status == ActionStatus.EXECUTED
    assert len(calendar.updates if operation == "update_event" else calendar.deletes) == 1


@pytest.mark.asyncio
async def test_unavailable_conditional_calendar_does_not_fall_back_to_legacy_write(
        session_factory):
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(session_factory, calendar, action_id="r53-no-fallback")
    unavailable = UnavailableCalendarAdapter("provider_unavailable")
    executor = ActionExecutor(
        unavailable, SqlAlchemyActionLedger(session_factory), session_factory,
        calendar_modify_coordinator=CalendarModifyCoordinator(
            unavailable, session_factory, sleep=_noop))
    result = await executor.execute(proposal)
    assert result.status == ActionStatus.FAILED
    assert result.result["failure_category"] == "preparation_unavailable"
    assert not calendar.updates


@pytest.mark.asyncio
async def test_update_412_manual_edit_parks_without_reapproval(session_factory):
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(session_factory, calendar, action_id="r53-412")
    calendar.event["etag"] = '"manual-v2"'
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    result = await service.execute(proposal)
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert row.outcome == "PARKED_REVIEW" and ledger.status == "executing"
    assert calendar.updates[0][2] == '"v1"'
    assert len(calendar.updates) == 1


@pytest.mark.asyncio
async def test_update_accepted_timeout_observed_postcondition(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_after")
    proposal = await _proposal(session_factory, calendar, action_id="r53-up-timeout")
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    result = await service.execute(proposal)
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert (row.confirmation_source, row.outcome, ledger.status) == (
        "observed_postcondition", "CONFIRMED", "executed")
    assert len(calendar.updates) == 1


@pytest.mark.asyncio
async def test_crash_after_arm_restarts_with_original_etag_and_patch(session_factory):
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(session_factory, calendar, action_id="r53-armed")
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    intent = CalendarEvidenceIntent.from_approved_proposal(
        proposal, "normal", "owner@example.test")
    await service.store.prepare_modification(intent, {"summary": "After"})
    stale = await service.store.arm(proposal.action_id, "crashed")
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    restarted = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    assert (await restarted.recover(proposal.action_id)).status == ActionStatus.EXECUTED
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert row.attempt_no == 2 and row.approved_baseline_etag == '"v1"'
    assert ledger.status == "executed" and calendar.updates[0][2] == '"v1"'
    assert not await service.store.finalize_modification(
        proposal.action_id, stale, "CONFIRMED", confirmation_source="provider_response")


@pytest.mark.asyncio
async def test_update_changed_etag_without_postcondition_parks(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_before")
    proposal = await _proposal(session_factory, calendar, action_id="r53-changed")
    original = calendar.update_event_if_match

    async def changed_then_timeout(*args):
        calendar.event["etag"] = '"manual-v2"'
        return await original(*args)

    calendar.update_event_if_match = changed_then_timeout
    result = await CalendarModifyCoordinator(calendar, session_factory, sleep=_noop).execute(proposal)
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert row.outcome == "PARKED_REVIEW" and ledger.status == "executing"


@pytest.mark.asyncio
async def test_approved_etag_and_patch_cannot_be_replaced(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_before")
    calendar.scope_available = True
    proposal = await _proposal(session_factory, calendar, action_id="r53-immutable")
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    original = await service.execute(proposal)
    assert original.status == ActionStatus.EXECUTING
    changed = proposal.model_copy(deep=True)
    changed.approved_calendar_baseline.etag = '"later"'
    changed.args["title"] = "Different"
    result = await service.execute(changed)
    assert result.result["remote_state"] == "APPROVAL_MISMATCH"
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert row.approved_baseline_etag == '"v1"' and ledger.status == "executing"


@pytest.mark.asyncio
async def test_delete_empty_success_and_412_after_manual_edit(session_factory):
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(session_factory, calendar, "delete_event", "r53-delete-ok")
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    assert (await service.execute(proposal)).status == ActionStatus.EXECUTED
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert row.confirmation_source == "provider_response" and ledger.status == "executed"
    assert calendar.deletes == [("owner@example.test", "synthetic-event", '"v1"')]

    other = FakeConditionalCalendar()
    new = await _proposal(session_factory, other, "delete_event", "r53-delete-412")
    other.event["etag"] = '"manual-v2"'
    result = await CalendarModifyCoordinator(other, session_factory, sleep=_noop).execute(new)
    row, ledger = await _rows(session_factory, new.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert row.outcome == "PARKED_REVIEW" and ledger.status == "executing"


@pytest.mark.asyncio
async def test_delete_timeout_cancelled_and_verified_absence(session_factory):
    cancelled = FakeConditionalCalendar(mode="timeout_after")
    proposal = await _proposal(session_factory, cancelled, "delete_event", "r53-del-cancel")
    service = CalendarModifyCoordinator(cancelled, session_factory, sleep=_noop)
    assert (await service.execute(proposal)).status == ActionStatus.EXECUTED
    row, _ = await _rows(session_factory, proposal.action_id)
    assert row.confirmation_source == "observed_cancelled" and len(cancelled.deletes) == 1

    absent = FakeConditionalCalendar(mode="timeout_after")
    proposal2 = await _proposal(session_factory, absent, "delete_event", "r53-del-absent")
    original = absent.delete_event_if_match

    async def accepted_then_404(*args):
        try:
            return await original(*args)
        finally:
            absent.event = None

    absent.delete_event_if_match = accepted_then_404
    assert (await CalendarModifyCoordinator(absent, session_factory, sleep=_noop).execute(
        proposal2)).status == ActionStatus.EXECUTED
    row, _ = await _rows(session_factory, proposal2.action_id)
    assert row.confirmation_source == "observed_absence"


@pytest.mark.asyncio
async def test_delete_inaccessible_404_is_not_absence(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_before")
    proposal = await _proposal(session_factory, calendar, "delete_event", "r53-del-access")
    calendar.get_error = "not_found_or_inaccessible"
    calendar.scope_available = False
    # Preparation must use the approved scope; simulate a previously armed
    # action rather than allowing a fresh approval under unavailable scope.
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    intent = CalendarEvidenceIntent.from_approved_proposal(
        proposal, "normal", "owner@example.test")
    await service.store.prepare_modification(intent, {})
    token = await service.store.arm(proposal.action_id, "crashed")
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    assert (await service.recover(proposal.action_id)).status == ActionStatus.EXECUTING
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert row.outcome == "UNCERTAIN" and ledger.status == "executing"
    assert not await service.store.finalize_modification(
        proposal.action_id, token, "CONFIRMED", confirmation_source="observed_absence")


@pytest.mark.asyncio
async def test_delete_verified_410_after_armed_attempt_is_observed_absence(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_before")
    proposal = await _proposal(session_factory, calendar, "delete_event", "r53-del-410")
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    intent = CalendarEvidenceIntent.from_approved_proposal(
        proposal, "normal", "owner@example.test")
    await service.store.prepare_modification(intent, {})
    await service.store.arm(proposal.action_id, "crashed")
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    calendar.get_error = "gone"
    assert (await service.recover(proposal.action_id)).status == ActionStatus.EXECUTED
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert row.confirmation_source == "observed_absence"
    assert ledger.status == "executed" and not calendar.deletes


@pytest.mark.asyncio
async def test_delete_unchanged_preimage_retries_same_etag_then_parks(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_before")
    proposal = await _proposal(session_factory, calendar, "delete_event", "r53-del-retry")
    service = CalendarModifyCoordinator(calendar, session_factory, sleep=_noop)
    result = await service.execute(proposal)
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert row.outcome == "PARKED_REVIEW" and ledger.status == "executing"
    assert row.attempt_no == 2 and len(calendar.deletes) == 2
    assert {attempt[2] for attempt in calendar.deletes} == {'"v1"'}


@pytest.mark.asyncio
async def test_delete_changed_preimage_parks_without_retry(session_factory):
    calendar = FakeConditionalCalendar(mode="timeout_before")
    proposal = await _proposal(session_factory, calendar, "delete_event", "r53-del-changed")
    original = calendar.delete_event_if_match

    async def changed_then_timeout(*args):
        calendar.event["etag"] = '"manual-v2"'
        return await original(*args)

    calendar.delete_event_if_match = changed_then_timeout
    result = await CalendarModifyCoordinator(calendar, session_factory, sleep=_noop).execute(proposal)
    row, ledger = await _rows(session_factory, proposal.action_id)
    assert result.status == ActionStatus.EXECUTING
    assert row.outcome == "PARKED_REVIEW" and ledger.status == "executing"
    assert len(calendar.deletes) == 1


@pytest.mark.asyncio
async def test_recurring_master_and_instance_fail_before_proposal(session_factory):
    for recurring in ({"recurrence": ["RRULE:FREQ=DAILY"]},
                      {"recurringEventId": "series", "originalStartTime": {"dateTime": "x"}}):
        calendar = FakeConditionalCalendar()
        calendar.event.update(recurring)
        proposal = ActionProposal(
            action_id="recurrence-probe", domain="calendar", operation="delete_event",
            args={"user_id": "synthetic", "eventId": "synthetic-event"},
            human_readable_preview="Delete synthetic", reason="test",
            risk_level=RiskLevel.DESTRUCTIVE)
        with pytest.raises(CalendarError, match="version changed before approval"):
            await bind_approved_calendar_version(calendar, proposal)

    calendar = FakeConditionalCalendar()
    proposal = ActionProposal(
        action_id="recurrence-scope", domain="calendar", operation="delete_event",
        args={"user_id": "synthetic", "eventId": "synthetic-event",
              "modificationScope": "thisAndFollowing"},
        human_readable_preview="Delete synthetic", reason="test",
        risk_level=RiskLevel.DESTRUCTIVE)
    with pytest.raises(CalendarError, match="multi-resource"):
        await bind_approved_calendar_version(calendar, proposal)
