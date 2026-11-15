"""R5.3 E4: isolated PG18 and synthetic conditional provider only."""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar_action_evidence import CalendarEvidenceIntent
from app.agent.calendar_modify import CalendarModifyCoordinator, _approved_patch
from app.agent.workflow import bind_approved_calendar_version
from app.db.models import ActionExecution, CalendarActionEvidence
from tests.test_r5_3_calendar_modify import FakeConditionalCalendar

pytestmark = pytest.mark.integration


async def _proposal(user_id, calendar, operation):
    args = {"user_id": user_id, "eventId": "synthetic-event"}
    if operation == "update_event":
        args["title"] = "After"
    proposal = ActionProposal(
        action_id="r53pg" + uuid4().hex, domain="calendar", operation=operation,
        args=args, human_readable_preview="Synthetic", reason="E4",
        risk_level=RiskLevel.DESTRUCTIVE if operation == "delete_event" else RiskLevel.EXTERNAL_WRITE)
    await bind_approved_calendar_version(calendar, proposal)
    return proposal


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update_event", "delete_event"])
async def test_pg18_two_workers_one_conditional_write_and_joined_truth(pg_harness, operation):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(user.id, calendar, operation)
    first = CalendarModifyCoordinator(calendar, factory)
    second = CalendarModifyCoordinator(calendar, factory)
    results = await asyncio.gather(first.execute(proposal), second.execute(proposal))
    assert ActionStatus.EXECUTED in {result.status for result in results}
    calls = calendar.updates if operation == "update_event" else calendar.deletes
    assert len(calls) == 1 and calls[0][2] == '"v1"'
    row, ledger = await first.store.load_create(proposal.action_id)
    assert row.outcome == "CONFIRMED" and ledger.status == "executed"
    assert row.approved_baseline_etag == '"v1"'
    assert ledger.external_id == row.event_id


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update_event", "delete_event"])
async def test_pg18_lost_success_restarts_reconciles_and_rejects_stale_fence(pg_harness, operation):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = FakeConditionalCalendar(mode="timeout_after")
    proposal = await _proposal(user.id, calendar, operation)
    first = CalendarModifyCoordinator(calendar, factory)
    intent = CalendarEvidenceIntent.from_approved_proposal(
        proposal, "normal", "owner@example.test")
    patch = _approved_patch(proposal)
    await first.store.prepare_modification(intent, patch)
    old_token = await first.store.arm(proposal.action_id, "crashed-worker")
    assert old_token
    try:
        if operation == "update_event":
            await calendar.update_event_if_match(intent.calendar_id, intent.event_id,
                                                 intent.approved_baseline.etag, patch)
        else:
            await calendar.delete_event_if_match(intent.calendar_id, intent.event_id,
                                                 intent.approved_baseline.etag)
    except Exception:
        pass  # Real provider may have accepted, but local result was lost.
    async with factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    fresh = CalendarModifyCoordinator(calendar, factory)
    result = await fresh.recover(proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    row, ledger = await fresh.store.load_create(proposal.action_id)
    assert row.confirmation_source == (
        "observed_postcondition" if operation == "update_event" else "observed_cancelled")
    assert (row.outcome, ledger.status) == ("CONFIRMED", "executed")
    assert not await first.store.finalize_modification(
        proposal.action_id, old_token, "CONFIRMED", confirmation_source="provider_response")
    calls = calendar.updates if operation == "update_event" else calendar.deletes
    assert len(calls) == 1 and (await fresh.execute(proposal)).status == ActionStatus.EXECUTED


@pytest.mark.asyncio
async def test_pg18_provider_io_does_not_hold_evidence_row_lock(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = FakeConditionalCalendar()
    proposal = await _proposal(user.id, calendar, "update_event")
    original = calendar.update_event_if_match

    async def provider_with_independent_row_lock(*args):
        async with factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == proposal.action_id).with_for_update(nowait=True))
            assert row.outcome == "IN_FLIGHT"
            await db.rollback()
        return await original(*args)

    calendar.update_event_if_match = provider_with_independent_row_lock
    result = await CalendarModifyCoordinator(calendar, factory).execute(proposal)
    assert result.status == ActionStatus.EXECUTED
