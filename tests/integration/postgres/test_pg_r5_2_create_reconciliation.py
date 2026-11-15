"""R5.2 E4: real isolated PG18 transaction/fence/restart, fake Google only."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar_action_evidence import CalendarActionEvidenceStore, CalendarEvidenceIntent
from app.agent.calendar_create import CalendarCreateCoordinator
from app.db.models import ActionExecution, CalendarActionEvidence

from tests.test_r5_2_create_reconciliation import FakeCalendar

pytestmark = pytest.mark.integration


def _proposal(user_id: str, action_id: str):
    return ActionProposal(
        action_id=action_id, domain="calendar", operation="create_event",
        args={"user_id": user_id, "calendarId": "primary", "title": "R5.2 PG synthetic",
              "start": "2026-10-12T15:00:00+08:00",
              "end": "2026-10-12T15:30:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="Synthetic", reason="E4", risk_level=RiskLevel.EXTERNAL_WRITE)


async def _noop():
    return None


@pytest.mark.asyncio
async def test_pg18_two_workers_one_action_and_joined_terminal_truth(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    proposal = _proposal(user.id, "r52-" + user.id.replace("-", "")[:29])
    calendar = FakeCalendar()
    worker_a = CalendarCreateCoordinator(calendar, factory, sleep=lambda _: _noop())
    worker_b = CalendarCreateCoordinator(calendar, factory, sleep=lambda _: _noop())
    result_a, result_b = await asyncio.gather(worker_a.execute(proposal), worker_b.execute(proposal))
    assert len(calendar.creates) == 1
    assert ActionStatus.EXECUTED in {result_a.status, result_b.status}
    row, ledger = await worker_a.store.load_create(proposal.action_id)
    assert row.outcome == "CONFIRMED" and ledger.status == "executed"
    assert ledger.external_id == row.event_id == calendar.creates[0][0]
    assert (await worker_b.execute(proposal)).status == ActionStatus.EXECUTED
    assert len(calendar.creates) == 1


@pytest.mark.asyncio
async def test_pg18_armed_restart_readback_and_stale_fence(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    proposal = _proposal(user.id, "r52-" + user.id.replace("-", "")[:29])
    calendar = FakeCalendar("accepted_then_timeout")
    first = CalendarCreateCoordinator(calendar, factory, sleep=lambda _: _noop())
    account, resolved = await calendar.resolve_create_scope("primary")
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, account, resolved)
    body = {"account": account, "calendarId": resolved, "summary": proposal.args["title"],
            "start": proposal.args["start"], "end": proposal.args["end"],
            "timeZone": proposal.args["timeZone"]}
    await first.store.prepare_create(intent, body, proposal.action_id)
    old_fence = await first.store.arm(proposal.action_id, "worker-before-crash")
    # Fake provider accepted, but process dies before any evidence/ledger completion.
    try:
        await calendar.create_event_prepared(intent.event_id, body, proposal.action_id)
    except Exception:
        pass
    assert len(calendar.creates) == 1
    async with factory() as db:
        row = await db.get(CalendarActionEvidence, proposal.action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    fresh = CalendarCreateCoordinator(calendar, factory, sleep=lambda _: _noop())
    result = await fresh.recover(proposal.action_id)
    assert result.status == ActionStatus.EXECUTED
    assert len(calendar.creates) == 1
    assert not await first.store.finalize_create(
        proposal.action_id, old_fence, "CONFIRMED", confirmation_source="provider_response")
    row, ledger = await fresh.store.load_create(proposal.action_id)
    assert (row.outcome, ledger.status, row.confirmation_source) == (
        "CONFIRMED", "executed", "exact_readback")
    changed = proposal.model_copy(deep=True)
    changed.args["title"] = "unapproved"
    assert (await fresh.execute(changed)).status == ActionStatus.EXECUTING
    assert len(calendar.creates) == 1
