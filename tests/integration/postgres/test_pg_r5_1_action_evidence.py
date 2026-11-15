"""R5.1 E4: real isolated PG18 claims, restart and immutable approval proof."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.agent.actions import ActionProposal, ApprovedCalendarBaseline, RiskLevel
from app.agent.calendar_action_evidence import CalendarActionEvidenceStore, CalendarEvidenceIntent
from app.db.models import CalendarActionEvidence

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_pg18_two_workers_fence_restart_and_approved_etag(pg_harness):
    factory, new_user, _engine = pg_harness
    user = await new_user()
    proposal = ActionProposal(action_id="r5-" + user.id.replace("-", "")[:30],
                              domain="calendar", operation="update_event",
                              args={"user_id": user.id, "eventId": "synthetic-event",
                                    "title": "After approval"},
                              human_readable_preview="Synthetic update", reason="test",
                              risk_level=RiskLevel.EXTERNAL_WRITE,
                              approved_calendar_baseline=ApprovedCalendarBaseline(
                                  calendar_id="primary", event_id="synthetic-event",
                                  etag='"approved-v1"', status="confirmed",
                                  summary="Before approval"))
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "primary")
    worker_a = CalendarActionEvidenceStore(factory)
    worker_b = CalendarActionEvidenceStore(factory)
    await worker_a.prepare(intent)
    claims = await asyncio.gather(worker_a.arm(intent.action_id, "worker-a"),
                                  worker_b.arm(intent.action_id, "worker-b"))
    assert sum(token is not None for token in claims) == 1
    old_token = next(token for token in claims if token)
    # A separate connection can read while provider I/O would occur: no long
    # transaction or row lock remains held after arm() returns.
    async with factory() as db:
        row = await db.scalar(select(CalendarActionEvidence).where(
            CalendarActionEvidence.action_id == intent.action_id).with_for_update(nowait=True))
        assert row.outcome == "IN_FLIGHT"
        assert row.dispatch_started_at is None
        assert row.approved_baseline_etag == '"approved-v1"'
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    # Fresh repository/runtime after a crash must reconcile before any repeat
    # provider dispatch, including the no-dispatch timestamp case.
    restarted = CalendarActionEvidenceStore(factory)
    assert await restarted.arm(intent.action_id, "restarted-worker") is None
    new_token = await restarted.claim_reconciliation(intent.action_id, "restarted-worker")
    assert new_token and new_token != old_token
    assert not await worker_a.finalize(intent.action_id, old_token, "CONFIRMED",
                                       external_id="synthetic-event",
                                       confirmation_source="provider_response")
    assert await restarted.finalize(intent.action_id, new_token, "UNCERTAIN",
                                    failure_category="remote_outcome_uncertain")
    async with factory() as db:
        row = await db.get(CalendarActionEvidence, intent.action_id)
        assert row.outcome == "UNCERTAIN"
        assert row.attempt_no == 1
        assert row.approved_baseline_etag == '"approved-v1"'
    changed = intent.model_copy(update={"approved_baseline": intent.approved_baseline.model_copy(
        update={"etag": '"unapproved-v2"'})})
    with pytest.raises(ValueError, match="changed after preparation"):
        await restarted.prepare(changed)
