"""Opt-in E5: one synthetic Google insert, dropped local result, exact-ID cleanup."""
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar import CalendarError, MCPCalendarAdapter, discover_calendar_tools
from app.agent.calendar_action_evidence import CalendarEvidenceIntent
from app.agent.calendar_conditional import (GoogleCalendarConditionalTransport,
                                            McpOAuthTokenSource)
from app.agent.calendar_create import CalendarCreateCoordinator
from app.config import get_settings
from app.db.models import CalendarActionEvidence

pytestmark = [pytest.mark.calendar_integration, pytest.mark.integration]


@pytest.mark.asyncio
async def test_real_one_insert_lost_local_result_reconciles_and_cleans_exact_id(pg_harness):
    if os.environ.get("RUN_R5_2_REAL_GOOGLE") != "1":
        pytest.skip("Explicit R5.2 real Google acceptance opt-in required")
    factory, new_user, _ = pg_harness
    user = await new_user()
    settings = get_settings()
    calendar = MCPCalendarAdapter(
        await discover_calendar_tools(settings),
        conditional=GoogleCalendarConditionalTransport(
            McpOAuthTokenSource(settings.google_credentials_file)))
    coordinator = CalendarCreateCoordinator(calendar, factory)
    zone = ZoneInfo(settings.default_timezone)
    start = (datetime.now(zone) + timedelta(days=14)).replace(
        hour=11, minute=37, second=0, microsecond=0)
    end = start + timedelta(minutes=10)
    action_id = "r52e5" + uuid4().hex
    proposal = ActionProposal(
        action_id=action_id, domain="calendar", operation="create_event",
        args={"user_id": user.id, "calendarId": "primary",
              "title": "R5.2 synthetic reconciliation " + action_id[-8:],
              "start": start.isoformat(), "end": end.isoformat(),
              "timeZone": settings.default_timezone},
        human_readable_preview="Synthetic test event", reason="R5.2 E5",
        risk_level=RiskLevel.EXTERNAL_WRITE)
    account, resolved = await calendar.resolve_create_scope("primary")
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, account, resolved)
    body = {"account": account, "calendarId": resolved,
            "summary": proposal.args["title"], "start": proposal.args["start"],
            "end": proposal.args["end"], "timeZone": proposal.args["timeZone"]}
    await coordinator.store.prepare_create(intent, body, action_id)
    token = await coordinator.store.arm(action_id, "e5-lost-result")
    assert token and await coordinator.store.mark_dispatch_started(action_id, token)
    created = False
    try:
        # Exactly one real insertion. Deliberately do not finalize local result:
        # this simulates provider success + local crash without a second insert.
        response = await calendar.create_event_prepared(intent.event_id, body, action_id)
        assert response["id"] == intent.event_id
        created = True
        exact = await calendar.get_event(resolved, intent.event_id)
        assert exact.id == intent.event_id
        assert exact.extended_properties["private"]["assistant_action"] == action_id
        assert exact.summary == body["summary"]
        async with factory() as db:
            row = await db.get(CalendarActionEvidence, action_id)
            row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        fresh = CalendarCreateCoordinator(calendar, factory)
        recovered = await fresh.recover(action_id)
        assert recovered.status == ActionStatus.EXECUTED
        again = await fresh.execute(proposal)
        assert again.status == ActionStatus.EXECUTED
        assert recovered.result["id"] == again.result["id"] == intent.event_id
        row, ledger = await fresh.store.load_create(action_id)
        assert row.attempt_no == 1 and row.confirmation_source == "exact_readback"
        assert ledger.status == "executed" and ledger.external_id == intent.event_id
    finally:
        # Exact-ID cleanup only when marker proves this test owns the event.
        try:
            exact = await calendar.get_event(resolved, intent.event_id)
        except CalendarError as exc:
            if str(exc) != "not_found_or_inaccessible":
                raise
            exact = None
        if exact is not None and exact.extended_properties.get("private", {}).get(
                "assistant_action") == action_id:
            await calendar.delete_event_if_match(resolved, intent.event_id, exact.etag)
            # Google may retain a cancelled event tombstone at the exact ID.
            try:
                after_delete = await calendar.get_event(resolved, intent.event_id)
            except CalendarError as exc:
                assert str(exc) == "not_found_or_inaccessible"
            else:
                assert after_delete.status == "cancelled"
        elif created:
            pytest.fail("Synthetic event ownership could not be verified for cleanup")
