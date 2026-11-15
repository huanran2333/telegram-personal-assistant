"""Opt-in E5: one synthetic Google event, one PATCH and one DELETE."""
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.agent.actions import ActionProposal, ActionStatus, RiskLevel
from app.agent.calendar import CalendarError, MCPCalendarAdapter, discover_calendar_tools
from app.agent.calendar_action_evidence import CalendarEvidenceIntent, planned_google_event_id
from app.agent.calendar_conditional import (GoogleCalendarConditionalTransport,
                                            McpOAuthTokenSource)
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.calendar_modify import CalendarModifyCoordinator, _approved_patch
from app.agent.workflow import bind_approved_calendar_version
from app.config import get_settings
from app.db.models import CalendarActionEvidence

pytestmark = [pytest.mark.calendar_integration, pytest.mark.integration]


async def _expire(factory, action_id):
    async with factory() as db:
        row = await db.get(CalendarActionEvidence, action_id)
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()


@pytest.mark.asyncio
async def test_real_conditional_update_delete_lost_local_results(pg_harness):
    if os.environ.get("RUN_R5_3_REAL_GOOGLE") != "1":
        pytest.skip("Explicit R5.3 real Google acceptance opt-in required")
    factory, new_user, _ = pg_harness
    user = await new_user()
    settings = get_settings()
    calendar = MCPCalendarAdapter(
        await discover_calendar_tools(settings),
        conditional=GoogleCalendarConditionalTransport(
            McpOAuthTokenSource(settings.google_credentials_file)))
    account, resolved = await calendar.resolve_create_scope("primary")
    seed_action = "r53e5seed" + uuid4().hex
    event_id = planned_google_event_id(seed_action, account, resolved)
    zone = ZoneInfo(settings.default_timezone)
    start = (datetime.now(zone) + timedelta(days=19)).replace(
        hour=11, minute=37, second=0, microsecond=0)
    end = start + timedelta(minutes=10)
    title = "R5.3 synthetic conditional " + seed_action[-8:]
    create = ActionProposal(
        action_id=seed_action, domain="calendar", operation="create_event",
        args={"user_id": user.id, "calendarId": resolved, "title": title,
              "start": start.isoformat(), "end": end.isoformat(),
              "timeZone": settings.default_timezone},
        human_readable_preview="Create isolated synthetic event",
        reason="R5.4 combined E5", risk_level=RiskLevel.EXTERNAL_WRITE)
    created = False
    try:
        create_coordinator = CalendarCreateCoordinator(calendar, factory)
        assert (await create_coordinator.execute(create)).status == ActionStatus.EXECUTED
        create_row, create_ledger = await create_coordinator.store.load_create(seed_action)
        assert create_row.outcome == "CONFIRMED" and create_ledger.status == "executed"
        assert create_row.event_id == create_ledger.external_id == event_id
        created = True
        original = await calendar.get_event(resolved, event_id)
        assert original.etag and original.extended_properties["private"]["assistant_action"] == seed_action

        update = ActionProposal(
            action_id="r53e5update" + uuid4().hex, domain="calendar",
            operation="update_event",
            args={"user_id": user.id, "account": account, "calendarId": resolved,
                  "eventId": event_id, "title": title + " updated"},
            human_readable_preview="Update isolated synthetic event",
            reason="R5.3 E5", risk_level=RiskLevel.EXTERNAL_WRITE)
        await bind_approved_calendar_version(calendar, update)
        assert update.approved_calendar_baseline.etag == original.etag
        coordinator = CalendarModifyCoordinator(calendar, factory)
        intent = CalendarEvidenceIntent.from_approved_proposal(update, account, resolved)
        patch = _approved_patch(update)
        await coordinator.store.prepare_modification(intent, patch)
        token = await coordinator.store.arm(update.action_id, "e5-update-lost-result")
        assert token and await coordinator.store.mark_dispatch_started(update.action_id, token)
        changed = await calendar.update_event_if_match(resolved, event_id, original.etag, patch)
        assert changed.id == event_id and changed.summary == patch["summary"]
        # Discard the success response: recovery must read back, never PATCH again.
        await _expire(factory, update.action_id)
        assert (await CalendarModifyCoordinator(calendar, factory).recover(
            update.action_id)).status == ActionStatus.EXECUTED
        update_row, update_ledger = await coordinator.store.load_create(update.action_id)
        assert update_row.attempt_no == 1
        assert update_row.confirmation_source == "observed_postcondition"
        assert update_ledger.status == "executed"
        assert (await coordinator.execute(update)).status == ActionStatus.EXECUTED

        delete = ActionProposal(
            action_id="r53e5delete" + uuid4().hex, domain="calendar",
            operation="delete_event",
            args={"user_id": user.id, "account": account, "calendarId": resolved,
                  "eventId": event_id},
            human_readable_preview="Delete isolated synthetic event",
            reason="R5.3 E5", risk_level=RiskLevel.DESTRUCTIVE)
        await bind_approved_calendar_version(calendar, delete)
        assert delete.approved_calendar_baseline.etag == changed.etag
        delete_intent = CalendarEvidenceIntent.from_approved_proposal(delete, account, resolved)
        await coordinator.store.prepare_modification(delete_intent, {})
        delete_token = await coordinator.store.arm(delete.action_id, "e5-delete-lost-result")
        assert delete_token and await coordinator.store.mark_dispatch_started(delete.action_id, delete_token)
        accepted = await calendar.delete_event_if_match(resolved, event_id, changed.etag)
        assert accepted == {"eventId": event_id, "accepted": True}
        # Discard the empty-body success; observed cancelled/absence is enough.
        await _expire(factory, delete.action_id)
        assert (await CalendarModifyCoordinator(calendar, factory).recover(
            delete.action_id)).status == ActionStatus.EXECUTED
        delete_row, delete_ledger = await coordinator.store.load_create(delete.action_id)
        assert delete_row.attempt_no == 1
        assert delete_row.confirmation_source in {"observed_absence", "observed_cancelled"}
        assert delete_ledger.status == "executed"
        assert (await coordinator.execute(delete)).status == ActionStatus.EXECUTED
    finally:
        if created:
            try:
                remaining = await calendar.get_event(resolved, event_id)
            except CalendarError as exc:
                if str(exc) not in {"not_found_or_inaccessible", "gone"}:
                    raise
            else:
                if remaining.extended_properties.get("private", {}).get(
                        "assistant_action") != seed_action:
                    pytest.fail("Synthetic event ownership could not be verified for cleanup")
                if remaining.status != "cancelled":
                    await calendar.delete_event_if_match(resolved, event_id, remaining.etag)
                    try:
                        after = await calendar.get_event(resolved, event_id)
                    except CalendarError as exc:
                        assert str(exc) in {"not_found_or_inaccessible", "gone"}
                    else:
                        assert after.status == "cancelled"
