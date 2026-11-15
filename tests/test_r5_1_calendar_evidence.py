"""R5.1 deterministic provider/approved-version/fault contracts, no Google write."""
import re
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import select

from app.agent.actions import ActionProposal, ApprovedCalendarBaseline, RiskLevel
from app.agent.calendar import MCPCalendarAdapter
from app.agent.calendar_action_evidence import (
    CalendarActionEvidenceStore, CalendarEvidenceIntent, planned_google_event_id,
)
from app.agent.calendar_conditional import (
    CalendarEventSnapshot, CalendarProviderError, GoogleCalendarConditionalTransport,
)
from app.agent.workflow import bind_approved_calendar_version
from app.db import crud
from app.db.models import CalendarActionEvidence


def _event(event_id="event-1", etag='"approved-v1"'):
    return {"id": event_id, "etag": etag, "status": "confirmed",
            "summary": "Synthetic meeting", "start": {"dateTime": "2026-10-10T15:00:00+08:00"},
            "end": {"dateTime": "2026-10-10T15:30:00+08:00"},
            "extendedProperties": {"private": {"assistant_action": "act-1"}}}


def _proposal(operation="update_event", action_id="act-1"):
    return ActionProposal(action_id=action_id, domain="calendar", operation=operation,
                          args={"user_id": "user-1", "eventId": "event-1",
                                "title": "Updated meeting"},
                          human_readable_preview="Update synthetic meeting",
                          reason="test", risk_level=RiskLevel.EXTERNAL_WRITE)


def test_stable_google_event_id_is_valid_and_action_scoped():
    a = planned_google_event_id("same-action", "normal", "calendar@example.test")
    assert re.fullmatch(r"[a-v0-9]{5,1024}", a)
    assert a == planned_google_event_id("same-action", "normal", "calendar@example.test")
    assert a != planned_google_event_id("other-action", "normal", "calendar@example.test")
    assert a != planned_google_event_id("same-action", "normal", "other-calendar")


@pytest.mark.asyncio
async def test_exact_get_preserves_etag_marker_and_conditional_writes_send_if_match():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json=_event())

    async def token():
        return "synthetic-test-token"

    transport = GoogleCalendarConditionalTransport(
        token, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    got = await transport.get_event("primary", "event-1")
    assert (got.id, got.etag, got.status, got.extended_properties["private"]["assistant_action"]) == (
        "event-1", '"approved-v1"', "confirmed", "act-1")
    await transport.update_event("primary", "event-1", '"approved-v1"', {"summary": "new"})
    await transport.delete_event("primary", "event-1", '"approved-v1"')
    assert [request.method for request in seen] == ["GET", "PATCH", "DELETE"]
    assert seen[1].headers["If-Match"] == '"approved-v1"'
    assert seen[2].headers["If-Match"] == '"approved-v1"'
    with pytest.raises(CalendarProviderError, match="approved_etag_required"):
        await transport.update_event("primary", "event-1", "", {"summary": "unsafe"})


@pytest.mark.asyncio
async def test_timeout_never_classified_as_definite_failure():
    async def token():
        return "synthetic-test-token"

    def timeout(_request):
        raise httpx.ReadTimeout("request may have reached Google")

    transport = GoogleCalendarConditionalTransport(
        token, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(timeout)))
    with pytest.raises(CalendarProviderError) as caught:
        await transport.update_event("primary", "event-1", '"approved-v1"', {"summary": "new"})
    assert caught.value.category == "remote_outcome_uncertain"
    assert "Google" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["oauth_unavailable", "oauth_refresh_failed"])
async def test_oauth_failure_never_dispatches_conditional_or_legacy_write(category):
    async def unavailable_token():
        raise CalendarProviderError(category)

    def forbidden_http(_request):
        raise AssertionError("No Google request is allowed without OAuth")

    transport = GoogleCalendarConditionalTransport(
        unavailable_token,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(forbidden_http)))
    adapter = MCPCalendarAdapter([], conditional=transport)
    from app.agent.calendar import CalendarError
    with pytest.raises(CalendarError, match=category):
        await adapter.update_event_if_match("test-calendar", "test-event", '"approved"',
                                            {"summary": "synthetic"})
    with pytest.raises(CalendarError, match=category):
        await adapter.delete_event_if_match("test-calendar", "test-event", '"approved"')


@pytest.mark.asyncio
async def test_etag_conflict_is_not_a_conditional_overwrite():
    async def token():
        return "synthetic-test-token"

    transport = GoogleCalendarConditionalTransport(
        token, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(412, json={"error": "changed"}))))
    with pytest.raises(CalendarProviderError) as caught:
        await transport.update_event("primary", "event-1", '"approved-v1"', {"summary": "new"})
    assert caught.value.category == "etag_conflict"
    assert "changed" not in str(caught.value)


@pytest.mark.asyncio
async def test_proposal_binds_exact_preapproval_version_not_later_etag():
    class Calendar:
        async def resolve_create_scope(self, calendar_id):
            return "normal", calendar_id

        async def get_event(self, calendar_id, event_id):
            assert (calendar_id, event_id) == ("primary", "event-1")
            return CalendarEventSnapshot.model_validate(_event())

    proposal = _proposal()
    await bind_approved_calendar_version(Calendar(), proposal)
    assert proposal.approved_calendar_baseline.etag == '"approved-v1"'
    assert proposal.args["display_title"] == "Synthetic meeting"
    assert "version \"approved-v1\"" in proposal.human_readable_preview
    delete = ActionProposal(
        domain="calendar", operation="delete_event",
        args={"user_id": "user-1", "eventId": "event-1"},
        human_readable_preview="Delete exact Calendar event", reason="test",
        risk_level=RiskLevel.DESTRUCTIVE)
    await bind_approved_calendar_version(Calendar(), delete)
    from app.localization import proposal_preview
    assert "Synthetic meeting" in proposal_preview(delete.model_dump())
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "primary")
    assert intent.approved_baseline.etag == '"approved-v1"'
    changed = proposal.model_copy(deep=True)
    changed.approved_calendar_baseline.etag = '"later-v2"'
    assert intent.approved_baseline.etag == '"approved-v1"'


@pytest.mark.asyncio
async def test_prepared_create_forwards_planned_id_and_marker_to_pinned_mcp_surface():
    calls = []

    class Tool:
        name = "create-event"

        async def ainvoke(self, args):
            calls.append(args)
            return {"event": {"id": args["eventId"]}}

    adapter = MCPCalendarAdapter([Tool()])
    event_id = planned_google_event_id("act-1", "normal", "primary")
    await adapter.create_event_prepared(event_id, {"account": "normal", "calendarId": "primary",
                                        "title": "Synthetic",
                                        "start": "2026-10-10T15:00:00+08:00",
                                        "end": "2026-10-10T15:30:00+08:00",
                                        "timeZone": "Asia/Shanghai"}, "act-1")
    assert calls[0]["eventId"] == event_id
    assert calls[0]["account"] == "normal"
    assert calls[0]["extendedProperties"] == {"private": {"assistant_action": "act-1"}}


@pytest.mark.asyncio
async def test_e3_evidence_preparation_is_immutable_and_expired_arm_reconciles_first(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "r5-e3-user")
        await db.commit()
        user_id = user.id
    proposal = _proposal()
    proposal.args["user_id"] = user_id
    proposal.approved_calendar_baseline = ApprovedCalendarBaseline(
        calendar_id="primary", event_id="event-1", etag='"approved-v1"',
        status="confirmed", summary="Synthetic meeting")
    intent = CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "primary")
    store = CalendarActionEvidenceStore(session_factory)
    await store.prepare(intent)
    # Crash after durable intent, before an attempted call: a fresh store can
    # still claim the same immutable plan, not create a new action identity.
    restarted_before_call = CalendarActionEvidenceStore(session_factory)
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, intent.action_id)
        assert row.outcome == "NOT_STARTED" and row.attempt_no == 0
    proposal.args["title"] = "Changed after approval"
    with pytest.raises(ValueError, match="changed after preparation"):
        await store.prepare(CalendarEvidenceIntent.from_approved_proposal(proposal, "normal", "primary"))
    first = await restarted_before_call.arm(intent.action_id, "worker-a", lease_seconds=60)
    assert first and await store.arm(intent.action_id, "worker-b") is None
    assert await store.claim_reconciliation(intent.action_id, "worker-b") is None
    # Deterministic crash after IN_FLIGHT but before fake provider dispatch.
    async with session_factory() as db:
        row = await db.get(CalendarActionEvidence, intent.action_id)
        assert row.dispatch_started_at is None
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
    second = await store.claim_reconciliation(intent.action_id, "restarted-worker")
    assert second and second != first
    assert await store.arm(intent.action_id, "old-worker") is None
    assert not await store.finalize(intent.action_id, first, "CONFIRMED",
                                    external_id="event-1", confirmation_source="exact_readback")
    with pytest.raises(ValueError, match="cannot be definite failure"):
        await store.finalize(intent.action_id, second, "FAILED_DEFINITE",
                             failure_category="remote_outcome_uncertain")
    assert await store.finalize(intent.action_id, second, "UNCERTAIN",
                                failure_category="remote_outcome_uncertain")
    async with session_factory() as db:
        row = await db.scalar(select(CalendarActionEvidence).where(
            CalendarActionEvidence.action_id == intent.action_id))
        assert row.outcome == "UNCERTAIN"
        assert row.approved_baseline_etag == '"approved-v1"'
        assert row.attempt_no == 1
