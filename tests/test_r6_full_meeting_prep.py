"""Deterministic R6 grounding cases; no provider/model fixture is called E5."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.calendar_conditional import CalendarEventSnapshot
from app.agent.gmail import GmailError, GmailMessage
from app.agent.meeting_prep_full import (EvidenceEnvelope, GroundingError,
    MeetingBrief, MeetingClaim, MeetingEvidenceBundle, collect_meeting_evidence,
    render_meeting_brief, revalidate_anchor, synthesize_meeting_brief,
    validate_meeting_brief)
from app.db import crud


NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def event(*, title="Project Sync", hour=16, attendees=None):
    return CalendarEventSnapshot.model_validate({
        "id": "event-1", "etag": '"v1"', "status": "confirmed", "summary": title,
        "start": {"dateTime": f"2026-10-08T{hour:02}:00:00+08:00"},
        "end": {"dateTime": f"2026-10-08T{hour + 1:02}:00:00+08:00"},
        "attendees": attendees or [{"email": "alice@example.test", "displayName": "Alex"}],
    })


class Calendar:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot or event()
        self.calls = 0

    async def resolve_read_scope(self, calendar_id):
        return "account-a", "calendar-a"

    async def get_event(self, calendar_id, event_id):
        self.calls += 1
        assert (calendar_id, event_id) == ("calendar-a", "event-1")
        return self.snapshot


class Gmail:
    def __init__(self, messages=(), *, failure=False):
        self.messages = list(messages)
        self.failure = failure
        self.queries = []
        self.expanded = []

    async def account_address(self):
        return "self@example.test"

    async def search(self, query, limit=5):
        self.queries.append(query)
        if self.failure:
            raise GmailError("provider_unavailable")
        return self.messages[:limit]

    async def thread(self, thread_id):
        self.expanded.append(thread_id)
        return [message for message in self.messages if message.thread_id == thread_id]


def mail(mid, sender, text, *, thread=None, subject="Project Sync", recipients=None):
    return GmailMessage(message_id=mid, thread_id=thread or mid, sender=sender,
                        recipients=recipients or ["self@example.test"], subject=subject,
                        body=text, snippet=text, timestamp=NOW)


def envelope(key, source_type, *, text="", fact=None, currentness="current", owner=None):
    return EvidenceEnvelope(
        key=key, source_type=source_type, source_id=key, owner_scope=owner or "user:u1",
        authority_domain=source_type, retrieved_at=NOW, currentness=currentness,
        safe_display_label=f"{source_type}: safe label", evidence_text=text,
        normalized_fact=fact or {})


def bundle(*items):
    return MeetingEvidenceBundle(user_id="u1", calendar_account="account-a",
        calendar_id="calendar-a", event_id="event-1", event_etag='"v1"',
        gmail_mailbox="self@example.test", evidence=list(items), availability={},
        selected_counts={}, dropped_counts={}, evidence_tokens=100)


def calendar_source():
    return EvidenceEnvelope(key="C1", source_type="calendar", source_id="event-1",
        source_sub_id="calendar-a", owner_scope="calendar:account-a:calendar-a",
        authority_domain="calendar_event", retrieved_at=NOW, currentness="current",
        safe_display_label="Calendar: Project Sync · Oct 08",
        normalized_fact={"title": "Project Sync", "start": "2026-10-08T16:00:00+08:00"})


@pytest.mark.asyncio
async def test_exact_anchor_focused_gmail_and_wrong_attendee_exclusion(session_factory):
    valid = mail("m1", "Alex <alice@example.test>", "Benchmark results are ready")
    wrong = mail("m2", "Alex <other@example.test>", "Unrelated Project Sync")
    unrelated = mail("m3", "alice@example.test", "Unrelated weather", subject="Weather")
    gmail = Gmail([valid, wrong, unrelated])
    result = await collect_meeting_evidence(
        user_id="u1", telegram_id="owner", timezone_name="Asia/Shanghai",
        candidate={"id": "event-1", "calendarId": "primary"},
        calendar=Calendar(), gmail=gmail, gmail_owner_telegram_id="owner",
        session_factory=session_factory, memory_items=[], memory_degraded=[])
    assert result.event_id == "event-1" and result.calendar_id == "calendar-a"
    assert [item.source_id for item in result.evidence if item.source_type == "gmail"] == ["m1"]
    assert gmail.expanded == ["m1"]
    assert all("alice@example.test" in query for query in gmail.queries)
    assert result.dropped_counts["gmail"] >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("gmail,availability", [
    (Gmail([]), "not_found"), (Gmail([], failure=True), "unavailable"),
])
async def test_gmail_missing_is_distinct_from_unavailable(session_factory, gmail, availability):
    result = await collect_meeting_evidence(
        user_id="u1", telegram_id="owner", timezone_name="Asia/Shanghai",
        candidate={"id": "event-1"}, calendar=Calendar(), gmail=gmail,
        gmail_owner_telegram_id="owner", session_factory=session_factory,
        memory_items=[], memory_degraded=[])
    assert result.availability["gmail"] == availability


@pytest.mark.asyncio
async def test_stale_and_superseded_memory_dropped(session_factory):
    memory = [{"source_type": "structured_memory", "source_id": "old", "text": "meeting 15:00",
               "currentness": "superseded"},
              {"source_type": "transcript", "source_id": "hidden", "text": "meeting 15:00",
               "currentness": "tombstoned"},
              {"source_type": "structured_memory", "source_id": "decision", "text": "use plan B",
               "currentness": "current", "kind": "decision"}]
    result = await collect_meeting_evidence(
        user_id="u1", telegram_id="owner", timezone_name="Asia/Shanghai",
        candidate={"id": "event-1"}, calendar=Calendar(), gmail=None,
        gmail_owner_telegram_id=None, session_factory=session_factory,
        memory_items=memory, memory_degraded=[])
    ids = {item.source_id for item in result.evidence}
    assert "old" not in ids and "hidden" not in ids and "decision" in ids
    assert result.dropped_counts["memory"] == 2


@pytest.mark.asyncio
async def test_todo_selection_uses_current_scoped_candidates_not_unrelated_rows(session_factory):
    async with session_factory() as db:
        owner = await crud.get_or_create_user(db, "r6-owner")
        other = await crud.get_or_create_user(db, "r6-other")
        for index in range(210):
            await crud.create_todo(db, owner.id, f"Unrelated item {index}")
        await crud.create_todo(db, owner.id, "Project Sync benchmark")
        await crud.create_todo(db, other.id, "Project Sync private distractor")
        await db.commit()
        owner_id = owner.id
    result = await collect_meeting_evidence(
        user_id=owner_id, telegram_id="owner", timezone_name="Asia/Shanghai",
        candidate={"id": "event-1"}, calendar=Calendar(), gmail=None,
        gmail_owner_telegram_id=None, session_factory=session_factory,
        memory_items=[], memory_degraded=[])
    selected = [item.normalized_fact["title"] for item in result.evidence if item.source_type == "todo"]
    assert selected == ["Project Sync benchmark"]


@pytest.mark.asyncio
async def test_evidence_budget_drops_tail_without_dropping_calendar_anchor(session_factory):
    memory = [{"source_type": "structured_memory", "source_id": str(index),
               "kind": "project", "currentness": "current", "text": "Project Sync " + "detail " * 120}
              for index in range(20)]
    result = await collect_meeting_evidence(
        user_id="u1", telegram_id="owner", timezone_name="Asia/Shanghai",
        candidate={"id": "event-1"}, calendar=Calendar(), gmail=None,
        gmail_owner_telegram_id=None, session_factory=session_factory,
        memory_items=memory, memory_degraded=[], token_budget=512)
    assert result.evidence_tokens <= 512
    assert result.evidence[0].source_type == "calendar"
    assert result.dropped_counts["memory"] > 0
    assert result.collection_latency_ms is not None


@pytest.mark.parametrize("claim", [
    MeetingClaim(kind="SOURCE_FACT", text="invented", source_keys=["X"]),
    MeetingClaim(kind="SOURCE_FACT", text="meeting 15:00", source_keys=["C1"],
                 fact_field="start", fact_value="2026-10-08T15:00:00+08:00"),
    MeetingClaim(kind="SOURCE_FACT", text="Alice promised launch", source_keys=["G1"],
                 evidence_quote="promised launch"),
    MeetingClaim(kind="SOURCE_FACT", text="email says done", source_keys=["G1"]),
    MeetingClaim(kind="SYNTHESIS", text="decided", source_keys=[]),
    MeetingClaim(kind="RECOMMENDATION", text="must launch", fact_field="start"),
])
def test_unsupported_claim_and_fabricated_quote_rejected(claim):
    gmail = envelope("G1", "gmail", text="Alice asked for benchmark",
                     owner="gmail:self@example.test")
    with pytest.raises(GroundingError):
        validate_meeting_brief(MeetingBrief(claims=[claim]), bundle(calendar_source(), gmail))


def test_calendar_authority_over_old_memory_and_todo_current_state():
    old = envelope("M1", "transcript", text="Meeting starts 15:00", currentness="historical")
    todo = envelope("T1", "todo", fact={"title": "Benchmark", "status": "completed"})
    claims = [MeetingClaim(kind="SOURCE_FACT", text="Meeting time", source_keys=["C1"],
                           fact_field="start", fact_value="2026-10-08T16:00:00+08:00"),
              MeetingClaim(kind="SOURCE_FACT", text="Task done", source_keys=["T1"],
                           fact_field="status", fact_value="completed")]
    result = render_meeting_brief(MeetingBrief(claims=claims), bundle(calendar_source(), old, todo), None)
    assert "16:00" in result and "15:00" not in result and "completed" in result


def test_conflicting_gmail_threads_remain_attributable():
    friday = envelope("G1", "gmail", text="Launch Friday", owner="gmail:self@example.test")
    monday = envelope("G2", "gmail", text="Launch Monday", owner="gmail:self@example.test")
    claims = [MeetingClaim(kind="SOURCE_FACT", text="Friday proposed", source_keys=["G1"],
                           evidence_quote="Launch Friday"),
              MeetingClaim(kind="SOURCE_FACT", text="Monday proposed", source_keys=["G2"],
                           evidence_quote="Launch Monday"),
              MeetingClaim(kind="SYNTHESIS", text="Launch date conflicts", source_keys=["G1", "G2"])]
    result = render_meeting_brief(MeetingBrief(claims=claims), bundle(calendar_source(), friday, monday), None)
    assert "Launch Friday" in result and "Launch Monday" in result


@pytest.mark.asyncio
async def test_anchor_changed_during_collection_cannot_finalize():
    calendar = Calendar()
    source = bundle(calendar_source())
    assert await revalidate_anchor(calendar, source)
    calendar.snapshot = event(hour=15)
    calendar.snapshot = calendar.snapshot.model_copy(update={"etag": '"v2"'})
    assert not await revalidate_anchor(calendar, source)


def test_wrong_owner_and_stale_memory_cannot_cite():
    wrong = envelope("G1", "gmail", text="private", owner="gmail:other@example.test")
    stale = envelope("M1", "structured_memory", text="Old decision", currentness="stale")
    for source in (wrong, stale):
        with pytest.raises(GroundingError):
            validate_meeting_brief(MeetingBrief(claims=[MeetingClaim(
                kind="SOURCE_FACT", text="private", source_keys=[source.key], evidence_quote=source.evidence_text)]),
                bundle(calendar_source(), source))


def test_summary_cannot_be_exact_quote_or_current_domain_fact():
    summary = envelope("M1", "summary", text="Meeting may be 15:00", currentness="historical")
    with pytest.raises(GroundingError):
        validate_meeting_brief(MeetingBrief(claims=[MeetingClaim(
            kind="SOURCE_FACT", text="meeting 15:00", source_keys=["M1"],
            evidence_quote="Meeting may be 15:00")]), bundle(calendar_source(), summary))


class FixedLLM:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def with_structured_output(self, schema):
        assert issubclass(schema, MeetingBrief)
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.result


@pytest.mark.asyncio
async def test_model_output_is_validated_not_rendered_bare():
    model = FixedLLM({"claims": [{"kind": "SOURCE_FACT", "text": "invented time",
                                  "source_keys": ["C1"], "fact_field": "start", "fact_value": "15:00"}]})
    with pytest.raises(GroundingError):
        await synthesize_meeting_brief(bundle(calendar_source()), "Prepare me", None, model)
    assert model.calls == 2


@pytest.mark.asyncio
async def test_bounded_repair_receives_rejected_claim_and_exact_validation_error():
    class RepairingLLM(FixedLLM):
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"claims": [{"kind": "SOURCE_FACT", "text": "unsupported description",
                                    "source_keys": ["C1"], "fact_field": "text",
                                    "fact_value": "not a source field"}]}
            import json
            assert messages[-2].type == "ai" and json.loads(messages[-2].content)["claims"][0]["fact_field"] == "text"
            assert "literal_error" in messages[-1].content
            assert "evidence_quote=null" in messages[-1].content
            return {"claims": [{"kind": "SOURCE_FACT", "text": "meeting title",
                                "source_keys": ["C1"], "fact_field": "title",
                                "fact_value": "Project Sync", "evidence_quote": None}]}
    model = RepairingLLM(None)
    result = await synthesize_meeting_brief(bundle(calendar_source()), "Prepare me", None, model)
    assert result.claims[0].fact_field == "title" and model.calls == 2


def test_generation_schema_limits_source_keys_and_fields_without_changing_storage_contract():
    from pydantic import ValidationError
    from app.agent.meeting_prep_full import _generation_schema
    schema = _generation_schema(bundle(calendar_source()))
    valid = {"claims": [{"kind": "SOURCE_FACT", "text": "title", "source_keys": ["C1"],
                         "fact_field": "title", "fact_value": "Project Sync"}]}
    parsed = schema.model_validate(valid)
    validate_meeting_brief(MeetingBrief.model_validate(parsed.model_dump()), bundle(calendar_source()))
    for field, value in (("fact_field", "text"), ("source_keys", ["OTHER_USER_SOURCE"])):
        invalid = {"claims": [{**valid["claims"][0], field: value}]}
        with pytest.raises(ValidationError):
            schema.model_validate(invalid)
    # Correct field with a fabricated value still fails the independent validator.
    parsed.claims[0].fact_value = "invented"
    with pytest.raises(GroundingError):
        validate_meeting_brief(parsed, bundle(calendar_source()))
