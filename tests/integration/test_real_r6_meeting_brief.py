"""Opt-in real-model R6 synthesis over harmless synthetic evidence."""
from datetime import datetime, timezone

import pytest

from app.agent.meeting_prep_full import (EvidenceEnvelope, MeetingEvidenceBundle,
    render_meeting_brief, synthesize_meeting_brief, validate_meeting_brief)
from app.agent.nodes import _build_chat_llm
from app.config import get_settings

pytestmark = [pytest.mark.llm_integration, pytest.mark.asyncio]


@pytest.mark.parametrize("model_options", [{"meeting_structured": True}, {}],
                         ids=["current-prepare-meeting", "ordinary-model-control"])
async def test_real_model_structured_grounded_brief_without_tools(model_options):
    now = datetime.now(timezone.utc)
    sources = [
        EvidenceEnvelope(key="C1", source_type="calendar", source_id="synthetic-event",
            source_sub_id="synthetic-calendar", owner_scope="calendar:synthetic-account:synthetic-calendar",
            authority_domain="calendar_event", retrieved_at=now, currentness="current",
            safe_display_label="Calendar: Test Sync · Oct 08",
            normalized_fact={"title": "Test Sync", "start": "2026-10-08T16:00:00+08:00",
                             "end": "2026-10-08T17:00:00+08:00", "status": "confirmed"}),
        EvidenceEnvelope(key="G1", source_type="gmail", source_id="synthetic-message",
            source_sub_id="synthetic-thread", owner_scope="gmail:self@example.test",
            authority_domain="gmail_message", retrieved_at=now, currentness="current",
            safe_display_label="Gmail: Alice · benchmark · Oct 07",
            evidence_text="Subject: benchmark\nAlice requested benchmark results for Test Sync."),
        EvidenceEnvelope(key="M1", source_type="structured_memory", source_id="synthetic-memory",
            owner_scope="user:synthetic-user", authority_domain="conversation_context",
            retrieved_at=now, currentness="historical",
            safe_display_label="Memory: previous decision",
            evidence_text="The team previously preferred plan B."),
    ]
    bundle = MeetingEvidenceBundle(
        user_id="synthetic-user", calendar_account="synthetic-account",
        calendar_id="synthetic-calendar", event_id="synthetic-event", event_etag='"v1"',
        gmail_mailbox="self@example.test", evidence=sources,
        availability={"calendar": "available", "gmail": "available", "memory": "available"},
        selected_counts={"calendar": 1, "gmail": 1, "structured_memory": 1},
        dropped_counts={}, evidence_tokens=150)
    settings = get_settings()
    model = _build_chat_llm(settings.llm_provider, settings.llm_model, **model_options)
    brief = await synthesize_meeting_brief(
        bundle, "Prepare me for Test Sync; include the verified meeting time, email request, and an agenda recommendation.",
        None, model)
    assert brief.claims
    validate_meeting_brief(brief, bundle)
    result = render_meeting_brief(brief, bundle, None)
    assert "16:00" in result and "15:00" not in result
    assert "synthetic-message" not in result and "synthetic-memory" not in result


async def test_current_meeting_model_handles_three_bounded_synthetic_sources():
    now = datetime.now(timezone.utc)
    bundle = MeetingEvidenceBundle(
        user_id="synthetic-user", calendar_account="synthetic-account",
        calendar_id="synthetic-calendar", event_id="synthetic-event", event_etag='"v1"',
        gmail_mailbox="self@example.test",
        evidence=[
            EvidenceEnvelope(
                key="C1", source_type="calendar", source_id="synthetic-event",
                source_sub_id="synthetic-calendar",
                owner_scope="calendar:synthetic-account:synthetic-calendar",
                authority_domain="calendar_event", retrieved_at=now,
                currentness="current", safe_display_label="Calendar: Project review",
                normalized_fact={"title": "Project review",
                                 "start": "2026-10-20T16:30:00+08:00",
                                 "end": "2026-10-20T16:40:00+08:00",
                                 "status": "confirmed"}),
            EvidenceEnvelope(
                key="G1", source_type="gmail", source_id="synthetic-message",
                source_sub_id="synthetic-thread", owner_scope="gmail:self@example.test",
                authority_domain="gmail_message", retrieved_at=now,
                currentness="current", safe_display_label="Gmail: review request",
                evidence_text=("Subject: Project review\n"
                               "Please bring the release checklist and discuss the test results.\n"
                               + "Background context for the discussion. " * 35)),
            EvidenceEnvelope(
                key="T1", source_type="todo", source_id="synthetic-todo",
                owner_scope="user:synthetic-user", authority_domain="todo",
                retrieved_at=now, currentness="current",
                safe_display_label="Todo: prepare checklist",
                normalized_fact={"title": "Prepare release checklist",
                                 "status": "pending"}),
        ],
        availability={"calendar": "available", "gmail": "available", "todo": "available"},
        selected_counts={"calendar": 1, "gmail": 1, "todo": 1},
        dropped_counts={}, evidence_tokens=900,
    )
    settings = get_settings()
    model = _build_chat_llm(settings.llm_provider, settings.llm_model,
                            meeting_structured=True)
    brief = await synthesize_meeting_brief(
        bundle, "Prepare a concise, source-checked draft for Project review.", None, model)
    validate_meeting_brief(brief, bundle)
    assert render_meeting_brief(brief, bundle, None, strict_factual=True)
