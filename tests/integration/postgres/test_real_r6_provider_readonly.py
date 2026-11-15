"""Optional Calendar→Gmail→brief acceptance on an existing safe test meeting.

The bounded discovery uses only test-labelled Calendar candidates. It never
prints titles, addresses, message contents or model input. No provider writes.
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.agent.calendar import MCPCalendarAdapter, discover_calendar_tools
from app.agent.calendar_conditional import GoogleCalendarConditionalTransport, McpOAuthTokenSource
from app.agent.gmail import OAuthGmailSource
from app.agent.meeting_prep_full import (collect_meeting_evidence, revalidate_anchor,
    revalidate_todos, render_meeting_brief, synthesize_meeting_brief)
from app.agent.nodes import _build_chat_llm
from app.config import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.calendar_integration,
              pytest.mark.gmail_integration, pytest.mark.llm_integration, pytest.mark.asyncio]


_SAFE_TEST_TITLE = re.compile(r"(?i)(?:\bppa\b|\br[456](?:\.|\b)|\btest\b|\bsynthetic\b|\bdemo\b|测试|验收|演示)")
_BOUNDED_QUERIES = ("PPA", "R6", "test", "测试", "验收")


async def _safe_candidates(calendar, gmail, settings):
    zone = ZoneInfo(settings.default_timezone)
    today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    begin = (today - timedelta(days=60)).isoformat()
    end = (today + timedelta(days=30)).isoformat()
    self_address = (await gmail.account_address()).lower()
    seen = set()
    safe = []
    for query in _BOUNDED_QUERIES:
        for row in (await calendar.search_events(query, begin, end))[:50]:
            event_id = row.get("id")
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            if not _SAFE_TEST_TITLE.search(str(row.get("summary") or "")):
                continue
            # Exact GET is required before attendee identity or Gmail lookup.
            scope = str(row.get("calendarId") or "primary")
            account, calendar_id = await calendar.resolve_read_scope(scope)
            exact = await calendar.get_event(calendar_id, event_id)
            if (not account or exact.id != event_id or exact.status != "confirmed"
                    or not exact.etag or not _SAFE_TEST_TITLE.search(exact.summary or "")):
                continue
            external = {str(person.get("email") or "").strip().lower()
                        for person in exact.attendees if isinstance(person, dict)}
            external.discard(self_address)
            external.discard("")
            if external and len(external) <= 5:
                safe.append({"id": event_id, "calendarId": scope})
            if len(safe) >= 3:
                return safe, len(seen)
    return safe, len(seen)


async def test_existing_meeting_full_readonly_provider_path(pg_harness):
    if os.environ.get("R6_PRIVATE_PROVIDER_MODEL_APPROVED") != "1":
        pytest.skip("Explicit private Calendar/Gmail-to-model authorization is absent")
    settings = get_settings()
    if not (settings.gmail_enabled and settings.gmail_token_file and
            Path(settings.gmail_token_file).is_file() and
            settings.gmail_owner_telegram_id and settings.google_calendar_owner_telegram_id and
            settings.gmail_owner_telegram_id == settings.google_calendar_owner_telegram_id):
        pytest.skip("No shared authorized Calendar/Gmail owner")
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = MCPCalendarAdapter(
        await discover_calendar_tools(settings),
        conditional=GoogleCalendarConditionalTransport(
            McpOAuthTokenSource(settings.google_credentials_file)))
    gmail = OAuthGmailSource(Path(settings.gmail_token_file))
    candidates, scanned = await _safe_candidates(calendar, gmail, settings)
    if not candidates:
        pytest.skip(f"SAFE_FIXTURE_NOT_FOUND: no test-labelled attendee meeting among {scanned} bounded candidates")
    evidence = None
    for selected in candidates:
        trial = await collect_meeting_evidence(
            user_id=user.id, telegram_id=settings.gmail_owner_telegram_id,
            timezone_name=settings.default_timezone, candidate=selected,
            calendar=calendar, gmail=gmail,
            gmail_owner_telegram_id=settings.gmail_owner_telegram_id,
            session_factory=factory, memory_items=[], memory_degraded=[],
            token_budget=settings.meeting_prep_evidence_token_budget,
            gmail_lookback_days=settings.meeting_prep_gmail_lookback_days,
            gmail_candidate_limit=settings.meeting_prep_gmail_candidate_limit)
        if trial.selected_counts.get("gmail", 0) > 0:
            evidence = trial
            break
    if evidence is None:
        pytest.skip(f"SAFE_FIXTURE_NOT_FOUND: {len(candidates)} safe meetings, no attendee-linked Gmail evidence")
    llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    brief = await synthesize_meeting_brief(evidence, "Prepare a grounded brief for this meeting", None, llm)
    if not await revalidate_anchor(calendar, evidence) or not await revalidate_todos(factory, evidence):
        pytest.fail("Current authoritative source changed during read-only preparation")
    result = render_meeting_brief(brief, evidence, None)
    assert result and evidence.selected_counts["calendar"] == 1
    assert "gmail:" not in result and evidence.evidence_tokens <= settings.meeting_prep_evidence_token_budget
