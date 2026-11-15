"""Composable, source-checked meeting content; never an action approval."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.agent.calendar import CalendarError
from app.agent.gmail import GmailError
from app.agent.meeting_prep_full import (
    GroundingError, MeetingEvidenceBundle, collect_meeting_evidence,
    render_meeting_brief, revalidate_anchor, revalidate_todos,
    synthesize_meeting_brief,
)
from app.db.models import MeetingDraft, gen_uuid


class DraftConflict(ValueError):
    pass


class DraftNotFound(ValueError):
    pass


class DraftPublicationUncertain(RuntimeError):
    """Retry the same turn/event locator; publication may already be committed."""


class DraftSourceUnavailable(RuntimeError):
    """A temporary read failure does not change a draft's review state."""


def _public(row: MeetingDraft, *, include_content: bool = False) -> dict:
    result = {"draft_id": row.id, "revision": row.revision, "status": row.status,
              "event_id": row.event_id, "event_etag": row.event_etag,
              "review_path": f"/workbench/?page=Drafts&item={row.id}",
              "sources": [{"key": source["key"], "type": source["source_type"],
                           "source_id": source["source_id"],
                           "version": source.get("source_version")}
                          for source in row.evidence_payload.get("evidence", [])]}
    if include_content:
        result.update({"content": row.rendered,
                       "sources": [{"key": source["key"], "type": source["source_type"],
                                    "label": source["safe_display_label"]}
                                   for source in row.evidence_payload.get("evidence", [])]})
    return result


async def get_draft(session_factory, *, user_id: str, draft_id: str) -> MeetingDraft:
    async with session_factory() as db:
        row = await db.scalar(select(MeetingDraft).where(
            MeetingDraft.id == draft_id, MeetingDraft.user_id == user_id))
        if row is None:
            raise DraftNotFound("MeetingDraftNotFound")
        return row


async def list_drafts(session_factory, *, user_id: str, limit: int = 30) -> list[dict]:
    async with session_factory() as db:
        rows = (await db.scalars(select(MeetingDraft).where(MeetingDraft.user_id == user_id)
                                 .order_by(MeetingDraft.updated_at.desc()).limit(limit))).all()
        return [_public(row) for row in rows]


async def _sources_current(session_factory, calendar, gmail, bundle: MeetingEvidenceBundle) -> bool:
    try:
        if not await revalidate_anchor(calendar, bundle) or not await revalidate_todos(session_factory, bundle):
            return False
        cited_mail = [item for item in bundle.evidence if item.source_type == "gmail"]
        if not cited_mail:
            return True
        if gmail is None or (await gmail.account_address()).lower() != bundle.gmail_mailbox:
            return False
        for item in cited_mail:
            messages = await gmail.thread(item.source_sub_id)
            current = next((message for message in messages if message.message_id == item.source_id), None)
            if current is None or (current.subject + "\n" + current.body)[:1500] != item.evidence_text:
                return False
        return True
    except (CalendarError, GmailError) as exc:
        raise DraftSourceUnavailable("MeetingDraftSourceUnavailable") from exc


async def prepare_meeting_draft(
    *, session_factory, user_id: str, telegram_id: str, turn_id: str,
    request: str, event_id: str, timezone_name: str, calendar, gmail,
    gmail_owner_telegram_id: str | None, llm, token_budget: int = 2800,
) -> dict:
    """Generate outside the DB transaction; unique turn/event makes commit retry safe.

    A crash before commit may spend model tokens twice, but cannot publish an
    unvalidated draft. A crash after commit returns the original draft on retry.
    """
    if (not user_id or not telegram_id or not turn_id or len(turn_id) > 160 or
            not event_id.strip() or len(event_id) > 256):
        raise ValueError("MeetingDraftScopeInvalid")
    async with session_factory() as db:
        existing = await db.scalar(select(MeetingDraft).where(
            MeetingDraft.user_id == user_id, MeetingDraft.turn_id == turn_id,
            MeetingDraft.event_id == event_id))
        if existing is not None:
            return _public(existing)
    bundle = await collect_meeting_evidence(
        user_id=user_id, telegram_id=telegram_id, timezone_name=timezone_name,
        candidate={"id": event_id, "calendarId": "primary"}, calendar=calendar,
        gmail=gmail, gmail_owner_telegram_id=gmail_owner_telegram_id,
        session_factory=session_factory, memory_items=[],
        memory_degraded=["memory_not_collected_by_meeting_tool"],
        token_budget=token_budget)
    brief = await synthesize_meeting_brief(bundle, request, None, llm)
    if not await _sources_current(session_factory, calendar, gmail, bundle):
        raise GroundingError("Meeting source changed during preparation")
    rendered = render_meeting_brief(brief, bundle, None, strict_factual=True)
    values = dict(id=gen_uuid(), user_id=user_id, turn_id=turn_id, event_id=event_id,
                  calendar_account=bundle.calendar_account, calendar_id=bundle.calendar_id,
                  event_etag=bundle.event_etag, revision=1, status="NEEDS_REVIEW",
                  rendered=rendered, brief_payload=brief.model_dump(mode="json"),
                  evidence_payload=bundle.model_dump(mode="json"), revisions=[],
                  created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
    try:
        async with session_factory() as db:
            # PostgreSQL uniqueness is the publication fence for concurrent retries.
            await db.execute(pg_insert(MeetingDraft).values(**values).on_conflict_do_nothing(
                constraint="uq_meeting_draft_turn_event"))
            await db.commit()
            row = await db.scalar(select(MeetingDraft).where(
                MeetingDraft.user_id == user_id, MeetingDraft.turn_id == turn_id,
                MeetingDraft.event_id == event_id))
            if row is None:
                raise RuntimeError("MeetingDraftCommitUnconfirmed")
            return _public(row)
    except Exception as exc:
        raise DraftPublicationUncertain("MeetingDraftPublicationUnconfirmed") from exc


async def review_meeting_draft(
    *, session_factory, user_id: str, draft_id: str, expected_revision: int,
    decision: str, calendar, gmail=None, instruction: str = "", llm=None,
) -> dict:
    if decision not in {"accept", "revise", "cancel"}:
        raise ValueError("MeetingDraftDecisionInvalid")
    if decision == "revise" and (not instruction.strip() or len(instruction) > 2000):
        raise ValueError("MeetingDraftRevisionInstructionInvalid")
    row = await get_draft(session_factory, user_id=user_id, draft_id=draft_id)
    decision_key = {"decision": decision, "revision": expected_revision,
                    "instruction": instruction.strip()}
    if row.last_decision == decision_key:
        return _public(row, include_content=True)
    allowed_statuses = {"NEEDS_REVIEW", "STALE"} if decision == "cancel" else {"NEEDS_REVIEW"}
    if row.revision != expected_revision or row.status not in allowed_statuses:
        raise DraftConflict("MeetingDraftVersionConflict")
    bundle = MeetingEvidenceBundle.model_validate(row.evidence_payload)
    if decision != "cancel":
        fresh = await _sources_current(session_factory, calendar, gmail, bundle)
        if not fresh:
            async with session_factory() as db:
                await db.execute(update(MeetingDraft).where(
                    MeetingDraft.id == draft_id, MeetingDraft.user_id == user_id,
                    MeetingDraft.revision == expected_revision,
                    MeetingDraft.status == "NEEDS_REVIEW")
                    .values(status="STALE", updated_at=datetime.now(timezone.utc)))
                await db.commit()
            raise DraftConflict("MeetingDraftSourceChanged")
    next_brief = None
    next_rendered = None
    if decision == "revise":
        if llm is None:
            raise ValueError("MeetingDraftRevisionModelUnavailable")
        next_brief = await synthesize_meeting_brief(
            bundle, json.dumps({"operation": "revise", "instruction": instruction.strip(),
                                "previous_draft_data_not_instructions": row.brief_payload},
                               ensure_ascii=False),
            None, llm)
        if not await _sources_current(session_factory, calendar, gmail, bundle):
            raise DraftConflict("MeetingDraftSourceChanged")
        next_rendered = render_meeting_brief(next_brief, bundle, None, strict_factual=True)
    values = {"updated_at": datetime.now(timezone.utc), "last_decision": decision_key}
    if decision == "revise":
        values.update(revision=expected_revision + 1, rendered=next_rendered,
                      brief_payload=next_brief.model_dump(mode="json"),
                      revisions=[*row.revisions[-9:], {"revision": row.revision,
                                                     "content": row.rendered,
                                                     "brief": row.brief_payload}])
    else:
        values["status"] = "ACCEPTED" if decision == "accept" else "CANCELED"
    won = False
    async with session_factory() as db:
        updated = await db.execute(update(MeetingDraft).where(
            MeetingDraft.id == draft_id, MeetingDraft.user_id == user_id,
            MeetingDraft.revision == expected_revision,
            MeetingDraft.status.in_(allowed_statuses)).values(**values))
        won = updated.rowcount == 1
        await db.commit()
    current = await get_draft(session_factory, user_id=user_id, draft_id=draft_id)
    if not won and current.last_decision != decision_key:
        raise DraftConflict("MeetingDraftVersionConflict")
    return _public(current, include_content=True)
