"""R5.2 Calendar create: one approved action, one stable remote resource ID.

The Calendar adapter owns transport, Saver/HITL owns authorization, and the
evidence store joins provider outcome with ActionExecution terminal truth.
No Google/LLM I/O occurs while a PostgreSQL row transaction is open.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select

from app.agent.actions import ActionProposal, ActionResult, ActionStatus
from app.agent.calendar import CalendarAdapter, CalendarError
from app.agent.calendar_action_evidence import CalendarActionEvidenceStore, CalendarEvidenceIntent
from app.agent.calendar_conditional import CalendarProviderError
from app.db.models import CalendarActionEvidence

logger = logging.getLogger(__name__)
_LEASE_SECONDS = 120
_MAX_PROVIDER_ATTEMPTS = 2
_SETTLING_DELAYS = (0.0, 0.25, 0.5)


def _same_instant(expected: str, observed: dict) -> bool:
    value = observed.get("dateTime") or observed.get("date")
    if not isinstance(value, str):
        return False
    if "T" not in expected or "T" not in value:
        return expected[:10] == value[:10]
    try:
        left, right = datetime.fromisoformat(expected), datetime.fromisoformat(value)
        return (left.tzinfo is not None and right.tzinfo is not None and
                left.astimezone(timezone.utc) == right.astimezone(timezone.utc))
    except ValueError:
        return False


def _matches_approved_create(event: object, row: CalendarActionEvidence) -> bool:
    """A provider ID alone (including 409) never proves this action succeeded."""
    if event is None:
        return False
    data = event if isinstance(event, dict) else event.model_dump(by_alias=True)
    body = row.immutable_body or {}
    if (data.get("id") != row.event_id or data.get("status") != "confirmed" or
            data.get("summary") != body.get("summary") or
            (data.get("calendarId") is not None and
             data.get("calendarId") != row.calendar_id) or
            (data.get("accountId") is not None and
             data.get("accountId") != row.account_identity) or
            data.get("extendedProperties", {}).get("private", {}).get("assistant_action") !=
            row.private_action_marker or
            not _same_instant(body.get("start", ""), data.get("start") or {}) or
            not _same_instant(body.get("end", ""), data.get("end") or {})):
        return False
    try:
        zone = ZoneInfo(body["timeZone"])
        for key in ("start", "end"):
            planned = datetime.fromisoformat(body[key])
            if planned.tzinfo is None or planned.utcoffset() != planned.astimezone(zone).utcoffset():
                return False
            remote_zone = (data.get(key) or {}).get("timeZone")
            if remote_zone and remote_zone != body["timeZone"]:
                return False
    except (KeyError, ValueError, TypeError):
        return False
    return True


class CalendarCreateCoordinator:
    def __init__(self, calendar: CalendarAdapter, session_factory,
                 *, sleep=asyncio.sleep):
        self.calendar = calendar
        self.store = CalendarActionEvidenceStore(session_factory)
        self.session_factory = session_factory
        self.sleep = sleep

    async def _status(self, action_id: str) -> ActionResult:
        row, ledger = await self.store.load_create(action_id)
        if row is None and ledger is not None:
            if ledger.status == "executed":
                return ActionResult(action_id=action_id, status=ActionStatus.EXECUTED,
                                    message="Previously completed Calendar action.",
                                    result=ledger.result or {})
            return ActionResult(action_id=action_id, status=ActionStatus.EXECUTING,
                                message="Earlier Calendar outcome is unresolved; review required.",
                                result={"remote_state": "LEGACY_UNRESOLVED"})
        if row is None or ledger is None:
            return ActionResult(action_id=action_id, status=ActionStatus.EXECUTING,
                                message="Calendar action evidence is unavailable; review required.",
                                result={"remote_state": "EVIDENCE_UNAVAILABLE"})
        if row.outcome == "CONFIRMED" and ledger.status == "executed":
            return ActionResult(action_id=action_id, status=ActionStatus.EXECUTED,
                                message="Calendar event confirmed.", result=ledger.result or {})
        if row.outcome == "FAILED_DEFINITE" and ledger.status == "failed":
            return ActionResult(action_id=action_id, status=ActionStatus.FAILED,
                                message="Calendar create was definitely rejected before insertion.",
                                result={"failure_category": row.failure_category})
        if ledger.status in {"executed", "failed"}:
            logger.error("action_id=%s Calendar terminal ledger/evidence mismatch", action_id)
            return ActionResult(action_id=action_id, status=ActionStatus.EXECUTING,
                                message="Calendar outcome requires operator review.",
                                result={"remote_state": "STATE_CONFLICT"})
        return ActionResult(action_id=action_id, status=ActionStatus.EXECUTING,
                            message="Calendar outcome is not yet confirmed; do not submit a replacement action.",
                            result={"remote_state": row.outcome})

    async def execute(self, proposal: ActionProposal) -> ActionResult:
        if proposal.domain != "calendar" or proposal.operation != "create_event":
            raise ValueError("Expected approved Calendar create")
        row, ledger = await self.store.load_create(proposal.action_id)
        if row is None and ledger is not None:
            return await self._status(proposal.action_id)
        try:
            if row is None:
                account, calendar_id = await self.calendar.resolve_create_scope(
                    proposal.args.get("calendarId", "primary"))
            else:
                account, calendar_id = row.account_identity, row.calendar_id
            intent = CalendarEvidenceIntent.from_approved_proposal(
                proposal, account, calendar_id)
            body = {"account": account, "calendarId": calendar_id,
                    "summary": proposal.args.get("summary") or proposal.args["title"],
                    "start": proposal.args["start"], "end": proposal.args["end"],
                    "timeZone": proposal.args["timeZone"]}
            await self.store.prepare_create(intent, body, proposal.action_id)
        except ValueError as exc:
            logger.warning("action_id=%s create preparation category=%s",
                           proposal.action_id, type(exc).__name__)
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.EXECUTING,
                                message="Approved Calendar create identity changed; review required.",
                                result={"remote_state": "APPROVAL_MISMATCH"})
        except (KeyError, CalendarError, CalendarProviderError) as exc:
            logger.warning("action_id=%s create preparation category=%s",
                           proposal.action_id, type(exc).__name__)
            if row is None:
                return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED,
                                    message="Calendar unavailable before provider dispatch.",
                                    result={"failure_category": "preparation_unavailable"})
            return await self._status(proposal.action_id)
        return await self.recover(proposal.action_id)

    async def recover(self, action_id: str) -> ActionResult:
        row, _ = await self.store.load_create(action_id)
        if row is None or row.operation != "create_event":
            return await self._status(action_id)
        owner = f"calendar-create:{uuid4().hex}"
        if row.outcome == "NOT_STARTED":
            token = await self.store.arm(action_id, owner, _LEASE_SECONDS)
            if token:
                return await self._send(action_id, token)
        elif row.outcome in {"IN_FLIGHT", "UNCERTAIN"}:
            token = await self.store.claim_reconciliation(action_id, owner, _LEASE_SECONDS)
            if token:
                return await self._reconcile(action_id, token, allow_retry=True)
        return await self._status(action_id)

    async def recover_due(self, limit: int = 20) -> int:
        """Bounded startup/maintenance sweep; no new action or provider ID."""
        now = datetime.now(timezone.utc)
        async with self.session_factory() as db:
            ids = (await db.scalars(select(CalendarActionEvidence.action_id).where(
                CalendarActionEvidence.operation == "create_event",
                or_(CalendarActionEvidence.outcome == "NOT_STARTED",
                    ((CalendarActionEvidence.outcome.in_(["IN_FLIGHT", "UNCERTAIN"])) &
                     or_(CalendarActionEvidence.lease_until.is_(None),
                         CalendarActionEvidence.lease_until <= now))))
                .order_by(CalendarActionEvidence.created_at).limit(limit))).all()
        for action_id in ids:
            try:
                await self.recover(action_id)
            except Exception as exc:
                logger.warning("action_id=%s create recovery category=%s",
                               action_id, type(exc).__name__)
        return len(ids)

    async def _send(self, action_id: str, token: str) -> ActionResult:
        row, _ = await self.store.load_create(action_id)
        if row is None or not await self.store.mark_dispatch_started(action_id, token):
            return await self._status(action_id)
        try:
            response = await self.calendar.create_event_prepared(
                row.event_id, row.immutable_body, row.private_action_marker)
        except CalendarProviderError as exc:
            if exc.status_code in {401, 403}:
                if row.attempt_no == 1:
                    await self.store.finalize_create(
                        action_id, token, "FAILED_DEFINITE",
                        failure_category="provider_permission_denied")
                    return await self._status(action_id)
                # A previous dispatched attempt may have succeeded despite
                # losing its response. This rejection proves only this attempt.
                return await self._reconcile(action_id, token, allow_retry=False)
            if exc.category == "similarity_precheck":
                if row.attempt_no == 1:
                    await self.store.finalize_create(action_id, token, "FAILED_DEFINITE",
                                                     failure_category="similarity_precheck")
                    return await self._status(action_id)
                # Only this retry was rejected before insert. An earlier armed
                # attempt may already have succeeded, so it remains uncertain.
                return await self._reconcile(action_id, token, allow_retry=False)
            # MCP 409 is only a reason to read back the exact planned ID.
            return await self._reconcile(action_id, token,
                                         allow_retry=exc.category != "resource_conflict")
        except Exception as exc:
            logger.warning("action_id=%s create transport category=%s", action_id, type(exc).__name__)
            return await self._reconcile(action_id, token, allow_retry=True)
        event = response.get("event") if isinstance(response, dict) else None
        if _matches_approved_create(event, row):
            await self.store.finalize_create(action_id, token, "CONFIRMED",
                                             confirmation_source="provider_response")
            return await self._status(action_id)
        # A malformed/mismatched success response is not a definite failure.
        return await self._reconcile(action_id, token, allow_retry=False)

    async def _reconcile(self, action_id: str, token: str,
                         *, allow_retry: bool) -> ActionResult:
        row, _ = await self.store.load_create(action_id)
        if row is None:
            return await self._status(action_id)
        for delay in _SETTLING_DELAYS:
            if delay:
                await self.sleep(delay)
            try:
                event = await self.calendar.get_event(row.calendar_id, row.event_id)
            except CalendarError as exc:
                if str(exc) != "not_found_or_inaccessible":
                    await self.store.finalize(action_id, token, "UNCERTAIN",
                                              failure_category="provider_unavailable")
                    return await self._status(action_id)
                try:
                    account, resolved = await self.calendar.resolve_create_scope(row.calendar_id)
                except CalendarError:
                    await self.store.finalize(action_id, token, "UNCERTAIN",
                                              failure_category="calendar_scope_unavailable")
                    return await self._status(action_id)
                if (account, resolved) != (row.account_identity, row.calendar_id):
                    await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                              failure_category="calendar_scope_unavailable")
                    return await self._status(action_id)
                continue
            if _matches_approved_create(event, row):
                await self.store.finalize_create(action_id, token, "CONFIRMED",
                                                 confirmation_source="exact_readback")
            else:
                await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                          failure_category="resource_conflict")
            return await self._status(action_id)
        if allow_retry and row.attempt_no < _MAX_PROVIDER_ATTEMPTS:
            # The original fence remains current; reuse identical immutable body/ID.
            if row.outcome == "IN_FLIGHT":
                await self.store.finalize(action_id, token, "UNCERTAIN",
                                          failure_category="remote_outcome_uncertain")
                token = await self.store.claim_reconciliation(action_id, f"calendar-retry:{uuid4().hex}",
                                                              _LEASE_SECONDS)
            if token:
                next_token = await self.store.retry_create_after_verified_absence(
                    action_id, token, max_attempts=_MAX_PROVIDER_ATTEMPTS,
                    lease_seconds=_LEASE_SECONDS)
                if next_token:
                    return await self._send(action_id, next_token)
        await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                  failure_category="manual_review_required")
        return await self._status(action_id)
