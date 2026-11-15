"""R5.3 single-resource conditional Calendar update/delete reconciliation."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import or_, select

from app.agent.actions import ActionProposal, ActionResult, ActionStatus
from app.agent.calendar import CalendarAdapter, CalendarError
from app.agent.calendar_action_evidence import CalendarActionEvidenceStore, CalendarEvidenceIntent
from app.agent.calendar_create import _same_instant
from app.agent.calendar_conditional import CalendarEventSnapshot, CalendarProviderError
from app.db.models import CalendarActionEvidence

logger = logging.getLogger(__name__)
_LEASE_SECONDS = 120
_MAX_PROVIDER_ATTEMPTS = 2
_SETTLING_DELAYS = (0.0, 0.25, 0.5)
_META_ARGS = frozenset({"user_id", "account", "calendarId", "eventId", "display_title",
                        "source_message_id", "source_thread_id"})
_PATCH_ARGS = frozenset({"title", "summary", "start", "end", "timeZone",
                         "description", "location"})


def _approved_patch(proposal: ActionProposal) -> dict:
    if proposal.operation == "delete_event":
        if set(proposal.args) - _META_ARGS:
            raise ValueError("Delete contains unsupported multi-resource or edit fields")
        return {}
    if proposal.operation != "update_event" or set(proposal.args) - _META_ARGS - _PATCH_ARGS:
        raise ValueError("Update contains unsupported Calendar fields")
    args = proposal.args
    patch = {}
    if "title" in args or "summary" in args:
        patch["summary"] = args.get("summary") or args.get("title")
    for key in ("description", "location"):
        if key in args:
            patch[key] = args[key]
    if "start" in args or "end" in args or "timeZone" in args:
        if not all(args.get(key) for key in ("start", "end", "timeZone")):
            raise ValueError("Calendar time patch requires start/end/timeZone")
        patch["start"] = {"dateTime": args["start"], "timeZone": args["timeZone"]}
        patch["end"] = {"dateTime": args["end"], "timeZone": args["timeZone"]}
    if not patch:
        raise ValueError("Approved update has no changed fields")
    return patch


def _matches_update(event: CalendarEventSnapshot, row: CalendarActionEvidence) -> bool:
    if (event.id != row.event_id or event.status != "confirmed" or not event.etag or
            event.recurrence or event.recurring_event_id or event.original_start_time):
        return False
    patch = row.immutable_body or {}
    if "summary" in patch and event.summary != patch["summary"]:
        return False
    for key in ("description", "location"):
        if key in patch and getattr(event, key, None) != patch[key]:
            return False
    for key in ("start", "end"):
        if key in patch:
            planned = patch[key]
            observed = getattr(event, key)
            if not _same_instant(planned["dateTime"], observed):
                return False
            if observed.get("timeZone") and observed["timeZone"] != planned["timeZone"]:
                return False
    return True


class CalendarModifyCoordinator:
    def __init__(self, calendar: CalendarAdapter, session_factory, *, sleep=asyncio.sleep):
        self.calendar = calendar
        self.store = CalendarActionEvidenceStore(session_factory)
        self.session_factory = session_factory
        self.sleep = sleep

    async def status(self, action_id: str) -> ActionResult:
        row, ledger = await self.store.load_create(action_id)
        if row is None or ledger is None or row.operation not in {"update_event", "delete_event"}:
            return ActionResult(action_id=action_id, status=ActionStatus.EXECUTING,
                                message="Calendar action evidence unavailable; review required.",
                                result={"remote_state": "EVIDENCE_UNAVAILABLE"})
        if row.outcome == "CONFIRMED" and ledger.status == "executed":
            return ActionResult(action_id=action_id, status=ActionStatus.EXECUTED,
                                message="Calendar modification confirmed.", result=ledger.result or {})
        if row.outcome == "FAILED_DEFINITE" and ledger.status == "failed":
            return ActionResult(action_id=action_id, status=ActionStatus.FAILED,
                                message="Calendar modification definitely rejected.",
                                result={"failure_category": row.failure_category})
        return ActionResult(action_id=action_id, status=ActionStatus.EXECUTING,
                            message="Calendar modification is not yet confirmed; do not submit a replacement.",
                            result={"remote_state": row.outcome})

    async def execute(self, proposal: ActionProposal) -> ActionResult:
        if proposal.domain != "calendar" or proposal.operation not in {"update_event", "delete_event"}:
            raise ValueError("Expected approved Calendar modification")
        row, ledger = await self.store.load_create(proposal.action_id)
        if row is None and ledger is not None:
            return await self.status(proposal.action_id)
        try:
            baseline = proposal.approved_calendar_baseline
            if baseline is None or proposal.args.get("calendarId") != baseline.calendar_id:
                raise ValueError("Missing approved Calendar resource version")
            if row is None:
                account, resolved = await self.calendar.resolve_create_scope(baseline.calendar_id)
                if (resolved != baseline.calendar_id or
                        proposal.args.get("account") != account):
                    raise ValueError("Approved Calendar account/scope changed")
            else:
                account, resolved = row.account_identity, row.calendar_id
            intent = CalendarEvidenceIntent.from_approved_proposal(proposal, account, resolved)
            await self.store.prepare_modification(intent, _approved_patch(proposal))
        except ValueError:
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.EXECUTING,
                                message="Approved Calendar modification changed; review required.",
                                result={"remote_state": "APPROVAL_MISMATCH"})
        except CalendarError:
            if row is None:
                return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED,
                                    message="Calendar scope unavailable before provider dispatch.",
                                    result={"failure_category": "preparation_unavailable"})
            return await self.status(proposal.action_id)
        return await self.recover(proposal.action_id)

    async def recover(self, action_id: str) -> ActionResult:
        row, _ = await self.store.load_create(action_id)
        if row is None or row.operation not in {"update_event", "delete_event"}:
            return await self.status(action_id)
        owner = f"calendar-modify:{uuid4().hex}"
        if row.outcome == "NOT_STARTED":
            token = await self.store.arm(action_id, owner, _LEASE_SECONDS)
            if token:
                return await self._send(action_id, token)
        elif row.outcome in {"IN_FLIGHT", "UNCERTAIN"}:
            token = await self.store.claim_reconciliation(action_id, owner, _LEASE_SECONDS)
            if token:
                return await self._reconcile(action_id, token)
        return await self.status(action_id)

    async def recover_due(self, limit: int = 20) -> int:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as db:
            ids = (await db.scalars(select(CalendarActionEvidence.action_id).where(
                CalendarActionEvidence.operation.in_(["update_event", "delete_event"]),
                or_(CalendarActionEvidence.outcome == "NOT_STARTED",
                    ((CalendarActionEvidence.outcome.in_(["IN_FLIGHT", "UNCERTAIN"])) &
                     or_(CalendarActionEvidence.lease_until.is_(None),
                         CalendarActionEvidence.lease_until <= now))))
                .order_by(CalendarActionEvidence.created_at).limit(limit))).all()
        for action_id in ids:
            try:
                await self.recover(action_id)
            except Exception as exc:
                logger.warning("action_id=%s Calendar modification recovery category=%s",
                               action_id, type(exc).__name__)
        return len(ids)

    async def _send(self, action_id: str, token: str) -> ActionResult:
        row, _ = await self.store.load_create(action_id)
        if row is None or not await self.store.mark_dispatch_started(action_id, token):
            return await self.status(action_id)
        try:
            if row.operation == "update_event":
                event = await self.calendar.update_event_if_match(
                    row.calendar_id, row.event_id, row.approved_baseline_etag,
                    row.immutable_body)
                if _matches_update(event, row):
                    await self.store.finalize_modification(
                        action_id, token, "CONFIRMED", confirmation_source="provider_response",
                        confirmed_etag=event.etag)
                    return await self.status(action_id)
            else:
                accepted = await self.calendar.delete_event_if_match(
                    row.calendar_id, row.event_id, row.approved_baseline_etag)
                if accepted.get("accepted") and accepted.get("eventId") == row.event_id:
                    await self.store.finalize_modification(
                        action_id, token, "CONFIRMED", confirmation_source="provider_response")
                    return await self.status(action_id)
        except CalendarError as exc:
            provider_error = exc.__cause__
            if (isinstance(provider_error, CalendarProviderError) and
                    provider_error.status_code in {401, 403}):
                if row.attempt_no == 1:
                    await self.store.finalize_modification(
                        action_id, token, "FAILED_DEFINITE",
                        failure_category="provider_permission_denied")
                    return await self.status(action_id)
                # A prior write attempt remains unconfirmed; reconcile it.
                return await self._reconcile(action_id, token)
            if str(exc) == "etag_conflict":
                await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                          failure_category="etag_conflict")
                return await self.status(action_id)
        except Exception as exc:
            logger.warning("action_id=%s modification transport category=%s",
                           action_id, type(exc).__name__)
        return await self._reconcile(action_id, token)

    async def _scope_matches(self, row: CalendarActionEvidence) -> bool:
        try:
            account, resolved = await self.calendar.resolve_create_scope(row.calendar_id)
        except CalendarError:
            return False
        return (account, resolved) == (row.account_identity, row.calendar_id)

    async def _reconcile(self, action_id: str, token: str) -> ActionResult:
        row, _ = await self.store.load_create(action_id)
        if row is None:
            return await self.status(action_id)
        for delay in _SETTLING_DELAYS:
            if delay:
                await self.sleep(delay)
            try:
                event = await self.calendar.get_event(row.calendar_id, row.event_id)
            except CalendarError as exc:
                if str(exc) in {"not_found_or_inaccessible", "gone"}:
                    if not await self._scope_matches(row):
                        await self.store.finalize(action_id, token, "UNCERTAIN",
                                                  failure_category="calendar_scope_unavailable")
                    elif row.operation == "delete_event":
                        await self.store.finalize_modification(
                            action_id, token, "CONFIRMED", confirmation_source="observed_absence")
                    else:
                        await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                                  failure_category="not_found_or_inaccessible")
                else:
                    await self.store.finalize(action_id, token, "UNCERTAIN",
                                              failure_category="provider_unavailable")
                return await self.status(action_id)
            if event.id != row.event_id or not await self._scope_matches(row):
                await self.store.finalize(action_id, token, "UNCERTAIN",
                                          failure_category="calendar_scope_unavailable")
                return await self.status(action_id)
            if event.recurrence or event.recurring_event_id or event.original_start_time:
                await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                          failure_category="resource_conflict")
                return await self.status(action_id)
            if event.status == "cancelled":
                if row.operation == "delete_event":
                    await self.store.finalize_modification(
                        action_id, token, "CONFIRMED", confirmation_source="observed_cancelled")
                else:
                    await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                              failure_category="resource_conflict")
                return await self.status(action_id)
            if row.operation == "update_event" and _matches_update(event, row):
                await self.store.finalize_modification(
                    action_id, token, "CONFIRMED", confirmation_source="observed_postcondition",
                    confirmed_etag=event.etag)
                return await self.status(action_id)
            if event.etag != row.approved_baseline_etag:
                await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                          failure_category="etag_conflict")
                return await self.status(action_id)
        if row.attempt_no < _MAX_PROVIDER_ATTEMPTS:
            # An unchanged approved ETag after bounded observations is the
            # preimage proof. A retry must still use that original If-Match.
            if row.outcome == "IN_FLIGHT":
                await self.store.finalize(action_id, token, "UNCERTAIN",
                                          failure_category="remote_outcome_uncertain")
                token = await self.store.claim_reconciliation(
                    action_id, f"calendar-modify-retry:{uuid4().hex}", _LEASE_SECONDS)
            if token:
                next_token = await self.store.retry_modification_after_verified_preimage(
                    action_id, token, max_attempts=_MAX_PROVIDER_ATTEMPTS,
                    lease_seconds=_LEASE_SECONDS)
                if next_token:
                    return await self._send(action_id, next_token)
        await self.store.finalize(action_id, token, "PARKED_REVIEW",
                                  failure_category="manual_review_required")
        return await self.status(action_id)
