"""Durable approved-version and provider-attempt evidence substrate.

Create, update and delete share one evidence authority with ActionExecution.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.agent.actions import ActionProposal, ApprovedCalendarBaseline
from app.db.models import ActionExecution, CalendarActionEvidence

Operation = Literal["create_event", "update_event", "delete_event"]
Outcome = Literal["CONFIRMED", "FAILED_DEFINITE", "UNCERTAIN", "PARKED_REVIEW"]
Confirmation = Literal["provider_response", "exact_readback", "postcondition",
                       "observed_postcondition", "observed_absence", "observed_cancelled"]
SAFE_FAILURES = frozenset({
    "validation", "provider_rejected", "provider_permission_denied", "oauth_unavailable", "etag_conflict",
    "not_found_or_inaccessible", "remote_outcome_uncertain", "provider_unavailable",
    "resource_conflict", "manual_review_required", "oauth_refresh_failed",
    "invalid_identity", "approved_etag_required", "conflict", "gone",
    "missing_resource_version", "similarity_precheck", "calendar_scope_unavailable",
})
DEFINITE_FAILURES = frozenset({
    "validation", "provider_rejected", "provider_permission_denied", "oauth_unavailable", "oauth_refresh_failed",
    "etag_conflict", "invalid_identity", "approved_etag_required",
    "similarity_precheck",
})


def planned_google_event_id(action_id: str, account_identity: str, calendar_id: str) -> str:
    """Stable, collision-resistant and Google base32hex-valid create identity."""
    if not all((action_id, account_identity, calendar_id)):
        raise ValueError("Calendar provider identity is incomplete")
    digest = hashlib.sha256(json.dumps(
        ["r5v1", account_identity, calendar_id, action_id],
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")).digest()
    return "p" + base64.b32hexencode(digest).decode("ascii").lower().rstrip("=")


class CalendarEvidenceIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_id: str
    user_id: str
    operation: Operation
    account_identity: str
    calendar_id: str
    event_id: str
    approved_args: dict[str, Any] = Field(default_factory=dict)
    approved_baseline: ApprovedCalendarBaseline | None = None
    desired_postcondition: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_approved_version(self):
        if not all((self.action_id, self.user_id, self.account_identity,
                    self.calendar_id, self.event_id)):
            raise ValueError("Calendar intent identity is incomplete")
        if self.operation != "create_event":
            baseline = self.approved_baseline
            if (baseline is None or not baseline.etag or
                    baseline.calendar_id != self.calendar_id or
                    baseline.event_id != self.event_id):
                raise ValueError("Update/delete require the exact approved baseline ETag")
        return self

    @classmethod
    def from_approved_proposal(cls, proposal: ActionProposal,
                               account_identity: str, calendar_id: str) -> "CalendarEvidenceIntent":
        if proposal.domain != "calendar" or proposal.operation not in {
                "create_event", "update_event", "delete_event"}:
            raise ValueError("Expected an approved Calendar proposal")
        baseline = proposal.approved_calendar_baseline
        event_id = (planned_google_event_id(proposal.action_id, account_identity, calendar_id)
                    if proposal.operation == "create_event" else proposal.args["eventId"])
        desired = ({key: proposal.args[key] for key in
                    ("title", "summary", "start", "end", "timeZone", "description", "location",
                     "source_message_id", "source_thread_id")
                    if key in proposal.args} if proposal.operation != "delete_event" else
                   {"status": "absent"})
        return cls(action_id=proposal.action_id, user_id=proposal.args["user_id"],
                   operation=proposal.operation, account_identity=account_identity,
                   calendar_id=calendar_id, event_id=event_id,
                   approved_args=copy.deepcopy(proposal.args),
                   approved_baseline=baseline.model_copy(deep=True) if baseline else None,
                   desired_postcondition=copy.deepcopy(desired))

    @property
    def fingerprint(self) -> str:
        frozen = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


def _aware(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


class CalendarActionEvidenceStore:
    """Short PG transactions; never performs provider I/O inside a row lock."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def load_create(self, action_id: str):
        async with self.session_factory() as db:
            return (await db.get(CalendarActionEvidence, action_id),
                    await db.get(ActionExecution, action_id))

    async def prepare_create(self, intent: CalendarEvidenceIntent,
                             immutable_body: dict[str, Any], marker: str) -> None:
        """Commit the provider plan and executing ledger in one short transaction."""
        if intent.operation != "create_event" or marker != intent.action_id:
            raise ValueError("Create evidence identity/marker mismatch")
        body = copy.deepcopy(immutable_body)
        if (body.get("calendarId") != intent.calendar_id or
                body.get("account") != intent.account_identity or
                not body.get("summary") or not body.get("start") or not body.get("end")):
            raise ValueError("Create body disagrees with approved identity")
        async with self.session_factory() as db:
            existing = await db.get(CalendarActionEvidence, intent.action_id)
            if existing is None:
                legacy = await db.get(ActionExecution, intent.action_id)
                if legacy is not None:
                    raise ValueError("Legacy Calendar action has no R5 evidence; do not replay")
                db.add(CalendarActionEvidence(
                    action_id=intent.action_id, user_id=intent.user_id,
                    operation=intent.operation, account_identity=intent.account_identity,
                    calendar_id=intent.calendar_id, event_id=intent.event_id,
                    approved_args_fingerprint=intent.fingerprint,
                    approved_baseline=None, approved_baseline_etag=None,
                    desired_postcondition=intent.desired_postcondition,
                    immutable_body=body, private_action_marker=marker,
                    outcome="NOT_STARTED", attempt_no=0))
                db.add(ActionExecution(
                    action_id=intent.action_id, user_id=intent.user_id,
                    domain="calendar", operation="create_event", status="executing"))
                try:
                    await db.commit()
                    return
                except IntegrityError:
                    await db.rollback()
                    existing = await db.get(CalendarActionEvidence, intent.action_id)
                    if existing is None:
                        raise
            ledger = await db.get(ActionExecution, intent.action_id)
            if (existing.approved_args_fingerprint != intent.fingerprint or
                    existing.event_id != intent.event_id or
                    existing.calendar_id != intent.calendar_id or
                    existing.account_identity != intent.account_identity or
                    existing.immutable_body != body or
                    existing.private_action_marker != marker or
                    ledger is None or ledger.user_id != intent.user_id or
                    ledger.operation != "create_event"):
                raise ValueError("Approved Calendar create changed after preparation")

    async def retry_create_after_verified_absence(self, action_id: str, token: str,
                                                  *, max_attempts: int = 2,
                                                  lease_seconds: int = 60) -> str | None:
        """Only the current reconciler may arm an identical second attempt."""
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (row is None or row.operation != "create_event" or row.outcome != "UNCERTAIN" or
                    row.fence_token != token or _aware(row.lease_until) is None or
                    _aware(row.lease_until) <= now or row.attempt_no >= max_attempts):
                return None
            next_token = str(uuid4())
            row.outcome = "IN_FLIGHT"
            row.fence_token = next_token
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.attempt_no += 1
            row.dispatch_started_at = None
            row.updated_at = now
            await db.commit()
            return next_token

    async def finalize_create(self, action_id: str, token: str,
                              outcome: Literal["CONFIRMED", "FAILED_DEFINITE"],
                              *, failure_category: str | None = None,
                              confirmation_source: Confirmation | None = None) -> bool:
        """Evidence terminal and ActionExecution terminal truth commit atomically."""
        if outcome == "FAILED_DEFINITE" and failure_category not in DEFINITE_FAILURES:
            raise ValueError("Ambiguous provider outcome cannot be definite failure")
        if outcome == "CONFIRMED" and confirmation_source not in {
                "provider_response", "exact_readback"}:
            raise ValueError("Create confirmation needs exact provider proof")
        async with self.session_factory() as db:
            evidence = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            ledger = await db.scalar(select(ActionExecution).where(
                ActionExecution.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (evidence is None or ledger is None or evidence.operation != "create_event" or
                    evidence.outcome not in {"IN_FLIGHT", "UNCERTAIN"} or
                    evidence.fence_token != token or _aware(evidence.lease_until) is None or
                    _aware(evidence.lease_until) <= now or ledger.status != "executing"):
                return False
            evidence.outcome = outcome
            evidence.failure_category = failure_category
            evidence.confirmed_external_id = evidence.event_id if outcome == "CONFIRMED" else None
            evidence.confirmation_source = confirmation_source
            evidence.owner = evidence.fence_token = evidence.lease_until = None
            evidence.updated_at = now
            if outcome == "CONFIRMED":
                ledger.status = "executed"
                ledger.external_id = evidence.event_id
                ledger.result = {"id": evidence.event_id}
                for key in ("source_message_id", "source_thread_id"):
                    if key in evidence.desired_postcondition:
                        ledger.result[key] = evidence.desired_postcondition[key]
                ledger.executed_at = now
            else:
                ledger.status = "failed"
                ledger.result = {"error_category": failure_category}
            await db.commit()
            return True

    async def prepare_modification(self, intent: CalendarEvidenceIntent,
                                   immutable_patch: dict[str, Any]) -> None:
        """One commit joins approved baseline/patch to the executing ledger."""
        if intent.operation not in {"update_event", "delete_event"}:
            raise ValueError("Expected approved single-resource modification")
        baseline = intent.approved_baseline
        if (baseline is None or baseline.calendar_id != intent.calendar_id or
                baseline.event_id != intent.event_id or
                intent.approved_args.get("account") != intent.account_identity or
                intent.approved_args.get("calendarId") != intent.calendar_id or
                any(key in intent.approved_args for key in
                    ("modificationScope", "recurrence", "futureStartDate")) or
                (intent.operation == "update_event" and not immutable_patch) or
                (intent.operation == "delete_event" and immutable_patch)):
            raise ValueError("Approved modification identity or patch is invalid")
        patch = copy.deepcopy(immutable_patch)
        async with self.session_factory() as db:
            row = await db.get(CalendarActionEvidence, intent.action_id)
            if row is None:
                if await db.get(ActionExecution, intent.action_id) is not None:
                    raise ValueError("Legacy Calendar action has no R5 evidence; do not replay")
                db.add(CalendarActionEvidence(
                    action_id=intent.action_id, user_id=intent.user_id,
                    operation=intent.operation, account_identity=intent.account_identity,
                    calendar_id=intent.calendar_id, event_id=intent.event_id,
                    approved_args_fingerprint=intent.fingerprint,
                    approved_baseline_etag=baseline.etag,
                    approved_baseline=baseline.model_dump(mode="json"),
                    desired_postcondition=copy.deepcopy(intent.desired_postcondition),
                    immutable_body=patch, outcome="NOT_STARTED", attempt_no=0))
                db.add(ActionExecution(
                    action_id=intent.action_id, user_id=intent.user_id,
                    domain="calendar", operation=intent.operation, status="executing"))
                try:
                    await db.commit()
                    return
                except IntegrityError:
                    await db.rollback()
                    row = await db.get(CalendarActionEvidence, intent.action_id)
                    if row is None:
                        raise
            ledger = await db.get(ActionExecution, intent.action_id)
            if (row.approved_args_fingerprint != intent.fingerprint or
                    row.approved_baseline_etag != baseline.etag or
                    row.account_identity != intent.account_identity or
                    row.calendar_id != intent.calendar_id or row.event_id != intent.event_id or
                    row.immutable_body != patch or ledger is None or
                    ledger.user_id != intent.user_id or ledger.operation != intent.operation):
                raise ValueError("Approved Calendar modification changed after preparation")

    async def retry_modification_after_verified_preimage(
            self, action_id: str, token: str, *, max_attempts: int = 2,
            lease_seconds: int = 120) -> str | None:
        """Only a fenced reconciler may reuse the same baseline ETag and patch."""
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (row is None or row.operation not in {"update_event", "delete_event"} or
                    row.outcome != "UNCERTAIN" or row.fence_token != token or
                    _aware(row.lease_until) is None or _aware(row.lease_until) <= now or
                    row.attempt_no >= max_attempts):
                return None
            next_token = str(uuid4())
            row.outcome = "IN_FLIGHT"
            row.owner = "calendar-modification-retry"
            row.fence_token = next_token
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.attempt_no += 1
            row.dispatch_started_at = None
            row.updated_at = now
            await db.commit()
            return next_token

    async def finalize_modification(self, action_id: str, token: str,
                                    outcome: Literal["CONFIRMED", "FAILED_DEFINITE"],
                                    *, confirmation_source: Confirmation | None = None,
                                    confirmed_etag: str | None = None,
                                    failure_category: str | None = None) -> bool:
        """Evidence and business ledger reach one terminal truth atomically."""
        if outcome == "CONFIRMED" and confirmation_source not in {
                "provider_response", "observed_postcondition", "observed_absence",
                "observed_cancelled"}:
            raise ValueError("Modification confirmation requires provider/observation proof")
        if outcome == "FAILED_DEFINITE" and failure_category not in DEFINITE_FAILURES:
            raise ValueError("Ambiguous modification cannot be definite failure")
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            ledger = await db.scalar(select(ActionExecution).where(
                ActionExecution.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (row is None or ledger is None or
                    row.operation not in {"update_event", "delete_event"} or
                    row.outcome not in {"IN_FLIGHT", "UNCERTAIN"} or
                    row.fence_token != token or _aware(row.lease_until) is None or
                    _aware(row.lease_until) <= now or ledger.status != "executing"):
                return False
            row.outcome = outcome
            row.failure_category = failure_category
            row.confirmed_external_id = row.event_id if outcome == "CONFIRMED" else None
            row.confirmed_etag = confirmed_etag
            row.confirmation_source = confirmation_source
            row.owner = row.fence_token = row.lease_until = None
            row.updated_at = now
            if outcome == "CONFIRMED":
                ledger.status = "executed"
                ledger.external_id = row.event_id
                ledger.result = {"id": row.event_id}
                ledger.executed_at = now
            else:
                ledger.status = "failed"
                ledger.result = {"error_category": failure_category}
            await db.commit()
            return True

    async def prepare(self, intent: CalendarEvidenceIntent) -> None:
        snapshot = intent.model_dump(mode="json")
        baseline = snapshot["approved_baseline"]
        async with self.session_factory() as db:
            db.add(CalendarActionEvidence(
                action_id=intent.action_id, user_id=intent.user_id,
                operation=intent.operation, account_identity=intent.account_identity,
                calendar_id=intent.calendar_id, event_id=intent.event_id,
                approved_args_fingerprint=intent.fingerprint,
                approved_baseline_etag=baseline["etag"] if baseline else None,
                approved_baseline=baseline,
                desired_postcondition=snapshot["desired_postcondition"],
                outcome="NOT_STARTED", attempt_no=0,
            ))
            try:
                await db.commit()
                return
            except IntegrityError:
                await db.rollback()
                row = await db.get(CalendarActionEvidence, intent.action_id)
                if row is None:
                    raise
                if (row.approved_args_fingerprint != intent.fingerprint or
                        row.approved_baseline_etag != (baseline["etag"] if baseline else None) or
                        row.event_id != intent.event_id):
                    raise ValueError("Approved Calendar action changed after preparation")

    async def arm(self, action_id: str, owner: str, lease_seconds: int = 60) -> str | None:
        if not owner or lease_seconds <= 0:
            raise ValueError("Invalid Calendar action lease")
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            if row is None or row.outcome != "NOT_STARTED":
                return None
            now = datetime.now(timezone.utc)
            token = str(uuid4())
            row.outcome = "IN_FLIGHT"
            row.owner = owner
            row.fence_token = token
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.attempt_no += 1
            row.updated_at = now
            await db.commit()
            return token

    async def mark_dispatch_started(self, action_id: str, token: str) -> bool:
        """Diagnostic only; absence of this timestamp cannot prove unsent."""
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (row is None or row.outcome != "IN_FLIGHT" or row.fence_token != token or
                    _aware(row.lease_until) is None or _aware(row.lease_until) <= now):
                return False
            row.dispatch_started_at = now
            row.updated_at = now
            await db.commit()
            return True

    async def claim_reconciliation(self, action_id: str, owner: str,
                                   lease_seconds: int = 60) -> str | None:
        """Expired IN_FLIGHT is UNKNOWN even if dispatch_started_at is null."""
        if not owner or lease_seconds <= 0:
            raise ValueError("Invalid reconciliation lease")
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (row is None or row.outcome not in {"IN_FLIGHT", "UNCERTAIN"} or
                    (_aware(row.lease_until) is not None and _aware(row.lease_until) > now)):
                return None
            token = str(uuid4())
            row.outcome = "UNCERTAIN"
            row.owner = owner
            row.fence_token = token
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.updated_at = now
            await db.commit()
            return token

    async def finalize(self, action_id: str, token: str, outcome: Outcome,
                       *, failure_category: str | None = None,
                       external_id: str | None = None,
                       confirmation_source: Confirmation | None = None) -> bool:
        if failure_category is not None and failure_category not in SAFE_FAILURES:
            raise ValueError("Unsafe provider failure category")
        if outcome == "FAILED_DEFINITE" and failure_category not in DEFINITE_FAILURES:
            raise ValueError("Ambiguous provider outcome cannot be definite failure")
        if outcome == "CONFIRMED" and (not external_id or confirmation_source is None):
            raise ValueError("Confirmation requires provider resource proof")
        async with self.session_factory() as db:
            row = await db.scalar(select(CalendarActionEvidence).where(
                CalendarActionEvidence.action_id == action_id).with_for_update())
            now = datetime.now(timezone.utc)
            if (row is None or row.outcome not in {"IN_FLIGHT", "UNCERTAIN"} or
                    row.fence_token != token or _aware(row.lease_until) is None or
                    _aware(row.lease_until) <= now):
                return False
            row.outcome = outcome
            row.failure_category = failure_category
            row.confirmed_external_id = external_id
            row.confirmation_source = confirmation_source
            row.owner = None
            row.fence_token = None
            row.lease_until = None
            row.updated_at = now
            await db.commit()
            return True
