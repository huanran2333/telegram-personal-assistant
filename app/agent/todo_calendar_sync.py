"""Opt-in projection of scheduled Todos to Google Calendar.

Todo remains canonical. PostgreSQL triggers enqueue desired state atomically;
this worker only projects it. A provider ambiguity is reconciled by stable
event identity and exact readback, never by searching a similar title.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import or_, select

from app.agent.calendar import CalendarError
from app.agent.calendar_action_evidence import planned_google_event_id
from app.db.models import TodoCalendarMirror, User

logger = logging.getLogger(__name__)
LEASE = timedelta(minutes=2)
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class Claim:
    todo_id: str
    user_id: str
    revision: int
    operation: str
    payload: dict | None
    account_identity: str | None
    calendar_id: str
    event_id: str | None
    etag: str | None
    token: str
    attempt: int


def _body(claim: Claim, event_id: str) -> dict:
    payload = claim.payload or {}
    if not all(payload.get(field) for field in ("summary", "start", "end")):
        raise ValueError("InvalidTimeBlockPayload")
    title = str(payload["summary"])
    return {"id": event_id,
            "summary": ("[已完成] " if payload.get("completed") else "") + title,
            "start": {"dateTime": payload["start"]},
            "end": {"dateTime": payload["end"]},
            "extendedProperties": {"private": {"pa_todo_id": claim.todo_id}},
            "reminders": {"useDefault": False, "overrides": []}}


def _matches(snapshot, body: dict, todo_id: str) -> bool:
    def instant(value: str | None):
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
    private = (snapshot.extended_properties or {}).get("private") or {}
    start = instant(snapshot.start.get("dateTime"))
    end = instant(snapshot.end.get("dateTime"))
    return (private.get("pa_todo_id") == todo_id and
            snapshot.summary == body["summary"] and
            start is not None and end is not None and
            start == instant(body["start"]["dateTime"]) and
            end == instant(body["end"]["dateTime"]))


def _owned(snapshot, todo_id: str) -> bool:
    return ((snapshot.extended_properties or {}).get("private") or {}).get("pa_todo_id") == todo_id


class TodoCalendarSync:
    def __init__(self, session_factory, calendar, owner_telegram_id: str):
        self.session_factory = session_factory
        self.calendar = calendar
        self.owner_telegram_id = owner_telegram_id

    async def _claim(self) -> Claim | None:
        if not self.owner_telegram_id:
            return None
        now = datetime.now(timezone.utc)
        async with self.session_factory() as db:
            row = await db.scalar(select(TodoCalendarMirror).join(
                User, User.id == TodoCalendarMirror.user_id).where(
                User.telegram_id == self.owner_telegram_id,
                User.calendar_time_block_sync.is_(True),
                or_(TodoCalendarMirror.status.in_(("PENDING", "RETRY")),
                    (TodoCalendarMirror.status == "LEASED") &
                    (TodoCalendarMirror.lease_until < now)),
                or_(TodoCalendarMirror.next_attempt_at.is_(None),
                    TodoCalendarMirror.next_attempt_at <= now))
                .order_by(TodoCalendarMirror.updated_at, TodoCalendarMirror.todo_id)
                .with_for_update(of=TodoCalendarMirror, skip_locked=True).limit(1))
            if row is None:
                return None
            row.status = "LEASED"
            row.lease_token = str(uuid4())
            row.lease_until = now + LEASE
            row.attempt_count += 1
            claim = Claim(row.todo_id, row.user_id, row.desired_revision,
                          row.desired_operation, row.desired_payload,
                          row.account_identity, row.calendar_id, row.event_id,
                          row.etag, row.lease_token, row.attempt_count)
            await db.commit()
            return claim

    async def _still_enabled(self, claim: Claim) -> bool:
        async with self.session_factory() as db:
            row = await db.get(User, claim.user_id)
            return bool(row and row.calendar_time_block_sync and
                        row.telegram_id == self.owner_telegram_id)

    async def _finish(self, claim: Claim, *, state: str, category: str | None = None,
                      account: str | None = None, calendar_id: str | None = None,
                      event_id: str | None = None, etag: str | None = None) -> None:
        async with self.session_factory() as db:
            row = await db.get(TodoCalendarMirror, claim.todo_id, with_for_update=True)
            if row is None or row.lease_token != claim.token:
                return  # stale worker must not overwrite a newer lease
            if account and row.account_identity and row.account_identity != account:
                state, category = "REVIEW", "account_identity_changed"
            elif account:
                row.account_identity = account
            if calendar_id and row.calendar_id not in {"primary", calendar_id}:
                state, category = "REVIEW", "calendar_identity_changed"
            elif calendar_id:
                row.calendar_id = calendar_id
            if event_id:
                row.event_id = event_id
            if etag:
                row.etag = etag
            if row.desired_revision != claim.revision and state in {"SYNCED", "REMOVED"}:
                state = "PENDING"  # remote got the prior projection; process newest desire next
            elif state in {"SYNCED", "REMOVED"}:
                row.synced_revision = claim.revision
            if state == "RETRY" and claim.attempt >= MAX_ATTEMPTS:
                state, category = "REVIEW", "retry_exhausted"
            row.status = state
            row.last_error_category = category
            row.next_attempt_at = (datetime.now(timezone.utc) +
                timedelta(seconds=min(10 * (2 ** min(claim.attempt - 1, 5)), 300))
                if state == "RETRY" else None)
            row.lease_token = None
            row.lease_until = None
            row.updated_at = datetime.now(timezone.utc)
            await db.commit()

    async def _event(self, calendar_id: str, event_id: str):
        try:
            return await self.calendar.get_event(calendar_id, event_id)
        except CalendarError as exc:
            if str(exc) == "not_found_or_inaccessible":
                return None
            raise

    async def _project(self, claim: Claim) -> tuple:
        if claim.operation == "DELETE" and not claim.event_id:
            return "REMOVED", None, None, None, None
        account, concrete = await self.calendar.resolve_create_scope("primary")
        if (claim.account_identity and claim.account_identity != account) or (
                claim.calendar_id not in {"primary", concrete}):
            return "REVIEW", "account_or_calendar_changed", None, None, None
        event_id = claim.event_id or planned_google_event_id(
            "todo:" + claim.todo_id, account, concrete)
        if claim.operation == "DELETE":
            if not claim.event_id:
                return "REMOVED", None, None, None, None
            existing = await self._event(concrete, event_id)
            if existing is None:
                return "REMOVED", None, account, concrete, event_id
            if not _owned(existing, claim.todo_id) or (claim.etag and existing.etag != claim.etag):
                return "REVIEW", "remote_event_changed", account, concrete, event_id
            try:
                await self.calendar.delete_event_if_match(concrete, event_id, existing.etag)
            except CalendarError as exc:
                if str(exc) in {"etag_conflict", "provider_rejected"}:
                    return "REVIEW", str(exc), account, concrete, event_id
                if str(exc) == "remote_outcome_uncertain":
                    after = await self._event(concrete, event_id)
                    if after is None:
                        return "REMOVED", None, account, concrete, event_id
                raise
            return "REMOVED", None, account, concrete, event_id

        body = _body(claim, event_id)
        existing = await self._event(concrete, event_id)
        if existing is not None:
            if _matches(existing, body, claim.todo_id):
                return "SYNCED", None, account, concrete, event_id, existing.etag
            if not _owned(existing, claim.todo_id) or (claim.etag and existing.etag != claim.etag):
                return "REVIEW", "remote_event_changed", account, concrete, event_id
            patch = {key: body[key] for key in ("summary", "start", "end", "reminders")}
            try:
                changed = await self.calendar.update_event_if_match(
                    concrete, event_id, existing.etag, patch)
            except CalendarError as exc:
                if str(exc) in {"etag_conflict", "provider_rejected"}:
                    return "REVIEW", str(exc), account, concrete, event_id
                if str(exc) == "remote_outcome_uncertain":
                    after = await self._event(concrete, event_id)
                    if after is not None and _matches(after, body, claim.todo_id):
                        return "SYNCED", None, account, concrete, event_id, after.etag
                    return "REVIEW", "update_outcome_unknown", account, concrete, event_id
                raise
            if not _matches(changed, body, claim.todo_id):
                return "REVIEW", "provider_postcondition_mismatch", account, concrete, event_id
            return "SYNCED", None, account, concrete, event_id, changed.etag

        if claim.event_id:  # Previously confirmed event disappeared: never recreate silently.
            return "REVIEW", "remote_event_missing", account, concrete, event_id
        try:
            created = await self.calendar.create_time_block(concrete, event_id, body)
        except CalendarError as exc:
            if str(exc) in {"conflict", "remote_outcome_uncertain"}:
                after = await self._event(concrete, event_id)
                if after is not None and _matches(after, body, claim.todo_id):
                    return "SYNCED", None, account, concrete, event_id, after.etag
                if str(exc) == "conflict":
                    return "REVIEW", "remote_id_conflict", account, concrete, event_id
            raise
        if not _matches(created, body, claim.todo_id):
            return "REVIEW", "provider_postcondition_mismatch", account, concrete, event_id
        return "SYNCED", None, account, concrete, event_id, created.etag

    async def tick(self, *, limit: int = 10) -> int:
        processed = 0
        for _ in range(limit):
            claim = await self._claim()
            if claim is None:
                break
            if not await self._still_enabled(claim):
                await self._finish(claim, state="PENDING")
                continue
            try:
                state, category, account, calendar_id, event_id, *etag = await self._project(claim)
            except Exception as exc:
                # A failed projection must release its lease and enter the
                # bounded retry path even for an unexpected adapter error.
                # Never log provider bodies or credentials.
                logger.warning("todo_calendar_sync category=%s", type(exc).__name__)
                await self._finish(claim, state="RETRY", category=type(exc).__name__)
            else:
                await self._finish(claim, state=state, category=category, account=account,
                                   calendar_id=calendar_id, event_id=event_id,
                                   etag=etag[0] if etag else None)
            processed += 1
        return processed

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("todo_calendar_sync sweep category=%s", type(exc).__name__)
            try:
                await asyncio.wait_for(stop.wait(), timeout=15)
            except asyncio.TimeoutError:
                pass
