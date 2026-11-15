"""Post-approval read failure is distinct from proposal preflight and remote dispatch."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.agent import agent_approval, official_guarded_tools, workflow_runtime
from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarEventSnapshot, CalendarProviderError
from app.agent.calendar_create import CalendarCreateCoordinator
from app.agent.calendar_modify import CalendarModifyCoordinator
from app.agent.workflow import bind_approved_calendar_version
from app.db import crud
from app.db.models import ActionExecution, CalendarActionEvidence


@pytest.mark.asyncio
async def test_post_approval_calendar_recheck_failure_is_paired_without_write(monkeypatch):
    proposal = ActionProposal(
        action_id="synthetic-action", domain="calendar", operation="create_event",
        args={"user_id": "owner", "title": "synthetic meeting",
              "start": "2026-12-10T15:00:00+08:00",
              "end": "2026-12-10T16:00:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="Create synthetic meeting", reason="user asked",
        risk_level=RiskLevel.EXTERNAL_WRITE)

    class Executor:
        async def calendar_create_prepared(self, _action_id):
            return False

        async def execute(self, _proposal):
            raise AssertionError("remote write must not start")

    class Calendar:
        async def availability(self, *_):
            raise CalendarError("synthetic service unavailable") from CalendarProviderError(
                "provider_unavailable", 503)

    monkeypatch.setattr(agent_approval, "get_settings", lambda: SimpleNamespace(
        google_calendar_enabled=True, google_calendar_owner_telegram_id="telegram",
        default_timezone="Asia/Shanghai", gmail_enabled=False))
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: Executor())
    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: Calendar())
    state = {"messages": [HumanMessage(id="turn", content="create meeting")],
             "prepared_actions": {"call": {"turn_id": "turn",
                                           "proposal": proposal.model_dump(mode="json")}},
             "user_id": "owner", "telegram_id": "telegram", "chat_id": "chat",
             "request_origin": "telegram"}
    runtime = SimpleNamespace(state=state, tool_call_id="call",
                              context={"user_id": "owner", "telegram_id": "telegram",
                                       "chat_id": "chat", "origin": "telegram"})
    result = await official_guarded_tools._execute(runtime, "create_event")
    assert result.tool_call_id == "call" and result.status == "error"
    assert "日历核验失败，本次写入尚未发送" in result.content
    assert "synthetic service unavailable" not in result.content


@pytest.mark.asyncio
async def test_approved_calendar_permission_failure_has_safe_paired_reason(monkeypatch):
    proposal = ActionProposal(
        action_id="synthetic-denied", domain="calendar", operation="create_event",
        args={"user_id": "owner", "title": "test",
              "start": "2026-12-10T15:00:00+08:00",
              "end": "2026-12-10T16:00:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="test", reason="test", risk_level=RiskLevel.EXTERNAL_WRITE)

    class Executor:
        async def calendar_create_prepared(self, _action_id):
            return True

        async def execute(self, _proposal):
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED,
                                message="definite rejection",
                                result={"failure_category": "provider_permission_denied"})

    monkeypatch.setattr(agent_approval, "get_settings", lambda: SimpleNamespace(
        google_calendar_enabled=True, google_calendar_owner_telegram_id="telegram",
        default_timezone="Asia/Shanghai", gmail_enabled=False))
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: Executor())
    runtime = SimpleNamespace(
        state={"messages": [HumanMessage(id="turn", content="create meeting")],
               "prepared_actions": {"call": {"turn_id": "turn",
                                             "proposal": proposal.model_dump(mode="json")}},
               "user_id": "owner", "telegram_id": "telegram", "chat_id": "chat",
               "request_origin": "telegram"},
        tool_call_id="call", context={"user_id": "owner", "telegram_id": "telegram",
                                 "chat_id": "chat", "origin": "telegram"})
    result = await official_guarded_tools._execute(runtime, "create_event")
    assert result.tool_call_id == "call" and result.status == "error"
    assert "日历服务拒绝访问，本次写入未执行" in result.content


async def _no_delay(_seconds):
    return None


class PermissionDeniedCalendar:
    def __init__(self, status_code=403):
        self.status_code = status_code
        self.create_calls = self.update_calls = self.delete_calls = 0
        self.event = CalendarEventSnapshot.model_validate({
            "id": "existing-event", "etag": '"v1"', "status": "confirmed",
            "summary": "Before", "start": {"dateTime": "2026-12-10T15:00:00+08:00"},
            "end": {"dateTime": "2026-12-10T16:00:00+08:00"},
        })

    async def resolve_create_scope(self, _calendar_id):
        return "normal", "owner@example.test"

    async def get_event(self, _calendar_id, event_id):
        if event_id == "existing-event":
            return self.event
        raise CalendarError("not_found_or_inaccessible")

    async def create_event_prepared(self, *_):
        self.create_calls += 1
        raise CalendarProviderError("permission_denied", self.status_code)

    async def update_event_if_match(self, *_):
        self.update_calls += 1
        raise CalendarError("permission_denied") from CalendarProviderError(
            "permission_denied", self.status_code)

    async def delete_event_if_match(self, *_):
        self.delete_calls += 1
        raise CalendarError("permission_denied") from CalendarProviderError(
            "permission_denied", self.status_code)


async def _owner(factory):
    async with factory() as db:
        user = await crud.get_or_create_user(db, "runtime-403-" + uuid4().hex)
        await db.commit()
        return user.id


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_calendar_create_auth_rejection_is_definite_without_retry(
        session_factory, status_code):
    owner = await _owner(session_factory)
    calendar = PermissionDeniedCalendar(status_code)
    proposal = ActionProposal(
        action_id="probe-create-" + uuid4().hex, domain="calendar",
        operation="create_event", args={
            "user_id": owner, "title": "test", "calendarId": "primary",
            "start": "2026-12-10T15:00:00+08:00",
            "end": "2026-12-10T16:00:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="test", reason="test", risk_level=RiskLevel.EXTERNAL_WRITE)
    result = await CalendarCreateCoordinator(
        calendar, session_factory, sleep=_no_delay).execute(proposal)
    assert calendar.create_calls == 1
    assert result.result["failure_category"] == "provider_permission_denied"
    async with session_factory() as db:
        evidence = await db.get(CalendarActionEvidence, proposal.action_id)
        ledger = await db.get(ActionExecution, proposal.action_id)
        assert evidence.outcome == "FAILED_DEFINITE" and evidence.attempt_no == 1
        assert ledger.status == "failed"


@pytest.mark.asyncio
async def test_calendar_create_unexpected_dispatch_exception_remains_unconfirmed(
        session_factory):
    owner = await _owner(session_factory)

    class UnexpectedCalendar(PermissionDeniedCalendar):
        async def create_event_prepared(self, *_):
            self.create_calls += 1
            raise RuntimeError("synthetic internal error")

    calendar = UnexpectedCalendar()
    proposal = ActionProposal(
        action_id="probe-unexpected-" + uuid4().hex, domain="calendar",
        operation="create_event", args={
            "user_id": owner, "title": "test", "calendarId": "primary",
            "start": "2026-12-10T15:00:00+08:00",
            "end": "2026-12-10T16:00:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="test", reason="test", risk_level=RiskLevel.EXTERNAL_WRITE)
    result = await CalendarCreateCoordinator(
        calendar, session_factory, sleep=_no_delay).execute(proposal)
    assert calendar.create_calls == 2
    assert result.result["remote_state"] == "PARKED_REVIEW"
    async with session_factory() as db:
        assert (await db.get(ActionExecution, proposal.action_id)).status == "executing"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.parametrize("operation,counter", [
    ("update_event", "update_calls"), ("delete_event", "delete_calls")])
async def test_calendar_modify_auth_rejection_is_definite_without_retry(
        session_factory, operation, counter, status_code):
    owner = await _owner(session_factory)
    calendar = PermissionDeniedCalendar(status_code)
    args = {"user_id": owner, "calendarId": "primary", "eventId": "existing-event"}
    if operation == "update_event":
        args["title"] = "After"
    proposal = ActionProposal(
        action_id="probe-modify-" + uuid4().hex, domain="calendar",
        operation=operation, args=args, human_readable_preview="test", reason="test",
        risk_level=(RiskLevel.EXTERNAL_WRITE if operation == "update_event"
                    else RiskLevel.DESTRUCTIVE))
    await bind_approved_calendar_version(calendar, proposal)
    result = await CalendarModifyCoordinator(
        calendar, session_factory, sleep=_no_delay).execute(proposal)
    assert getattr(calendar, counter) == 1
    assert result.result["failure_category"] == "provider_permission_denied"
    async with session_factory() as db:
        evidence = await db.get(CalendarActionEvidence, proposal.action_id)
        ledger = await db.get(ActionExecution, proposal.action_id)
        assert evidence.outcome == "FAILED_DEFINITE" and evidence.attempt_no == 1
        assert ledger.status == "failed"


@pytest.mark.asyncio
async def test_create_prior_unknown_then_403_stays_unconfirmed(session_factory):
    owner = await _owner(session_factory)

    class UnknownThenDenied(PermissionDeniedCalendar):
        async def create_event_prepared(self, *args):
            self.create_calls += 1
            if self.create_calls == 1:
                raise RuntimeError("response lost after dispatch")
            raise CalendarProviderError("permission_denied", 403)

    calendar = UnknownThenDenied()
    proposal = ActionProposal(
        action_id="probe-unknown-create-" + uuid4().hex, domain="calendar",
        operation="create_event", args={
            "user_id": owner, "title": "test", "calendarId": "primary",
            "start": "2026-12-10T15:00:00+08:00",
            "end": "2026-12-10T16:00:00+08:00", "timeZone": "Asia/Shanghai"},
        human_readable_preview="test", reason="test", risk_level=RiskLevel.EXTERNAL_WRITE)
    result = await CalendarCreateCoordinator(
        calendar, session_factory, sleep=_no_delay).execute(proposal)
    assert calendar.create_calls == 2
    assert result.result["remote_state"] == "PARKED_REVIEW"
    async with session_factory() as db:
        evidence = await db.get(CalendarActionEvidence, proposal.action_id)
        ledger = await db.get(ActionExecution, proposal.action_id)
        assert evidence.outcome == "PARKED_REVIEW" and evidence.attempt_no == 2
        assert ledger.status == "executing"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,counter", [
    ("update_event", "update_calls"), ("delete_event", "delete_calls")])
async def test_modify_prior_unknown_then_403_stays_unconfirmed(
        session_factory, operation, counter):
    owner = await _owner(session_factory)

    class UnknownThenDenied(PermissionDeniedCalendar):
        async def update_event_if_match(self, *args):
            self.update_calls += 1
            if self.update_calls == 1:
                raise CalendarError("timeout")
            raise CalendarError("permission_denied") from CalendarProviderError(
                "permission_denied", 403)

        async def delete_event_if_match(self, *args):
            self.delete_calls += 1
            if self.delete_calls == 1:
                raise CalendarError("timeout")
            raise CalendarError("permission_denied") from CalendarProviderError(
                "permission_denied", 403)

    calendar = UnknownThenDenied()
    args = {"user_id": owner, "calendarId": "primary", "eventId": "existing-event"}
    if operation == "update_event":
        args["title"] = "After"
    proposal = ActionProposal(
        action_id="probe-unknown-modify-" + uuid4().hex, domain="calendar",
        operation=operation, args=args, human_readable_preview="test", reason="test",
        risk_level=(RiskLevel.EXTERNAL_WRITE if operation == "update_event"
                    else RiskLevel.DESTRUCTIVE))
    await bind_approved_calendar_version(calendar, proposal)
    result = await CalendarModifyCoordinator(
        calendar, session_factory, sleep=_no_delay).execute(proposal)
    assert getattr(calendar, counter) == 2
    assert result.result["remote_state"] == "PARKED_REVIEW"
    async with session_factory() as db:
        evidence = await db.get(CalendarActionEvidence, proposal.action_id)
        ledger = await db.get(ActionExecution, proposal.action_id)
        assert evidence.outcome == "PARKED_REVIEW" and evidence.attempt_no == 2
        assert ledger.status == "executing"
