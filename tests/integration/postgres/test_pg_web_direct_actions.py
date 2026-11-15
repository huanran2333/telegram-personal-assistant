"""Web form writes are direct; Telegram-origin approvals stay separate."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response
from sqlalchemy import select

from app.agent.actions import ActionResult, ActionStatus
from app.agent.actions import ActionProposal, RiskLevel
from app.agent.calendar import CalendarError
from app.agent.calendar_modify import CalendarModifyCoordinator
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.workflow import bind_approved_calendar_version
from app.db.models import Todo, WebMutation, WebRequest
from app.web import calendar as calendar_api, todos as todo_api
from app.web import api as web_api
from app.web.runs import submit_run
from tests.test_r5_3_calendar_modify import FakeConditionalCalendar
from app.config import get_settings
from zoneinfo import ZoneInfo


pytestmark = pytest.mark.integration


def request(key: str, etag: str | None = None):
    headers = {"idempotency-key": key}
    if etag:
        headers["if-match"] = etag
    return SimpleNamespace(headers=headers)


@pytest.mark.asyncio
async def test_web_todo_delete_is_direct_versioned_and_idempotent(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    owner, other = await new_user(), await new_user()
    monkeypatch.setattr(todo_api, "AsyncSessionLocal", factory)

    async def auth_owner(_request, *, write=False):
        return owner, None

    monkeypatch.setattr(todo_api, "_authenticated", auth_owner)
    created = await todo_api.create_web_todo(
        request(uuid4().hex), Response(), todo_api.TodoCreate(title="direct delete"))
    with pytest.raises(HTTPException) as stale:
        await todo_api.delete_web_todo(request(uuid4().hex, '"0"'), Response(), created["id"])
    assert stale.value.status_code == 412
    async with factory() as db:
        assert await db.get(Todo, created["id"]) is not None

    async def auth_other(_request, *, write=False):
        return other, None

    monkeypatch.setattr(todo_api, "_authenticated", auth_other)
    with pytest.raises(HTTPException) as hidden:
        await todo_api.delete_web_todo(request(uuid4().hex, created["etag"]), Response(), created["id"])
    assert hidden.value.status_code == 404
    monkeypatch.setattr(todo_api, "_authenticated", auth_owner)
    key = uuid4().hex
    first = await todo_api.delete_web_todo(request(key, created["etag"]), Response(), created["id"])
    again = await todo_api.delete_web_todo(request(key, created["etag"]), Response(), created["id"])
    assert first == again == {"deleted": True, "title": "direct delete"}
    with pytest.raises(HTTPException) as conflict:
        await todo_api.delete_web_todo(request(key, '"999"'), Response(), created["id"])
    assert conflict.value.status_code == 409
    async with factory() as db:
        assert await db.get(Todo, created["id"]) is None
        assert len((await db.scalars(select(WebMutation).where(
            WebMutation.user_id == owner.id, WebMutation.operation == "todo_delete"))).all()) == 1
        assert not (await db.scalars(select(WebRequest).where(
            WebRequest.user_id == owner.id, WebRequest.operation == "form_proposal"))).all()


@pytest.mark.asyncio
async def test_web_calendar_direct_reuses_one_action_identity_after_response_loss(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    monkeypatch.setattr(calendar_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return owner, None

    monkeypatch.setattr(calendar_api, "_authenticated", auth)

    class Calendar:
        scope_reads = 0

        async def resolve_create_scope(self, _calendar_id):
            self.scope_reads += 1
            return "synthetic-account", "primary"

    calendar = Calendar()

    async def get_calendar(_request):
        return calendar

    monkeypatch.setattr(calendar_api, "_calendar_for", get_calendar)

    class Executor:
        ids = []

        async def execute(self, proposal):
            self.ids.append(proposal.action_id)
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.EXECUTED,
                                message="confirmed")

    executor = Executor()
    monkeypatch.setattr(calendar_api, "get_action_executor", lambda: executor)
    start = datetime.now(timezone(timedelta(hours=8))) + timedelta(days=2)
    body = calendar_api.CalendarForm(
        operation="create", calendar_id="primary",
        fields=calendar_api.CalendarFields(
            title="synthetic event", start=start, end=start + timedelta(hours=1),
            timeZone="Asia/Shanghai"))
    key = uuid4().hex
    first = await calendar_api.calendar_action(request(key), Response(), body)
    again = await calendar_api.calendar_action(request(key), Response(), body)
    assert first["status"] == again["status"] == "executed"
    assert executor.ids == [executor.ids[0], executor.ids[0]]
    assert calendar.scope_reads == 1
    async with factory() as db:
        assert len((await db.scalars(select(WebMutation).where(
            WebMutation.user_id == owner.id, WebMutation.operation == "calendar_action"))).all()) == 1
        assert not (await db.scalars(select(WebRequest).where(
            WebRequest.user_id == owner.id, WebRequest.operation == "form_proposal"))).all()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Retired Mini App approval listing; direct Web actions remain covered")
async def test_web_approval_list_only_reads_telegram_threads(pg_harness, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    monkeypatch.setattr(web_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return user, None

    monkeypatch.setattr(web_api, "_authenticated", auth)

    class Graph:
        threads = []

        async def aget_state(self, config):
            self.threads.append(config["configurable"]["thread_id"])
            return SimpleNamespace(values={})

    graph = Graph()
    web_run, _ = await submit_run(factory, user_id=user.id, operation="approval",
                                  idempotency_key=uuid4().hex, body={"origin": "web"})
    telegram_run, _ = await submit_run(
        factory, user_id=user.id, operation="approval", idempotency_key=uuid4().hex,
        body={"origin": "telegram"})
    import app.agent.official_runtime as official_runtime
    import app.agent.workflow_runtime as workflow_runtime
    async def get_graph():
        return graph

    monkeypatch.setattr(official_runtime, "get_graph", get_graph)
    monkeypatch.setattr(workflow_runtime, "get_productivity_graph", get_graph)
    result = await web_api.approvals(request(uuid4().hex), Response())
    assert result["items"] == []
    assert [item["run_id"] for item in result["recent"]] == [telegram_run.id]
    assert all(user.telegram_id in thread and "web:" not in thread for thread in graph.threads)
    assert (await web_api.get_run(request(uuid4().hex), Response(), telegram_run.id))["run_id"] == telegram_run.id
    with pytest.raises(HTTPException) as hidden:
        await web_api.get_run(request(uuid4().hex), Response(), web_run.id)
    assert hidden.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["web", "telegram"])
@pytest.mark.parametrize("operation", ["update", "delete"])
async def test_web_and_telegram_calendar_changes_reject_stale_baseline(
        pg_harness, monkeypatch, first, operation):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    calendar = FakeConditionalCalendar()
    coordinator = CalendarModifyCoordinator(calendar, factory)
    executor = ActionExecutor(
        calendar, SqlAlchemyActionLedger(factory), factory,
        calendar_modify_coordinator=coordinator)
    monkeypatch.setattr(calendar_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(calendar_api, "get_action_executor", lambda: executor)

    async def auth(_request, *, write=False):
        return owner, None

    async def calendar_for(_request):
        return calendar

    monkeypatch.setattr(calendar_api, "_authenticated", auth)
    monkeypatch.setattr(calendar_api, "_calendar_for", calendar_for)
    telegram = ActionProposal(
        action_id=f"synthetic-telegram-{uuid4().hex}", domain="calendar",
        operation=f"{operation}_event",
        args={"user_id": owner.id, "calendarId": "primary",
              "eventId": "synthetic-event",
              **({"title": "Telegram revision"} if operation == "update" else {})},
        human_readable_preview="Change synthetic event",
        reason="Telegram approval", risk_level=(
            RiskLevel.EXTERNAL_WRITE if operation == "update" else RiskLevel.DESTRUCTIVE))
    await bind_approved_calendar_version(calendar, telegram)
    zone = ZoneInfo(get_settings().default_timezone)
    start = datetime.now(zone) + timedelta(days=2)
    web_body = calendar_api.CalendarForm(
        operation=operation, calendar_id="primary", event_id="synthetic-event",
        baseline_etag='"v1"', fields=(calendar_api.CalendarFields(
            title="Web revision", start=start, end=start + timedelta(hours=1),
            timeZone=get_settings().default_timezone) if operation == "update" else None))
    if first == "web":
        web = await calendar_api.calendar_action(
            request(uuid4().hex), Response(), web_body)
        assert web["status"] == "executed"
        later = await executor.execute(telegram)
        assert later.status != ActionStatus.EXECUTED
        assert (calendar.event["summary"] == "Web revision" if operation == "update"
                else calendar.event["status"] == "cancelled")
    else:
        assert (await executor.execute(telegram)).status == ActionStatus.EXECUTED
        with pytest.raises(HTTPException) as stale:
            await calendar_api.calendar_action(
                request(uuid4().hex), Response(), web_body)
        assert stale.value.status_code in {409, 412}
        assert (calendar.event["summary"] == "Telegram revision" if operation == "update"
                else calendar.event["status"] == "cancelled")
    assert len(calendar.updates if operation == "update" else calendar.deletes) <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update", "delete"])
async def test_todo_managed_calendar_block_cannot_be_edited_as_independent_event(
        pg_harness, monkeypatch, operation):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    calendar = FakeConditionalCalendar()
    calendar.event["extendedProperties"] = {
        "private": {"pa_todo_id": str(uuid4())}}
    monkeypatch.setattr(calendar_api, "AsyncSessionLocal", factory)

    async def auth(_request, *, write=False):
        return owner, None

    async def calendar_for(_request):
        return calendar

    monkeypatch.setattr(calendar_api, "_authenticated", auth)
    monkeypatch.setattr(calendar_api, "_calendar_for", calendar_for)
    proposal = ActionProposal(
        action_id=f"managed-telegram-{uuid4().hex}", domain="calendar",
        operation=f"{operation}_event",
        args={"user_id": owner.id, "calendarId": "primary",
              "eventId": "synthetic-event",
              **({"title": "wrong direct edit"} if operation == "update" else {})},
        human_readable_preview="Synthetic independent event action",
        reason="Synthetic Telegram request", risk_level=(
            RiskLevel.EXTERNAL_WRITE if operation == "update" else RiskLevel.DESTRUCTIVE))
    with pytest.raises(CalendarError, match="ManagedTodoCalendarEvent"):
        await bind_approved_calendar_version(calendar, proposal)
    zone = ZoneInfo(get_settings().default_timezone)
    start = datetime.now(zone) + timedelta(days=2)
    body = calendar_api.CalendarForm(
        operation=operation, calendar_id="primary", event_id="synthetic-event",
        baseline_etag='"v1"', fields=(calendar_api.CalendarFields(
            title="wrong direct edit", start=start, end=start + timedelta(hours=1),
            timeZone=get_settings().default_timezone) if operation == "update" else None))
    with pytest.raises(HTTPException) as denied:
        await calendar_api.calendar_action(request(uuid4().hex), Response(), body)
    assert denied.value.status_code == 409
    assert not calendar.updates and not calendar.deletes
    async with factory() as db:
        assert not (await db.scalars(select(WebMutation).where(
            WebMutation.user_id == owner.id,
            WebMutation.operation == "calendar_action"))).all()
