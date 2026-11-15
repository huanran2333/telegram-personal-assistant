"""Isolated PG18: publication fence, owner scope, versioned content review."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.calendar_conditional import CalendarEventSnapshot
from app.agent.meeting_prep_full import GroundingError
from app.agent.meeting_drafts import (DraftConflict, DraftNotFound, DraftPublicationUncertain, DraftSourceUnavailable, get_draft,
                                      prepare_meeting_draft, review_meeting_draft)
from app.db.models import ActionExecution, MeetingDraft, Todo, WebCredential, WebSession

pytestmark = [pytest.mark.integration,
              pytest.mark.skip(reason="Retired meeting draft feature and Mini App review page")]


class Calendar:
    def __init__(self):
        self.etag = '"v1"'

    async def resolve_read_scope(self, calendar_id):
        return "synthetic-account", "synthetic-calendar"

    async def get_event(self, calendar_id, event_id):
        assert event_id == "event-1"
        return CalendarEventSnapshot.model_validate({
            "id": event_id, "etag": self.etag, "status": "confirmed",
            "summary": "Project Sync", "start": {"dateTime": "2026-10-18T14:00:00+08:00"},
            "end": {"dateTime": "2026-10-18T15:00:00+08:00"}, "attendees": []})


class Model:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return {"claims": [{"kind": "SOURCE_FACT", "section": "meeting",
                            "text": "Project Sync", "source_keys": ["C1"],
                            "fact_field": "title", "fact_value": "Project Sync"}]}


@pytest.mark.asyncio
async def test_temporary_source_failure_preserves_draft_and_cancel_needs_no_provider(pg_harness):
    from app.agent.calendar import CalendarError
    factory, new_user, _ = pg_harness
    owner = await new_user()
    calendar = Calendar()
    draft = await prepare_meeting_draft(
        session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
        turn_id="synthetic-temporary-source", request="prepare meeting", event_id="event-1",
        timezone_name="Asia/Shanghai", calendar=calendar, gmail=None,
        gmail_owner_telegram_id=None, llm=Model())

    class UnavailableCalendar(Calendar):
        async def get_event(self, *_args):
            raise CalendarError("Synthetic temporary outage")
    with pytest.raises(DraftSourceUnavailable):
        await review_meeting_draft(session_factory=factory, user_id=owner.id,
            draft_id=draft["draft_id"], expected_revision=1, decision="accept",
            calendar=UnavailableCalendar())
    assert (await get_draft(factory, user_id=owner.id, draft_id=draft["draft_id"])).status == "NEEDS_REVIEW"
    canceled = await review_meeting_draft(session_factory=factory, user_id=owner.id,
        draft_id=draft["draft_id"], expected_revision=1, decision="cancel", calendar=None)
    assert canceled["status"] == "CANCELED"
    with pytest.raises(DraftConflict):
        await review_meeting_draft(session_factory=factory, user_id=owner.id,
            draft_id=draft["draft_id"], expected_revision=1, decision="accept", calendar=calendar)


@pytest.mark.asyncio
async def test_todo_source_version_and_description_change_require_new_draft(pg_harness):
    from sqlalchemy import update
    factory, new_user, _ = pg_harness
    owner = await new_user()
    async with factory() as db:
        todo = Todo(user_id=owner.id, title="Project Sync followup", description="Initial context")
        db.add(todo)
        await db.commit()
        todo_id = todo.id
    calendar = Calendar()
    draft = await prepare_meeting_draft(
        session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
        turn_id="synthetic-todo-source", request="prepare meeting", event_id="event-1",
        timezone_name="Asia/Shanghai", calendar=calendar, gmail=None,
        gmail_owner_telegram_id=None, llm=Model())
    assert any(source["type"] == "todo" and source["version"] for source in draft["sources"])
    async with factory() as db:
        await db.execute(update(Todo).where(Todo.id == todo_id).values(
            description="Changed context", version=Todo.version + 1))
        await db.commit()
    with pytest.raises(DraftConflict, match="SourceChanged"):
        await review_meeting_draft(session_factory=factory, user_id=owner.id,
            draft_id=draft["draft_id"], expected_revision=1, decision="accept", calendar=calendar)


@pytest.mark.asyncio
async def test_draft_retry_owner_review_revision_and_stale_source(pg_harness):
    factory, new_user, _ = pg_harness
    owner, outsider = await new_user(), await new_user()
    calendar, model = Calendar(), Model()
    params = dict(session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
                  turn_id="synthetic-turn-1", request="准备 Project Sync 会议",
                  event_id="event-1", timezone_name="Asia/Shanghai", calendar=calendar,
                  gmail=None, gmail_owner_telegram_id=None, llm=model)
    first = await prepare_meeting_draft(**params)
    assert first["status"] == "NEEDS_REVIEW" and first["revision"] == 1
    assert await prepare_meeting_draft(**params) == first
    assert model.calls == 1  # committed draft survives a repeated tool call
    with pytest.raises(DraftNotFound):
        await get_draft(factory, user_id=outsider.id, draft_id=first["draft_id"])
    changed = await review_meeting_draft(
        session_factory=factory, user_id=owner.id, draft_id=first["draft_id"],
        expected_revision=1, decision="revise", instruction="简短一些", calendar=calendar,
        llm=model)
    assert changed["revision"] == 2 and changed["status"] == "NEEDS_REVIEW"
    assert model.calls == 2
    assert (await review_meeting_draft(
        session_factory=factory, user_id=owner.id, draft_id=first["draft_id"],
        expected_revision=1, decision="revise", instruction="简短一些", calendar=calendar,
        llm=model))["revision"] == 2
    assert model.calls == 2
    with pytest.raises(DraftConflict):
        await review_meeting_draft(
            session_factory=factory, user_id=owner.id, draft_id=first["draft_id"],
            expected_revision=1, decision="accept", calendar=calendar)
    calendar.etag = '"v2"'
    with pytest.raises(DraftConflict, match="SourceChanged"):
        await review_meeting_draft(
            session_factory=factory, user_id=owner.id, draft_id=first["draft_id"],
            expected_revision=2, decision="accept", calendar=calendar)
    assert (await get_draft(factory, user_id=owner.id,
                            draft_id=first["draft_id"])).status == "STALE"
    canceled = await review_meeting_draft(
        session_factory=factory, user_id=owner.id, draft_id=first["draft_id"],
        expected_revision=2, decision="cancel", calendar=calendar)
    assert canceled["status"] == "CANCELED"


@pytest.mark.asyncio
async def test_concurrent_review_has_one_winner_and_no_business_write(pg_harness):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    calendar, model = Calendar(), Model()
    draft = await prepare_meeting_draft(
        session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
        turn_id="synthetic-turn-2", request="准备 Project Sync 会议", event_id="event-1",
        timezone_name="Asia/Shanghai", calendar=calendar, gmail=None,
        gmail_owner_telegram_id=None, llm=model)
    async def decide(choice):
        try:
            return (await review_meeting_draft(
                session_factory=factory, user_id=owner.id, draft_id=draft["draft_id"],
                expected_revision=1, decision=choice, calendar=calendar))["status"]
        except DraftConflict:
            return "CONFLICT"
    results = await asyncio.gather(decide("accept"), decide("cancel"))
    assert sorted(results) in (["ACCEPTED", "CONFLICT"], ["CANCELED", "CONFLICT"])
    row = await get_draft(factory, user_id=owner.id, draft_id=draft["draft_id"])
    assert row.revision == 1 and row.status in {"ACCEPTED", "CANCELED"}
    async with factory() as db:
        assert await db.scalar(select(func.count()).select_from(ActionExecution).where(
            ActionExecution.user_id == owner.id)) == 0
        assert await db.scalar(select(func.count()).select_from(Todo).where(Todo.user_id == owner.id)) == 0


@pytest.mark.asyncio
async def test_invalid_generation_never_publishes_and_retry_can_recover(pg_harness):
    factory, new_user, _ = pg_harness
    owner = await new_user()

    class InvalidModel(Model):
        async def ainvoke(self, messages):
            value = await super().ainvoke(messages)
            value["claims"][0]["fact_value"] = "fabricated title"
            return value

    model = InvalidModel()
    params = dict(session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
                  turn_id="synthetic-invalid", request="prepare meeting", event_id="event-1",
                  timezone_name="Asia/Shanghai", calendar=Calendar(), gmail=None,
                  gmail_owner_telegram_id=None, llm=model)
    with pytest.raises(GroundingError):
        await prepare_meeting_draft(**params)
    assert model.calls == 2
    async with factory() as db:
        assert await db.scalar(select(func.count()).select_from(MeetingDraft).where(
            MeetingDraft.user_id == owner.id)) == 0
    params["llm"] = Model()
    assert (await prepare_meeting_draft(**params))["status"] == "NEEDS_REVIEW"


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_after_commit", [False, True])
async def test_publication_crash_boundary_and_retry(pg_harness, crash_after_commit):
    factory, new_user, engine = pg_harness
    owner = await new_user()

    class LostResponseSession(AsyncSession):
        async def commit(self):
            if crash_after_commit:
                await super().commit()
            raise RuntimeError("Synthetic publication crash")

    crashing = async_sessionmaker(engine, class_=LostResponseSession, expire_on_commit=False)
    model = Model()
    params = dict(user_id=owner.id, telegram_id=owner.telegram_id,
                  turn_id=f"synthetic-crash-{crash_after_commit}", request="prepare meeting",
                  event_id="event-1", timezone_name="Asia/Shanghai", calendar=Calendar(),
                  gmail=None, gmail_owner_telegram_id=None, llm=model)
    with pytest.raises(DraftPublicationUncertain):
        await prepare_meeting_draft(session_factory=crashing, **params)
    async with factory() as db:
        rows = (await db.scalars(select(MeetingDraft).where(MeetingDraft.user_id == owner.id))).all()
        assert len(rows) == int(crash_after_commit)
    recovered = await prepare_meeting_draft(session_factory=factory, **params)
    assert recovered["status"] == "NEEDS_REVIEW"
    assert model.calls == (1 if crash_after_commit else 2)


@pytest.mark.asyncio
async def test_concurrent_publication_and_same_review_are_idempotent(pg_harness):
    factory, new_user, _ = pg_harness
    owner = await new_user()
    params = dict(session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
                  turn_id="synthetic-race", request="prepare meeting", event_id="event-1",
                  timezone_name="Asia/Shanghai", calendar=Calendar(), gmail=None,
                  gmail_owner_telegram_id=None, llm=Model())
    first, second = await asyncio.gather(prepare_meeting_draft(**params), prepare_meeting_draft(**params))
    assert first == second
    review = dict(session_factory=factory, user_id=owner.id, draft_id=first["draft_id"],
                  expected_revision=1, decision="accept", calendar=params["calendar"])
    a, b = await asyncio.gather(review_meeting_draft(**review), review_meeting_draft(**review))
    assert a == b and a["status"] == "ACCEPTED"


@pytest.mark.asyncio
async def test_gmail_source_changes_fail_closed_on_content_review(pg_harness):
    from app.agent.gmail import GmailMessage
    factory, new_user, _ = pg_harness
    owner = await new_user()

    class MailCalendar(Calendar):
        async def get_event(self, calendar_id, event_id):
            event = await super().get_event(calendar_id, event_id)
            return event.model_copy(update={"attendees": [{"email": "alice@example.test"}]})

    class Gmail:
        body = "Agenda ready"

        async def account_address(self):
            return "self@example.test"

        async def search(self, query, limit=5):
            return await self.thread("thread-1")

        async def thread(self, thread_id):
            return [GmailMessage(message_id="mail-1", thread_id="thread-1",
                                 sender="alice@example.test", recipients=["self@example.test"],
                                 subject="Project Sync", body=self.body, snippet=self.body,
                                 timestamp=datetime.now(timezone.utc))]

    gmail, calendar = Gmail(), MailCalendar()
    draft = await prepare_meeting_draft(
        session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
        turn_id="synthetic-mail-source", request="prepare meeting", event_id="event-1",
        timezone_name="Asia/Shanghai", calendar=calendar, gmail=gmail,
        gmail_owner_telegram_id=owner.telegram_id, llm=Model())
    assert any(source["type"] == "gmail" for source in draft["sources"])
    gmail.body = "Agenda has changed"
    with pytest.raises(DraftConflict, match="SourceChanged"):
        await review_meeting_draft(
            session_factory=factory, user_id=owner.id, draft_id=draft["draft_id"],
            expected_revision=1, decision="accept", calendar=calendar, gmail=gmail)
    assert (await get_draft(factory, user_id=owner.id, draft_id=draft["draft_id"])).status == "STALE"


@pytest.mark.asyncio
async def test_http_review_auth_csrf_origin_owner_and_version(pg_harness, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.web import api
    from app.web.auth import SESSION_COOKIE, digest
    from app.agent import workflow_runtime
    from app.config import get_settings

    factory, new_user, _ = pg_harness
    owner, outsider = await new_user(), await new_user()
    calendar = Calendar()
    draft = await prepare_meeting_draft(
        session_factory=factory, user_id=owner.id, telegram_id=owner.telegram_id,
        turn_id="synthetic-api", request="prepare meeting", event_id="event-1",
        timezone_name="Asia/Shanghai", calendar=calendar, gmail=None,
        gmail_owner_telegram_id=None, llm=Model())
    owner_token, outsider_token = uuid4().hex, uuid4().hex
    async with factory() as db:
        for user, token in ((owner, owner_token), (outsider, outsider_token)):
            db.add(WebCredential(user_id=user.id, password_hash="unused-synthetic-hash", version=1))
            db.add(WebSession(token_hash=digest(token), user_id=user.id, credential_version=1,
                              csrf_token="synthetic-csrf", auth_source="password",
                              expires_at=datetime.now(timezone.utc) + timedelta(hours=1)))
        await db.commit()
    settings = get_settings().model_copy(update={
        "web_public_origin": "https://assistant.example.test", "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": owner.telegram_id, "gmail_enabled": False})
    monkeypatch.setattr(api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: calendar)
    app = FastAPI()
    app.include_router(api.router)
    app.state.web_ready = True
    app.state.web_worker = SimpleNamespace(accepting=True)
    url = f"/api/v3/meeting-drafts/{draft['draft_id']}"
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="https://assistant.example.test") as client:
        assert (await client.get(url)).status_code == 401
        client.cookies.set(SESSION_COOKIE, outsider_token)
        assert (await client.get(url)).status_code == 404
        client.cookies.set(SESSION_COOKIE, owner_token)
        detail = await client.get(url)
        assert detail.status_code == 200 and "title: Project Sync" in detail.json()["content"]
        assert detail.headers["cache-control"] == "no-store"
        body = {"expected_revision": 1, "decision": "accept"}
        assert (await client.post(url + "/review", json=body)).status_code == 403
        assert (await client.post(url + "/review", json=body, headers={
            "origin": "https://untrusted.example.test", "x-csrf-token": "synthetic-csrf"})).status_code == 403
        headers = {"origin": settings.web_public_origin, "x-csrf-token": "synthetic-csrf"}
        assert (await client.post(url + "/review", json=body | {"expected_revision": 2},
                                  headers=headers)).status_code == 409
        accepted = await client.post(url + "/review", json=body, headers=headers)
        assert accepted.status_code == 200 and accepted.json()["status"] == "ACCEPTED"
        assert (await client.post(url + "/review", json=body, headers=headers)).json() == accepted.json()


class MainModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.observed = []
        self.offered = []

    def bind_tools(self, tools):
        self.offered.append({tool.name for tool in tools})
        return self

    async def ainvoke(self, messages):
        self.observed.append(list(messages))
        return self.responses.pop(0)

    def with_structured_output(self, schema):
        return self.synthesis


async def full_graph(monkeypatch, factory, owner, main_model, synth_model):
    from app.agent import graph, nodes, unified_loop, unified_tools, workflow_runtime
    from app.config import get_settings
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True, "google_calendar_owner_telegram_id": owner.telegram_id,
        "gmail_enabled": False})
    main_model.synthesis = synth_model
    monkeypatch.setattr(graph, "settings", settings)
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **kw:
                        synth_model if kw.get("routing") else main_model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_kw: main_model)
    monkeypatch.setattr(unified_loop, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow_runtime, "get_calendar_adapter", lambda: Calendar())
    return await graph.build_graph(factory, durable=True)


@pytest.mark.asyncio
@pytest.mark.skip(reason=(
    "Retired custom graph; equivalent current create_agent flow is in "
    "test_pg_official_meeting.py"))
async def test_full_graph_real_tool_draft_then_todo_and_saver_replay(pg_harness, monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage
    from app.agent import graph
    factory, new_user, _ = pg_harness
    owner = await new_user()
    synth = Model()
    main = MainModel(
        AIMessage(content="", tool_calls=[
            {"name": "prepare_meeting", "id": "prepare-1", "args": {"event_id": "event-1"}, "type": "tool_call"},
            {"name": "create_todo", "id": "create-1", "args": {"title": "Synthetic followup"}, "type": "tool_call"}]),
        AIMessage(content="会议草稿待审核；Synthetic followup 已创建。"))
    compiled = await full_graph(monkeypatch, factory, owner, main, synth)
    thread, turn = f"web:ordinary:{owner.id}", f"web-request:{uuid4()}"
    try:
        args = (compiled, owner.id, owner.telegram_id, "", "准备会议并创建 Synthetic followup")
        reply, state = await graph.run_agent(*args, thread_id=thread, turn_id=turn,
                                             origin="web", return_state=True)
        replay = await graph.run_agent(*args, thread_id=thread, turn_id=turn, origin="web")
        assert reply == replay and "待审核" in reply and "Synthetic followup" in reply
        assert len(main.observed) == 2 and synth.calls == 1
        assert not {"meeting_evidence", "submit_meeting_brief"} & main.offered[-1]
        assert [message.tool_call_id for message in state["messages"] if isinstance(message, ToolMessage)] == [
            "prepare-1", "create-1"]
        assert not state.get("validated_meeting_response")
        async with factory() as db:
            assert await db.scalar(select(func.count()).select_from(MeetingDraft).where(
                MeetingDraft.user_id == owner.id)) == 1
            assert await db.scalar(select(func.count()).select_from(Todo).where(
                Todo.user_id == owner.id, Todo.title == "Synthetic followup")) == 1
    finally:
        await compiled.checkpointer.adelete_thread(thread)
        await compiled.checkpointer.conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.skip(reason=(
    "Retired custom graph; equivalent current create_agent flow is in "
    "test_pg_official_meeting.py"))
async def test_full_graph_draft_survives_action_approval_and_continues(pg_harness, monkeypatch, decision):
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.types import Command
    from app.agent import graph, workflow_runtime
    from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
    from app.db import crud
    factory, new_user, _ = pg_harness
    owner = await new_user()
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=owner.id, title="Synthetic deletion target")
        await db.commit()
        todo_id = todo.id
    synth = Model()
    main = MainModel(
        AIMessage(content="", tool_calls=[
            {"name": "prepare_meeting", "id": "prepare-1", "args": {"event_id": "event-1"}, "type": "tool_call"},
            {"name": "propose_todo_delete", "id": "delete-1", "args": {"target": todo_id}, "type": "tool_call"},
            {"name": "list_todos", "id": "list-1", "args": {}, "type": "tool_call"}]),
        AIMessage(content=f"会议草稿待审核；删除结果 {decision}；待办已查询。"))
    compiled = await full_graph(monkeypatch, factory, owner, main, synth)
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda:
                        ActionExecutor(Calendar(), SqlAlchemyActionLedger(factory), factory))
    thread = f"web:ordinary:{owner.id}"
    try:
        _, pending = await graph.run_agent(
            compiled, owner.id, owner.telegram_id, "", "准备会议、删除指定任务、再列待办",
            thread_id=thread, turn_id=f"web-request:{uuid4()}", origin="web", return_state=True)
        assert pending["__interrupt__"][0].value["kind"] == "action_proposal"
        async with factory() as db:
            draft = await db.scalar(select(MeetingDraft).where(MeetingDraft.user_id == owner.id))
            assert draft is not None and draft.status == "NEEDS_REVIEW"
            assert await db.get(Todo, todo_id) is not None
        result = await compiled.ainvoke(Command(resume={"decision": decision}),
                                         {"configurable": {"thread_id": thread}})
        assert [message.tool_call_id for message in result["messages"] if isinstance(message, ToolMessage)] == [
            "prepare-1", "delete-1", "list-1"]
        assert synth.calls == 1 and len(main.observed) == 2
        assert "待审核" in result["messages"][-1].content
        async with factory() as db:
            assert (await db.get(Todo, todo_id) is None) == (decision == "approve")
            assert (await db.get(MeetingDraft, draft.id)).status == "NEEDS_REVIEW"
            assert await db.scalar(select(func.count()).select_from(ActionExecution).where(
                ActionExecution.user_id == owner.id)) == int(decision == "approve")
    finally:
        await compiled.checkpointer.adelete_thread(thread)
        await compiled.checkpointer.conn.close()
