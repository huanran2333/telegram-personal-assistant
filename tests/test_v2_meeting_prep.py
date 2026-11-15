from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarEventSnapshot
from app.agent.gmail import GmailMessage
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.meeting_prep import MeetingPrepSnapshot, format_meeting_prep, select_meeting
from app.agent.memory_context import ContextBundle
from app.agent.structured_search import StructuredHit
from app.agent.planner import RoutedRequest, analyze_route, route_request
from app.agent.workflow import build_productivity_graph, start_workflow
from app.db import crud
from app.db.models import StructuredMemory


class MeetingCalendar:
    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail
        self.writes = 0
        self.calls = 0

    async def search_events(self, query, start=None, end=None):
        if self.fail:
            raise CalendarError("OAuth expired")
        assert query == ""
        return self.events

    async def create_event(self, args):
        self.writes += 1
        raise AssertionError("Meeting prep must be read-only")

    async def resolve_read_scope(self, calendar_id):
        return "test-account", "test-calendar"

    async def get_event(self, calendar_id, event_id):
        self.calls += 1
        assert calendar_id == "test-calendar"
        match = next((item for item in self.events if item["id"] == event_id), None)
        if match is None:
            raise CalendarError("not_found")
        return CalendarEventSnapshot.model_validate({**match, "etag": '"v1"', "status": "confirmed"})


class MeetingLLM:
    def __init__(self):
        self.evidence = None

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        import json
        payload = json.loads(messages[1].content)
        evidence = payload["evidence"]
        self.evidence = evidence
        event = next(item for item in evidence if item["type"] == "calendar")
        claims = [{"kind": "SOURCE_FACT", "section": "meeting", "text": "Meeting time confirmed",
                   "source_keys": [event["key"]], "fact_field": "start",
                   "fact_value": event["facts"]["start"]}]
        todo = next((item for item in evidence if item["type"] == "todo"), None)
        if todo:
            claims.append({"kind": "SOURCE_FACT", "section": "open_loops", "text": "Relevant Todo",
                           "source_keys": [todo["key"]], "fact_field": "title",
                           "fact_value": todo["facts"]["title"]})
        return {"claims": claims}


class ProfileStore:
    def __init__(self, user_id):
        self.user_id = user_id

    async def aget(self, namespace, key):
        assert namespace == ("profile", self.user_id)
        assert key == "user_profile"
        return type("Item", (), {"value": {"preferences": ["concise answers"]}})()


def event(event_id="event-1", title="论文讨论"):
    return {"id": event_id, "summary": title,
            "start": {"dateTime": "2026-10-01T15:00:00+08:00"},
            "end": {"dateTime": "2026-10-01T16:00:00+08:00"},
            "attendees": [{"displayName": "导师"}]}


def test_meeting_prep_plan_is_read_only():
    plan = analyze_route(RoutedRequest(intent="meeting_prep", query="论文讨论",
                                       date_reference="today", hour=15), "Asia/Shanghai")
    assert [step.operation for step in plan.actions] == ["search_events", "list_related_todos"]
    assert all(not step.mutation for step in plan.actions)


@pytest.mark.asyncio
async def test_structured_llm_meeting_prep_route():
    class LLM:
        def with_structured_output(self, schema):
            assert schema is RoutedRequest
            return self

        async def ainvoke(self, messages):
            assert "导师" in messages[-1].content
            return {"intent": "meeting_prep", "query": "导师讨论", "date_reference": "today",
                    "hour": 15}

    route = await route_request("帮我准备下午三点和导师的讨论", "Asia/Shanghai", LLM())
    assert route.intent == "meeting_prep" and route.hour == 15


def test_meeting_selection_clarifies_ambiguity():
    meeting, error = select_meeting([event(), event("event-2")], "Asia/Shanghai", 15, query="论文")
    assert meeting is None
    assert "Multiple meetings" in error
    meeting, error = select_meeting([event()], "Asia/Shanghai", 17, query="论文")
    assert meeting is None and "No matching" in error
    attendee_match = event(title="实验进展")
    meeting, error = select_meeting([attendee_match], "Asia/Shanghai", 15, query="导师讨论")
    assert meeting is attendee_match and error is None


def test_meeting_prep_timezone_and_missing_context_are_explicit():
    source = event()
    source["start"]["dateTime"] = "2026-10-01T07:00:00+00:00"
    snapshot = MeetingPrepSnapshot(user_id="u1", event=source, todos=[], profile=None,
                                   recent_turns=[], timezone="Asia/Shanghai")
    text = format_meeting_prep(snapshot)
    assert "2026-10-01 15:00" in text
    assert "No related Todo found" in text
    assert "do not assume" in text


def test_meeting_prep_applies_chinese_concise_profile_to_final_answer():
    source = event()
    related = [{"id": f"todo-{index}", "title": f"准备论文讨论 {index}",
                "status": "pending", "priority": "high", "due_date": "2026-10-02T10:00:00+08:00"}
               for index in range(4)]
    base = dict(user_id="u1", event=source, todos=related, timezone="Asia/Shanghai",
                recent_turns=[{"source": "turn-1", "text": "上次讨论论文进度"}])
    verbose = format_meeting_prep(MeetingPrepSnapshot(**base, profile=None))
    response = format_meeting_prep(MeetingPrepSnapshot(
        **base, profile={"preferences": ["以后请用简短中文回答"]}))
    assert response.startswith("会议准备：论文讨论")
    assert "相关任务：" in response and "讨论建议：" in response
    assert "[日程 event-1]" in response and "[待办 todo-0]" in response
    assert "turn-1" in response
    assert "todo-2" not in response
    assert len(response) < len(verbose)


def test_meeting_prep_english_concise_preference_reduces_output():
    base = dict(user_id="u1", event=event(), todos=[], recent_turns=[], timezone="Asia/Shanghai")
    verbose = format_meeting_prep(MeetingPrepSnapshot(**base, profile=None))
    concise = format_meeting_prep(MeetingPrepSnapshot(
        **base, profile={"preferences": ["concise answers"]}))
    assert concise.startswith("Meeting prep:")
    assert "Calendar source: event-1" in concise
    assert len(concise) < len(verbose)


@pytest.mark.asyncio
async def test_prep_uses_user_todos_profile_and_recent_source_without_writes(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "prep-a")
        other = await crud.get_or_create_user(db, "prep-b")
        await crud.create_todo(db, user.id, "修改论文", due_date=datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(days=1))
        await crud.create_todo(db, other.id, "另一个用户的论文", due_date=datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(days=1))
        await db.commit()
        user_id = user.id
    calendar = MeetingCalendar([event()])
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(), store=ProfileStore(user_id), meeting_llm=MeetingLLM())
    route = RoutedRequest(intent="meeting_prep", query="论文讨论", date_reference="today", hour=15)
    result = await start_workflow(graph, user_id, "帮我准备论文讨论", "Asia/Shanghai", "meeting-a",
                                  route=route, recent_context=[{"source": "turn-1", "text": "我还在修改论文"}])
    assert "__interrupt__" not in result
    response = result["response"]
    assert "Calendar: 论文讨论" in response
    assert "修改论文" in response
    assert "另一个用户的论文" not in response
    assert "Meeting prep:" in response
    assert calendar.writes == 0


@pytest.mark.asyncio
async def test_meeting_memory_query_is_anchored_after_exact_get(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "prep-memory-owner")
        db.add(StructuredMemory(id="memory-1", user_id=user.id, memory_key="meeting-test",
                                kind="decision", payload={},
                                canonical_text="Exact Project Sync chose plan B",
                                source_refs=[{"source_type": "checkpoint", "source_id": "message-1"}],
                                status="current"))
        await db.commit()
        user_id = user.id
    source = event(title="Exact Project Sync")
    source["attendees"] = [{"email": "alice@example.test"}]
    calendar = MeetingCalendar([source])
    model = MeetingLLM()

    class Memory:
        request = None

        async def assemble(self, request):
            self.request = request
            return ContextBundle(
                current_request=request.request, recent_messages=(), summary=None,
                history=(), structured=(StructuredHit(
                    memory_id="memory-1", user_id=user_id, kind="decision", payload={},
                    canonical_text="Exact Project Sync chose plan B",
                    source_refs=({"source_type": "checkpoint", "source_id": "message-1"},),
                    updated_at=datetime.now(ZoneInfo("Asia/Shanghai")), score=0.9,
                    retrieval_method="structured_semantic"),),
                provenance=(), estimated_tokens=50, token_budget=request.token_budget,
                degraded=(), summary_updated=False)

    memory = Memory()
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(), store=ProfileStore(user_id), meeting_llm=model,
        memory_orchestrator=memory)
    route = RoutedRequest(intent="meeting_prep", query="Exact Project Sync", date_reference="today", hour=15)
    result = await start_workflow(graph, user_id, "准备 Exact Project Sync", "Asia/Shanghai",
                                  "meeting-memory", route=route)
    assert "Meeting prep:" in result["response"]
    assert memory.request is not None and "alice@example.test" in memory.request.search_query
    assert calendar.calls >= 2
    assert any(item["key"].startswith("M") for item in model.evidence)


@pytest.mark.asyncio
async def test_full_workflow_uses_focused_attendee_gmail_thread(session_factory):
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "prep-gmail-owner")
        await db.commit()
        user_id = user.id
    source = event(title="Project Sync")
    source["attendees"] = [{"email": "alice@example.test"}]
    message = GmailMessage(message_id="message-1", thread_id="thread-1",
                           sender="Alice <alice@example.test>",
                           recipients=["self@example.test"], subject="Project Sync",
                           body="Please bring the benchmark results.")

    class Gmail:
        queries = []

        async def account_address(self):
            return "self@example.test"

        async def search(self, query, limit=5):
            self.queries.append(query)
            return [message]

        async def thread(self, thread_id):
            assert thread_id == "thread-1"
            return [message]

    gmail = Gmail()
    model = MeetingLLM()
    calendar = MeetingCalendar([source])
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(), store=ProfileStore(user_id), meeting_llm=model,
        gmail=gmail, gmail_owner_telegram_id="prep-owner")
    route = RoutedRequest(intent="meeting_prep", query="Project Sync", date_reference="today", hour=15)
    result = await start_workflow(graph, user_id, "Prepare Project Sync", "Asia/Shanghai",
                                  "meeting-gmail", route=route, telegram_id="prep-owner")
    assert "Meeting prep:" in result["response"]
    assert len(gmail.queries) == 1 and "alice@example.test" in gmail.queries[0]
    assert any(item["type"] == "gmail" for item in model.evidence)
    assert calendar.writes == 0


@pytest.mark.asyncio
async def test_calendar_failure_does_not_invent_meeting(session_factory):
    calendar = MeetingCalendar([], fail=True)
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver())
    route = RoutedRequest(intent="meeting_prep", query="论文讨论", date_reference="today", hour=15)
    result = await start_workflow(graph, "user-a", "准备会议", "Asia/Shanghai", "meeting-fail", route=route)
    assert "Calendar unavailable" in result["response"]
    assert "Meeting prep:" not in result["response"]
    assert calendar.writes == 0
