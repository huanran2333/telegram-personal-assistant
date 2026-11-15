import pytest
import logging
from datetime import datetime, timedelta, timezone
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.actions import ActionProposal, ActionResult, ActionStatus, PlanStep, RequestAnalysis
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarEventSnapshot
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.planner import CalendarProposalEdit, RoutedRequest, analyze_route, proposal_from_plan


def test_todo_delete_proposals_bind_approved_versions():
    exact = RequestAnalysis(actions=[PlanStep(step_id="delete_todo", domain="todo",
                                              operation="delete_todo", mutation=True)])
    exact_proposal = proposal_from_plan(
        exact, {"todos": [{"id": "synthetic-1", "title": "One", "version": 4}]}, "UTC")
    assert exact_proposal.args["approved_todo_version"] == 4

    bulk = RequestAnalysis(actions=[PlanStep(step_id="delete_todos_by_tag", domain="todo",
                                             operation="delete_todos_by_tag",
                                             args={"tag": "work"}, mutation=True)])
    bulk_proposal = proposal_from_plan(
        bulk, {"todos": [{"id": "synthetic-1", "title": "One", "version": 4},
                         {"id": "synthetic-2", "title": "Two", "version": 7}]}, "UTC")
    assert bulk_proposal.args["approved_todo_versions"] == {"synthetic-1": 4, "synthetic-2": 7}
from app.agent.workflow import build_productivity_graph, resume_workflow, start_workflow
from app.db import crud


class FakeCalendar:
    def __init__(self, events=None, fail=False):
        self.events = events or []
        self.fail = fail
        self.created = []
        self.updated = []
        self.deleted = []

    async def availability(self, start, end, timezone):
        if self.fail:
            raise CalendarError("OAuth unavailable")
        return {"calendars": {"primary": {"busy": []}}}

    async def search_events(self, query, start=None, end=None):
        if self.fail:
            raise CalendarError("OAuth unavailable")
        return self.events

    async def resolve_create_scope(self, calendar_id):
        if self.fail:
            raise CalendarError("OAuth unavailable")
        return "normal", calendar_id

    async def get_event(self, calendar_id, event_id):
        if self.fail:
            raise CalendarError("OAuth unavailable")
        event = next((item for item in self.events if item.get("id") == event_id), None)
        if event is None:
            raise CalendarError("Calendar event not found")
        return CalendarEventSnapshot.model_validate({
            "id": event_id, "etag": '"synthetic-approved-v1"',
            "status": "confirmed", "summary": event.get("summary"),
            "start": event.get("start") or {}, "end": event.get("end") or {},
        })

    async def create_event(self, args):
        self.created.append(args)
        return {"id": f"event-{len(self.created)}"}

    async def update_event(self, args):
        self.updated.append(args)
        return {"id": args["eventId"]}

    async def delete_event(self, args):
        self.deleted.append(args)
        return {"id": args["eventId"]}


class FakeLedger:
    def __init__(self):
        self.rows = {}

    async def begin(self, proposal):
        if proposal.action_id in self.rows:
            row = self.rows[proposal.action_id]
            return ActionResult(action_id=proposal.action_id, status=row["status"], message="Action was already processed.", result=row.get("result", {}))
        self.rows[proposal.action_id] = {"status": ActionStatus.EXECUTING}
        return None

    async def complete(self, proposal, result):
        self.rows[proposal.action_id] = {"status": ActionStatus.EXECUTED, "result": result}

    async def fail(self, proposal, message):
        self.rows[proposal.action_id] = {"status": ActionStatus.FAILED, "result": {"error": message}}


def interrupt_value(result):
    return result["__interrupt__"][0].value


class FakeEditLLM:
    def with_structured_output(self, schema):
        assert schema is CalendarProposalEdit
        return self

    async def ainvoke(self, messages):
        assert messages[-1].content == "时间改为五点半"
        return {"hour": 5, "minute": 30}


def make_graph(calendar):
    return build_productivity_graph(calendar, ActionExecutor(calendar, FakeLedger(), None),
                                    InMemorySaver(), edit_llm=FakeEditLLM())


READ_ROUTE = RoutedRequest(intent="calendar_read", date_reference="weekday", weekday=4,
                           day_part="afternoon", read_kind="availability")
_TODAY = datetime.now(timezone.utc).date()
NEXT_FRIDAY = (_TODAY + timedelta(days=((4 - _TODAY.weekday()) % 7 or 7))).isoformat()
CREATE_ROUTE = RoutedRequest(intent="cross_domain", title="写论文", date_reference=NEXT_FRIDAY,
                             day_part="afternoon", duration_minutes=120,
                             read_kind="availability", conditional_on_free=True)


def test_simple_request_fast_path():
    assert not analyze_route(RoutedRequest(intent="todo", todo_operation="create",
                                           title="groceries"), "UTC").requires_plan


@pytest.mark.asyncio
async def test_workflow_logs_do_not_contain_proposal_or_plan_payload(caplog):
    caplog.set_level(logging.INFO)
    graph = make_graph(FakeCalendar())
    route = CREATE_ROUTE.model_copy(update={"title": "PRIVATE_CALENDAR_TITLE"})
    await start_workflow(graph, "u1", "PRIVATE_USER_REQUEST", "UTC",
                         "privacy-log-thread", route=route)
    app_logs = "\n".join(record.getMessage() for record in caplog.records
                         if record.name.startswith("app."))
    assert "PRIVATE_CALENDAR_TITLE" not in app_logs
    assert "PRIVATE_USER_REQUEST" not in app_logs
    assert "proposal=create_event" not in app_logs  # metadata includes domain.operation, not raw args


@pytest.mark.asyncio
async def test_cross_domain_plan():
    plan = analyze_route(CREATE_ROUTE, "UTC")
    assert plan.requires_plan
    assert [step.operation for step in plan.actions] == ["availability", "create_event"]


@pytest.mark.asyncio
async def test_calendar_read_no_approval():
    graph = make_graph(FakeCalendar())
    result = await start_workflow(graph, "u1", "周五下午有会吗？", "UTC", "calendar-read", route=READ_ROUTE)
    assert "__interrupt__" not in result
    assert result["response"] == "Calendar is free for the requested time."


@pytest.mark.asyncio
async def test_calendar_event_listing_with_exactly_one_event():
    event = {"id": "event-1", "summary": "测试会议",
             "start": {"dateTime": "2026-10-01T16:00:00+00:00"},
             "end": {"dateTime": "2026-10-01T16:30:00+00:00"}}
    graph = make_graph(FakeCalendar(events=[event]))
    route = RoutedRequest(intent="calendar_read", date_reference="tomorrow", read_kind="events")
    result = await start_workflow(graph, "u1", "看看我明天有什么日程", "UTC", "list-one", route=route)
    assert "__interrupt__" not in result
    assert "16:00–16:30 测试会议" in result["response"]
    assert "Calendar is free" not in result["response"]


@pytest.mark.asyncio
async def test_calendar_create_requires_approval_and_checkpoint_resume():
    calendar = FakeCalendar()
    graph = make_graph(calendar)
    thread_id = "create-thread"
    result = await start_workflow(graph, "u1", "看看我周五下午有没有会，没有的话安排两小时写论文。", "UTC", thread_id, route=CREATE_ROUTE)
    payload = interrupt_value(result)
    assert payload["kind"] == "action_proposal"
    assert payload["risk_level"] == "external_write"
    resumed = await resume_workflow(graph, thread_id, {"decision": "approve"})
    assert resumed["response"] == "Action completed."
    assert len(calendar.created) == 1


@pytest.mark.asyncio
async def test_reject_does_not_execute():
    calendar = FakeCalendar()
    graph = make_graph(calendar)
    result = await start_workflow(graph, "u1", "看看我周五下午有没有会，没有的话安排两小时写论文。", "UTC", "reject-thread", route=CREATE_ROUTE)
    assert interrupt_value(result)["kind"] == "action_proposal"
    resumed = await resume_workflow(graph, "reject-thread", {"decision": "reject"})
    assert resumed["response"] == "Action rejected; nothing was changed."
    assert calendar.created == []


@pytest.mark.asyncio
async def test_edit_reproposes():
    calendar = FakeCalendar()
    graph = make_graph(calendar)
    thread_id = "edit-thread"
    route = RoutedRequest(intent="calendar_create", title="测试会议", date_reference="tomorrow",
                          hour=17, minute=0, duration_minutes=30)
    first = await start_workflow(graph, "u1", "明天下午五点安排半小时测试会议", "UTC", thread_id, route=route)
    first_id = interrupt_value(first)["action_id"]
    edit_prompt = await resume_workflow(graph, thread_id, {"decision": "edit"})
    assert interrupt_value(edit_prompt)["kind"] == "edit_request"
    assert calendar.created == []
    reproposal = await resume_workflow(graph, thread_id, "时间改为五点半")
    assert interrupt_value(reproposal)["kind"] == "action_proposal"
    assert interrupt_value(reproposal)["action_id"] != first_id
    assert reproposal["proposal"]["args"]["start"].endswith("T17:30:00+00:00")
    assert reproposal["proposal"]["args"]["end"].endswith("T18:00:00+00:00")
    assert calendar.created == []
    rejected = await resume_workflow(graph, thread_id, {"decision": "reject"})
    assert rejected["response"] == "Action rejected; nothing was changed."
    assert calendar.created == []


@pytest.mark.asyncio
async def test_edit_calendar_read_failure_cannot_repropose_or_write():
    calendar = FakeCalendar()
    graph = make_graph(calendar)
    thread_id = "edit-calendar-failure"
    route = RoutedRequest(intent="calendar_create", title="测试会议", date_reference="tomorrow",
                          hour=17, duration_minutes=30)
    first = await start_workflow(graph, "u1", "明天下午五点安排测试会议", "UTC", thread_id, route=route)
    assert interrupt_value(first)["kind"] == "action_proposal"
    await resume_workflow(graph, thread_id, {"decision": "edit"})
    calendar.fail = True
    failed = await resume_workflow(graph, thread_id, "时间改为五点半")
    assert "__interrupt__" not in failed
    assert "OAuth unavailable" in failed["response"]
    assert failed["proposal"] == {}
    assert calendar.created == []


@pytest.mark.asyncio
async def test_duplicate_approve_is_idempotent():
    calendar = FakeCalendar()
    ledger = FakeLedger()
    executor = ActionExecutor(calendar, ledger, None)
    proposal = ActionProposal(action_id="same-action", domain="calendar", operation="create_event",
                              args={"title": "Write thesis"}, human_readable_preview="Create", reason="test",
                              risk_level="external_write")
    first = await executor.execute(proposal)
    second = await executor.execute(proposal)
    assert first.status == ActionStatus.EXECUTED
    assert second.status == ActionStatus.EXECUTED
    assert len(calendar.created) == 1


@pytest.mark.asyncio
async def test_calendar_update_and_delete_require_approval():
    event = {"id": "event-1", "summary": "Thesis discussion"}
    for request, route in (
        ("move thesis discussion meeting to five", RoutedRequest(intent="calendar_update", query="Thesis discussion", date_reference="tomorrow", hour=17)),
        ("delete thesis discussion meeting", RoutedRequest(intent="calendar_delete", query="Thesis discussion")),
    ):
        graph = make_graph(FakeCalendar(events=[event]))
        result = await start_workflow(graph, "u1", request, "UTC", f"thread-{request[:2]}", route=route)
        payload = interrupt_value(result)
        assert payload["kind"] == "action_proposal"


@pytest.mark.asyncio
async def test_calendar_delete_multiple_candidates_clarifies_and_zero_is_not_found():
    route = RoutedRequest(intent="calendar_delete", query="review meeting",
                          date_reference="tomorrow")
    for events, expected in (([{"id": "a", "summary": "review meeting"},
                               {"id": "b", "summary": "review meeting"}],
                              "Multiple matching events"),
                             ([], "No matching Calendar event")):
        calendar = FakeCalendar(events=events)
        graph = make_graph(calendar)
        result = await start_workflow(graph, "u1", "Delete tomorrow's review meeting",
                                      "UTC", "delete-ambiguity-" + str(len(events)), route=route)
        assert "__interrupt__" not in result
        assert expected in result["response"]
        assert calendar.deleted == []


@pytest.mark.asyncio
async def test_mcp_failure_does_not_fail_open():
    graph = make_graph(FakeCalendar(fail=True))
    result = await start_workflow(graph, "u1", "看看我周五下午有没有会，没有的话安排两小时写论文。", "UTC", "failure-thread", route=CREATE_ROUTE)
    assert "__interrupt__" not in result
    assert "Calendar unavailable" in result["response"]


@pytest.mark.asyncio
async def test_calendar_todo_reminder_plan_executes_after_approval(session_factory):
    calendar = FakeCalendar()
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(),
    )
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "cross-domain-user")
        await db.commit()
        user_id = user.id
    # Two/three days ahead keeps the approved Calendar slot and 20:00
    # reminder in the future regardless of the CI wall clock.
    event_weekday = (datetime.now(timezone.utc).weekday() + 2) % 7
    due_weekday = (event_weekday + 1) % 7
    route = RoutedRequest(
        intent="cross_domain", todo_operation="create", title="修改论文", todo_title="提交论文",
        date_reference="weekday", weekday=event_weekday, day_part="afternoon", duration_minutes=120,
        todo_due_date_reference="weekday", todo_due_weekday=due_weekday,
        reminder_requested=True, reminder_date_reference="weekday", reminder_weekday=event_weekday,
        reminder_hour=20,
    )
    thread_id = "cross-domain-approval"
    first = await start_workflow(graph, user_id, "周五论文截止，周四下午安排两小时修改论文，周四晚上八点提醒", "UTC",
                                 thread_id, route=route)
    assert interrupt_value(first)["kind"] == "action_proposal"
    assert not calendar.created
    async with session_factory() as db:
        assert await crud.list_todos(db, user_id) == []
    result = await resume_workflow(graph, thread_id, {"decision": "approve"})
    assert result["response"] == "Calendar event and Todo/reminder created."
    assert len(calendar.created) == 1
    async with session_factory() as db:
        todos = await crud.list_todos(db, user_id)
        assert len(todos) == 1
        assert todos[0].title == "提交论文"
        assert todos[0].due_date.weekday() == due_weekday
        assert todos[0].reminder_at.weekday() == event_weekday
        assert todos[0].reminder_at.hour == 20


@pytest.mark.asyncio
async def test_cross_domain_reject_creates_neither_event_nor_todo(session_factory):
    calendar = FakeCalendar()
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(),
    )
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "cross-domain-reject-user")
        await db.commit()
        user_id = user.id
    route = RoutedRequest(
        intent="cross_domain", todo_operation="create", title="写论文", todo_title="交论文",
        date_reference="tomorrow", day_part="afternoon", duration_minutes=120,
        todo_due_date_reference="day_after_tomorrow", reminder_requested=True,
        reminder_date_reference="tomorrow", reminder_hour=20,
    )
    await start_workflow(graph, user_id, "schedule and remind", "UTC", "cross-domain-reject", route=route)
    result = await resume_workflow(graph, "cross-domain-reject", {"decision": "reject"})
    assert result["response"] == "Action rejected; nothing was changed."
    assert calendar.created == []
    async with session_factory() as db:
        assert await crud.list_todos(db, user_id) == []


@pytest.mark.asyncio
async def test_todo_delete_requires_approval_and_reject_preserves_row(session_factory):
    calendar = FakeCalendar()
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(),
    )
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "todo-delete-approval")
        todo = await crud.create_todo(db, user.id, "论文初稿")
        await db.commit()
        user_id, todo_id = user.id, todo.id
    route = RoutedRequest(intent="todo", todo_operation="delete", title="论文初稿")
    first = await start_workflow(graph, user_id, "删除论文初稿", "UTC", "todo-delete-reject", route=route)
    assert interrupt_value(first)["risk_level"] == "destructive"
    async with session_factory() as db:
        assert await crud.get_todo(db, todo_id, user_id) is not None
    await resume_workflow(graph, "todo-delete-reject", {"decision": "reject"})
    async with session_factory() as db:
        assert await crud.get_todo(db, todo_id, user_id) is not None
    await start_workflow(graph, user_id, "删除论文初稿", "UTC", "todo-delete-approve", route=route)
    approved = await resume_workflow(graph, "todo-delete-approve", {"decision": "approve"})
    assert approved["response"] == "Action completed."
    async with session_factory() as db:
        assert await crud.get_todo(db, todo_id, user_id) is None


@pytest.mark.asyncio
async def test_ambiguous_todo_delete_never_proposes(session_factory):
    calendar = FakeCalendar()
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(),
    )
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "todo-delete-ambiguous")
        await crud.create_todo(db, user.id, "论文初稿")
        await crud.create_todo(db, user.id, "论文终稿")
        await db.commit()
        user_id = user.id
    route = RoutedRequest(intent="todo", todo_operation="delete", title="论文")
    result = await start_workflow(graph, user_id, "删除论文", "UTC", "todo-delete-ambiguous", route=route)
    assert "__interrupt__" not in result
    assert "multiple matching Todos" in result["response"]
    async with session_factory() as db:
        assert len(await crud.list_todos(db, user_id)) == 2


@pytest.mark.asyncio
async def test_tag_bulk_delete_requires_approval_and_uses_captured_ids(session_factory):
    calendar = FakeCalendar()
    executor = ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory)
    graph = build_productivity_graph(calendar, executor, InMemorySaver())
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "todo-bulk-delete")
        other = await crud.get_or_create_user(db, "todo-bulk-other")
        first = await crud.create_todo(db, user.id, "First", tags=["work"])
        second = await crud.create_todo(db, user.id, "Second", tags=["work"])
        unrelated = await crud.create_todo(db, user.id, "Unrelated", tags=["home"])
        other_user = await crud.create_todo(db, other.id, "Private", tags=["work"])
        await db.commit()
        user_id = user.id
    route = RoutedRequest(intent="todo", todo_operation="delete", todo_tag="work")

    rejected = await start_workflow(graph, user_id, "删除所有 #work 任务", "UTC", "bulk-reject", route=route)
    assert interrupt_value(rejected)["risk_level"] == "destructive"
    await resume_workflow(graph, "bulk-reject", {"decision": "reject"})
    async with session_factory() as db:
        assert await crud.get_todo(db, first.id, user_id) is not None
        assert await crud.get_todo(db, second.id, user_id) is not None

    pending = await start_workflow(graph, user_id, "删除所有 #work 任务", "UTC", "bulk-approve", route=route)
    proposal = ActionProposal.model_validate(pending["proposal"])
    assert set(proposal.args["todo_ids"]) == {first.id, second.id}
    assert "First" in proposal.human_readable_preview
    assert "Second" in proposal.human_readable_preview
    async with session_factory() as db:
        new_match = await crud.create_todo(db, user_id, "Created after proposal", tags=["work"])
        await db.commit()
    approved = await resume_workflow(graph, "bulk-approve", {"decision": "approve"})
    assert approved["response"] == "Action completed."
    duplicate = await executor.execute(proposal)
    assert duplicate.message == "Action was already processed."
    async with session_factory() as db:
        assert await crud.get_todo(db, first.id, user_id) is None
        assert await crud.get_todo(db, second.id, user_id) is None
        assert await crud.get_todo(db, unrelated.id, user_id) is not None
        assert await crud.get_todo(db, new_match.id, user_id) is not None
        assert await crud.get_todo(db, other_user.id, other.id) is not None


@pytest.mark.asyncio
async def test_tag_bulk_delete_fails_if_approved_task_changes(session_factory):
    calendar = FakeCalendar()
    graph = build_productivity_graph(
        calendar, ActionExecutor(calendar, SqlAlchemyActionLedger(session_factory), session_factory),
        InMemorySaver(),
    )
    async with session_factory() as db:
        user = await crud.get_or_create_user(db, "todo-bulk-stale")
        first = await crud.create_todo(db, user.id, "First", tags=["work"])
        second = await crud.create_todo(db, user.id, "Second", tags=["work"])
        await db.commit()
        user_id = user.id
    route = RoutedRequest(intent="todo", todo_operation="delete", todo_tag="work")
    await start_workflow(graph, user_id, "删除所有 #work 任务", "UTC", "bulk-stale", route=route)
    async with session_factory() as db:
        await crud.update_todo(db, second.id, user_id, tags=["home"])
        await db.commit()
    failed = await resume_workflow(graph, "bulk-stale", {"decision": "approve"})
    assert "changed since approval" in failed["response"]
    async with session_factory() as db:
        assert await crud.get_todo(db, first.id, user_id) is not None
        assert await crud.get_todo(db, second.id, user_id) is not None
