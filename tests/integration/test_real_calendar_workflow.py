"""Opt-in UTF-8 real Calendar acceptance; never part of default pytest."""
import asyncio
import sys
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import delete

from app.agent.actions import ActionProposal
from app.agent.planner import analyze_route, route_request
from app.agent.workflow import resume_workflow, start_workflow
from app.agent.workflow_runtime import get_calendar_adapter, get_productivity_graph
from app.config import get_settings
from app.db.models import ActionExecution, Todo, User
from app.db import crud
from app.db.session import AsyncSessionLocal
from app.main import app, lifespan
from app.scheduler.daily_brief import render_daily_occurrence

pytestmark = [pytest.mark.calendar_integration, pytest.mark.asyncio(loop_scope="session"),
              pytest.mark.provider_lifespan]


@pytest.fixture(scope="session")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


async def test_real_structured_calendar_analysis():
    timezone = get_settings().default_timezone
    now = datetime.now(ZoneInfo(timezone))
    read = await route_request("看看我明天有什么日程。", timezone)
    assert read.intent == "calendar_read"
    assert read.read_kind == "events"
    read_plan = analyze_route(read, timezone, now)
    expected = (now + timedelta(days=1)).date()
    assert datetime.fromisoformat(read_plan.actions[0].args["start"]).date() == expected

    create = await route_request("明天下午 4 点安排一个 30 分钟的测试会议。", timezone)
    assert create.intent == "calendar_create"
    assert create.title == "测试会议"
    plan = analyze_route(create, timezone, now)
    event = plan.actions[-1].args
    start = datetime.fromisoformat(event["start"])
    end = datetime.fromisoformat(event["end"])
    assert start.date() == expected
    assert start.hour == 16
    assert end - start == timedelta(minutes=30)


def _event(events, event_id):
    return next((event for event in events if event.get("id") == event_id), None)


async def _calendar_owner():
    """Use the real OAuth account's Telegram mapping, never a synthetic identity."""
    telegram_id = get_settings().google_calendar_owner_telegram_id
    if not telegram_id:
        pytest.skip("GOOGLE_CALENDAR_OWNER_TELEGRAM_ID is not configured")
    async with AsyncSessionLocal() as db:
        owner = await crud.get_user_by_telegram_id(db, telegram_id)
        if owner is None:
            pytest.skip("Calendar owner has no existing Telegram user mapping")
        return owner.id, telegram_id


async def test_real_calendar_read_create_update_delete():
    timezone = get_settings().default_timezone
    zone = ZoneInfo(timezone)
    tomorrow = (datetime.now(zone) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_start, day_end = tomorrow.isoformat(), (tomorrow + timedelta(days=1)).isoformat()
    uid, owner_telegram_id = await _calendar_owner()
    thread = f"calendar-e2e-{uid}"
    action_ids = []
    created_id = None
    baseline = set()
    async with lifespan(app):
        graph = await get_productivity_graph()
        calendar = get_calendar_adapter()
        try:
            baseline = {event["id"] for event in await calendar.search_events("", day_start, day_end)}
            read = await start_workflow(graph, uid, "看看我明天有什么日程。", timezone, thread,
                                        telegram_id=owner_telegram_id)
            assert read["analysis"]["mode"] == "calendar_read"
            read_step = read["analysis"]["actions"][0]
            assert read_step["operation"] == "search_events"
            assert datetime.fromisoformat(read_step["args"]["start"]).date() == tomorrow.date()
            assert "events" in read["read_results"]
            print("REAL_READ_RANGE", read_step["args"]["start"], read_step["args"]["end"])

            create = await start_workflow(graph, uid, "明天下午 4 点安排一个 30 分钟的测试会议。", timezone, thread,
                                          telegram_id=owner_telegram_id)
            assert create["__interrupt__"][0].value["kind"] == "action_proposal"
            proposal = ActionProposal.model_validate(create["proposal"])
            action_ids.append(proposal.action_id)
            assert proposal.operation == "create_event"
            assert proposal.args["title"] == "测试会议"
            assert datetime.fromisoformat(proposal.args["start"]).astimezone(zone).hour == 16
            assert datetime.fromisoformat(proposal.args["end"]) - datetime.fromisoformat(proposal.args["start"]) == timedelta(minutes=30)
            assert {event["id"] for event in await calendar.search_events("", day_start, day_end)} == baseline
            approved = await resume_workflow(graph, thread, {"decision": "approve"})
            assert approved["response"] == "Action completed."
            async with AsyncSessionLocal() as db:
                row = await db.get(ActionExecution, proposal.action_id)
                assert row.status == "executed" and row.external_id
                created_id = row.external_id
            events = await calendar.search_events("", day_start, day_end)
            found = _event(events, created_id)
            assert found and found["summary"] == "测试会议"
            assert datetime.fromisoformat(found["start"]["dateTime"]).astimezone(zone).hour == 16
            assert len({event["id"] for event in events} - baseline) == 1
            print("REAL_CREATE", proposal.action_id, created_id, found["summary"])

            prep = await start_workflow(graph, uid, "帮我准备明天下午四点的测试会议。", timezone,
                                        thread, telegram_id=owner_telegram_id)
            assert "__interrupt__" not in prep
            assert prep["analysis"]["mode"] == "meeting_prep"
            assert created_id in prep["response"]
            assert ("Suggested discussion points" in prep["response"] or
                    "讨论建议" in prep["response"])
            print("REAL_MEETING_PREP", created_id)

            update = await start_workflow(graph, uid, "把刚才那个测试会议改到下午 6 点。", timezone, thread,
                                          telegram_id=owner_telegram_id)
            print("UPDATE_PLAN", update.get("analysis", {}).get("mode"),
                  len(update.get("read_results", {}).get("events", [])), flush=True)
            assert update["__interrupt__"][0].value["kind"] == "action_proposal", update
            update_proposal = ActionProposal.model_validate(update["proposal"])
            action_ids.append(update_proposal.action_id)
            assert update_proposal.operation == "update_event"
            assert update_proposal.args["eventId"] == created_id
            assert datetime.fromisoformat(update_proposal.args["start"]).astimezone(zone).hour == 18
            updated = await resume_workflow(graph, thread, {"decision": "approve"})
            assert updated["response"] == "Action completed."
            events = await calendar.search_events("", day_start, day_end)
            found = _event(events, created_id)
            assert found and datetime.fromisoformat(found["start"]["dateTime"]).astimezone(zone).hour == 18
            assert len({event["id"] for event in events} - baseline) == 1
            print("REAL_UPDATE", update_proposal.action_id, created_id)

            reject = await start_workflow(graph, uid, "删除刚才那个测试会议。", timezone, thread,
                                          telegram_id=owner_telegram_id)
            assert reject["__interrupt__"][0].value["kind"] == "action_proposal", reject
            reject_proposal = ActionProposal.model_validate(reject["proposal"])
            assert reject_proposal.operation == "delete_event" and reject_proposal.args["eventId"] == created_id
            rejected = await resume_workflow(graph, thread, {"decision": "reject"})
            assert "rejected" in rejected["response"]
            assert _event(await calendar.search_events("", day_start, day_end), created_id)
            print("REAL_DELETE_REJECT", reject_proposal.action_id)

            deletion = await start_workflow(graph, uid, "删除刚才那个测试会议。", timezone, thread,
                                            telegram_id=owner_telegram_id)
            assert deletion["__interrupt__"][0].value["kind"] == "action_proposal", deletion
            delete_proposal = ActionProposal.model_validate(deletion["proposal"])
            action_ids.append(delete_proposal.action_id)
            assert delete_proposal.args["eventId"] == created_id
            deleted = await resume_workflow(graph, thread, {"decision": "approve"})
            assert deleted["response"] == "Action completed."
            assert not _event(await calendar.search_events("", day_start, day_end), created_id)
            print("REAL_DELETE_APPROVE", delete_proposal.action_id, created_id)
        finally:
            # If an assertion fails after create, remove only this test event
            # by its exact external ID, never another user's event by title.
            if created_id and _event(await calendar.search_events("", day_start, day_end), created_id):
                await calendar.delete_event({"eventId": created_id})
            async with AsyncSessionLocal() as db:
                for action_id in action_ids:
                    await db.execute(delete(ActionExecution).where(ActionExecution.action_id == action_id))
                await db.commit()


async def test_real_conditional_cross_domain_calendar():
    timezone = get_settings().default_timezone
    uid, owner_telegram_id = await _calendar_owner()
    thread = f"calendar-cross-e2e-{uuid4().hex}"
    created_id = None
    action_id = None
    async with lifespan(app):
        graph = await get_productivity_graph()
        calendar = get_calendar_adapter()
        try:
            # A bare "Friday" becomes the current Friday when this test runs
            # on Friday afternoon, making the slot stale before approval.
            today = datetime.now(ZoneInfo(timezone)).date()
            days_until_future_friday = (4 - today.weekday()) % 7 or 7
            future_friday = today + timedelta(days=days_until_future_friday)
            request = f"看看{future_friday.isoformat()}下午有没有会，没有的话安排两小时写论文。"
            route = await route_request(request, timezone)
            assert route.intent == "cross_domain"
            assert route.conditional_on_free
            assert route.title == "写论文"
            plan = analyze_route(route, timezone)
            operations = [step.operation for step in plan.actions]
            assert operations[:2] == ["availability", "create_event"]
            assert operations[2:] in ([], ["create_todo"])
            window = plan.actions[0].args
            before = {event["id"] for event in await calendar.search_events(
                "", window["start"], window["end"])}
            result = await start_workflow(graph, uid, request, timezone, thread, route=route,
                                          telegram_id=owner_telegram_id)
            assert "availability" in result["read_results"], result
            busy = result["read_results"]["availability"]["calendars"]["primary"]["busy"]
            if "__interrupt__" not in result:
                assert "__interrupt__" not in result
                assert "busy" in result["response"]
                print("REAL_CROSS_DOMAIN_CONFLICT_NO_WRITE", len(busy))
            else:
                assert result["__interrupt__"][0].value["kind"] == "action_proposal", result
                proposal = ActionProposal.model_validate(result["proposal"])
                action_id = proposal.action_id
                assert proposal.operation == "create_event" and proposal.args["title"] == "写论文"
                assert {event["id"] for event in await calendar.search_events(
                    "", window["start"], window["end"])} == before
                approved = await resume_workflow(graph, thread, {"decision": "approve"})
                expected = ("Calendar event and Todo/reminder created."
                            if "create_todo" in operations else "Action completed.")
                assert approved["response"] == expected
                async with AsyncSessionLocal() as db:
                    row = await db.get(ActionExecution, action_id)
                    created_id = row.external_id
                assert created_id and created_id not in before
                assert _event(await calendar.search_events("", window["start"], window["end"]), created_id)
                print("REAL_CROSS_DOMAIN_CREATED", action_id, created_id)
        finally:
            if created_id:
                await calendar.delete_event({"eventId": created_id})
                assert not _event(await calendar.search_events(
                    "", window["start"], window["end"]), created_id)
            async with AsyncSessionLocal() as db:
                if action_id:
                    await db.execute(delete(ActionExecution).where(ActionExecution.action_id == action_id))
                    todo_row = await db.get(ActionExecution, f"{action_id}:todo")
                    if todo_row and todo_row.external_id:
                        await db.execute(delete(Todo).where(Todo.id == todo_row.external_id,
                                                            Todo.user_id == uid))
                    await db.execute(delete(ActionExecution).where(ActionExecution.action_id == f"{action_id}:todo"))
                await db.commit()


async def test_real_smart_schedule_uses_google_freebusy_without_writing():
    uid, owner_telegram_id = await _calendar_owner()
    timezone = get_settings().default_timezone
    thread = f"schedule-e2e-{uuid4().hex}"
    async with lifespan(app):
        graph = await get_productivity_graph()
        route = await route_request("未来七天找一个一小时空档学习 LangGraph，最好下午。", timezone)
        assert route.intent == "smart_schedule"
        result = await start_workflow(graph, uid, "未来七天找一个一小时空档学习 LangGraph，最好下午。",
                                      timezone, thread, route=route, telegram_id=owner_telegram_id)
        assert "availability" in result["read_results"]
        assert result["__interrupt__"][0].value["kind"] == "slot_selection"
        assert 2 <= len(result["slot_candidates"]) <= 3
        chosen = await resume_workflow(graph, thread, {"selection_index": 0})
        assert chosen["__interrupt__"][0].value["kind"] == "action_proposal"
        assert chosen["proposal"]["operation"] == "create_event"
        # No approval: a candidate and proposal alone cannot write to Google.
        print("REAL_SMART_SCHEDULING", result["slot_candidates"])


async def test_real_calendar_todo_reminder_combined_workflow():
    timezone = get_settings().default_timezone
    uid, owner_telegram_id = await _calendar_owner()
    thread = f"calendar-todo-reminder-e2e-{uid}"
    request = "周五论文截止，看看周四下午有没有空，安排两小时让我修改论文，并在周四晚上八点提醒我。"
    event_id = None
    action_id = None
    async with lifespan(app):
        graph = await get_productivity_graph()
        calendar = get_calendar_adapter()
        try:
            route = await route_request(request, timezone)
            assert (route.weekday, route.todo_due_weekday, route.reminder_weekday) == (3, 4, 3)
            result = await start_workflow(graph, uid, request, timezone, thread, route=route,
                                          telegram_id=owner_telegram_id)
            assert result["__interrupt__"][0].value["kind"] == "action_proposal", result
            proposal = ActionProposal.model_validate(result["proposal"])
            action_id = proposal.action_id
            window = result["analysis"]["actions"][0]["args"]
            before = {event["id"] for event in await calendar.search_events("", window["start"], window["end"])}
            async with AsyncSessionLocal() as db:
                existing_todo_ids = {todo.id for todo in await crud.list_todos(db, uid)}
            approved = await resume_workflow(graph, thread, {"decision": "approve"})
            assert approved["response"] == "Calendar event and Todo/reminder created."
            async with AsyncSessionLocal() as db:
                event_row = await db.get(ActionExecution, action_id)
                todo_row = await db.get(ActionExecution, f"{action_id}:todo")
                assert event_row.status == todo_row.status == "executed"
                event_id = event_row.external_id
                todos = await crud.list_todos(db, uid)
                created_todos = [todo for todo in todos if todo.id not in existing_todo_ids]
                assert len(created_todos) == 1
                local_due = created_todos[0].due_date.astimezone(ZoneInfo(timezone))
                local_reminder = created_todos[0].reminder_at.astimezone(ZoneInfo(timezone))
                assert local_due.weekday() == 4
                assert local_reminder.weekday() == 3
                assert local_reminder.hour == 20
            after = {event["id"] for event in await calendar.search_events("", window["start"], window["end"])}
            assert event_id in after - before
            print("REAL_THREE_DOMAIN", action_id, event_id)
        finally:
            if action_id and not event_id:
                async with AsyncSessionLocal() as db:
                    row = await db.get(ActionExecution, action_id)
                    event_id = row.external_id if row else None
            if event_id:
                await calendar.delete_event({"eventId": event_id})
                if "window" in locals():
                    assert not _event(await calendar.search_events(
                        "", window["start"], window["end"]), event_id)
            async with AsyncSessionLocal() as db:
                if action_id:
                    todo_row = await db.get(ActionExecution, f"{action_id}:todo")
                    if todo_row and todo_row.external_id:
                        await db.execute(delete(Todo).where(Todo.id == todo_row.external_id,
                                                            Todo.user_id == uid))
                    await db.execute(delete(ActionExecution).where(
                        ActionExecution.action_id.in_([action_id, f"{action_id}:todo"])))
                await db.commit()


async def test_real_calendar_and_postgres_daily_brief_without_telegram():
    from types import SimpleNamespace
    timezone = get_settings().default_timezone
    now = datetime.now(ZoneInfo(timezone))
    uid, owner_telegram_id = await _calendar_owner()

    async with lifespan(app):
        calendar = get_calendar_adapter()
        async with AsyncSessionLocal() as db:
            todo = await crud.create_todo(db, uid, "itest daily brief task",
                                          due_date=now.replace(hour=23, minute=0))
            todo_id = todo.id
            await db.commit()
        try:
            text, locale = await render_daily_occurrence(
                SimpleNamespace(user_id=uid, timezone_name=timezone,
                                planned_at=now), calendar=calendar,
                calendar_owner_telegram_id=owner_telegram_id)
            assert "itest daily brief task" in text
            assert "Calendar unavailable" not in text
        finally:
            async with AsyncSessionLocal() as db:
                await db.execute(delete(Todo).where(Todo.id == todo_id, Todo.user_id == uid))
                await db.commit()
