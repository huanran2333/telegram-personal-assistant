"""Synthetic current create_agent acceptance; no real model or Google calls."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from app.agent import official_read_tools, unified_tools
from app.agent.calendar import CalendarError
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.config import get_settings
from app.db import crud
from app.db.models import ActionExecution, OutboundDelivery, ScheduledOccurrence, Todo
from app.scheduler import proactive
from app.web.processing_context import web_run_scope
from app.web.runs import claim_one, submit_run
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

pytestmark = pytest.mark.integration


class CalendarWorld:
    def __init__(self, *, fail: bool = False, fully_booked: bool = False):
        self.fail = fail
        self.fully_booked = fully_booked
        self.reads = 0

    async def search_events(self, query, start, end):
        self.reads += 1
        if self.fail:
            raise CalendarError("SyntheticCalendarReadFailure")
        return [{"id": "occupied", "summary": "验收占用",
                 "start": {"dateTime": "2026-11-04T14:00:00+08:00"},
                 "end": {"dateTime": "2026-11-04T15:00:00+08:00"},
                 "status": "confirmed"}]

    async def availability(self, start, end, timezone_name):
        self.reads += 1
        if self.fail:
            raise CalendarError("SyntheticCalendarReadFailure")
        if self.fully_booked:
            return {"calendars": {"primary": {"busy": [
                {"start": start, "end": end}]}}}
        return {"calendars": {"primary": {"busy": [{
            "start": "2026-11-04T14:00:00+08:00",
            "end": "2026-11-04T15:00:00+08:00"}]}}}


async def _invoke(factory, target, monkeypatch, user, responses, request, calendar,
                  *, evidence_id=None):
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id,
        "gmail_enabled": False})
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: calendar)
    conn = await psycopg.AsyncConnection.connect(
        target.set(drivername="postgresql").render_as_string(hide_password=False),
        autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=ScriptedModel(responses), session_factory=factory,
            settings=settings, checkpointer=saver, interrupt_mode="per_call")
        answer, state = await run_agent(
            graph, user.id, user.telegram_id, "", request, origin="web",
            turn_id=f"core-{uuid4().hex}", thread_id=f"web:ordinary:{user.id}",
            return_state=True)
        async with factory() as db:
            todos = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
            rows = [{"title": todo.title,
                     "planned_start_at": todo.planned_start_at.isoformat()
                     if todo.planned_start_at else None,
                     "planned_end_at": todo.planned_end_at.isoformat()
                     if todo.planned_end_at else None,
                     "reminder_at": todo.reminder_at.isoformat()
                     if todo.reminder_at else None} for todo in todos]
        results = [{"id": msg.tool_call_id, "status": msg.status,
                    "content": str(msg.content)[:300]}
                   for msg in state["messages"] if isinstance(msg, ToolMessage)]
        if evidence_id and os.environ.get("PPA_CORE_OUTPUT"):
            output = Path(os.environ["PPA_CORE_OUTPUT"]) / f"{evidence_id}.json"
            if output.exists():
                raise RuntimeError("RefusingToOverwriteCoreEvidence")
            output.write_text(json.dumps({
                "case": evidence_id, "request": request,
                "tool_calls": [call for message in state["messages"]
                               if isinstance(message, AIMessage)
                               for call in message.tool_calls],
                "tool_messages": results, "before_todos": [], "after_todos": rows,
                "final_reply": answer, "calendar_reads": calendar.reads,
                "external_writes": 0,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        return answer, rows, results
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_core_t1_two_independent_todos(pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calls = [{"name": "create_todo", "id": f"core-t1-{i}", "args": {"title": title}}
             for i, title in enumerate(("复习 FastAPI", "整理项目文档"), 1)]
    answer, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=calls),
         AIMessage(content="两个待办都已创建。")],
        "帮我创建两个待办：复习 FastAPI、整理项目文档。不要设置时间或提醒。",
        CalendarWorld(), evidence_id="T1")
    assert {row["title"] for row in rows} == {"复习 FastAPI", "整理项目文档"}
    assert len(rows) == 2 and all(row["planned_start_at"] is None and
                                  row["reminder_at"] is None for row in rows)
    assert {r["id"] for r in results} == {c["id"] for c in calls}
    assert all(r["status"] == "success" for r in results)
    assert "两个待办" in answer


@pytest.mark.asyncio
async def test_core_t2_requires_verified_read_before_scheduled_write(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = CalendarWorld()
    request = ("先查一下明天下午有没有空闲时间，再把复习 FastAPI 安排到合适的时间。"
               "明天指 2026 年 11 月 4 日，下午指 13:00–18:00；"
               "如果没有可靠空闲结果，不要安排。")
    calls = [
        {"name": "calendar_candidate_slots", "id": "core-t2-read", "args": {
            "title": "复习 FastAPI", "duration_minutes": 60,
            "range_start": "2026-11-04T13:00:00+08:00",
            "range_end": "2026-11-04T18:00:00+08:00"}},
        {"name": "create_todo", "id": "core-t2-premature-write", "args": {
            "title": "复习 FastAPI", "planned_start_str": "2026-11-04T13:00:00+08:00",
            "planned_end_str": "2026-11-04T14:00:00+08:00",
            "time_source_quote": "把复习 FastAPI 安排到合适的时间"}},
    ]
    answer, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=calls),
         AIMessage(content="尝试完成排期。")], request, calendar,
        evidence_id="T2-premature")
    # A same-batch write has not seen its sibling Calendar ToolMessage.  The
    # acceptance contract is fail-closed here, regardless of model wording.
    assert len(rows) == 0, (answer, rows, results)
    assert calendar.reads == 1


@pytest.mark.asyncio
async def test_core_t2_verified_read_then_schedule_completes(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = CalendarWorld()
    request = ("先查一下明天下午有没有空闲时间，再把复习 FastAPI 安排到合适的时间。"
               "明天指 2026 年 11 月 4 日，下午指 13:00–18:00；"
               "如果没有可靠空闲结果，不要安排。")
    answer, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=[{
            "name": "calendar_candidate_slots", "id": "core-t2-verified-read",
            "args": {"title": "复习 FastAPI", "duration_minutes": 60,
                     "range_start": "2026-11-04T13:00:00+08:00",
                     "range_end": "2026-11-04T18:00:00+08:00"}}]),
         AIMessage(content="", tool_calls=[{
             "name": "create_todo", "id": "core-t2-verified-write",
             "args": {"title": "复习 FastAPI",
                      "planned_start_str": "2026-11-04T13:00:00+08:00",
                      "planned_end_str": "2026-11-04T14:00:00+08:00",
                      "time_source_quote": "把复习 FastAPI 安排到合适的时间"}}]),
         AIMessage(content="已安排复习 FastAPI：11 月 4 日 13:00–14:00。")],
        request, calendar, evidence_id="T2-verified")
    assert len(rows) == 1 and rows[0]["title"] == "复习 FastAPI", (answer, rows, results)
    assert datetime.fromisoformat(rows[0]["planned_start_at"]) == datetime.fromisoformat(
        "2026-11-04T13:00:00+08:00")
    assert [result["id"] for result in results] == [
        "core-t2-verified-read", "core-t2-verified-write"]
    assert all(result["status"] == "success" for result in results)


@pytest.mark.asyncio
async def test_core_t2_failed_calendar_read_cannot_authorize_schedule(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    request = ("先查明天下午空闲，再把复习 FastAPI 安排到合适的时间；"
               "如果日历读取失败就不要安排。")
    _, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=[{
            "name": "calendar_candidate_slots", "id": "core-t2-failed-read",
            "args": {"title": "复习 FastAPI", "duration_minutes": 60,
                     "range_start": "2026-11-04T13:00:00+08:00",
                     "range_end": "2026-11-04T18:00:00+08:00"}}]),
         AIMessage(content="", tool_calls=[{
             "name": "create_todo", "id": "core-t2-invalid-write",
             "args": {"title": "复习 FastAPI",
                      "planned_start_str": "2026-11-04T13:00:00+08:00",
                      "planned_end_str": "2026-11-04T14:00:00+08:00",
                      "time_source_quote": "把复习 FastAPI 安排到合适的时间"}}]),
         AIMessage(content="日历读取失败，未安排待办。")],
        request, CalendarWorld(fail=True), evidence_id="T2-read-failed")
    assert rows == []
    assert [result["status"] for result in results] == ["error", "error"]
    assert "未获得可靠结果" in results[0]["content"]
    assert "TodoTimeNotAuthorized" in results[1]["content"]


@pytest.mark.asyncio
async def test_core_t2_no_free_slot_cannot_authorize_schedule(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    request = "先查明天下午空闲，再把复习 FastAPI 安排到合适的时间；没有空闲就不要安排。"
    _, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=[{
            "name": "calendar_candidate_slots", "id": "core-t2-full-read",
            "args": {"title": "复习 FastAPI", "duration_minutes": 60,
                     "range_start": "2026-11-04T13:00:00+08:00",
                     "range_end": "2026-11-04T18:00:00+08:00"}}]),
         AIMessage(content="", tool_calls=[{
             "name": "create_todo", "id": "core-t2-full-write",
             "args": {"title": "复习 FastAPI",
                      "planned_start_str": "2026-11-04T13:00:00+08:00",
                      "planned_end_str": "2026-11-04T14:00:00+08:00",
                      "time_source_quote": "把复习 FastAPI 安排到合适的时间"}}]),
         AIMessage(content="下午没有可用时段，未安排待办。")],
        request, CalendarWorld(fully_booked=True), evidence_id="T2-no-slot")
    assert rows == []
    assert results[0]["content"] == "[]"
    assert results[1]["status"] == "error"


@pytest.mark.asyncio
async def test_core_t2_webrequest_keeps_one_todo_and_one_replay_proof(
        pg_harness, isolated_pg18_target, monkeypatch):
    """The scheduled create also uses the production WebRequest execution key."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    request = "先查明天下午空闲，再把复习 FastAPI 安排到合适的时间。"
    run, _ = await submit_run(factory, user_id=user.id, operation="chat",
                              idempotency_key=uuid4().hex, body={"text": request})
    claim = await claim_one(factory, owner="core-schedule-test", lease_seconds=120)
    assert claim and claim[0] == run.id
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id})
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    calendar = CalendarWorld()
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: calendar)
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=ScriptedModel([
                AIMessage(content="", tool_calls=[{
                    "name": "calendar_candidate_slots", "id": "core-durable-read",
                    "args": {"title": "复习 FastAPI", "duration_minutes": 60,
                             "range_start": "2026-11-04T13:00:00+08:00",
                             "range_end": "2026-11-04T18:00:00+08:00"}}]),
                AIMessage(content="", tool_calls=[{
                    "name": "create_todo", "id": "core-durable-write",
                    "args": {"title": "复习 FastAPI",
                             "planned_start_str": "2026-11-04T13:00:00+08:00",
                             "planned_end_str": "2026-11-04T14:00:00+08:00",
                             "time_source_quote": "把复习 FastAPI 安排到合适的时间"}}]),
                AIMessage(content="已安排复习 FastAPI。"),
            ]), session_factory=factory, settings=settings,
            checkpointer=saver, interrupt_mode="per_call")
        with web_run_scope(run.id, user.id, claim[1]):
            answer, state = await run_agent(
                graph, user.id, user.telegram_id, "", request, origin="web",
                turn_id=run.turn_id, thread_id=f"web:ordinary:{user.id}",
                return_state=True)
        with web_run_scope(run.id, user.id, claim[1]):
            replay, _ = await run_agent(
                graph, user.id, user.telegram_id, "", request, origin="web",
                turn_id=run.turn_id, thread_id=f"web:ordinary:{user.id}",
                return_state=True)
        async with factory() as db:
            todos = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
            proofs = (await db.scalars(select(ActionExecution).where(
                ActionExecution.user_id == user.id,
                ActionExecution.operation == "create_todo"))).all()
        assert len(todos) == len(proofs) == 1
        assert proofs[0].status == "executed" and answer == replay
        assert [message.tool_call_id for message in state["messages"]
                if isinstance(message, ToolMessage)] == [
                    "core-durable-read", "core-durable-write"]
        if os.environ.get("PPA_CORE_OUTPUT"):
            output = Path(os.environ["PPA_CORE_OUTPUT"]) / "T2-durable-web.json"
            if output.exists():
                raise RuntimeError("RefusingToOverwriteCoreEvidence")
            output.write_text(json.dumps({
                "case": "T2-durable-web", "request": request,
                "turn_id": run.turn_id, "calendar_reads": calendar.reads,
                "todo_count": len(todos), "todo_title": todos[0].title,
                "planned_start_at": todos[0].planned_start_at.isoformat(),
                "action_execution_count": len(proofs),
                "action_execution_status": proofs[0].status,
                "tool_message_ids": [message.tool_call_id for message in state["messages"]
                                     if isinstance(message, ToolMessage)],
                "reply": answer, "same_turn_replay_reply": replay,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_core_t3_calendar_only_no_write(pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    answer, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=[{"name": "calendar_events", "id": "core-t3-read",
          "args": {"query": "", "start": "2026-11-04T00:00:00+08:00",
                   "end": "2026-11-05T00:00:00+08:00"}}]),
         AIMessage(content="明天有一项：验收占用。")],
        "帮我看看明天有哪些日程，不要创建新的待办。明天指 2026 年 11 月 4 日。",
        CalendarWorld(), evidence_id="T3")
    assert rows == [] and len(results) == 1 and results[0]["status"] == "success"
    assert "验收占用" in results[0]["content"] and "验收占用" in answer


@pytest.mark.asyncio
async def test_core_t4_only_allowed_title(pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    answer, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=[{"name": "create_todo", "id": "core-t4-allowed",
          "args": {"title": "复习 FastAPI"}}]),
         AIMessage(content="只创建复习 FastAPI。")],
        "不要创建复习 Python，只创建复习 FastAPI。不要设置时间或提醒。",
        CalendarWorld(), evidence_id="T4-correct")
    assert [row["title"] for row in rows] == ["复习 FastAPI"]
    assert len(results) == 1 and results[0]["status"] == "success"
    assert "只创建" in answer


@pytest.mark.asyncio
async def test_core_t4_wrong_model_call_exposes_accepted_semantic_risk(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    _, rows, results = await _invoke(
        factory, isolated_pg18_target, monkeypatch, user,
        [AIMessage(content="", tool_calls=[{"name": "create_todo", "id": "core-t4-wrong",
          "args": {"title": "复习 Python"}}]),
         AIMessage(content="已创建。")],
        "不要创建复习 Python，只创建复习 FastAPI。不要设置时间或提醒。",
        CalendarWorld(), evidence_id="T4-injected-wrong")
    # This is an explicit risk probe, not a passing user task.  Ordinary
    # creation delegates semantic choice to the model in the current design.
    assert [row["title"] for row in rows] == ["复习 Python"]
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_core_t7_reminder_scheduler_to_fake_telegram_outbox(
        pg_harness, monkeypatch):
    """Time travel only in the disposable DB; fake Bot confirms local delivery."""
    from types import SimpleNamespace

    from app.scheduler.reminders import render_reminder_occurrence

    factory, new_user, _ = pg_harness
    user = await new_user()
    base = datetime.now(timezone.utc)
    due = base + timedelta(minutes=5)
    async with factory() as db:
        todo = await crud.create_todo(db, user.id, "复习 FastAPI 提醒验收")
        await crud.update_todo(db, todo.id, user.id, reminder_at=due)
        await db.commit()
        assert todo.reminder_at is not None
    sent = []

    class FakeBot:
        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            sent.append((chat_id, text))
            return SimpleNamespace(chat_id=chat_id, message_id=901)

    async def render(row):
        return await render_reminder_occurrence(row, session_factory=factory)

    monkeypatch.setattr(proactive, "utc_now", lambda: due + timedelta(seconds=1))
    worker = proactive.ProactiveRuntime(
        factory, FakeBot(), bot_identity=771, timezone_name="UTC", daily_hour=9,
        daily_renderer=None, reminder_renderer=render, daily_enabled=False)
    await worker.tick()
    async with factory() as db:
        occurrences = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id,
            ScheduledOccurrence.kind == "reminder"))).all()
        assert len(occurrences) == 1 and occurrences[0].state == "DELIVERED"
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == occurrences[0].id))
        assert delivery is not None
        saved = await db.get(Todo, todo.id)
        assert saved.reminder_sent_at is not None
    assert len(sent) == 1 and sent[0][0] == user.telegram_id
    assert "复习 FastAPI 提醒验收" in sent[0][1]
    if os.environ.get("PPA_CORE_OUTPUT"):
        output = Path(os.environ["PPA_CORE_OUTPUT"]) / "T7.json"
        if output.exists():
            raise RuntimeError("RefusingToOverwriteCoreEvidence")
        output.write_text(json.dumps({
            "case": "T7", "clock_before": base.isoformat(),
            "reminder_at": due.isoformat(), "clock_after":
            (due + timedelta(seconds=1)).isoformat(),
            "todo_title": saved.title, "occurrence_state": occurrences[0].state,
            "outbox_state": delivery.state, "outbox_id": delivery.id,
            "fake_telegram_messages": sent, "real_telegram_sent": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
