"""Two observed real-Qwen fault cases; the test pass means capture, not task success."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from sqlalchemy import select

from app.agent import agent_approval, nodes, official_read_tools, unified_tools, workflow_runtime
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError
from app.agent.nodes import _build_chat_llm
from app.agent.official_approval_view import official_decision_for_action, pending_official_approvals
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.agent.provider_clients import provider_client_scope
from app.config import get_settings
from app.db.models import Todo

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm_integration,
    pytest.mark.skipif(
        not os.environ.get("PPA_BC_OUTPUT"),
        reason="Opt-in real-model experiment; set PPA_BC_OUTPUT to run",
    ),
]


class ForbiddenGmail:
    def __init__(self):
        self.reads = 0

    async def account_address(self):
        return "synthetic@example.test"

    async def thread(self, _thread_id):
        self.reads += 1
        request = httpx.Request("GET", "https://synthetic.example.test/mail")
        response = httpx.Response(403, request=request)
        try:
            raise httpx.HTTPStatusError("synthetic forbidden", request=request, response=response)
        except httpx.HTTPStatusError as cause:
            raise GmailError("synthetic forbidden") from cause

    async def search(self, _query, limit=5):
        return await self.thread("synthetic-thread")

    async def unread(self, _query, limit=5):
        return await self.thread("synthetic-thread")


class CalendarFailsAfterApproval:
    def __init__(self):
        self.reads = 0
        self.fail_recheck = False
        self.writes = 0

    async def availability(self, *_args):
        self.reads += 1
        if self.fail_recheck:
            raise CalendarError("synthetic 503") from CalendarProviderError(
                "provider_unavailable", 503)
        return {"calendars": {"primary": {"busy": []}}}

    async def search_events(self, *_args):
        return []

    async def resolve_create_scope(self, _calendar_id):
        return "normal", "synthetic-calendar"

    async def create_event_prepared(self, *_args):
        self.writes += 1
        raise AssertionError("Remote Calendar write must not start")


class NoWriteExecutor:
    def __init__(self):
        self.writes = 0

    async def calendar_create_prepared(self, _action_id):
        return False

    async def execute(self, _proposal):
        self.writes += 1
        raise AssertionError("Calendar executor must not write after failed recheck")


def safe_messages(messages):
    result = []
    for msg in messages:
        if not isinstance(msg, (AIMessage, ToolMessage)):
            continue
        row = {"type": type(msg).__name__, "content": str(msg.content)}
        if isinstance(msg, AIMessage):
            row["tool_calls"] = msg.tool_calls
            row["usage"] = msg.usage_metadata
        else:
            row["tool_call_id"] = msg.tool_call_id
            row["status"] = msg.status
        result.append(row)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ["B", "C"])
async def test_real_qwen_g14_case(case_id, pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    base = get_settings()
    slot = os.environ.get("PPA_BC_MODEL_SLOT", "primary")
    if slot == "backup1":
        original = base.model_copy(update={
            "llm_model": base.llm_fallback_1_model,
            "openai_base_url": base.llm_fallback_1_base_url,
            "openai_api_key": base.llm_fallback_1_api_key,
        })
    else:
        assert slot == "primary"
        original = base
    assert original.llm_provider == "openai" and original.llm_model
    monkeypatch.setattr(nodes, "settings", original)
    settings = original.model_copy(update={
        "app_env": "test", "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id,
        "gmail_enabled": True, "gmail_owner_telegram_id": user.telegram_id,
        "memory_enabled": False, "tavily_api_key": "",
    })
    monkeypatch.setattr(official_read_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_approval, "get_settings", lambda: settings)
    gmail, calendar, executor = ForbiddenGmail(), CalendarFailsAfterApproval(), NoWriteExecutor()
    monkeypatch.setattr(workflow_runtime, "_action_executor", executor)
    title = "BC-邮件会议准备" if case_id == "B" else "BC-独立复习待办"
    request = (
        "先读取刚收到的邮件，只有实际读到邮件正文并确认会议准备事项后，才据此创建待办『BC-邮件会议准备』。"
        "如果 Gmail 拒绝访问，不能猜邮件内容、不能创建这条待办；请告诉我各项实际结果。"
        if case_id == "B" else
        "请做两件互不依赖的事：创建一个不设时间的普通待办『BC-独立复习待办』；"
        "另在 Google 日历 2026 年 12 月 10 日 15:00–16:00（北京时间）创建『BC-项目讨论』事件，"
        "该日历操作请先让我审批。两件事分别报告，不能把日历失败说成待办失败。"
    )
    record = {"case": case_id, "request": request, "model": original.llm_model,
              "model_parameters": {"temperature": 0, "max_tokens": 4096,
                                   "thinking": False, "request_timeout_seconds": 60},
              "status": "RUNNING"}
    output = Path(os.environ["PPA_BC_OUTPUT"]) / f"{case_id}.json"
    assert not output.exists()
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    start = time.perf_counter()
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        with (patch.object(workflow_runtime, "get_gmail_source", return_value=gmail),
              patch.object(official_read_tools, "get_gmail_source", return_value=gmail),
              patch.object(workflow_runtime, "get_calendar_adapter", return_value=calendar),
              patch.object(official_read_tools, "get_calendar_adapter", return_value=calendar)):
            async with provider_client_scope():
                model = _build_chat_llm(original.llm_provider, original.llm_model,
                                        disable_qwen_thinking=True, request_timeout=60)
                graph, _ = build_official_agent(
                    model=model, session_factory=factory, settings=settings,
                    checkpointer=saver, interrupt_mode="per_call")
                thread_id = f"web:ordinary:{user.id}"
                _, state = await asyncio.wait_for(run_agent(
                    graph, user.id, user.telegram_id, "", request,
                    origin="web", turn_id=f"bc-{uuid4().hex}",
                    thread_id=thread_id, return_state=True), timeout=180)
                config = {"configurable": {"thread_id": f"official-v1:{thread_id}"}}
                if case_id == "C":
                    snapshot = await graph.aget_state(config)
                    pending = pending_official_approvals(snapshot, user_id=user.id)
                    record["pending_before_decision"] = [
                        {"operation": p.proposal.operation, "action_id": p.proposal.action_id}
                        for p in pending]
                    calendar.fail_recheck = True
                    for item in pending:
                        decision = official_decision_for_action(
                            snapshot, user_id=user.id, action_id=item.proposal.action_id,
                            decision="approve", checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
                        state = await asyncio.wait_for(graph.ainvoke(
                            Command(resume=decision), config=config,
                            context={"user_id": user.id, "telegram_id": user.telegram_id,
                                     "chat_id": "", "origin": "web"}), timeout=180)
                        snapshot = await graph.aget_state(config)
                async with factory() as db:
                    todos = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
                record.update(status="CAPTURED", messages=safe_messages(state.get("messages", [])),
                              todos=[{"title": t.title, "planned_start_at": str(t.planned_start_at),
                                      "reminder_at": str(t.reminder_at)} for t in todos],
                              gmail_reads=gmail.reads, calendar_reads=calendar.reads,
                              calendar_write_calls=calendar.writes + executor.writes,
                              pending_after=len(pending_official_approvals(
                                  await graph.aget_state(config), user_id=user.id)))
    except Exception as exc:
        record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc)[:200])
    finally:
        record["elapsed_ms"] = round((time.perf_counter() - start) * 1000)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        await conn.close()
    print(f"{case_id} {record['status']} {record['elapsed_ms']}ms")
    assert record["status"] == "CAPTURED", record.get("error_type")
