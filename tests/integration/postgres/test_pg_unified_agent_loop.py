"""The new model/tool loop persists an owner-scoped Todo on isolated PG18."""
import asyncio
from uuid import uuid4
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select
from fastapi import Response

from app.db.models import Todo

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_telegram_guard_serializes_with_web_agent_approval(isolated_pg18_target):
    from app.telegram.durable import telegram_workflow_guard, thread_invocation_guard

    entered, release, web_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    identity = f"{uuid4().int % 10**10:010d}"

    async def telegram_turn():
        async with telegram_workflow_guard(
                isolated_pg18_target, "telegram_session", f"telegram_user_{identity}"):
            entered.set()
            await release.wait()

    async def web_decision():
        async with thread_invocation_guard(
                isolated_pg18_target, "workflow", f"user_{identity}"):
            web_entered.set()

    telegram = asyncio.create_task(telegram_turn())
    await asyncio.wait_for(entered.wait(), timeout=3)
    web = asyncio.create_task(web_decision())
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(web_entered.wait(), timeout=0.15)
    finally:
        release.set()
        await asyncio.wait_for(telegram, timeout=3)
        await asyncio.wait_for(web, timeout=3)
    assert web_entered.is_set()


@pytest.mark.asyncio
async def test_telegram_guard_serializes_with_versioned_official_web_approval(
        isolated_pg18_target):
    from app.telegram.durable import telegram_workflow_guard, thread_invocation_guard

    entered, release, web_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    identity = f"{uuid4().int % 10**10:010d}"

    async def telegram_turn():
        async with telegram_workflow_guard(
                isolated_pg18_target, "telegram_session", f"telegram_user_{identity}"):
            entered.set()
            await release.wait()

    async def web_decision():
        async with thread_invocation_guard(
                isolated_pg18_target, "workflow", f"official-v1:user_{identity}"):
            web_entered.set()

    telegram = asyncio.create_task(telegram_turn())
    await asyncio.wait_for(entered.wait(), timeout=3)
    web = asyncio.create_task(web_decision())
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(web_entered.wait(), timeout=0.15)
    finally:
        release.set()
        await asyncio.wait_for(telegram, timeout=3)
        await asyncio.wait_for(web, timeout=3)
    assert web_entered.is_set()


@pytest.mark.asyncio
@pytest.mark.skip(reason=(
    "Retired unified_loop graph; current create_agent Todo replay is covered "
    "by test_pg_official_real_todo.py"))
async def test_unified_loop_creates_one_todo_and_replay_does_not_duplicate(pg_harness, monkeypatch):
    from app.agent import graph, nodes, unified_loop

    class SyntheticModel:
        def __init__(self):
            self.calls = 0
            self.offered = []

        def bind_tools(self, tools):
            self.offered.append({item.name for item in tools})
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_todo", "args": {"title": title},
                    "id": "synthetic-create", "type": "tool_call"}])
            return AIMessage(content="任务已创建")

    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"synthetic unified {uuid4().hex[:12]}"
    model = SyntheticModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    ordinary = await graph.build_graph(factory, durable=True)
    thread_id, turn_id = f"web:ordinary:{user.id}", f"web-request:{uuid4()}"
    try:
        args = (ordinary, user.id, user.telegram_id, "", f"创建任务：{title}")
        first = await graph.run_agent(*args, thread_id=thread_id,
                                      turn_id=turn_id, origin="web")
        replay = await graph.run_agent(*args, thread_id=thread_id,
                                       turn_id=turn_id, origin="web")
        assert first == replay == "任务已创建"
        assert model.calls == 2
        assert "create_todo" in model.offered[0]
        assert "delete_todo" not in model.offered[0]
        async with factory() as db:
            rows = list((await db.scalars(select(Todo).where(Todo.title == title))).all())
        assert len(rows) == 1 and rows[0].user_id == user.id
    finally:
        await ordinary.checkpointer.adelete_thread(thread_id)
        await ordinary.checkpointer.conn.close()


@pytest.mark.asyncio
@pytest.mark.skip(reason=(
    "Retired unified_loop graph; current create_agent time authorization is "
    "covered by test_pg_official_oa11.py"))
async def test_compound_request_cannot_borrow_meeting_time_for_todo(pg_harness, monkeypatch):
    from app.agent import graph, nodes, unified_loop

    class SyntheticModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_todo", "args": {
                        "title": title, "planned_start_str": "2026-11-15 14:00",
                        "planned_end_str": "2026-11-15 15:00",
                        "time_source_quote": "2026年11月15日14点"},
                    "id": "contaminated-create", "type": "tool_call"}])
            if self.calls == 2:
                assert any(isinstance(item, ToolMessage) and
                           item.tool_call_id == "contaminated-create" and
                           item.status == "error" for item in _messages)
                return AIMessage(content="", tool_calls=[{
                    "name": "create_todo", "args": {"title": title},
                    "id": "untimed-create", "type": "tool_call"}])
            return AIMessage(content="已创建未排期的待办。会议时间没有写入待办。")

    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"整理会议资料 {uuid4().hex[:8]}"
    model = SyntheticModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    ordinary = await graph.build_graph(factory, durable=True)
    thread = f"web:ordinary:{user.id}:time-scope"
    try:
        reply, state = await graph.run_agent(
            ordinary, user.id, user.telegram_id, "",
            f"为2026年11月15日14点的 Project Sync 准备会议资料，另外创建待办：{title}",
            thread_id=thread, turn_id=f"web-request:{uuid4().hex}", origin="web",
            return_state=True)
        async with factory() as db:
            rows = list((await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all())
        assert len(rows) == 1
        assert rows[0].due_date is None
        assert rows[0].planned_start_at is None and rows[0].planned_end_at is None
        assert rows[0].reminder_at is None
        assert [item.tool_call_id for item in state["messages"]
                if isinstance(item, ToolMessage)] == ["contaminated-create", "untimed-create"]
        assert "未排期" in reply and model.calls == 3
    finally:
        await ordinary.checkpointer.adelete_thread(thread)
        await ordinary.checkpointer.conn.close()


@pytest.mark.asyncio
@pytest.mark.skip(reason=(
    "Retired unified_loop approval gate; current per-call HITL is covered "
    "by test_pg_official_approval_entry.py"))
async def test_unified_delete_proposal_interrupts_without_deleting(pg_harness, monkeypatch):
    from app.agent import graph, nodes, unified_loop
    from app.db import crud
    from langgraph.types import Command

    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"synthetic approval {uuid4().hex[:12]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id = todo.id
    class SyntheticModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "propose_todo_delete", "args": {"target": title},
                    "id": "synthetic-delete", "type": "tool_call"}])
            return AIMessage(content="已拒绝删除")

    model = SyntheticModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    ordinary = await graph.build_graph(factory, durable=True)
    thread = f"web:ordinary:{user.id}"
    turn = f"web-request:{uuid4()}"
    try:
        _, pending = await graph.run_agent(
            ordinary, user.id, user.telegram_id, "", f"删除任务：{title}",
            thread_id=thread, turn_id=turn, origin="web", return_state=True)
        assert pending["__interrupt__"][0].value["kind"] == "action_proposal"
        assert pending["proposal"]["args"]["approved_todo_version"] >= 1
        _, replay = await graph.run_agent(
            ordinary, user.id, user.telegram_id, "", f"删除任务：{title}",
            thread_id=thread, turn_id=turn, origin="web", return_state=True)
        assert replay["__interrupt__"][0].value["action_id"] == pending["proposal"]["action_id"]
        assert model.calls == 1
        snapshot = await ordinary.aget_state({"configurable": {"thread_id": thread}})
        assert "agent_approval_gate" in snapshot.next
        result = await ordinary.ainvoke(Command(resume={"decision": "reject"}),
                                        {"configurable": {"thread_id": thread}},
                                        context={"user_id": user.id,
                                                 "telegram_id": user.telegram_id,
                                                 "chat_id": "", "origin": "web"})
        assert result["proposal"]["status"] == "rejected"
        assert model.calls == 2
        async with factory() as db:
            assert await db.get(Todo, todo_id) is not None
    finally:
        await ordinary.checkpointer.adelete_thread(thread)
        await ordinary.checkpointer.conn.close()


@pytest.mark.asyncio
@pytest.mark.skip(reason=(
    "Retired unified_loop Web Reject; current create_agent Web Reject is "
    "covered by test_pg_official_approval_entry.py"))
async def test_web_reject_resumes_same_agent_saver_thread(pg_harness, monkeypatch):
    from app.agent import graph, nodes, unified_loop, workflow_runtime
    from app.db import crud
    from app.db.models import WebRequest, ActionExecution
    from app.web import api, runs

    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"synthetic cross-entry {uuid4().hex[:12]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id = todo.id

    class Model:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "propose_todo_delete", "args": {"target": title},
                    "id": "web-delete", "type": "tool_call"}])
            return AIMessage(content="拒绝后任务保留")

    model = Model()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: model)
    compiled = await graph.build_graph(factory, durable=True)
    thread = f"web:ordinary:{user.id}"
    try:
        parent, _ = await runs.submit_run(
            factory, user_id=user.id, operation="chat", idempotency_key=uuid4().hex,
            body={"text": f"删除任务：{title}"})
        _, pending = await graph.run_agent(
            compiled, user.id, user.telegram_id, "", f"删除任务：{title}",
            thread_id=thread, turn_id=parent.turn_id, origin="web", return_state=True)
        assert pending["__interrupt__"][0].value["kind"] == "action_proposal"
        async with factory() as db:
            row = await db.get(WebRequest, parent.id, with_for_update=True)
            row.state, row.thread_id, row.graph_kind = "WAITING_APPROVAL", thread, "ordinary"
            row.route_payload = {"agent_mode": "unified"}
            await db.commit()

        monkeypatch.setattr(api, "AsyncSessionLocal", factory)
        monkeypatch.setattr(api, "_authenticated", AsyncMock(return_value=(user, None)))
        monkeypatch.setattr(workflow_runtime, "get_productivity_graph", AsyncMock(
            return_value=SimpleNamespace(aget_state=AsyncMock(return_value=None))))
        request = SimpleNamespace(headers={"idempotency-key": uuid4().hex},
                                  app=SimpleNamespace(state=SimpleNamespace(
                                      web_worker=SimpleNamespace(wake=lambda: None))))
        listed = await api.approvals(request, Response())
        item = next(entry for entry in listed["items"] if entry["run_id"] == parent.id)
        admitted = await api._decision_admission(
            request, Response(), action_id=item["action_id"], run_id=parent.id,
            checkpoint_version=item["checkpoint_version"],
            operation="approval", payload={"decision": "reject"})
        async with factory() as db:
            decision = await db.get(WebRequest, admitted["run_id"])
        claimed = await runs.claim_one(factory, owner="synthetic-worker", lease_seconds=120)
        assert claimed and claimed[0] == decision.id
        await runs.WebRunWorker(factory, graph.settings.database_url).process(*claimed)
        snapshot = await compiled.aget_state({"configurable": {"thread_id": thread}})
        assert snapshot.values["proposal"]["status"] == "rejected"
        assert model.calls == 2
        async with factory() as db:
            assert await db.get(Todo, todo_id) is not None
            assert await db.get(ActionExecution, item["action_id"]) is None
            assert (await db.get(WebRequest, decision.id)).state == "COMPLETED"
            assert (await db.get(WebRequest, parent.id)).state == "COMPLETED"
    finally:
        await compiled.checkpointer.adelete_thread(thread)
        await compiled.checkpointer.conn.close()
