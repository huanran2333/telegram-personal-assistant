"""OA11: concurrent read failure must not erase a committed Todo result."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select

from app.agent import official_read_tools, unified_tools
from app.agent.calendar import CalendarError
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import run_agent
from app.agent.tools import make_todo_tools
from app.config import get_settings
from app.db.models import Todo, TelegramInboundReceipt, TelegramThreadLease
from app.db.models import ActionExecution
from app.telegram import durable
from app.telegram.processing_context import receipt_scope, tool_call_scope
from app.web.processing_context import web_run_scope
from app.web.runs import submit_run, claim_one
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel

pytestmark = pytest.mark.integration


def _synthetic_update(update_id: int, actor: int, text: str) -> dict:
    return {"update_id": update_id, "message": {
        "message_id": 1, "date": 0,
        "from": {"id": actor, "is_bot": False, "first_name": "Synthetic"},
        "chat": {"id": actor, "type": "private"}, "text": text}}


class FailingCalendar:
    async def search_events(self, query, start, end):
        raise CalendarError("SyntheticCalendarReadFailure: private adapter details")


class TransientCalendar:
    def __init__(self):
        self.calls = 0

    async def search_events(self, query, start, end):
        self.calls += 1
        if self.calls == 1:
            raise CalendarError("TemporaryCalendarRead") from httpx.ReadTimeout("synthetic")
        return [{"id": "synthetic-event", "summary": "verified event",
                 "start": {"dateTime": start}, "end": {"dateTime": end},
                 "status": "confirmed"}]


@pytest.mark.asyncio
async def test_transient_calendar_read_recovers_inside_one_tool_call(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    calendar = TransientCalendar()
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", lambda: calendar)
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id})
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "calendar_events", "id": "read-once",
                  "args": {"query": "verified", "start": "2026-11-10T00:00:00+08:00",
                           "end": "2026-11-10T23:59:59+08:00"}}]),
        AIMessage(content="已核实 verified event。"),
    ])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, session_factory=factory, settings=settings,
            checkpointer=saver, interrupt_mode="per_call")
        reply, state = await run_agent(
            graph, user.id, user.telegram_id, "", "只读查询 verified event。",
            origin="web", turn_id=f"transient-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        assert "verified event" in reply
        reads = [item for item in state["messages"] if isinstance(item, ToolMessage)
                 and item.tool_call_id == "read-once"]
        assert len(reads) == 1 and reads[0].status == "success"
        assert calendar.calls == 2
        async with factory() as db:
            assert not (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [0, 1, 2], ids=[
    "read_only", "read_plus_create", "read_plus_unauthorized_create"])
async def test_calendar_failure_is_paired_and_independent_todo_commits_once(
        pg_harness, isolated_pg18_target, monkeypatch, batch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"OA11-regression-{uuid4().hex[:10]}"
    calls = [{"name": "calendar_events", "id": "oa11-calendar",
              "args": {"query": "2026-11-10", "start": "2026-11-10T00:00:00+08:00",
                       "end": "2026-11-10T23:59:59+08:00"}}]
    if batch:
        calls.append({"name": "create_todo", "id": "oa11-todo",
                      "args": {"title": title}})
    model = ScriptedModel([
        AIMessage(content="", tool_calls=calls),
        AIMessage(content=(f"Created {title}; calendar read failed." if batch == 1 else
                           "Calendar read failed; no reliable result.")),
    ])
    settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": user.telegram_id,
    })
    monkeypatch.setattr(official_read_tools, "get_calendar_adapter", FailingCalendar)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings)
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=settings,
            checkpointer=saver, interrupt_mode="per_call")
        reply, state = await run_agent(
            graph, user.id, user.telegram_id, "", 
            f"查日历；{'创建不设时间的待办 ' + title if batch == 1 else '只读，不要创建待办'}；如实报告。",
            origin="web", turn_id=f"oa11-{uuid4().hex}",
            thread_id=f"web:ordinary:{user.id}", return_state=True)
        messages = [item for item in state["messages"] if isinstance(item, ToolMessage)]
        assert {item.tool_call_id for item in messages} == {call["id"] for call in calls}
        read = next(item for item in messages if item.tool_call_id == "oa11-calendar")
        assert read.status == "error"
        assert "未获得可靠结果" in read.content
        assert "calendar_events" in read.content and "read_failed" in read.content
        assert "private adapter details" not in read.content
        assert "calendar read failed" in reply.lower()
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            assert len(rows) == int(batch != 0)
            if batch != 0:
                assert rows[0].due_date is None
                assert rows[0].planned_start_at is None
                assert rows[0].reminder_at is None
                assert next(item for item in messages if item.tool_call_id == "oa11-todo").status != "error"
            # batch 2 is deliberately a read-only user request with an erroneous
            # model create call. Without semantic authorization it commits.
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_same_web_execution_recovers_committed_todo_without_duplicate(
        pg_harness, isolated_pg18_target, monkeypatch):
    """Crash after Todo/proof commit but before ToolNode returns its ToolMessage."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"OA11-replay-{uuid4().hex[:10]}"
    run, _ = await submit_run(factory, user_id=user.id, operation="chat",
                              idempotency_key=uuid4().hex, body={"text": f"创建待办 {title}"})
    claim = await claim_one(factory, owner="oa11-test-worker", lease_seconds=120)
    assert claim and claim[0] == run.id
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "create_todo", "id": "oa11-replay-call",
                                           "args": {"title": title}}]),
        AIMessage(content=f"Created {title}"),
    ])
    original_commit = durable.commit_todo_tool_operation
    injected = False

    async def commit_then_lose_feedback(*args, **kwargs):
        nonlocal injected
        result = await original_commit(*args, **kwargs)
        if not injected:
            injected = True
            raise RuntimeError("SyntheticAfterCommitBeforeToolMessage")
        return result

    monkeypatch.setattr(durable, "commit_todo_tool_operation", commit_then_lose_feedback)
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=get_settings(),
            checkpointer=saver, interrupt_mode="per_call")
        thread = f"web:ordinary:{user.id}"
        request = f"创建一个不设时间的待办 {title}"
        with web_run_scope(run.id, user.id, claim[1]):
            with pytest.raises(RuntimeError, match="SyntheticAfterCommitBeforeToolMessage"):
                await run_agent(graph, user.id, user.telegram_id, "", request,
                                origin="web", turn_id=run.turn_id, thread_id=thread)
        config = {"configurable": {"thread_id": f"official-v1:{thread}"}}
        interrupted = await graph.aget_state(config)
        assert interrupted.next and not any(isinstance(item, ToolMessage)
                                            for item in interrupted.values["messages"])
        async with factory() as db:
            assert len((await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()) == 1
        with web_run_scope("different-web-request", user.id, claim[1]):
            with pytest.raises(RuntimeError, match="IncompleteAgentTurnRequiresDurableIngress"):
                await run_agent(graph, user.id, user.telegram_id, "", request,
                                origin="web", turn_id=run.turn_id, thread_id=thread)
        with web_run_scope(run.id, user.id, claim[1]):
            reply, state = await run_agent(
                graph, user.id, user.telegram_id, "", request,
                origin="web", turn_id=run.turn_id, thread_id=thread, return_state=True)
        assert title in reply
        assert any(isinstance(item, ToolMessage) and
                   item.tool_call_id == "oa11-replay-call" and item.status != "error"
                   for item in state["messages"])
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            proofs = (await db.scalars(select(ActionExecution).where(
                ActionExecution.user_id == user.id,
                ActionExecution.domain == "todo"))).all()
            assert len(rows) == len(proofs) == 1
        async with factory() as db:
            async def must_not_execute():
                raise AssertionError("Different parameters must not execute")

            with web_run_scope(run.id, user.id, claim[1]), tool_call_scope("oa11-replay-call"):
                with pytest.raises(durable.DurableSafetyError,
                                   match="WebTodoReplayProofMismatch"):
                    await original_commit(
                        db, name="create_todo",
                        args={"user_id": user.id, "title": title + "-different"},
                        perform=must_not_execute)
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("utterance_template", "allow_a", "allow_b"), [
    ("不创建待办", False, False),
    ("只读，不创建待办", False, False),
    ("不要创建待办", False, False),
    ("现在不创建待办 A", False, False),
    ("请先不要创建待办 A", False, False),
    ("我不想创建待办 A", False, False),
    ("暂时不要记录这件事", False, False),
    ("只是讨论计划，不要求创建待办", False, False),
    ("不创建 A，只创建 B", False, True),
    ("创建 A，但不要创建 B", True, False),
    ("创建两个没有时间的待办：A 和 B", True, True),
    ("不创建日历事件，但创建 Todo A 和 B", True, True),
])
async def test_wrong_model_create_call_commits_without_semantic_authorization(
        pg_harness, isolated_pg18_target, utterance_template, allow_a, allow_b):
    factory, new_user, _ = pg_harness
    user = await new_user()
    bystander = await new_user()
    marker = uuid4().hex[:10]
    a, b = f"A-{marker}", f"B-{marker}"
    utterance = utterance_template.replace("A", a).replace("B", b)
    if "B" not in utterance_template:
        calls = [{"name": "create_todo", "id": "negated-a",
                                       "args": {"title": a}}]
    else:
        calls = [{"name": "create_todo", "id": "negated-a",
                  "args": {"title": a}},
                 {"name": "create_todo", "id": "allowed-b",
                  "args": {"title": b}}]
    model = ScriptedModel([AIMessage(content="", tool_calls=calls),
                           AIMessage(content="Done")])
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=model, todo_tools=make_todo_tools(factory),
            session_factory=factory, settings=get_settings(),
            checkpointer=saver, interrupt_mode="per_call")
        run, _ = await submit_run(factory, user_id=user.id, operation="chat",
                                  idempotency_key=uuid4().hex, body={"text": utterance})
        claim = await claim_one(factory, owner="todo-intent-test", lease_seconds=120)
        assert claim and claim[0] == run.id
        with web_run_scope(run.id, user.id, claim[1]):
            _, state = await run_agent(
                graph, user.id, user.telegram_id, "", utterance,
                origin="web", turn_id=run.turn_id,
                thread_id=f"web:ordinary:{user.id}", return_state=True)
        results = {item.tool_call_id: item for item in state["messages"]
                   if isinstance(item, ToolMessage)}
        assert set(results) == {call["id"] for call in calls}
        assert results["negated-a"].status == "success"
        if len(calls) == 2:
            assert results["allowed-b"].status == "success"
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(Todo.user_id == user.id,
                                                       Todo.title.in_([a, b])))).all()
            assert {row.title for row in rows} == (
                {a} | ({b} if len(calls) == 2 else set()))
            # allow_a/allow_b describe user intent, not a server guarantee.
            # Negative cases explicitly preserve evidence of unauthorized writes.
            if not allow_a:
                assert a in {row.title for row in rows}
            if len(calls) == 2 and not allow_b:
                assert b in {row.title for row in rows}
            assert all(row.due_date is None and row.planned_start_at is None
                       and row.reminder_at is None for row in rows)
            proofs = (await db.scalars(select(ActionExecution).where(
                ActionExecution.user_id == user.id,
                ActionExecution.domain == "todo"))).all()
            assert len(proofs) == len(rows)
            assert not (await db.scalars(select(Todo).where(
                Todo.user_id == bystander.id))).all()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_same_telegram_receipt_recovers_committed_todo_in_official_graph(
        pg_harness, isolated_pg18_target, monkeypatch):
    """Receipt retry re-enters the saved ToolNode, not a new model turn."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    other_user = await new_user()
    title = f"OA11-telegram-replay-{uuid4().hex[:10]}"
    request = f"创建一个不设时间的待办 {title}"
    thread = f"user_{user.telegram_id}"
    receipt = await durable.admit_receipt(
        factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
        data=_synthetic_update(int(uuid4().hex[:12], 16), 70001, request),
        graph_kind="telegram_session", thread_id=thread)
    other_receipt = None
    first = await durable.claim_receipt(factory, receipt.id, "oa11-worker-before")
    assert first
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "create_todo", "id": "oa11-telegram-call",
                                           "args": {"title": title}}]),
        AIMessage(content=f"已创建待办 {title}")])
    original_commit = durable.commit_todo_tool_operation
    injected = False

    async def commit_then_crash(*args, **kwargs):
        nonlocal injected
        result = await original_commit(*args, **kwargs)
        if not injected:
            injected = True
            raise RuntimeError("SyntheticTelegramAfterCommitBeforeToolMessage")
        return result

    monkeypatch.setattr(durable, "commit_todo_tool_operation", commit_then_crash)
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        def make_graph():
            return build_official_agent(
                model=model, todo_tools=make_todo_tools(factory),
                session_factory=factory, settings=get_settings(),
                checkpointer=saver, interrupt_mode="per_call")[0]
        graph = make_graph()
        with receipt_scope(receipt.id, first):
            with pytest.raises(RuntimeError, match="SyntheticTelegramAfterCommitBeforeToolMessage"):
                await run_agent(graph, user.id, user.telegram_id, "", request,
                                origin="telegram", thread_id=thread)
        snapshot = await graph.aget_state({"configurable": {"thread_id": f"official-v1:{thread}"}})
        assert snapshot.next and not any(isinstance(item, ToolMessage)
                                         for item in snapshot.values["messages"])
        # A new ingress identity may not take over this incomplete superstep.
        with receipt_scope("different-receipt", "different-fence"):
            with pytest.raises(RuntimeError, match="IncompleteAgentTurnRequiresOriginalIngress"):
                await run_agent(graph, user.id, user.telegram_id, "", request,
                                origin="telegram", thread_id=thread)
        async with factory() as db:
            assert len((await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()) == 1
            proof = await db.get(ActionExecution, durable.telegram_todo_action_id(
                receipt.id, "create_todo", "oa11-telegram-call"))
            assert proof is not None and proof.status == "executed"
            row = await db.get(TelegramInboundReceipt, receipt.id)
            lease = await db.get(TelegramThreadLease, {
                "graph_kind": "telegram_session", "thread_id": thread})
            row.lease_until = lease.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        second = await durable.claim_receipt(factory, receipt.id, "oa11-worker-after")
        assert second and second != first
        # Rebuilding the graph models a worker process reopening the Saver.
        with receipt_scope(receipt.id, second):
            reply, state = await run_agent(make_graph(), user.id, user.telegram_id, "", request,
                                           origin="telegram", thread_id=thread, return_state=True)
        assert title in reply and "已创建" in reply
        paired = [item for item in state["messages"] if isinstance(item, ToolMessage)
                  and item.tool_call_id == "oa11-telegram-call"]
        assert len(paired) == 1 and paired[0].status == "success" and title in paired[0].content
        async with factory() as db:
            rows = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            proofs = (await db.scalars(select(ActionExecution).where(
                ActionExecution.action_id.startswith(f"tgtodo:{receipt.id}:")))).all()
            assert len(rows) == len(proofs) == 1
            assert paired[0].content == proofs[0].result["text"]
            with receipt_scope(receipt.id, second), tool_call_scope("oa11-telegram-call"):
                async def must_not_execute():
                    raise AssertionError("Mismatched replay must not execute")
                with pytest.raises(durable.DurableSafetyError,
                                   match="TodoReplayProofMismatch"):
                    await original_commit(db, name="create_todo",
                                          args={"user_id": user.id, "title": title + "-other"},
                                          perform=must_not_execute)
        async with factory() as db:
            with receipt_scope(receipt.id, second), tool_call_scope("oa11-telegram-call"):
                with pytest.raises(durable.DurableSafetyError,
                                   match="TodoReplayProofMismatch"):
                    await original_commit(db, name="create_todo",
                                          args={"user_id": other_user.id, "title": title},
                                          perform=must_not_execute)
            assert not (await db.scalars(select(Todo).where(
                Todo.user_id == other_user.id))).all()
        # A distinct Receipt does not inherit the first Receipt's proof.
        other_receipt = await durable.admit_receipt(
            factory, bot_identity=800000 + int(uuid4().hex[:6], 16),
            data=_synthetic_update(int(uuid4().hex[:12], 16), 70002, request + " new"),
            graph_kind="telegram_session", thread_id=f"{thread}-other")
        other_fence = await durable.claim_receipt(factory, other_receipt.id, "other-worker")
        assert other_fence
        create_tool = next(tool for tool in make_todo_tools(factory)
                           if tool.name == "create_todo")
        with receipt_scope(other_receipt.id, other_fence), tool_call_scope("oa11-telegram-call"):
            other_result = await create_tool.ainvoke({"user_id": user.id,
                                                     "title": title + "-new-request"})
        assert title + "-new-request" in other_result
        async with factory() as db:
            assert await db.get(ActionExecution, durable.telegram_todo_action_id(
                other_receipt.id, "create_todo", "oa11-telegram-call")) is not None
    finally:
        await conn.close()
        async with factory() as db:
            for receipt_id in [receipt.id, *([other_receipt.id] if other_receipt else [])]:
                await db.execute(delete(ActionExecution).where(
                    ActionExecution.action_id.startswith(f"tgtodo:{receipt_id}:")))
            await db.execute(delete(TelegramThreadLease).where(
                TelegramThreadLease.receipt_id.in_(
                    [receipt.id, *([other_receipt.id] if other_receipt else [])])))
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id.in_(
                    [receipt.id, *([other_receipt.id] if other_receipt else [])])))
            await db.commit()
