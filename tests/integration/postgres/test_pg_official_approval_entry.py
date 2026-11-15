"""Official per-call Mini App decision through real WebRequest and PG Saver.

No Telegram or Google network call; business execution is a synthetic fake.
The database fixture verifies the disposable PG18 identity before this test.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import psycopg
import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select

from app.agent import unified_tools, workflow_runtime
from app.agent.actions import ActionResult, ActionStatus
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.official_graph import build_official_agent
from app.agent.official_approval_view import pending_official_approvals
from app.agent.official_runtime import run_agent
from app.agent.tools import make_todo_tools
from app.config import get_settings
from app.db import crud
from app.db.models import (ActionExecution, OutboundDelivery, OutboundDeliveryChunk,
                           TelegramInboundReceipt, TelegramThreadLease, Todo, WebRequest)
from app.telegram.durable import (DurableSafetyError, admit_receipt, claim_receipt,
                                  dispatch_delivery, enqueue_reply, finalize_receipt)
from app.telegram.processing_context import receipt_scope
from app.web import api, runs
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel
from experiments.official_agent_migration_20261020.verify_per_call_pg import _project_graph

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "interrupt_mode" not in inspect.signature(HumanInTheLoopMiddleware).parameters,
        reason="Requires isolated LangChain >=1.4.6 per-call HITL"),
]


@pytest.mark.asyncio
async def test_two_official_cards_require_both_durable_intents_before_dispatch(
        pg_harness, isolated_pg18_target):
    """No half-batch is acknowledged; each card has its own ordered reply."""
    factory, _, _ = pg_harness
    actor = int(uuid4().hex[:9], 16)
    receipt = await admit_receipt(factory, bot_identity=actor,
                                  data={"update_id": int(uuid4().hex[:12], 16),
                                        "message": {"message_id": 1, "date": 0,
                                                    "from": {"id": actor, "is_bot": False,
                                                             "first_name": "Synthetic"},
                                                    "chat": {"id": actor, "type": "private"},
                                                    "text": "synthetic dual approval"}},
                                  graph_kind="telegram_session",
                                  thread_id=f"telegram_user_{actor}")
    delivery_ids = []
    try:
        token = await claim_receipt(factory, receipt.id, "synthetic-worker")
        assert token
        for ordinal in range(2):
            action = f"synthetic-action-{ordinal}"
            delivery_ids.append(await enqueue_reply(
                factory, receipt_id=receipt.id, fence_token=token,
                reply_ordinal=ordinal, kind="hitl_proposal", chat_id=str(actor),
                locale="zh", text=f"待审批操作 {ordinal}",
                keyboard={"inline_keyboard": [[
                    {"text": "批准", "callback_data": f"act:{action}:approve"},
                    {"text": "拒绝", "callback_data": f"act:{action}:reject"}]]},
                outcome_kind="ordinary_checkpoint", outcome_ref="synthetic-checkpoint",
                required_reply_count=2))
            if ordinal == 0:
                with pytest.raises(DurableSafetyError, match="MissingBusinessOutcomeOrReplyIntent"):
                    await finalize_receipt(factory, receipt.id, token)
                async with factory() as db:
                    row = await db.get(TelegramInboundReceipt, receipt.id)
                    lease = await db.get(TelegramThreadLease, {
                        "graph_kind": "telegram_session",
                        "thread_id": f"telegram_user_{actor}"})
                    row.lease_until = lease.lease_until = (
                        datetime.now(timezone.utc) - timedelta(seconds=1))
                    await db.commit()
                token = await claim_receipt(factory, receipt.id, "restarted-worker")
                assert token
                replayed = await enqueue_reply(
                    factory, receipt_id=receipt.id, fence_token=token,
                    reply_ordinal=0, kind="hitl_proposal", chat_id=str(actor),
                    locale="zh", text="待审批操作 0",
                    keyboard={"inline_keyboard": [[
                        {"text": "批准", "callback_data": "act:synthetic-action-0:approve"},
                        {"text": "拒绝", "callback_data": "act:synthetic-action-0:reject"}]]},
                    outcome_kind="ordinary_checkpoint",
                    outcome_ref="synthetic-checkpoint", required_reply_count=2)
                assert replayed == delivery_ids[0]
        assert await finalize_receipt(factory, receipt.id, token)

        class FakeBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(message_id=800 + len(self.calls), chat_id=actor)

        bot = FakeBot()
        for delivery_id in delivery_ids:
            await dispatch_delivery(factory, delivery_id, "synthetic-sender", bot)
        assert len(bot.calls) == 2
        assert [call["text"] for call in bot.calls] == ["待审批操作 0", "待审批操作 1"]
        assert [call["reply_markup"].inline_keyboard[0][0].callback_data
                for call in bot.calls] == [
                    "act:synthetic-action-0:approve", "act:synthetic-action-1:approve"]
    finally:
        async with factory() as db:
            await db.execute(delete(OutboundDeliveryChunk).where(
                OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
            await db.execute(delete(OutboundDelivery).where(
                OutboundDelivery.id.in_(delivery_ids)))
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id == receipt.id))
            await db.commit()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Retired Mini App approval worker; Telegram Receipt owns Agent resume")
async def test_miniapp_decision_survives_worker_crash_after_pg_saver_commit(
        pg_harness, isolated_pg18_target, monkeypatch):
    factory, new_user, _ = pg_harness
    user = await new_user()
    thread = f"official-v1:user_{user.telegram_id}"
    todo = SimpleNamespace(id="synthetic-todo", title="Synthetic approval target", version=7)
    monkeypatch.setattr(unified_tools.crud, "list_todos", AsyncMock(return_value=[todo]))

    async def execute(proposal):
        return ActionResult(action_id=proposal.action_id,
                            status=ActionStatus.EXECUTED,
                            message="synthetic execution")

    executor = SimpleNamespace(execute=AsyncMock(side_effect=execute))
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False),
        autocommit=True)
    try:
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[{
                "name": "propose_todo_delete", "args": {"target": todo.title},
                "id": "delete-call"}]),
            AIMessage(content="The approved operation was handled."),
        ])
        graph = _project_graph(model, AsyncPostgresSaver(conn))
        identity = {"user_id": user.id, "telegram_id": user.telegram_id,
                    "chat_id": user.telegram_id, "origin": "telegram"}
        await graph.ainvoke({
            "messages": [HumanMessage(id=f"turn-{uuid4().hex}",
                                      content=f"Delete {todo.title}")],
            "user_id": user.id, "telegram_id": user.telegram_id,
            "chat_id": user.telegram_id, "request_origin": "telegram",
            "active_turn_id": "synthetic-user-turn",
        }, config={"configurable": {"thread_id": thread}}, context=identity)
        paused = await graph.aget_state({"configurable": {"thread_id": thread}})
        proposal = paused.values["prepared_actions"]["delete-call"]["proposal"]
        checkpoint_id = paused.config["configurable"]["checkpoint_id"]

        from app.agent import official_runtime as graph_module
        monkeypatch.setattr(graph_module, "get_graph", AsyncMock(return_value=graph))
        monkeypatch.setattr(api, "_authenticated", AsyncMock(return_value=(user, None)))
        decision_key = uuid4().hex
        monkeypatch.setattr(api, "_key", lambda _request: decision_key)
        monkeypatch.setattr(api, "AsyncSessionLocal", factory)
        wake = SimpleNamespace(wake=lambda: None)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(web_worker=wake)))
        from starlette.responses import Response
        admitted = await api._decision_admission(
            request, Response(), action_id=proposal["action_id"],
            run_id="telegram-agent-official", checkpoint_version=checkpoint_id,
            operation="approval", payload={"decision": "approve"})
        async with factory() as db:
            row = await db.scalar(select(WebRequest).where(WebRequest.id == admitted["run_id"]))
        assert row is not None and row.thread_id == thread
        assert row.request_body["official_approval"] is True

        first_state, first_result = await runs._execute(factory, row, "synthetic-fence")
        assert first_state == "COMPLETED"
        assert executor.execute.await_count == 1
        current = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert any(isinstance(item, ToolMessage) and item.tool_call_id == "delete-call"
                   for item in current.values["messages"])
        # Simulate a process death after Saver commit but before WebRequest
        # finish_run. The same durable row must be reconciled, never resumed.
        second_state, second_result = await runs._execute(factory, row, "new-fence")
        assert second_state == first_state
        # This intentionally small protocol graph lacks CandidateState's
        # active_turn_id, so readback uses a conservative settled response
        # instead of pretending it can reproduce the exact model reply.
        assert second_result["reconciled_from_saver"] is True
        assert second_result["action_id"] == proposal["action_id"]
        assert first_result["response"] == "The approved operation was handled."
        assert executor.execute.await_count == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Retired Mini App approval worker; Telegram Receipt owns Agent resume")
async def test_official_full_graph_web_approval_executes_real_todo_once(
        pg_harness, isolated_pg18_target, monkeypatch):
    """Web approval resumes the production-shaped graph and real PG ledger."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Official approval {uuid4().hex[:12]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id, approved_version = todo.id, todo.version
    monkeypatch.setattr(
        workflow_runtime, "_action_executor",
        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=ScriptedModel([
                AIMessage(content="", tool_calls=[{
                    "name": "propose_todo_delete", "args": {"target": title},
                    "id": "real-delete-call"}]),
                AIMessage(content=f"已删除待办 {title}"),
            ]),
            todo_tools=make_todo_tools(factory), session_factory=factory,
            settings=get_settings(), checkpointer=saver,
            interrupt_mode="per_call")
        thread = f"official-v1:web:ordinary:{user.id}"
        reply, result = await run_agent(
            graph, user.id, user.telegram_id, "", f"删除待办 {title}",
            thread_id=thread, turn_id=f"turn-{uuid4().hex}",
            origin="web", return_state=True)
        assert result.get("__interrupt__")
        paused = await graph.aget_state({"configurable": {"thread_id": thread}})
        approvals = pending_official_approvals(paused, user_id=user.id)
        assert len(approvals) == 1
        proposal = approvals[0].proposal
        assert proposal.args["todo_id"] == todo_id
        assert proposal.args["approved_todo_version"] == approved_version
        async with factory() as db:
            assert await db.get(Todo, todo_id) is not None
            assert await db.get(ActionExecution, proposal.action_id) is None

        monkeypatch.setattr(api, "_authenticated", AsyncMock(return_value=(user, None)))
        decision_key = uuid4().hex
        monkeypatch.setattr(api, "_key", lambda _request: decision_key)
        monkeypatch.setattr(api, "AsyncSessionLocal", factory)
        monkeypatch.setattr(workflow_runtime, "get_productivity_graph",
                            AsyncMock(return_value=SimpleNamespace()))
        from app.agent import official_runtime
        monkeypatch.setattr(official_runtime, "get_graph", AsyncMock(return_value=graph))
        wake = SimpleNamespace(wake=lambda: None)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(web_worker=wake)))
        from starlette.responses import Response
        parent, _ = await runs.submit_run(
            factory, user_id=user.id, operation="chat",
            idempotency_key=f"parent-{uuid4().hex}",
            body={"text": f"删除待办 {title}"}, thread_id=thread)
        async with factory() as db:
            parent_row = await db.get(WebRequest, parent.id)
            parent_row.state = "WAITING_APPROVAL"
            await db.commit()
        admitted = await api._decision_admission(
            request, Response(), action_id=proposal.action_id,
            run_id=parent.id,
            checkpoint_version=paused.config["configurable"]["checkpoint_id"],
            operation="approval", payload={"decision": "approve"})
        retried = await api._decision_admission(
            request, Response(), action_id=proposal.action_id,
            run_id=parent.id,
            checkpoint_version=paused.config["configurable"]["checkpoint_id"],
            operation="approval", payload={"decision": "approve"})
        assert retried["run_id"] == admitted["run_id"]
        async with factory() as db:
            row = await db.get(WebRequest, admitted["run_id"])
            assert row is not None and row.thread_id == thread
        first_state, first_result = await runs._execute(factory, row, "real-ledger-fence")
        assert first_state == "COMPLETED"
        assert title in first_result["response"]
        async with factory() as db:
            assert await db.get(Todo, todo_id) is None
            execution = await db.get(ActionExecution, proposal.action_id)
            assert execution is not None and execution.status == "executed"
            assert execution.result["id"] == todo_id
        current = await graph.aget_state({"configurable": {"thread_id": thread}})
        paired = [message for message in current.values["messages"]
                  if isinstance(message, ToolMessage) and
                  message.tool_call_id == "real-delete-call"]
        assert len(paired) == 1 and paired[0].status != "error"
        replay_state, replay_result = await runs._execute(factory, row, "replayed-fence")
        assert replay_state == "COMPLETED"
        # With the same active turn, recovery can return its checkpointed
        # final answer. `reconciled_from_saver` is reserved for a later turn.
        assert replay_result["response"] == first_result["response"]
        async with factory() as db:
            execution = await db.get(ActionExecution, proposal.action_id)
            assert execution is not None and execution.status == "executed"
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Retired Mini App approval worker; Telegram Receipt owns Agent resume")
async def test_official_full_graph_web_reject_preserves_todo_and_saver_thread(
        pg_harness, isolated_pg18_target, monkeypatch):
    """The current Web decision resumes the official checkpoint, not Workflow."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Official reject {uuid4().hex[:12]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id = todo.id
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=ScriptedModel([
                AIMessage(content="", tool_calls=[{
                    "name": "propose_todo_delete", "args": {"target": title},
                    "id": "web-reject-call"}]),
                AIMessage(content=f"已取消删除，{title} 保留。"),
            ]),
            todo_tools=make_todo_tools(factory), session_factory=factory,
            settings=get_settings(), checkpointer=saver, interrupt_mode="per_call")
        thread = f"official-v1:web:ordinary:{user.id}"
        _, initial = await run_agent(
            graph, user.id, user.telegram_id, "", f"删除待办 {title}",
            thread_id=thread, turn_id=f"web-request:{uuid4().hex}",
            origin="web", return_state=True)
        assert initial.get("__interrupt__")
        paused = await graph.aget_state({"configurable": {"thread_id": thread}})
        approval = pending_official_approvals(paused, user_id=user.id)[0]
        assert approval.proposal.args["todo_id"] == todo_id
        async with factory() as db:
            assert await db.get(Todo, todo_id) is not None
            assert await db.get(ActionExecution, approval.proposal.action_id) is None

        monkeypatch.setattr(api, "_authenticated", AsyncMock(return_value=(user, None)))
        monkeypatch.setattr(api, "_key", lambda _request: uuid4().hex)
        monkeypatch.setattr(api, "AsyncSessionLocal", factory)
        from app.agent import official_runtime
        monkeypatch.setattr(official_runtime, "get_graph", AsyncMock(return_value=graph))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            web_worker=SimpleNamespace(wake=lambda: None))))
        from starlette.responses import Response
        parent, _ = await runs.submit_run(
            factory, user_id=user.id, operation="chat",
            idempotency_key=f"parent-{uuid4().hex}",
            body={"text": f"删除待办 {title}"}, thread_id=thread)
        async with factory() as db:
            parent_row = await db.get(WebRequest, parent.id)
            parent_row.state = "WAITING_APPROVAL"
            await db.commit()
        admitted = await api._decision_admission(
            request, Response(), action_id=approval.proposal.action_id,
            run_id=parent.id,
            checkpoint_version=paused.config["configurable"]["checkpoint_id"],
            operation="approval", payload={"decision": "reject"})
        async with factory() as db:
            decision = await db.get(WebRequest, admitted["run_id"])
            assert decision.thread_id == thread
        state, result = await runs._execute(factory, decision, "reject-fence")
        assert state == "COMPLETED" and title in result["response"]
        finished = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert not finished.interrupts
        paired = [item for item in finished.values["messages"]
                  if isinstance(item, ToolMessage) and
                  item.tool_call_id == "web-reject-call"]
        assert len(paired) == 1
        async with factory() as db:
            assert await db.get(Todo, todo_id) is not None
            assert await db.get(ActionExecution, approval.proposal.action_id) is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_official_full_graph_telegram_button_executes_real_todo(
        pg_harness, isolated_pg18_target, monkeypatch):
    """The actual callback adapter resumes official HITL, not the old gate."""
    from app.telegram import webhook
    from app.agent import official_runtime
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"Telegram official {uuid4().hex[:10]}"
    async with factory() as db:
        todo = await crud.create_todo(db, user_id=user.id, title=title)
        await db.commit()
        todo_id = todo.id
    monkeypatch.setattr(
        workflow_runtime, "_action_executor",
        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=ScriptedModel([
                AIMessage(content="", tool_calls=[{
                    "name": "propose_todo_delete", "args": {"target": title},
                    "id": "telegram-delete-call"}]),
                AIMessage(content=f"已删除待办 {title}"),
            ]),
            todo_tools=make_todo_tools(factory), session_factory=factory,
            settings=get_settings(), checkpointer=saver,
            interrupt_mode="per_call")
        thread = f"official-v1:user_{user.telegram_id}"
        _, initial = await run_agent(
            graph, user.id, user.telegram_id, user.telegram_id,
            f"删除待办 {title}", origin="telegram", return_state=True)
        assert initial.get("__interrupt__")
        paused = await graph.aget_state({"configurable": {"thread_id": thread}})
        approval = pending_official_approvals(paused, user_id=user.id)[0]
        async with factory() as db:
            assert await db.get(Todo, todo_id) is not None
            assert await db.get(ActionExecution, approval.proposal.action_id) is None

        monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=graph))
        monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
        monkeypatch.setattr(webhook, "_send_agent_result", AsyncMock())
        callback = SimpleNamespace(
            data=f"act:{approval.proposal.action_id}:approve",
            from_user=SimpleNamespace(id=user.telegram_id),
            message=SimpleNamespace(chat_id=user.telegram_id),
            answer=AsyncMock())
        await webhook._handle_action_callback(callback)
        webhook._send_agent_result.assert_awaited_once()
        current = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert len([item for item in current.values["messages"]
                    if isinstance(item, ToolMessage) and
                    item.tool_call_id == "telegram-delete-call"]) == 1
        async with factory() as db:
            assert await db.get(Todo, todo_id) is None
            row = await db.get(ActionExecution, approval.proposal.action_id)
            assert row is not None and row.status == "executed"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_two_telegram_decisions_commit_partial_result_without_duplicate_resume(
        pg_harness, isolated_pg18_target, monkeypatch):
    """One ToolNode batch has two interrupts but no new checkpoint after the first."""
    from app.telegram import webhook
    from app.agent import official_runtime
    factory, new_user, _ = pg_harness
    user = await new_user()
    titles = [f"Batch A {uuid4().hex[:8]}", f"Batch B {uuid4().hex[:8]}"]
    async with factory() as db:
        todos = [await crud.create_todo(db, user_id=user.id, title=title)
                 for title in titles]
        await db.commit()
    monkeypatch.setattr(workflow_runtime, "_action_executor",
                        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    receipt_ids = []
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=ScriptedModel([
                AIMessage(content="", tool_calls=[
                    {"name": "propose_todo_delete", "args": {"target": titles[0]},
                     "id": "batch-delete-a"},
                    {"name": "propose_todo_delete", "args": {"target": titles[1]},
                     "id": "batch-delete-b"}]),
                AIMessage(content="One deleted, one rejected"),
            ]),
            todo_tools=make_todo_tools(factory), session_factory=factory,
            settings=get_settings(), checkpointer=saver,
            interrupt_mode="per_call")
        thread = f"official-v1:user_{user.telegram_id}"
        _, initial = await run_agent(
            graph, user.id, user.telegram_id, user.telegram_id,
            f"Delete {titles[0]} and {titles[1]}",
            origin="telegram", return_state=True)
        assert initial.get("__interrupt__")
        paused = await graph.aget_state({"configurable": {"thread_id": thread}})
        approvals = pending_official_approvals(paused, user_id=user.id)
        assert len(approvals) == 2
        ids = {item.tool_call_id: item.proposal.action_id for item in approvals}
        original_checkpoint = paused.config["configurable"]["checkpoint_id"]

        monkeypatch.setattr(official_runtime, "get_graph", AsyncMock(return_value=graph))
        monkeypatch.setattr(webhook, "get_graph", AsyncMock(return_value=graph))
        monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
        monkeypatch.setattr(webhook, "_answer_callback_best_effort", AsyncMock())
        replies = AsyncMock()
        duplicate_card = AsyncMock(side_effect=AssertionError("duplicate card"))
        monkeypatch.setattr(webhook, "send_message", replies)
        monkeypatch.setattr(webhook, "send_official_action_proposals", duplicate_card)

        async def decision(call_id, verdict, update_id):
            action_id = ids[call_id]
            receipt = TelegramInboundReceipt(
                bot_identity=999, update_id=update_id,
                envelope={"callback_query": {"data": f"act:{action_id}:{verdict}"}},
                state="PROCESSING", fence_token=f"fence-{update_id}",
                lease_until=datetime.now(timezone.utc) + timedelta(minutes=2),
                graph_kind="workflow", thread_id=f"telegram_user_{user.telegram_id}",
                action_id=action_id)
            async with factory() as db:
                db.add(receipt)
                await db.commit()
                receipt_ids.append(receipt.id)
            callback = SimpleNamespace(
                data=f"act:{action_id}:{verdict}",
                from_user=SimpleNamespace(id=user.telegram_id),
                message=SimpleNamespace(chat_id=user.telegram_id),
                answer=AsyncMock())
            with receipt_scope(receipt.id, f"fence-{update_id}") as run:
                await webhook._handle_action_callback(callback)
                return run.outcome_kind

        assert await decision("batch-delete-a", "approve", 900001) == "tool_result"
        middle = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert middle.config["configurable"]["checkpoint_id"] == original_checkpoint
        assert [item.tool_call_id for item in pending_official_approvals(
            middle, user_id=user.id)] == ["batch-delete-b"]
        assert replies.await_count == 1 and "1 项待确认" in replies.await_args.args[1]
        duplicate_card.assert_not_awaited()
        async with factory() as db:
            assert await db.get(Todo, todos[0].id) is None
            assert await db.get(Todo, todos[1].id) is not None
            execution = await db.get(ActionExecution, ids["batch-delete-a"])
            assert execution is not None and execution.status == "executed"

        # Replay of the same receipt finds the pending-write ToolMessage and
        # must not supply Command(resume) or execute the delete twice.
        with receipt_scope(receipt_ids[0], "fence-900001") as run:
            assert await webhook._replay_completed_resume(
                graph, thread, middle, user.telegram_id, agent=True,
                agent_context={"user_id": user.id,
                               "telegram_id": user.telegram_id,
                               "chat_id": user.telegram_id, "origin": "telegram"})
            assert run.outcome_kind == "tool_result"
        assert await decision("batch-delete-b", "reject", 900002) == "workflow_checkpoint"
        finished = await graph.aget_state({"configurable": {"thread_id": thread}})
        assert not finished.interrupts
        paired = {item.tool_call_id for item in finished.values["messages"]
                  if isinstance(item, ToolMessage)}
        assert {"batch-delete-a", "batch-delete-b"} <= paired
        assert "One deleted, one rejected" in replies.await_args.args[1]
        async with factory() as db:
            assert await db.get(Todo, todos[1].id) is not None
            assert await db.get(ActionExecution, ids["batch-delete-b"]) is None
    finally:
        if receipt_ids:
            async with factory() as db:
                await db.execute(delete(TelegramInboundReceipt).where(
                    TelegramInboundReceipt.id.in_(receipt_ids)))
                await db.commit()
        await conn.close()
