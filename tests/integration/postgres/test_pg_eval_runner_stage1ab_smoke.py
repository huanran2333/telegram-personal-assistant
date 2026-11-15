"""Disposable-PG smoke for the Evaluation runner's synthetic Telegram path.

The real create_agent graph, PostgreSQL Saver, Todo transaction, and
ActionExecution ledger run with a scripted model. No provider or Bot is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from app.agent import workflow_runtime
from app.agent.executor import ActionExecutor, SqlAlchemyActionLedger
from app.agent.official_approval_view import (
    official_decision_for_action, pending_official_approvals,
)
from app.agent.official_graph import build_official_agent
from app.agent.official_runtime import official_thread_id, run_agent
from app.config import get_settings
from app.db import crud
from app.db.models import ActionExecution, Todo
from experiments.official_agent_eval_rebuild_20261023.test_current_eval import (
    _case_evaluation_time, _invoke_at_evaluation_time,
    _synthetic_telegram_identity,
)
from experiments.official_agent_migration_20261020.test_official_runtime_contract import (
    ScriptedModel,
)


@pytest.mark.integration
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.asyncio
async def test_eval_stage1ab_real_pg_hitl_and_frozen_clock(
        pg_harness, isolated_pg18_target, monkeypatch, decision):
    factory, new_user, _ = pg_harness
    owner, bystander = await new_user(), await new_user()
    title = f"eval-stage1ab-{decision}-{uuid4().hex[:8]}"
    foreign_title = f"eval-foreign-{uuid4().hex[:8]}"
    async with factory() as db:
        owned = await crud.create_todo(db, user_id=owner.id, title=title)
        foreign = await crud.create_todo(db, user_id=bystander.id,
                                         title=foreign_title)
        await db.commit()
        owned_id, foreign_id = owned.id, foreign.id

    settings = get_settings().model_copy(update={
        "app_env": "test", "google_calendar_enabled": False,
        "gmail_enabled": False, "tavily_api_key": "", "memory_enabled": False,
    })
    fixed, zone_name = _case_evaluation_time({
        "evaluation_now": "2026-11-09T12:00:00+08:00",
        "evaluation_timezone": "Asia/Shanghai",
    }, default_timezone=settings.default_timezone)
    monkeypatch.setattr(workflow_runtime, "_action_executor",
                        ActionExecutor(None, SqlAlchemyActionLedger(factory), factory))
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "propose_todo_delete", "args": {"target": title},
            "id": f"eval-stage1ab-call-{decision}"}]),
        AIMessage(content="已处理这项审批"),
    ])
    thread_id, context = _synthetic_telegram_identity(owner)
    thread = official_thread_id(thread_id)
    config = {"configurable": {"thread_id": thread}}
    connection_url = isolated_pg18_target.set(
        drivername="postgresql").render_as_string(hide_password=False)
    first = second = None
    saver = None
    try:
        first = await psycopg.AsyncConnection.connect(
            connection_url, autocommit=True)
        first_saver = AsyncPostgresSaver(first)
        await first_saver.setup()
        graph, _ = build_official_agent(
            model=model, session_factory=factory, settings=settings,
            checkpointer=first_saver, interrupt_mode="per_call")
        _, initial = await _invoke_at_evaluation_time(
            run_agent(graph, owner.id, owner.telegram_id,
                      context["chat_id"], f"请确认删除 {title}",
                      thread_id=thread_id, origin=context["origin"],
                      return_state=True),
            now=fixed, zone_name=zone_name, settings=settings)
        assert initial.get("__interrupt__")
        assert "propose_todo_delete" in model._bound_tool_names[0]
        await first.close()
        first = None

        # A second Saver connection and graph must recover the persisted pause.
        second = await psycopg.AsyncConnection.connect(
            connection_url, autocommit=True)
        saver = AsyncPostgresSaver(second)
        resumed_graph, _ = build_official_agent(
            model=model, session_factory=factory, settings=settings,
            checkpointer=saver, interrupt_mode="per_call")
        paused = await resumed_graph.aget_state(config)
        pending = pending_official_approvals(paused, user_id=owner.id)
        assert len(pending) == 1
        assert pending[0].tool_call_id == f"eval-stage1ab-call-{decision}"
        assert pending[0].proposal.args["user_id"] == owner.id
        with pytest.raises(PermissionError, match="OfficialApprovalOwnerMismatch"):
            pending_official_approvals(paused, user_id=bystander.id)
        with pytest.raises(PermissionError, match="CheckpointOwnerMismatch"):
            await run_agent(resumed_graph, bystander.id, bystander.telegram_id,
                            bystander.telegram_id, "查看待办", thread_id=thread_id,
                            origin="telegram")
        async with factory() as db:
            assert await db.get(Todo, owned_id) is not None
            assert await db.get(Todo, foreign_id) is not None
            assert await db.get(ActionExecution, pending[0].proposal.action_id) is None

        resume = official_decision_for_action(
            paused, user_id=owner.id,
            action_id=pending[0].proposal.action_id, decision=decision,
            checkpoint_id=paused.config["configurable"]["checkpoint_id"])
        await _invoke_at_evaluation_time(
            resumed_graph.ainvoke(Command(resume=resume),
                                  config=config, context=context),
            now=fixed, zone_name=zone_name, settings=settings)
        finished = await resumed_graph.aget_state(config)
        assert not pending_official_approvals(finished, user_id=owner.id)
        paired = [message for message in finished.values["messages"]
                  if isinstance(message, ToolMessage) and
                  message.tool_call_id == f"eval-stage1ab-call-{decision}"]
        assert len(paired) == 1
        prompts = [next(str(message.content) for message in turn
                        if isinstance(message, SystemMessage))
                   for turn in model._received]
        assert len(prompts) == 2
        assert all(f"Current local time: {fixed.isoformat()}" in prompt
                   for prompt in prompts)
        incoming = next(message for message in finished.values["messages"]
                        if isinstance(message, HumanMessage))
        assert incoming.additional_kwargs["occurred_at_utc"] == fixed.astimezone(
            timezone.utc).isoformat()

        async with factory() as db:
            assert (await db.get(Todo, owned_id) is None) == (decision == "approve")
            assert await db.get(Todo, foreign_id) is not None
            action = await db.get(ActionExecution, pending[0].proposal.action_id)
            if decision == "approve":
                assert action is not None and action.status == "executed"
                assert paired[0].status == "success"
            else:
                assert action is None
                assert paired[0].status == "error"
    finally:
        if saver is not None:
            await saver.adelete_thread(thread)
            assert await saver.aget_tuple(config) is None
        if first is not None:
            await first.close()
        if second is not None:
            await second.close()
