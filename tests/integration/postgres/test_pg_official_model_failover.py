"""Real local vLLM failover through the current production create_agent graph."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
import httpx
import openai
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select
from pydantic import PrivateAttr

from app.agent.official_graph import build_official_agent
from app.agent.official_model_failover import OfficialModelFailover, configured_failover
from app.agent.official_runtime import run_agent
from app.agent.provider_clients import provider_client_scope, current_provider_owner
from app.config import get_settings
from app.db.models import ActionExecution, Todo
from app.web.processing_context import web_run_scope
from app.web.runs import claim_one, submit_run
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel


pytestmark = [pytest.mark.integration, pytest.mark.llm_integration]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("normal-primary", "failed-primary"))
async def test_current_model_order_executes_read_tool_and_returns_answer(
        pg_harness, isolated_pg18_target, scenario):
    factory, new_user, _ = pg_harness
    user = await new_user()
    original = get_settings()
    if not original.llm_fallback_1_base_url:
        pytest.skip("Operator did not configure the local backup model")
    settings = original.model_copy(update={
        "llm_provider": "openai",
        "llm_model": ("synthetic-unreachable-primary" if scenario == "failed-primary"
                      else original.llm_model),
        "openai_base_url": ("http://127.0.0.1:1/v1" if scenario == "failed-primary"
                            else original.openai_base_url),
        "openai_api_key": ("synthetic-primary-key" if scenario == "failed-primary"
                           else original.openai_api_key),
        "google_calendar_enabled": False, "gmail_enabled": False,
        "memory_enabled": False,
    })
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        async with provider_client_scope():
            sync_http, async_http = current_provider_owner().openai_http_clients()
            primary = ChatOpenAI(
                model=settings.llm_model, base_url=settings.openai_base_url,
                api_key=settings.openai_api_key, temperature=0, max_tokens=4096,
                request_timeout=settings.llm_primary_timeout_seconds, max_retries=0,
                **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                   if "qwen" in settings.llm_model.lower() else {}),
                http_client=sync_http, http_async_client=async_http)
            failover = configured_failover(settings)
            graph, _ = build_official_agent(
                model=primary, session_factory=factory, settings=settings,
                checkpointer=saver, model_failover=failover,
                interrupt_mode="per_call")
            answer, state = await asyncio.wait_for(run_agent(
                graph, user.id, user.telegram_id, "",
                "请调用 list_todos 列出我当前的待办，然后根据工具结果回答。只读，不要创建或修改。",
                origin="web", turn_id=f"fallback-test-{uuid4().hex}",
                thread_id=f"web:ordinary:{user.id}", return_state=True), timeout=120)
        calls = [call for message in state["messages"]
                 if isinstance(message, AIMessage) for call in message.tool_calls]
        results = [message for message in state["messages"]
                   if isinstance(message, ToolMessage)]
        async with factory() as db:
            todos = (await db.scalars(select(Todo).where(Todo.user_id == user.id))).all()
        assert todos == []
        assert any(call["name"] == "list_todos" for call in calls)
        assert any(message.tool_call_id in {call["id"] for call in calls}
                   for message in results)
        assert answer.strip()
        observed_models = [item.response_metadata.get("model_name")
                           for item in state["messages"] if isinstance(item, AIMessage)]
        expected_model = (original.llm_model if scenario == "normal-primary"
                          else original.llm_fallback_1_model)
        assert observed_models and all(name == expected_model for name in observed_models)
        if os.environ.get("PPA_CORE_OUTPUT"):
            output = Path(os.environ["PPA_CORE_OUTPUT"]) / f"real-{scenario}.json"
            if output.exists():
                raise RuntimeError("RefusingToOverwriteModelFallbackEvidence")
            output.write_text(json.dumps({
                "scenario": scenario,
                "expected_model": expected_model,
                "request": "只读列出待办", "tool_calls": calls,
                "tool_messages": [{"id": item.tool_call_id, "status": item.status,
                                   "content": str(item.content)[:300]} for item in results],
                "final_reply": answer, "todo_count_after": len(todos),
                "ai_model_names": observed_models,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        await conn.close()


class FailsAfterFirstToolCall(ScriptedModel):
    _attempts: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._attempts += 1
        if self._attempts > 1:
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://synthetic.invalid/v1"))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.asyncio
async def test_model_switch_after_web_todo_commit_keeps_one_proof_and_result(
        pg_harness, isolated_pg18_target):
    """Fault injection: switching on the next model turn cannot replay ToolNode."""
    factory, new_user, _ = pg_harness
    user = await new_user()
    title = f"failover-committed-{uuid4().hex[:8]}"
    request = f"创建一个不设时间的待办『{title}』，然后报告真实结果。"
    run, _ = await submit_run(factory, user_id=user.id, operation="chat",
                              idempotency_key=uuid4().hex, body={"text": request})
    claim = await claim_one(factory, owner="model-failover-test", lease_seconds=120)
    assert claim and claim[0] == run.id
    primary = FailsAfterFirstToolCall([AIMessage(content="", tool_calls=[{
        "name": "create_todo", "id": "failover-original-tool-call",
        "args": {"title": title}}])])
    backup = ScriptedModel([AIMessage(content=f"已创建待办『{title}』。")])
    settings = get_settings().model_copy(update={"memory_enabled": False})
    conn = await psycopg.AsyncConnection.connect(
        isolated_pg18_target.set(drivername="postgresql").render_as_string(
            hide_password=False), autocommit=True)
    try:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        graph, _ = build_official_agent(
            model=primary, session_factory=factory, settings=settings,
            checkpointer=saver, model_failover=OfficialModelFailover([backup]),
            interrupt_mode="per_call")
        with web_run_scope(run.id, user.id, claim[1]):
            reply, state = await run_agent(
                graph, user.id, user.telegram_id, "", request,
                origin="web", turn_id=run.turn_id,
                thread_id=f"web:ordinary:{user.id}", return_state=True)
        with web_run_scope(run.id, user.id, claim[1]):
            replay, _ = await run_agent(
                graph, user.id, user.telegram_id, "", request,
                origin="web", turn_id=run.turn_id,
                thread_id=f"web:ordinary:{user.id}", return_state=True)
        async with factory() as db:
            todos = (await db.scalars(select(Todo).where(
                Todo.user_id == user.id, Todo.title == title))).all()
            proofs = (await db.scalars(select(ActionExecution).where(
                ActionExecution.user_id == user.id,
                ActionExecution.operation == "create_todo"))).all()
        results = [item for item in state["messages"] if isinstance(item, ToolMessage)]
        assert len(todos) == len(proofs) == len(results) == 1
        assert proofs[0].status == "executed"
        assert results[0].tool_call_id == "failover-original-tool-call"
        assert title in reply and reply == replay
        assert primary._attempts == 3 and backup._received
    finally:
        await conn.close()
