"""Current create_agent model failover never replays a tool execution."""
from __future__ import annotations

import asyncio

import httpx
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from app.agent.official_model_failover import OfficialModelFailover
from app.agent.provider_clients import provider_client_scope
from app.config import Settings
from app.agent.official_model_failover import configured_failover
from app.agent import official_runtime
from app.agent import nodes


class SequenceModel(BaseChatModel):
    _responses: list = PrivateAttr()
    _attempts: int = PrivateAttr(default=0)

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self):
        return "synthetic-failover"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._attempts += 1
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ChatResult(generations=[ChatGeneration(message=outcome)])


def transient():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://test.invalid/v1"))


def timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://test.invalid/v1"))


def permanent():
    return openai.BadRequestError(
        "SyntheticInvalidToolSchema",
        response=httpx.Response(400, request=httpx.Request("POST", "https://test.invalid/v1")),
        body=None)


def status_error(status: int):
    return openai.APIStatusError(
        f"SyntheticHTTP{status}",
        response=httpx.Response(status, request=httpx.Request("POST", "https://test.invalid/v1")),
        body=None)


@pytest.mark.asyncio
async def test_transient_primary_failure_uses_backup_and_pairs_tool_result():
    primary = SequenceModel([transient(), transient(), transient(), transient()])
    backup = SequenceModel([
        AIMessage(content="", tool_calls=[{
            "name": "add_numbers", "id": "fallback-call",
            "args": {"a": 2, "b": 3}}]),
        AIMessage(content="结果是 5。"),
    ])
    executions = []

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two integers exactly once."""
        executions.append((a, b))
        return a + b

    graph = create_agent(
        primary, tools=[add_numbers],
        middleware=[OfficialModelFailover([backup], primary_attempts=2)],
        checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="计算 2+3")]},
        config={"configurable": {"thread_id": "fallback-tool-result"}})
    assert primary._attempts == 4  # The next model turn retries its own model request.
    assert backup._attempts == 2 and executions == [(2, 3)]
    messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert len(messages) == 1 and messages[0].tool_call_id == "fallback-call"
    assert result["messages"][-1].content == "结果是 5。"


@pytest.mark.asyncio
async def test_permanent_provider_error_does_not_retry_or_fallback():
    primary, backup = SequenceModel([permanent()]), SequenceModel([AIMessage(content="wrong")])
    graph = create_agent(
        primary, tools=[], middleware=[OfficialModelFailover([backup])],
        checkpointer=InMemorySaver())
    with pytest.raises(openai.BadRequestError):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"configurable": {"thread_id": "permanent-error"}})
    assert primary._attempts == 1 and backup._attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (401, 403))
async def test_authentication_or_permission_error_never_uses_another_model(status):
    primary = SequenceModel([status_error(status)])
    backup = SequenceModel([AIMessage(content="must not run")])
    graph = create_agent(primary, tools=[], middleware=[OfficialModelFailover([backup])])
    with pytest.raises(openai.APIStatusError):
        await graph.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert primary._attempts == 1 and backup._attempts == 0


@pytest.mark.asyncio
async def test_service_unavailable_retries_primary_then_uses_backup():
    primary = SequenceModel([status_error(503), status_error(503)])
    backup = SequenceModel([AIMessage(content="backup available")])
    graph = create_agent(primary, tools=[], middleware=[OfficialModelFailover([backup])])
    result = await graph.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert result["messages"][-1].content == "backup available"
    assert primary._attempts == 2 and backup._attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (timeout, lambda: status_error(429),
                                    lambda: status_error(500)))
async def test_other_transient_model_failures_switch_after_bounded_attempts(failure):
    primary = SequenceModel([failure(), failure()])
    backup = SequenceModel([AIMessage(content="备用模型回复")])
    graph = create_agent(primary, tools=[], middleware=[OfficialModelFailover([backup])])
    result = await graph.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert result["messages"][-1].content == "备用模型回复"
    assert (primary._attempts, backup._attempts) == (2, 1)


@pytest.mark.asyncio
async def test_both_models_unavailable_surfaces_error_without_tool_execution():
    primary = SequenceModel([transient(), transient()])
    backup = SequenceModel([status_error(503)])
    executed = []

    @tool
    def record_once(value: str) -> str:
        """Record a synthetic side effect."""
        executed.append(value)
        return value

    graph = create_agent(primary, tools=[record_once],
                         middleware=[OfficialModelFailover([backup])])
    with pytest.raises(openai.APIStatusError) as caught:
        await graph.ainvoke({"messages": [HumanMessage(content="hello")]})
    assert caught.value.status_code == 503
    assert (primary._attempts, backup._attempts) == (2, 1)
    assert executed == []


@pytest.mark.asyncio
async def test_model_switch_after_committed_tool_call_does_not_reexecute_it(caplog):
    primary = SequenceModel([
        AIMessage(content="", tool_calls=[{
            "name": "record_once", "id": "original-call",
            "args": {"value": "one"}}]),
        transient(), transient(),
    ])
    backup = SequenceModel([AIMessage(content="已记录 one")])
    executed = []

    @tool
    def record_once(value: str) -> str:
        """Record a synthetic side effect."""
        executed.append(value)
        return value

    graph = create_agent(primary, tools=[record_once],
                         middleware=[OfficialModelFailover([backup])])
    with caplog.at_level("INFO", logger="app.agent.official_model_failover"):
        result = await graph.ainvoke({"messages": [HumanMessage(content="record one")]})
    messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert executed == ["one"] and len(messages) == 1
    assert messages[0].tool_call_id == "original-call"
    assert result["messages"][-1].content == "已记录 one"
    assert (primary._attempts, backup._attempts) == (3, 1)
    assert "round=1 backend=primary" in caplog.text
    assert "round=2 index=1" in caplog.text
    assert "reason=APIConnectionError" in caplog.text


@pytest.mark.asyncio
async def test_primary_recovers_before_backup_and_two_backups_are_ordered():
    primary = SequenceModel([transient(), AIMessage(content="primary recovered")])
    backup = SequenceModel([AIMessage(content="unused")])
    graph = create_agent(
        primary, tools=[], middleware=[OfficialModelFailover([backup])],
        checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"thread_id": "primary-recovers"}})
    assert result["messages"][-1].content == "primary recovered"
    assert primary._attempts == 2 and backup._attempts == 0

    primary = SequenceModel([transient(), transient()])
    first, second = SequenceModel([transient()]), SequenceModel([AIMessage(content="second backup")])
    graph = create_agent(
        primary, tools=[], middleware=[OfficialModelFailover([first, second])],
        checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"thread_id": "second-backup"}})
    assert result["messages"][-1].content == "second backup"
    assert (primary._attempts, first._attempts, second._attempts) == (2, 1, 1)


@pytest.mark.asyncio
async def test_configured_backup_models_have_separate_endpoints_and_keys():
    settings = Settings(
        _env_file=None, llm_provider="openai", llm_model="primary",
        openai_api_key="primary-only-secret", memory_enabled=False,
        llm_fallback_1_base_url="http://10.0.0.1:3027/v1",
        llm_fallback_1_model="Qwen3.8-27B-AWQ-INT4",
        llm_fallback_1_api_key="local-placeholder-one",
        llm_fallback_2_base_url="http://10.0.0.1:25578/v1",
        llm_fallback_2_model="Qwen/Qwen3.5-9B",
        llm_fallback_2_api_key="local-placeholder-two")
    async with provider_client_scope():
        middleware = configured_failover(settings)
        assert middleware is not None and len(middleware.backups) == 2
        assert [item.model_name for item in middleware.backups] == [
            "Qwen3.8-27B-AWQ-INT4", "Qwen/Qwen3.5-9B"]
        assert [item.openai_api_key.get_secret_value() for item in middleware.backups] == [
            "local-placeholder-one", "local-placeholder-two"]
        assert all(item.openai_api_key.get_secret_value() != "primary-only-secret"
                   for item in middleware.backups)


@pytest.mark.asyncio
async def test_primary_qwen_uses_verified_non_thinking_request_option(monkeypatch):
    settings = Settings(_env_file=None, llm_provider="openai",
                        llm_model="Qwen3.8-27B-AWQ-INT4",
                        openai_base_url="http://10.0.0.1:3027/v1",
                        openai_api_key="local-placeholder", memory_enabled=False)
    monkeypatch.setattr(nodes, "settings", settings)
    async with provider_client_scope():
        primary = nodes._build_chat_llm(
            "openai", settings.llm_model, request_timeout=12,
            disable_qwen_thinking=True)
        assert primary.model_name == settings.llm_model
        assert primary.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert primary.request_timeout == 12 and primary.max_retries == 0


def test_incomplete_or_public_plain_http_backup_config_is_rejected():
    common = dict(_env_file=None, llm_provider="openai", llm_model="primary",
                  openai_api_key="synthetic", memory_enabled=False)
    with pytest.raises(ValueError, match="LLM_FALLBACK_1"):
        Settings(**common, llm_fallback_1_base_url="http://10.0.0.1:3027/v1")
    with pytest.raises(ValueError, match="LLM_FALLBACK_1"):
        Settings(**common, llm_fallback_1_base_url="http://example.com/v1",
                 llm_fallback_1_model="model", llm_fallback_1_api_key="synthetic")


@pytest.mark.asyncio
async def test_runtime_build_wires_failover_only_for_its_own_model(monkeypatch):
    settings = Settings(_env_file=None, llm_provider="openai", llm_model="primary",
                        openai_api_key="synthetic", memory_enabled=False,
                        app_env="test")
    marker = object()
    captured = {}
    monkeypatch.setattr(official_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(official_runtime, "configured_failover", lambda _settings: marker)

    def build_model(*args, **kwargs):
        captured["timeout"] = kwargs.get("request_timeout")
        captured["disable_qwen_thinking"] = kwargs.get("disable_qwen_thinking")
        return "synthetic-primary"

    def build_agent(**kwargs):
        captured["model"] = kwargs["model"]
        captured["failover"] = kwargs["model_failover"]
        return "synthetic-graph", []

    monkeypatch.setattr(official_runtime, "_build_chat_llm", build_model)
    monkeypatch.setattr(official_runtime, "build_official_agent", build_agent)
    try:
        assert await official_runtime.build_graph(None, durable=False) == "synthetic-graph"
        assert captured == {"timeout": 12.0, "disable_qwen_thinking": True,
                            "model": "synthetic-primary",
                            "failover": marker}
        captured.clear()
        assert await official_runtime.build_graph(
            None, durable=False, model="injected-test-model") == "synthetic-graph"
        assert captured == {"model": "injected-test-model", "failover": None}
    finally:
        official_runtime.clear_graph_runtime()
