"""The production tool boundary binds identity and rejects untrusted writes."""

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from app.agent.official_permissions import OfficialToolAuthorization
from app.agent.official_graph import _read_tool_error
from app.agent.official_context import OFFICIAL_PROMPT, TOOL_ORDERING_RULE
from app.agent.official_read_tools import make_official_read_tools
from app.agent.official_todo_tools import make_official_todo_tools
from app.agent.official_todo_approval import make_approved_todo_delete_tool
from app.agent.official_guarded_tools import make_official_guarded_tools
from app.agent.gmail import GmailError
from tavily.errors import TimeoutError as TavilyTimeoutError
from app.agent.official_graph import build_official_agent
from app.config import get_settings
from experiments.official_agent_migration_20261020.test_official_runtime_contract import ScriptedModel


def _runtime(request="创建待办『复习算法』，不设时间", call_id="call-1"):
    identity = {"user_id": "synthetic-user", "telegram_id": "123",
                "chat_id": "123", "origin": "telegram"}
    state = {"user_id": identity["user_id"], "telegram_id": identity["telegram_id"],
             "chat_id": identity["chat_id"], "request_origin": identity["origin"],
             "active_request": request}
    runtime = ToolRuntime(state=state, context=identity, config={},
                          stream_writer=lambda _: None, tool_call_id=call_id,
                          store=None)
    return state, runtime


def _authorization(tools):
    settings = SimpleNamespace(
        google_calendar_enabled=False, google_calendar_owner_telegram_id=None,
        gmail_enabled=False, gmail_owner_telegram_id=None, tavily_api_key=None)
    return OfficialToolAuthorization(tools, settings=settings)


async def _attempt(middleware, name, args, *, request, call_id="call-1"):
    state, runtime = _runtime(request, call_id)
    called = []

    async def handler(_):
        called.append(True)
        return ToolMessage(content="executed", tool_call_id=call_id)

    result = await middleware.awrap_tool_call(SimpleNamespace(
        state=state, runtime=runtime,
        tool_call={"id": call_id, "name": name, "args": args}), handler)
    return result, called


def test_production_tool_schemas_hide_identity_and_execution_key():
    todo = make_official_todo_tools(None)
    reads = make_official_read_tools(None, memory_orchestrator=object())
    guarded = [make_approved_todo_delete_tool(), *make_official_guarded_tools()]
    assert "search_memory" in {item.name for item in reads}
    assert "prepare_meeting" not in {item.name for item in reads}
    assert "control_memory" not in {item.name for item in reads}
    assert "control_memory" not in OFFICIAL_PROMPT
    for item in [*todo, *reads, *guarded]:
        assert "user_id" not in item.args
        assert "telegram_id" not in item.args
        assert "tool_call_id" not in item.args


@pytest.mark.asyncio
async def test_todo_model_tool_calls_plain_operation_with_trusted_identity_and_call_id(monkeypatch):
    observed = []

    async def plain_create(**kwargs):
        observed.append(kwargs)
        return "created"

    monkeypatch.setattr("app.agent.official_todo_tools.make_todo_operations",
                        lambda _factory: {"create_todo": plain_create})
    tool = next(item for item in make_official_todo_tools(None)
                if item.name == "create_todo")
    _, runtime = _runtime(call_id="checkpointed-call")
    result = await tool.coroutine(title="复习算法", runtime=runtime)
    assert result == "created"
    assert observed == [{"title": "复习算法", "description": "",
                         "due_date_str": "", "planned_start_str": "",
                         "planned_end_str": "", "reminder_at_str": "",
                         "priority": "medium", "tags": "",
                         "tool_call_id": "checkpointed-call",
                         "user_id": "synthetic-user"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(("utterance", "title"), [
    ("只读，不创建待办", "误建"),
    ("不创建 A，只创建 B", "A"),
    ("不创建 A，只创建 B", "B"),
    ("创建两个待办『复习算法』和『写总结』", "写总结"),
])
async def test_todo_creation_does_not_get_a_second_intent_check(utterance, title):
    middleware = _authorization(make_official_todo_tools(None))
    message, called = await _attempt(middleware, "create_todo", {"title": title},
                                     request=utterance)
    # The first two rows are accepted risk evidence, not proof of authorization.
    assert called
    assert message.tool_call_id == "call-1"
    assert message.status == "success"


@pytest.mark.asyncio
async def test_todo_time_source_is_checked_before_tool_execution():
    middleware = _authorization(make_official_todo_tools(None))
    message, called = await _attempt(
        middleware, "create_todo",
        {"title": "复习算法", "planned_start_str": "明天 10:00",
         "time_source_quote": "明天 10:00"},
        request="明天 10:00 开会，另创建待办『复习算法』")
    assert message.status == "error" and not called


@pytest.mark.asyncio
async def test_memory_words_do_not_disable_authorized_history_search():
    tools = make_official_read_tools(None, memory_orchestrator=object())
    middleware = _authorization(tools)
    result, called = await _attempt(
        middleware, "search_memory", {"query": "代号"},
        request="请忘掉代号")
    assert result.status == "success" and called


@pytest.mark.asyncio
async def test_h3_20_reaches_model_and_authorized_todo_tool(monkeypatch):
    written = []

    async def create(**kwargs):
        written.append(kwargs)
        return "created:H3-20-整理代码"

    monkeypatch.setattr("app.agent.official_todo_tools.make_todo_operations",
                        lambda _factory: {"create_todo": create})
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{
            "name": "create_todo", "id": "h3-20-call",
            "args": {"title": "H3-20-整理代码"}}]),
        AIMessage(content="已创建整理代码；未记录闲聊想法。"),
    ])
    graph, _ = build_official_agent(
        model=model, session_factory=None, settings=get_settings(),
        checkpointer=InMemorySaver())
    request = "请创建无时间待办『H3-20-整理代码』，但『H3-20-闲聊想法』不要记录，且不允许为任何待办加提醒。"
    result = await graph.ainvoke({
        "messages": [HumanMessage(id="h3-20-request", content=request)],
        "user_id": "synthetic-user", "telegram_id": "123", "chat_id": "123",
        "request_origin": "telegram", "active_request": request,
        "active_turn_id": "h3-20-request", "conversation_thread_id": "user_123",
    }, config={"configurable": {"thread_id": "h3-20-new-runtime"}},
        context={"user_id": "synthetic-user", "telegram_id": "123",
                 "chat_id": "123", "origin": "telegram"})
    assert len(model._received) == 2
    assert len(written) == 1 and written[0]["title"] == "H3-20-整理代码"
    assert any(isinstance(message, ToolMessage) and
               message.tool_call_id == "h3-20-call" and message.status == "success"
               for message in result["messages"])


@pytest.mark.asyncio
async def test_stale_checkpoint_owner_cannot_reach_tool_handler():
    middleware = _authorization(make_official_todo_tools(None))
    state, runtime = _runtime()
    state["user_id"] = "different-user"

    async def handler(_):
        pytest.fail("Mismatched identity reached the business operation")

    with pytest.raises(PermissionError, match="AgentRuntimeOwnerMismatch"):
        await middleware.awrap_tool_call(SimpleNamespace(
            state=state, runtime=runtime,
            tool_call={"id": "call-1", "name": "create_todo",
                       "args": {"title": "复习算法"}}), handler)


@pytest.mark.asyncio
async def test_gmail_read_error_is_paired_but_other_exceptions_propagate():
    middleware = ToolErrorMiddleware(
        on_error=_read_tool_error, tools=["gmail_search", "gmail_unread", "gmail_thread"])
    request = SimpleNamespace(tool=None, tool_call={"name": "gmail_unread", "id": "mail-1"})

    async def read_failure(_):
        raise GmailError("private OAuth details")

    result = await middleware.awrap_tool_call(request, read_failure)
    assert result.status == "error" and result.tool_call_id == "mail-1"
    assert "未获得可靠结果" in result.content
    assert "OAuth" not in result.content

    async def unexpected_failure(_):
        raise RuntimeError("unknown failure")

    with pytest.raises(RuntimeError, match="unknown failure"):
        await middleware.awrap_tool_call(request, unexpected_failure)


@pytest.mark.asyncio
async def test_web_read_timeout_is_paired_without_exposing_provider_detail():
    middleware = ToolErrorMiddleware(on_error=_read_tool_error, tools=["web_search"])
    request = SimpleNamespace(tool=None, tool_call={"name": "web_search", "id": "web-1"})

    async def read_timeout(_):
        raise TavilyTimeoutError(15)

    result = await middleware.awrap_tool_call(request, read_timeout)
    assert result.status == "error" and result.tool_call_id == "web-1"
    assert "未获得可靠结果" in result.content
    assert "15" not in result.content


@pytest.mark.asyncio
async def test_full_official_graph_does_not_intercept_memory_words_before_model():
    model = ScriptedModel([AIMessage(content="请说明具体目标。")])
    graph, tools = build_official_agent(
        model=model, session_factory=None, settings=get_settings(),
        checkpointer=InMemorySaver())
    assert "control_memory" not in {item.name for item in tools}
    state = await graph.ainvoke({
        "messages": [HumanMessage(id="request-1", content="请忘掉合成代号 Z-91")],
        "user_id": "synthetic-user", "telegram_id": "123", "chat_id": "123",
        "request_origin": "telegram", "active_request": "请忘掉合成代号 Z-91",
        "active_turn_id": "request-1", "conversation_thread_id": "user_123",
    }, config={"configurable": {"thread_id": "forget-retired"}},
        context={"user_id": "synthetic-user", "telegram_id": "123",
                 "chat_id": "123", "origin": "telegram"})
    assert state["messages"][-1].content == "请说明具体目标。"
    assert len(model._received) == 1


@pytest.mark.asyncio
async def test_full_official_graph_sends_tool_ordering_rule_to_model():
    model = ScriptedModel([AIMessage(content="收到")])
    graph, _ = build_official_agent(
        model=model, session_factory=None, settings=get_settings(),
        checkpointer=InMemorySaver())
    await graph.ainvoke({
        "messages": [HumanMessage(id="request-order-1", content="先查日历，再创建待办")],
        "user_id": "synthetic-user", "telegram_id": "123", "chat_id": "123",
        "request_origin": "telegram", "active_request": "先查日历，再创建待办",
        "active_turn_id": "request-order-1", "conversation_thread_id": "user_123",
    }, config={"configurable": {"thread_id": "tool-ordering-prompt"}},
        context={"user_id": "synthetic-user", "telegram_id": "123",
                 "chat_id": "123", "origin": "telegram"})
    assert len(model._received) == 1
    system = [message for message in model._received[0]
              if isinstance(message, SystemMessage)]
    assert len(system) == 1
    assert TOOL_ORDERING_RULE.strip() in str(system[0].content)
    assert OFFICIAL_PROMPT.count(TOOL_ORDERING_RULE) == 1
    assert "call calendar_candidate_slots first" in str(system[0].content)
