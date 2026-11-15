"""The new Agent can compose authorized domains without a single-intent gate."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent.gmail import GmailMessage
from app.agent.unified_loop import make_unified_nodes
from app.agent.unified_tools import (AgentToolScope, agent_tool_scope, available_tools,
                                     make_unified_tools)


def settings(**overrides):
    base = dict(google_calendar_enabled=True, google_calendar_owner_telegram_id="owner",
                gmail_enabled=True, gmail_owner_telegram_id="owner", tavily_api_key="",
                default_timezone="Asia/Shanghai", llm_provider="openai", llm_model="fake")
    return SimpleNamespace(**(base | overrides))


@asynccontextmanager
async def no_db():
    yield object()


class SequenceModel:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.bound = []
        self.observed = []

    def bind_tools(self, tools):
        self.bound.append({item.name for item in tools})
        return self

    async def ainvoke(self, _messages):
        self.observed.append(list(_messages))
        return self.answers.pop(0)


def state():
    return {"user_id": "synthetic-user", "telegram_id": "owner", "chat_id": "owner", "request_origin": "telegram",
            "conversation_thread_id": "user_owner", "active_turn_id": "synthetic-turn",
            "active_request": "查看日历和未读邮件", "messages": [HumanMessage(content="查看日历和未读邮件")],
            "unified_tool_rounds": 0}


def runtime():
    return Runtime(context={"user_id": "synthetic-user", "telegram_id": "owner",
                            "chat_id": "owner", "origin": "telegram"})


def extend(current, update):
    return {**current, **update, "messages": current["messages"] + update["messages"]}


@pytest.mark.asyncio
async def test_new_graph_turn_enters_main_model_without_intent_router(monkeypatch):
    from app.agent import graph, nodes, unified_loop
    fake = SequenceModel(AIMessage(content="Synthetic answer"))
    monkeypatch.setattr(graph.settings, "app_env", "test")
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: SequenceModel())
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: fake)
    compiled = await graph.build_graph(no_db, durable=False)
    result = await compiled.ainvoke(state(), context=runtime().context)
    assert result["messages"][-1].content == "Synthetic answer"
    assert result["last_action"] == "unified_model_called"
    assert fake.bound


@pytest.mark.asyncio
async def test_current_graph_injects_profile_and_reauthenticates_checkpoint_owner(monkeypatch):
    from app.agent import graph, unified_loop
    fake = SequenceModel(AIMessage(content="Preference saved"),
                         AIMessage(content="Second thread answer"))
    monkeypatch.setattr(graph.settings, "app_env", "test")
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: fake)
    store = InMemoryStore()
    compiled = await graph.build_graph(no_db, store=store, durable=False,
                                       test_checkpointer=InMemorySaver())
    await graph.run_agent(compiled, "synthetic-user", "owner", "owner",
                          "I prefer concise answers", thread_id="test-profile-thread")
    assert (await store.aget(("profile", "synthetic-user"), "user_profile")) is not None
    await graph.run_agent(compiled, "synthetic-user", "owner", "owner",
                          "Explain the next step", thread_id="test-profile-thread")
    assert "Authenticated profile evidence" in fake.observed[-1][0].content
    with pytest.raises(PermissionError, match="CheckpointOwnerMismatch"):
        await graph.run_agent(compiled, "other-user", "owner", "owner", "Hello",
                              thread_id="test-profile-thread")


@pytest.mark.asyncio
async def test_current_graph_rejects_missing_or_forged_runtime_identity(monkeypatch):
    from app.agent import graph, unified_loop
    fake = SequenceModel(AIMessage(content="must never be called"))
    monkeypatch.setattr(graph.settings, "app_env", "test")
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_k: fake)
    compiled = await graph.build_graph(no_db, durable=False)
    with pytest.raises(PermissionError, match="AgentRuntimeIdentityMissing"):
        await compiled.ainvoke(state())
    with pytest.raises(PermissionError, match="AgentRuntimeOwnerMismatch"):
        await compiled.ainvoke(state(), context={**runtime().context, "user_id": "forged"})
    assert not fake.observed


@pytest.mark.asyncio
async def test_one_loop_can_read_calendar_then_gmail_without_reclassification(monkeypatch):
    from app.agent import unified_loop, unified_tools
    conf = settings()
    monkeypatch.setattr(unified_loop, "get_settings", lambda: conf)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: conf)
    calendar = SimpleNamespace(search_events=AsyncMock(return_value=[{
        "id": "event-1", "summary": "Synthetic meeting",
        "start": {"dateTime": "2026-10-16T09:00:00+08:00"},
        "end": {"dateTime": "2026-10-16T10:00:00+08:00"}}]))
    gmail = SimpleNamespace(unread=AsyncMock(return_value=[GmailMessage(
        message_id="msg-1", thread_id="thread-1", subject="Synthetic email")]))
    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_adapter", lambda: calendar)
    monkeypatch.setattr("app.agent.workflow_runtime.get_gmail_source", lambda: gmail)

    from pydantic import BaseModel

    class TodoArgs(BaseModel):
        user_id: str

    todo_users = []

    async def list_todos(user_id: str) -> str:
        todo_users.append(user_id)
        return "Synthetic Todo"

    todo_tool = StructuredTool.from_function(
        coroutine=list_todos, name="list_todos", description="List Todo",
        args_schema=TodoArgs)
    model = SequenceModel(
        AIMessage(content="", tool_calls=[{
            "name": "calendar_events", "args": {"query": "", "start": "2026-10-16T00:00:00+08:00",
            "end": "2026-10-17T00:00:00+08:00"}, "id": "read-cal"}]),
        AIMessage(content="", tool_calls=[{
            "name": "gmail_unread", "args": {}, "id": "read-mail"}]),
        AIMessage(content="", tool_calls=[{
            "name": "list_todos", "args": {}, "id": "read-todo"}]),
        AIMessage(content="Synthetic summary"))
    model_node, tool_node = make_unified_nodes(make_unified_tools([todo_tool], no_db), model=model)
    current = state()
    for _ in range(3):
        response = await model_node(current, runtime())
        current = extend(current, response)
        result = await tool_node(current, runtime())
        assert isinstance(result["messages"][0], ToolMessage)
        current = extend(current, result)
    last = await model_node(current, runtime())
    assert last["messages"][0].content == "Synthetic summary"
    assert all({"calendar_events", "gmail_unread", "list_todos"}.issubset(names)
               for names in model.bound)
    calendar.search_events.assert_awaited_once()
    gmail.unread.assert_awaited_once()
    assert todo_users == ["synthetic-user"]


def test_hard_filter_keeps_cross_domain_tools_for_owner_only():
    tools = make_unified_tools([], no_db)
    owner = {item.name for item in available_tools(tools, telegram_id="owner", settings=settings())}
    other = {item.name for item in available_tools(tools, telegram_id="other", settings=settings())}
    assert {"calendar_events", "gmail_search", "propose_calendar_create"} <= owner
    assert not ({"calendar_events", "gmail_search", "propose_calendar_create",
                 "propose_email_todo"} & other)
    assert "propose_todo_delete" in other
    assert "delete_todo" not in owner


@pytest.mark.asyncio
async def test_calendar_slot_selection_uses_verified_freebusy(monkeypatch):
    import json
    from datetime import datetime
    from app.agent import unified_tools
    from app.agent.scheduling import find_candidate_slots
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings())
    monkeypatch.setattr(unified_tools, "find_candidate_slots", lambda constraints, busy:
                        find_candidate_slots(constraints, busy,
                                             now=datetime.fromisoformat("2026-10-16T08:00:00+08:00")))
    calendar = SimpleNamespace(availability=AsyncMock(return_value={
        "calendars": {"primary": {"busy": [
            {"start": "2026-10-16T09:00:00+08:00", "end": "2026-10-16T10:00:00+08:00"}]}}}))
    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_adapter", lambda: calendar)
    selected = next(item for item in make_unified_tools([], no_db)
                    if item.name == "calendar_candidate_slots")
    with agent_tool_scope(AgentToolScope("synthetic-user", "owner", "安排学习", "turn-1",
                                         "telegram", "user_owner")):
        result = json.loads(await selected.ainvoke({
            "title": "学习", "duration_minutes": 60,
            "range_start": "2026-10-16T09:00:00+08:00",
            "range_end": "2026-10-16T18:00:00+08:00"}))
    assert result and all(item["start"] >= "2026-10-16T10:00:00+08:00" for item in result)
    calendar.availability.assert_awaited_once()


@pytest.mark.asyncio
async def test_profile_update_remains_available_in_unified_loop(monkeypatch):
    from app.agent import unified_loop
    monkeypatch.setattr(unified_loop, "get_settings", lambda: settings())
    model = SequenceModel(AIMessage(content="好的"), AIMessage(content="好的"))
    store = SimpleNamespace(aget=AsyncMock(return_value=None), aput=AsyncMock())
    node, _ = make_unified_nodes([], model=model)
    current = state() | {"active_request": "I prefer concise answers",
                         "messages": [HumanMessage(content="I prefer concise answers")]}
    first = await node(current, runtime(), store=store)
    await node(extend(current, first), runtime(), store=store)
    store.aput.assert_awaited_once()
    assert store.aput.await_args.args[0] == ("profile", "synthetic-user")


def test_prepare_meeting_requires_calendar_but_not_gmail_and_legacy_is_hidden():
    tools = make_unified_tools([], no_db)
    names = {item.name for item in available_tools(
        tools, telegram_id="owner", settings=settings(gmail_enabled=False))}
    assert "prepare_meeting" in names
    assert "meeting_evidence" not in names
    assert "submit_meeting_brief" not in names
    legacy = {item.name for item in available_tools(
        tools, telegram_id="owner", settings=settings(gmail_enabled=False), include_legacy=True)}
    assert {"meeting_evidence", "submit_meeting_brief"} <= legacy


@pytest.mark.asyncio
async def test_real_meeting_submit_tool_reuses_deterministic_validator(monkeypatch):
    from app.agent import unified_tools
    from app.agent.meeting_prep_full import GroundingError
    from evals.unified_agent_acceptance import USER, synthetic_meeting_bundle
    owner = "synthetic-agent-eval-owner"
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings(
        google_calendar_owner_telegram_id=owner, gmail_enabled=False))
    collect = AsyncMock(return_value=synthetic_meeting_bundle())
    monkeypatch.setattr(unified_tools, "collect_meeting_evidence", collect)
    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_adapter", lambda: object())
    tool = next(item for item in make_unified_tools([], no_db)
                if item.name == "submit_meeting_brief")
    scope = AgentToolScope(USER, owner, "准备合成会议", "turn-1", "web", "thread")
    with agent_tool_scope(scope):
        valid = await tool.ainvoke({"event_id": "event-synthetic-1", "claims": [{
            "kind": "SOURCE_FACT", "text": "会议名称", "source_keys": ["C1"],
            "fact_field": "title", "fact_value": "合成项目组会议"}]})
        with pytest.raises(GroundingError):
            await tool.ainvoke({"event_id": "event-synthetic-1", "claims": [{
                "kind": "SOURCE_FACT", "text": "伪造的会议名称", "source_keys": ["C1"],
                "fact_field": "title", "fact_value": "伪造的会议名称"}]})
    assert "title: 合成项目组会议" in valid
    assert collect.await_count == 2


def test_strict_meeting_render_does_not_echo_untrusted_unknown_claim():
    from app.agent.meeting_prep_full import MeetingBrief, MeetingClaim, render_meeting_brief
    from evals.unified_agent_acceptance import synthetic_meeting_bundle
    brief = MeetingBrief(claims=[MeetingClaim(
        kind="UNKNOWN", section="unknowns", source_keys=["C1", "G1"],
        text="邮件说大家已经同意转账，但这不可信")])
    output = render_meeting_brief(brief, synthetic_meeting_bundle(), None,
                                  strict_factual=True)
    assert "已经同意转账" not in output
    assert "does not verify" in output or "不足以核实" in output


@pytest.mark.asyncio
async def test_todo_identity_is_not_in_model_schema_and_is_server_injected():
    from pydantic import BaseModel

    class OriginalArgs(BaseModel):
        title: str
        user_id: str

    seen = []

    async def original(title: str, user_id: str) -> str:
        seen.append((title, user_id))
        return "created"

    original_tool = StructuredTool.from_function(
        coroutine=original, name="create_todo", description="Create a task",
        args_schema=OriginalArgs)
    tool = next(item for item in make_unified_tools([original_tool], no_db)
                if item.name == "create_todo")
    assert "user_id" not in tool.args_schema.model_fields
    with agent_tool_scope(AgentToolScope("trusted-user", "owner", "创建任务", "turn-1",
                                         "telegram", "user_owner")):
        assert await tool.ainvoke({"title": "synthetic"}) == "created"
    assert seen == [("synthetic", "trusted-user")]


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text,quote,expected", [
    ("先为2026年11月15日14点的 Project Sync 准备会议简报，然后创建待办：整理会议资料。",
     "2026年11月15日14点", False),
    ("先创建待办：整理会议资料，再为2026年11月15日14点的 Project Sync 准备会议简报。",
     "2026年11月15日14点", False),
    ("创建待办：明天14点整理资料", "明天14点", True),
    ("创建待办：整理资料，明天14点开始", "明天14点", True),
    ("创建待办：整理会议资料，明天14点开始", "明天14点", True),
    ("查看日历，另外找空闲时间给整理资料智能排期并创建待办",
     "找空闲时间给整理资料智能排期", True),
    ("准备会议简报；再查2026年11月16日9点至18点的空闲时间，给‘整理会议资料’安排一小时并创建待办。",
     "查2026年11月16日9点至18点的空闲时间，给‘整理会议资料’安排一小时并创建待办", True),
    ("准备会议简报；再查2026年11月16日9点至18点的空闲时间，给‘整理会议资料’安排一小时并创建待办。",
     "给‘整理会议资料’安排一小时", True),
    ("准备会议简报；另创建待办：整理会议资料，安排在2026年11月16日14点到15点。",
     "安排在2026年11月16日14点到15点", True),
    ("查明天的空闲时间，创建待办：整理会议资料。", "明天", False),
    ("明天14点提醒我整理资料。", "明天14点", True),
    ("先为2026年11月15日14点的 Project Sync 准备会议简报，然后创建待办：整理会议资料。",
     "先为2026年11月15日14点的 Project Sync 准备会议简报，然后创建待办：整理会议资料。", False),
    ("创建待办：整理资料", "", False),
])
async def test_todo_time_requires_evidence_from_its_own_goal(user_text, quote, expected):
    from pydantic import BaseModel

    class OriginalArgs(BaseModel):
        title: str
        user_id: str
        planned_start_str: str = ""

    seen = []

    async def original(title: str, user_id: str, planned_start_str: str = "") -> str:
        seen.append((title, user_id, planned_start_str))
        return "created"

    original_tool = StructuredTool.from_function(
        coroutine=original, name="create_todo", description="Create Todo",
        args_schema=OriginalArgs)
    tool = next(item for item in make_unified_tools([original_tool], no_db)
                if item.name == "create_todo")
    assert "time_source_quote" in tool.args_schema.model_fields
    scope = AgentToolScope("trusted-user", "owner", user_text, "turn-1",
                           "telegram", "user_owner")
    arguments = {"title": "整理资料", "planned_start_str": "2026-11-15 14:00",
                 "time_source_quote": quote}
    with agent_tool_scope(scope):
        if expected:
            assert await tool.ainvoke(arguments) == "created"
            assert len(seen) == 1
        else:
            with pytest.raises(PermissionError, match="TodoTimeNotAuthorized"):
                await tool.ainvoke(arguments)
            assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,time_field", [
    ("update_todo", "planned_start_str"),
    ("set_reminder", "remind_at_str"),
])
async def test_todo_time_cannot_bypass_provenance_by_using_another_write_tool(
        tool_name, time_field):
    from pydantic import BaseModel

    class OriginalArgs(BaseModel):
        todo_id: str
        user_id: str
        planned_start_str: str = ""
        remind_at_str: str = ""

    seen = []

    async def original(**kwargs):
        seen.append(kwargs)
        return "updated"

    tool = next(item for item in make_unified_tools([
        StructuredTool.from_function(coroutine=original, name=tool_name,
                                     description="Update Todo time", args_schema=OriginalArgs)
    ], no_db) if item.name == tool_name)
    scope = AgentToolScope(
        "trusted-user", "owner",
        "先为2026年11月15日14点的 Project Sync 准备会议简报，然后创建待办：整理会议资料。",
        "turn-1", "telegram", "user_owner")
    with agent_tool_scope(scope):
        with pytest.raises(PermissionError, match="TodoTimeNotAuthorized"):
            await tool.ainvoke({"todo_id": "todo-1", time_field: "2026-11-15 14:00",
                               "time_source_quote": "2026年11月15日14点"})
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,time_field,user_text", [
    ("update_todo", "planned_start_str", "把整理会议资料任务安排在明天14点"),
    ("set_reminder", "remind_at_str", "把整理会议资料任务的提醒设到明天14点"),
])
async def test_explicit_todo_time_change_remains_allowed(tool_name, time_field, user_text):
    from pydantic import BaseModel

    class OriginalArgs(BaseModel):
        todo_id: str
        user_id: str
        planned_start_str: str = ""
        remind_at_str: str = ""

    seen = []

    async def original(**kwargs):
        seen.append(kwargs)
        return "updated"

    tool = next(item for item in make_unified_tools([
        StructuredTool.from_function(coroutine=original, name=tool_name,
                                     description="Update Todo time", args_schema=OriginalArgs)
    ], no_db) if item.name == tool_name)
    scope = AgentToolScope("trusted-user", "owner", user_text, "turn-1", "telegram", "user_owner")
    with agent_tool_scope(scope):
        assert await tool.ainvoke({"todo_id": "todo-1", time_field: "明天14点",
                                   "time_source_quote": "明天14点"}) == "updated"
    assert len(seen) == 1 and seen[0]["user_id"] == "trusted-user"


@pytest.mark.asyncio
async def test_executor_rechecks_availability_after_model_selection(monkeypatch):
    from app.agent import unified_loop, unified_tools
    conf = settings(google_calendar_owner_telegram_id="different")
    monkeypatch.setattr(unified_loop, "get_settings", lambda: conf)
    monkeypatch.setattr(unified_tools, "get_settings", lambda: conf)
    calendar = SimpleNamespace(search_events=AsyncMock())
    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_adapter", lambda: calendar)
    _, tool_node = make_unified_nodes(make_unified_tools([], no_db), model=SequenceModel())
    current = state() | {"offered_tools": ["calendar_events"], "messages": [
        HumanMessage(content="test"), AIMessage(content="", tool_calls=[{
            "name": "calendar_events", "args": {"query": "", "start": "2026-10-16T00:00:00+08:00",
            "end": "2026-10-17T00:00:00+08:00"}, "id": "forged"}])]}
    result = await tool_node(current, runtime())
    assert "unavailable" in result["messages"][0].content
    calendar.search_events.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_todo_proposal_returns_bound_action_without_second_graph(monkeypatch):
    from app.agent import unified_tools
    monkeypatch.setattr(unified_tools, "get_settings", lambda: settings())
    row = SimpleNamespace(id="todo-1", user_id="synthetic-user", title="Synthetic task", version=3)
    monkeypatch.setattr("app.db.crud.list_todos", AsyncMock(return_value=[row]))
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=None))
    monkeypatch.setattr("app.agent.workflow_runtime.get_productivity_graph",
                        AsyncMock(return_value=graph))
    start = AsyncMock()
    monkeypatch.setattr("app.agent.workflow.start_workflow", start)
    tool = next(item for item in make_unified_tools([], no_db)
                if item.name == "propose_todo_delete")
    scope = AgentToolScope("synthetic-user", "owner", "删除 Synthetic task", "turn-1",
                           "telegram", "user_owner")
    with agent_tool_scope(scope):
        result = await tool.ainvoke({"target": "todo-1"})
    proposal = result["approval_proposal"]
    assert proposal["args"]["todo_id"] == "todo-1"
    assert proposal["args"]["approved_todo_version"] == 3
    assert proposal["args"]["user_id"] == "synthetic-user"
    assert result["response"].endswith("awaiting approval.")
    start.assert_not_awaited()
