"""Read tools are selected per request, owner-scoped, and never grant writes."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import nodes, read_tools
from app.agent.capability_policy import capability_for_route
from app.agent.planner import RoutedRequest
from app.agent.planner import validate_route_clarity


class ReadModel:
    def __init__(self):
        self.bound = []

    def bind_tools(self, tools):
        self.bound.append([tool.name for tool in tools])
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="", tool_calls=[{
            "name": "read_calendar_events", "args": {}, "id": "read-1", "type": "tool_call",
        }])


def settings(**overrides):
    return SimpleNamespace(default_timezone="UTC", google_calendar_enabled=True,
                           google_calendar_owner_telegram_id="owner",
                           gmail_enabled=True, gmail_owner_telegram_id="owner",
                           llm_provider="openai", llm_model="fake", **overrides)


def state(route, telegram_id="owner"):
    return {"messages": [HumanMessage(content="查看明天的日程")],
            "precomputed_route": route.model_dump(), "active_request": "查看明天的日程",
            "intent": route.intent, "user_id": "internal-user", "telegram_id": telegram_id,
            "read_tool_rounds": 0}


def test_read_modes_do_not_offer_action_capability():
    for intent in ("calendar_read", "email_search", "email_unread", "email_thread",
                   "email_draft", "email_cross_domain", "read_cross_domain"):
        assert capability_for_route(RoutedRequest(intent=intent)) is None
    for intent in ("calendar_create", "calendar_update", "calendar_delete",
                   "email_to_todo", "email_to_calendar", "smart_schedule"):
        assert capability_for_route(RoutedRequest(intent=intent)) == "prepare_action_request"
    assert capability_for_route(RoutedRequest(intent="meeting_prep")) == "prepare_meeting_brief"
    assert capability_for_route(RoutedRequest(intent="todo", todo_operation="delete")) == "prepare_action_request"


@pytest.mark.asyncio
async def test_cross_domain_answer_requires_every_requested_source(monkeypatch):
    fake_settings = settings()
    monkeypatch.setattr(nodes, "settings", fake_settings)

    class FinishingModel(ReadModel):
        async def ainvoke(self, _messages):
            return AIMessage(content="Both sources checked")

    model = FinishingModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: model)
    llm_node = nodes.make_nodes([], llm_tools=[], read_tools=list(read_tools.READ_TOOLS))[2]
    route = RoutedRequest(intent="email_cross_domain", email_query="subject:report",
                          email_calendar_check=True, date_reference="tomorrow")
    current = state(route)
    current["messages"].extend([AIMessage(content="", tool_calls=[{
        "name": "read_gmail_evidence", "args": {}, "id": "gmail-1", "type": "tool_call",
    }]), ToolMessage(content="A synthetic email", tool_call_id="gmail-1")])
    current["read_sources_seen"] = ["gmail"]
    incomplete = await llm_node(current)
    assert incomplete["last_action"] == "read_missing_source_blocked"
    current["read_sources_seen"] = ["gmail", "calendar"]
    complete = await llm_node(current)
    assert complete["messages"][0].content == "Both sources checked"


@pytest.mark.asyncio
async def test_cross_domain_refuses_partial_authorization_before_model_call(monkeypatch):
    fake_settings = settings()
    fake_settings.google_calendar_owner_telegram_id = "different-owner"
    monkeypatch.setattr(nodes, "settings", fake_settings)
    model = ReadModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: model)
    llm_node = nodes.make_nodes([], llm_tools=[], read_tools=list(read_tools.READ_TOOLS))[2]
    route = RoutedRequest(intent="email_cross_domain", email_query="subject:report",
                          email_calendar_check=True, date_reference="tomorrow")
    denied = await llm_node(state(route))
    assert denied["last_action"] == "read_scope_incomplete"
    assert model.bound == [[]]


def test_read_cross_domain_cannot_launder_a_mutation():
    read_route = RoutedRequest(intent="read_cross_domain", date_reference="tomorrow",
                               todo_operation="list")
    with pytest.raises(ValueError, match="must not include a mutation"):
        validate_route_clarity(read_route, "删除任务并查看明天的日程")
    with pytest.raises(ValueError, match="must not include a mutation"):
        validate_route_clarity(RoutedRequest(intent="read_cross_domain",
                                            date_reference="tomorrow",
                                            todo_operation="delete"),
                               "查看明天的日程和任务")


@pytest.mark.asyncio
async def test_calendar_read_uses_agent_tool_and_forwards_provider_result(monkeypatch):
    fake_settings = settings()
    monkeypatch.setattr(nodes, "settings", fake_settings)
    monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
    calendar = SimpleNamespace(search_events=AsyncMock(return_value=[{
        "id": "event-1", "summary": "Synthetic meeting",
        "start": {"dateTime": "2026-10-15T09:00:00+00:00"},
        "end": {"dateTime": "2026-10-15T10:00:00+00:00"},
    }]))
    monkeypatch.setattr("app.agent.workflow_runtime.get_calendar_adapter", lambda: calendar)
    model = ReadModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: model)
    llm_node, execute = nodes.make_nodes([], llm_tools=[], read_tools=list(read_tools.READ_TOOLS))[2:4]
    route = RoutedRequest(intent="calendar_read", date_reference="tomorrow", read_kind="events")
    current = state(route)
    first = await llm_node(current)
    assert model.bound[-1] == ["read_calendar_events", "read_calendar_availability"]
    result = await execute({**current, "messages": current["messages"] + first["messages"]})
    assert isinstance(result["messages"][0], ToolMessage)
    assert "Synthetic meeting" in result["messages"][0].content
    assert calendar.search_events.await_count == 1
    final = await llm_node({**current, "messages": current["messages"] + first["messages"] + result["messages"],
                            "read_tool_rounds": 1})
    assert final["messages"][0].content == result["messages"][0].content


@pytest.mark.asyncio
async def test_unauthorized_read_is_hidden_before_model_and_rechecked_on_execution(monkeypatch):
    fake_settings = settings()
    monkeypatch.setattr(nodes, "settings", fake_settings)
    monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
    model = ReadModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: model)
    llm_node = nodes.make_nodes([], llm_tools=[], read_tools=list(read_tools.READ_TOOLS))[2]
    route = RoutedRequest(intent="calendar_read", date_reference="tomorrow")
    result = await llm_node(state(route, telegram_id="other"))
    assert result["last_action"] == "capability_unavailable"
    assert model.bound == [[]]  # existing Todo binding only; no private read schema
    with read_tools.read_scope(route=route.model_dump(), request="明天日程",
                               user_id="internal-user", telegram_id="other"):
        with pytest.raises(PermissionError, match="CalendarReadUnavailable"):
            await read_tools.read_calendar_events.ainvoke({})


@pytest.mark.asyncio
async def test_gmail_read_retains_existing_evidence_validator(monkeypatch):
    fake_settings = settings()
    monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
    from app.agent.gmail import GmailMessage
    source = SimpleNamespace(search=AsyncMock(return_value=[GmailMessage(
        message_id="m1", thread_id="t1", subject="Synthetic report", body="prepare a report")]))
    monkeypatch.setattr("app.agent.workflow_runtime.get_gmail_source", lambda: source)
    synthesize = AsyncMock(return_value="Validated source: m1 / t1")
    monkeypatch.setattr("app.agent.email_workflow.synthesize_email", synthesize)
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: object())
    route = RoutedRequest(intent="email_search", email_query="subject:report")
    with read_tools.read_scope(route=route.model_dump(), request="找报告邮件",
                               user_id="internal-user", telegram_id="owner"):
        answer = await read_tools.read_gmail_evidence.ainvoke({})
    assert answer == "Validated source: m1 / t1"
    source.search.assert_awaited_once_with("subject:report", limit=5)
    synthesize.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_cross_domain_binds_only_read_tools(monkeypatch):
    fake_settings = settings()
    monkeypatch.setattr(nodes, "settings", fake_settings)
    model = ReadModel()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_args, **_kwargs: model)
    todo = [SimpleNamespace(name=name) for name in
            ("create_todo", "list_todos", "update_todo", "delete_todo", "search_todos")]
    llm_node = nodes.make_nodes(todo, llm_tools=todo,
                                read_tools=list(read_tools.READ_TOOLS))[2]
    route = RoutedRequest(intent="read_cross_domain", date_reference="tomorrow",
                          email_query="subject:report", todo_operation="list")
    await llm_node(state(route))
    assert set(model.bound[-1]) == {"read_calendar_events", "read_calendar_availability",
                                    "read_gmail_evidence", "list_todos", "search_todos"}
