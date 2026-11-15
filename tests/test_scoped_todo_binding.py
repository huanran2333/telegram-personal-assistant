"""Production Todo binding is narrow without changing the typed workflow."""
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.agent import graph, nodes


class StubTool:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    async def ainvoke(self, _args):
        self.calls += 1
        return "ok"


class StubLLM:
    def __init__(self):
        self.bound_names = None

    def bind_tools(self, tools):
        self.bound_names = [tool.name for tool in tools]
        return self


@pytest.mark.asyncio
async def test_scoped_executor_rejects_unbound_calendar_tool():
    todo = StubTool("list_todos")
    calendar = StubTool("list-events")
    model = StubLLM()
    with patch.object(nodes, "_build_chat_llm", return_value=model):
        execute = nodes.make_nodes([todo, calendar], llm_tools=[todo])[3]
    assert model.bound_names == ["list_todos"]
    result = await execute({"messages": [AIMessage(content="", tool_calls=[{
        "name": "list-events", "args": {}, "id": "read-1", "type": "tool_call",
    }])], "telegram_id": "owner"})
    assert calendar.calls == 0
    assert "not available" in result["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_current_graph_has_no_legacy_intent_or_todo_binding_nodes(monkeypatch):
    todo = [StubTool(name) for name in
            ("create_todo", "list_todos", "update_todo", "delete_todo", "search_todos", "set_reminder")]
    monkeypatch.setattr(graph, "make_todo_tools", lambda _: todo)
    monkeypatch.setattr(graph, "make_unified_tools", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(graph, "make_unified_nodes", lambda *_args, **_kwargs: (
        lambda _state, _runtime: {}, lambda _state, _runtime: {}))
    monkeypatch.setattr(graph.settings, "app_env", "test")
    compiled = await graph.build_graph(None, durable=False)
    assert "agent_model" in compiled.nodes
    assert "agent_tools" in compiled.nodes
    assert "entry_dispatch" not in compiled.nodes
    assert "classify_intent" not in compiled.nodes
