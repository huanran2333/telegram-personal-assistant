"""Production nodes consume one scoped context contract without authority drift."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent import nodes
from app.agent.memory_context import ContextBundle


class MemoryStub:
    def __init__(self):
        self.requests = []

    async def assemble(self, request):
        self.requests.append(request)
        return ContextBundle(
            current_request=request.request,
            recent_messages=(request.messages[-1],), summary="Older project code AX-4731",
            history=(), structured=(), provenance=(), estimated_tokens=120,
            token_budget=request.token_budget, degraded=(), summary_updated=False)


class LLMStub:
    def __init__(self):
        self.calls = []

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return nodes.ChatResponse(content="AX-4731")


class StoreStub:
    async def aget(self, *_args):
        return None


class ProfileStoreStub:
    async def aget(self, *_args):
        return SimpleNamespace(value={"preferences": ["short answers"]})


@pytest.mark.asyncio
async def test_chat_uses_orchestrated_evidence_once_with_profile_authority(monkeypatch):
    memory, model = MemoryStub(), LLMStub()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    chat = nodes.make_nodes([], memory_orchestrator=memory)[5]
    state = {"user_id": "owner", "telegram_id": "777",
             "entities": {"memory_search_query": "Aurora project code",
                          "memory_search_purpose": "exact"},
             "messages": [HumanMessage(id="current", content="What was its code?")]}
    result = await chat(state, StoreStub())
    assert result["messages"][0].content == "AX-4731"
    assert len(memory.requests) == 1
    assert memory.requests[0].thread_id == "user_777"
    assert memory.requests[0].search_query == "Aurora project code"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert "Profile Store" in model.calls[0][0].content
    assert sum(isinstance(item, SystemMessage) for item in model.calls[0]) == 1
    assert [item.content for item in model.calls[0] if isinstance(item, HumanMessage)] == ["What was its code?"]


@pytest.mark.asyncio
async def test_todo_reuses_evidence_across_tool_loop_without_extra_search(monkeypatch):
    memory = MemoryStub()

    class TodoLLM(LLMStub):
        async def ainvoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="Done")

    model = TodoLLM()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    todo = nodes.make_nodes([], memory_orchestrator=memory)[2]
    state = {"user_id": "owner", "telegram_id": "777", "intent": "search",
             "entities": {"memory_search_query": "old internship task",
                          "memory_search_purpose": "topic"},
             "messages": [HumanMessage(id="current", content="Find that old internship task")],
             "memory_context_prepared": False, "memory_context_payload": ""}
    first = await todo(state)
    assert len(memory.requests) == 1
    state.update(first)
    state["messages"] = [*state["messages"], first["messages"][0]]
    await todo(state)
    assert len(memory.requests) == 1
    assert all(any("Older project code" in str(message.content) for message in call)
               for call in model.calls)


@pytest.mark.asyncio
async def test_todo_agent_can_request_scoped_historical_search_then_use_result(monkeypatch):
    memory = MemoryStub()

    class SearchLLM(LLMStub):
        def bind_tools(self, tools):
            self.offered = [tool.name for tool in tools]
            return self

        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "search_memory", "args": {
                        "query": "old Aurora project code", "purpose": "exact"},
                    "id": "search-1"}])
            return AIMessage(content="The old code was AX-4731.")

    model = SearchLLM()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    todo, execute = nodes.make_nodes([], memory_orchestrator=memory)[2:4]
    state = {"user_id": "owner", "telegram_id": "777", "intent": "search",
             "entities": {}, "messages": [HumanMessage(id="current", content="What was the old code?")],
             "memory_context_prepared": False, "memory_context_payload": "",
             "memory_search_rounds": 0}
    first = await todo(state, ProfileStoreStub())
    assert "search_memory" in model.offered
    assert "short answers" in str(model.calls[0][0].content)
    state = {**state, **first, "messages": [*state["messages"], first["messages"][0]]}
    searched = await execute(state)
    assert searched["memory_search_rounds"] == 1
    assert isinstance(searched["messages"][0], ToolMessage)
    assert "Older project code" in searched["messages"][0].content
    assert memory.requests[-1].user_id == "owner"
    assert memory.requests[-1].thread_id == "user_777"
    assert memory.requests[-1].search_query == "old Aurora project code"
    assert memory.requests[-1].allow_summary_update is False
    state = {**state, **searched, "messages": [*state["messages"], *searched["messages"]]}
    answer = await todo(state, ProfileStoreStub())
    assert answer["messages"][0].content == "The old code was AX-4731."
    assert len(memory.requests) == 2  # initial context, then one explicit search


@pytest.mark.asyncio
async def test_memory_search_budget_blocks_stale_extra_tool_call(monkeypatch):
    memory, model = MemoryStub(), LLMStub()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    execute = nodes.make_nodes([], memory_orchestrator=memory)[3]
    state = {"user_id": "owner", "telegram_id": "777", "intent": "search",
             "entities": {}, "messages": [HumanMessage(id="current", content="Find an old fact")],
             "memory_search_rounds": 2}
    result = await execute({**state, "messages": [*state["messages"], AIMessage(
        content="", tool_calls=[{"name": "search_memory", "args": {
            "query": "old fact"}, "id": "search-stale"}])]})
    assert len(memory.requests) == 0
    assert "limit reached" in result["messages"][0].content.lower()


def test_memory_tool_is_disclosed_only_for_historical_request():
    assert not nodes.memory_search_relevant({
        "messages": [HumanMessage(content="创建任务：整理资料")], "entities": {}})
    assert nodes.memory_search_relevant({
        "messages": [HumanMessage(content="按我们之前聊过的代号创建待办")], "entities": {}})
    assert nodes.memory_search_relevant({
        "messages": [HumanMessage(content="Create a note")],
        "entities": {"memory_search_query": "old project code"}})


@pytest.mark.asyncio
async def test_memory_tool_rejects_model_supplied_user_scope(monkeypatch):
    memory, model = MemoryStub(), LLMStub()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    execute = nodes.make_nodes([], memory_orchestrator=memory)[3]
    result = await execute({
        "user_id": "owner", "telegram_id": "777", "intent": "search",
        "messages": [HumanMessage(id="current", content="What was the old code?"),
                     AIMessage(content="", tool_calls=[{
                         "name": "search_memory", "args": {
                             "query": "old code", "user_id": "other-user"},
                         "id": "forged"}])],
        "memory_search_rounds": 0,
    })
    assert not memory.requests
    assert "ToolOutcomeUncertain" not in result["messages"][0].content


@pytest.mark.asyncio
async def test_memory_tool_reuses_identical_prefetch_instead_of_searching_twice(monkeypatch):
    memory, model = MemoryStub(), LLMStub()
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_k: model)
    execute = nodes.make_nodes([], memory_orchestrator=memory)[3]
    result = await execute({
        "user_id": "owner", "telegram_id": "777", "intent": "search",
        "entities": {"memory_search_query": "Aurora old code",
                     "memory_search_purpose": "exact"},
        "memory_context_payload": "Source-backed prefetched code AX-4731",
        "messages": [HumanMessage(id="current", content="Use the old Aurora code"),
                     AIMessage(content="", tool_calls=[{
                         "name": "search_memory", "args": {
                             "query": "Aurora old code", "purpose": "exact"},
                         "id": "reuse"}])],
        "memory_search_rounds": 0,
    })
    assert result["messages"][0].content == "Source-backed prefetched code AX-4731"
    assert not memory.requests
