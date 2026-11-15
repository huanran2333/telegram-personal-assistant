"""Installed-version contract tests for replacing the custom Agent tool loop.

Synthetic tools only: these tests never access project data or providers.
"""
from __future__ import annotations

import asyncio
from typing import Any, NotRequired, TypedDict

import pytest

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.prebuilt import ToolRuntime
from pydantic import PrivateAttr


class Identity(TypedDict):
    user_id: str
    allow_write: NotRequired[bool]


class ScriptedModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _received: list[list] = PrivateAttr(default_factory=list)
    _bound_tool_names: list[list[str]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage]):
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "synthetic_official_agent_contract"

    def bind_tools(self, tools, **kwargs):
        self._bound_tool_names.append([tool.name for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])


@tool
async def owner_scoped_read(value: str, runtime: ToolRuntime[Identity]) -> str:
    """Return a synthetic value under the server-provided owner identity."""
    return f"{runtime.context['user_id']}:{value}"


_synthetic_writes: list[tuple[str, str]] = []


@tool
async def owner_scoped_write(value: str, runtime: ToolRuntime[Identity]) -> str:
    """Record a synthetic write after the caller approves it."""
    _synthetic_writes.append((runtime.context["user_id"], value))
    return f"saved:{value}"


class PreparedState(AgentState[Any]):
    prepared_actions: NotRequired[dict[str, dict[str, Any]]]


class ExactTargetPreflight(AgentMiddleware[PreparedState, Identity, Any]):
    """The business preflight that standard HITL does not provide itself."""

    state_schema = PreparedState

    async def aafter_model(self, state: PreparedState, runtime) -> dict | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        proposals = {}
        for call in last.tool_calls:
            if call["name"] == "exact_write":
                # The synthetic authoritative business record is version 3;
                # the model's version argument is deliberately ignored.
                proposals[call["id"]] = {
                    "owner": runtime.context["user_id"],
                    "target_id": "todo-1", "title": "Synthetic Todo", "version": 3,
                }
        return {"prepared_actions": proposals} if proposals else None


class OwnerToolPermission(AgentMiddleware[AgentState[Any], Identity, Any]):
    """Visible tools are a model hint; this also enforces the server policy."""

    async def awrap_model_call(self, request, handler):
        tools = [t for t in request.tools if t.name != "owner_scoped_write" or
                 request.runtime.context.get("allow_write", False)]
        return await handler(request.override(tools=tools))

    async def awrap_tool_call(self, request, handler):
        if (request.tool_call["name"] == "owner_scoped_write" and
                not request.runtime.context.get("allow_write", False)):
            return ToolMessage(content="Not authorized; nothing executed.",
                               tool_call_id=request.tool_call["id"], status="error")
        return await handler(request)


@tool
async def exact_write(target_id: str, version: int, runtime: ToolRuntime[Identity, PreparedState]) -> str:
    """Execute a synthetic owner/version-bound operation after review."""
    proposal = runtime.state["prepared_actions"][runtime.tool_call_id]
    if proposal["owner"] != runtime.context["user_id"] or proposal["target_id"] != target_id:
        raise PermissionError("TargetOwnerMismatch")
    if proposal["version"] != 3:
        raise ValueError("TargetVersionChanged")
    _synthetic_writes.append((proposal["owner"], proposal["title"]))
    return "executed:todo-1:version-3"


def _graph(responses: list[AIMessage], *, approval: bool = False):
    middleware = [ToolCallLimitMiddleware(run_limit=10, exit_behavior="continue")]
    if approval:
        middleware.insert(0, HumanInTheLoopMiddleware(
            interrupt_on={"owner_scoped_write": {"allowed_decisions": ["approve", "reject"]}}))
    return create_agent(
        model=ScriptedModel(responses), tools=[owner_scoped_read, owner_scoped_write],
        middleware=middleware, context_schema=Identity, checkpointer=InMemorySaver())


def test_official_tool_runtime_hides_identity_and_pairs_tool_result():
    async def run():
        call = {"name": "owner_scoped_read", "args": {"value": "ok"}, "id": "read-1"}
        graph = _graph([AIMessage(content="", tool_calls=[call]), AIMessage(content="finished")])
        assert "user_id" not in owner_scoped_read.tool_call_schema.model_fields
        result = await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="read")]},
            config={"configurable": {"thread_id": "official-read"}},
            context={"user_id": "server-owner"}), timeout=10)
        replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(replies) == 1
        assert replies[0].tool_call_id == "read-1"
        assert replies[0].content == "server-owner:ok"
    asyncio.run(run())


def test_official_hitl_reject_and_resume_keep_original_tool_call_id():
    async def run():
        _synthetic_writes.clear()
        calls = [
            {"name": "owner_scoped_read", "args": {"value": "before"}, "id": "read-before"},
            {"name": "owner_scoped_write", "args": {"value": "secret"}, "id": "review-1"},
            {"name": "owner_scoped_read", "args": {"value": "after"}, "id": "read-after"},
        ]
        graph = _graph([AIMessage(content="", tool_calls=calls), AIMessage(content="not executed")], approval=True)
        config = {"configurable": {"thread_id": "official-review"}}
        await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="read")]}, config=config,
            context={"user_id": "server-owner"}), timeout=10)
        paused = await graph.aget_state(config)
        assert paused.interrupts
        assert not any(isinstance(item, ToolMessage) for item in paused.values["messages"])
        assert _synthetic_writes == []
        result = await asyncio.wait_for(graph.ainvoke(
            Command(resume={"decisions": [{"type": "reject"}]}), config=config,
            context={"user_id": "server-owner"}), timeout=10)
        replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert {m.tool_call_id for m in replies} == {c["id"] for c in calls}
        rejected = next(m for m in replies if m.tool_call_id == "review-1")
        assert rejected.status == "error"
        assert "not executed" in rejected.content.lower()
        assert _synthetic_writes == []
    asyncio.run(run())


def test_official_hitl_reviews_two_guarded_calls_before_either_executes():
    async def run():
        _synthetic_writes.clear()
        calls = [
            {"name": "owner_scoped_write", "args": {"value": "first"}, "id": "write-1"},
            {"name": "owner_scoped_write", "args": {"value": "second"}, "id": "write-2"},
        ]
        graph = _graph([AIMessage(content="", tool_calls=calls),
                        AIMessage(content="one executed")], approval=True)
        config = {"configurable": {"thread_id": "official-review-batch"}}
        await graph.ainvoke({"messages": [HumanMessage(content="two operations")]},
                            config=config, context={"user_id": "server-owner"})
        paused = await graph.aget_state(config)
        assert len(paused.interrupts[0].value["action_requests"]) == 2
        assert _synthetic_writes == []
        result = await graph.ainvoke(Command(resume={"decisions": [
            {"type": "approve"}, {"type": "reject"}]}),
            config=config, context={"user_id": "server-owner"})
        assert _synthetic_writes == [("server-owner", "first")]
        replies = {item.tool_call_id: item for item in result["messages"]
                   if isinstance(item, ToolMessage)}
        assert set(replies) == {"write-1", "write-2"}
        assert replies["write-2"].status == "error"

    asyncio.run(run())


def test_batched_hitl_cannot_resume_two_calls_with_one_button_decision():
    async def run():
        _synthetic_writes.clear()
        calls = [
            {"name": "owner_scoped_write", "args": {"value": "first"}, "id": "write-1"},
            {"name": "owner_scoped_write", "args": {"value": "second"}, "id": "write-2"},
        ]
        graph = _graph([AIMessage(content="", tool_calls=calls),
                        AIMessage(content="finished")], approval=True)
        config = {"configurable": {"thread_id": "batched-single-button-rejected"}}
        await graph.ainvoke({"messages": [HumanMessage(content="two operations")]},
                            config=config, context={"user_id": "server-owner"})
        with pytest.raises(ValueError, match="Number of human decisions"):
            await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}),
                                config=config, context={"user_id": "server-owner"})
        assert _synthetic_writes == []

    asyncio.run(run())


def test_official_limit_blocks_eleventh_call_before_execution():
    async def run():
        _synthetic_writes.clear()
        calls = [{"name": "owner_scoped_write", "args": {"value": str(i)}, "id": f"w-{i}"}
                 for i in range(11)]
        graph = _graph([AIMessage(content="", tool_calls=calls), AIMessage(content="partial")])
        result = await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="write eleven")]},
            config={"configurable": {"thread_id": "official-budget"}},
            context={"user_id": "server-owner"}), timeout=10)
        replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert {m.tool_call_id for m in replies} == {c["id"] for c in calls}
        assert _synthetic_writes == [("server-owner", str(i)) for i in range(10)]
        assert next(m for m in replies if m.tool_call_id == "w-10").status == "error"
    asyncio.run(run())


def test_official_limit_continues_across_hitl_resume_for_same_user_turn():
    async def run():
        _synthetic_writes.clear()
        batch = [{"name": "owner_scoped_read", "args": {"value": str(i)}, "id": f"read-{i}"}
                 for i in range(9)]
        batch.append({"name": "owner_scoped_write", "args": {"value": "approved"},
                      "id": "approval-10"})
        model = ScriptedModel([
            AIMessage(content="", tool_calls=batch),
            AIMessage(content="", tool_calls=[{
                "name": "owner_scoped_write", "args": {"value": "eleventh"},
                "id": "blocked-11"}]),
            AIMessage(content="partial"),
        ])
        seen_reviews = []
        def inspect_review(request):
            seen_reviews.append((request.tool_call["id"],
                                 request.state.get("thread_tool_call_count"),
                                 [item.tool_call_id for item in request.state["messages"]
                                  if isinstance(item, ToolMessage)]))
            return not any(
                isinstance(item, ToolMessage)
                and item.tool_call_id == request.tool_call["id"]
                for item in request.state["messages"]
            )
        graph = create_agent(
            model=model, tools=[owner_scoped_read, owner_scoped_write],
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"owner_scoped_write": {
                            "allowed_decisions": ["approve", "reject"],
                            "when": inspect_review}}),
                        ToolCallLimitMiddleware(thread_limit=10, exit_behavior="continue")],
            context_schema=Identity, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "budget-across-approval"}}
        await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="do ten then one")],
             "thread_tool_call_count": {"__all__": 0}},
            config=config, context={"user_id": "server-owner"}), 10)
        assert (await graph.aget_state(config)).interrupts
        result = await asyncio.wait_for(graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config, context={"user_id": "server-owner"}), 10)
        resumed = await graph.aget_state(config)
        outcomes = [(item.tool_call_id, item.content) for item in resumed.values["messages"]
                    if isinstance(item, ToolMessage)]
        assert not resumed.interrupts, (
            f"Budget-blocked call asked for approval; counts={resumed.values.get('thread_tool_call_count')}; "
            f"results={outcomes}; review_inputs={seen_reviews}")
        replies = [item for item in result["messages"] if isinstance(item, ToolMessage)]
        assert {item.tool_call_id for item in replies} == {call["id"] for call in batch} | {"blocked-11"}
        assert _synthetic_writes == [("server-owner", "approved")]
        assert next(item for item in replies if item.tool_call_id == "blocked-11").status == "error"
    asyncio.run(run())


def test_business_preflight_is_visible_to_official_hitl_and_tool_runtime():
    async def run():
        _synthetic_writes.clear()
        model = ScriptedModel([
            AIMessage(content="", tool_calls=[{
                "name": "exact_write", "args": {"target_id": "todo-1", "version": 999},
                "id": "exact-1"}]),
            AIMessage(content="done"),
        ])
        hitl = HumanInTheLoopMiddleware(interrupt_on={"exact_write": {
            "allowed_decisions": ["approve", "reject"],
            "description": lambda call, state, runtime: (
                f"Delete {state['prepared_actions'][call['id']]['title']} "
                f"version {state['prepared_actions'][call['id']]['version']}")}})
        graph = create_agent(
            model=model, tools=[exact_write], middleware=[hitl, ExactTargetPreflight()],
            context_schema=Identity, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "official-exact"}}
        await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="delete the Todo")]}, config=config,
            context={"user_id": "server-owner"}), timeout=10)
        paused = await graph.aget_state(config)
        request = paused.interrupts[0].value
        assert "Synthetic Todo version 3" in request["action_requests"][0]["description"]
        assert _synthetic_writes == []
        result = await asyncio.wait_for(graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config=config,
            context={"user_id": "server-owner"}), timeout=10)
        assert _synthetic_writes == [("server-owner", "Synthetic Todo")]
        reply = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert reply.tool_call_id == "exact-1"
        assert reply.content == "executed:todo-1:version-3"
    asyncio.run(run())


def test_official_middleware_hides_tool_and_rechecks_at_execution():
    async def run():
        _synthetic_writes.clear()
        call = {"name": "owner_scoped_write", "args": {"value": "forged"}, "id": "forged-1"}
        model = ScriptedModel([AIMessage(content="", tool_calls=[call]), AIMessage(content="denied")])
        graph = create_agent(
            model=model, tools=[owner_scoped_read, owner_scoped_write],
            middleware=[OwnerToolPermission()], context_schema=Identity,
            checkpointer=InMemorySaver())
        result = await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="read only")]},
            config={"configurable": {"thread_id": "official-permission"}},
            context={"user_id": "server-owner", "allow_write": False}), timeout=10)
        assert model._bound_tool_names[0] == ["owner_scoped_read"]
        assert _synthetic_writes == []
        reply = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert reply.tool_call_id == "forged-1"
        assert reply.status == "error"
    asyncio.run(run())


def test_official_tool_node_does_not_serialize_a_same_batch_dependency():
    """Document the behavior that prevents an untested drop-in graph cutover."""
    async def run():
        first_started = asyncio.Event()
        second_started = asyncio.Event()

        @tool
        async def first_step() -> str:
            """Perform the first synthetic step."""
            first_started.set()
            await asyncio.wait_for(second_started.wait(), timeout=2)
            return "first-finished"

        @tool
        async def second_step() -> str:
            """Perform the second synthetic step."""
            second_started.set()
            return "second-finished"

        model = ScriptedModel([
            AIMessage(content="", tool_calls=[
                {"name": "first_step", "args": {}, "id": "first"},
                {"name": "second_step", "args": {}, "id": "second"},
            ]),
            AIMessage(content="finished"),
        ])
        graph = create_agent(model=model, tools=[first_step, second_step],
                             checkpointer=InMemorySaver())
        result = await asyncio.wait_for(graph.ainvoke(
            {"messages": [HumanMessage(content="first, then second")]},
            config={"configurable": {"thread_id": "official-parallel-batch"}}), timeout=5)
        assert first_started.is_set() and second_started.is_set()
        assert {message.tool_call_id for message in result["messages"]
                if isinstance(message, ToolMessage)} == {"first", "second"}

    asyncio.run(run())
