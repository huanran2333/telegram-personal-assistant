"""Full-graph regressions for multi-goal meeting work and per-call tool budget."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.actions import ActionProposal, ActionResult, ActionStatus, RiskLevel


@asynccontextmanager
async def no_db():
    yield object()


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.observed = []

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        self.observed.append(list(_messages))
        return self.responses.pop(0)


async def setup(monkeypatch, tools, model):
    from app.agent import graph, nodes, unified_loop
    from app.config import get_settings
    synthetic_settings = get_settings().model_copy(update={
        "google_calendar_enabled": True,
        "google_calendar_owner_telegram_id": "owner",
    })
    monkeypatch.setattr(graph.settings, "app_env", "test")
    monkeypatch.setattr(graph, "make_todo_tools", lambda _factory: [])
    monkeypatch.setattr(graph, "make_unified_tools", lambda *_a, **_kw: tools)
    monkeypatch.setattr(nodes, "_build_chat_llm", lambda *_a, **_kw: ScriptedModel())
    monkeypatch.setattr(unified_loop, "_build_chat_llm", lambda *_a, **_kw: model)
    monkeypatch.setattr(unified_loop, "get_settings", lambda: synthetic_settings)
    return await graph.build_graph(no_db, durable=False,
                                   test_checkpointer=InMemorySaver())


def call(name, index, **args):
    return {"name": name, "args": args, "id": f"call-{index}", "type": "tool_call"}


@pytest.mark.asyncio
async def test_composable_prepare_meeting_tool_returns_to_parent_loop(monkeypatch):
    """Architecture proof: one ordinary tool result cannot terminate the parent graph."""
    from app.agent import graph
    events = []

    async def prepare_meeting(event_id: str):
        events.append("prepare_meeting")
        return '{"draft_id":"synthetic-draft","status":"NEEDS_REVIEW","revision":1}'

    async def list_todos():
        events.append("list_todos")
        return '[{"title":"Synthetic task"}]'

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (prepare_meeting, list_todos)]
    model = ScriptedModel(
        AIMessage(content="", tool_calls=[call("prepare_meeting", 0, event_id="event-1")]),
        AIMessage(content="", tool_calls=[call("list_todos", 1)]),
        AIMessage(content="会议草稿待审核；Synthetic task 已列出。"))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "准备会议，然后列出待办",
        thread_id="web:ordinary:synthetic-user:composable-meeting",
        turn_id="web-request:composable-meeting", origin="web", return_state=True)
    assert events == ["prepare_meeting", "list_todos"]
    assert model.calls == 3
    assert [item.tool_call_id for item in state["messages"] if isinstance(item, ToolMessage)] == [
        "call-0", "call-1"]
    assert "待审核" in reply and "Synthetic task" in reply


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_cursor", [0, 1])
async def test_saved_legacy_meeting_calls_resume_without_reoffering_them(monkeypatch, saved_cursor):
    """Seed the actual pending-call Saver protocol, rather than exposing old tools to new requests."""
    events = []

    async def meeting_evidence(event_id: str):
        events.append("meeting_evidence")
        return "synthetic evidence"

    async def submit_meeting_brief(event_id: str, claims: list[str]):
        events.append("submit_meeting_brief")
        return "Verified legacy brief"

    async def list_todos():
        events.append("list_todos")
        return '[{"title":"Synthetic legacy task"}]'

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (meeting_evidence, submit_meeting_brief, list_todos)]
    model = ScriptedModel(AIMessage(content="legacy model final"))
    compiled = await setup(monkeypatch, tools, model)
    calls = [call("meeting_evidence", 0, event_id="event-1"),
             call("submit_meeting_brief", 1, event_id="event-1", claims=[]),
             call("list_todos", 2)]
    messages = [HumanMessage(content="legacy synthetic request"), AIMessage(content="", tool_calls=calls)]
    if saved_cursor:
        messages.append(ToolMessage(content="saved evidence", tool_call_id="call-0"))
    config = {"configurable": {"thread_id": f"web:ordinary:synthetic-user:legacy-{saved_cursor}"}}
    await compiled.aupdate_state(config, {
        "messages": messages, "user_id": "synthetic-user", "telegram_id": "owner",
        "chat_id": "", "conversation_thread_id": config["configurable"]["thread_id"],
        "active_turn_id": "legacy-turn", "active_request": "legacy synthetic request",
        "request_origin": "web", "pending_tool_calls": calls, "pending_tool_index": saved_cursor,
        "offered_tools": [item.name for item in tools], "unified_tool_rounds": saved_cursor,
        "meeting_evidence_pending": bool(saved_cursor),
    }, as_node="agent_tools" if saved_cursor else "agent_model")
    result = await compiled.ainvoke(None, config, context={
        "user_id": "synthetic-user", "telegram_id": "owner", "chat_id": "", "origin": "web"})
    assert events == (["meeting_evidence"] if not saved_cursor else []) + [
        "submit_meeting_brief", "list_todos"]
    assert [item.tool_call_id for item in result["messages"] if isinstance(item, ToolMessage)] == [
        "call-0", "call-1", "call-2"]
    assert result["unified_tool_rounds"] == 3
    assert "Verified legacy brief" in result["messages"][-1].content
    assert "Synthetic legacy task" in result["messages"][-1].content
    assert not {"meeting_evidence", "submit_meeting_brief"} & set(result["offered_tools"])


@pytest.mark.asyncio
async def test_prepared_meeting_does_not_skip_same_batch_todo(monkeypatch):
    from app.agent import graph
    events = []

    async def prepare_meeting(event_id: str):
        events.append("prepare_meeting")
        return '{"draft_id":"synthetic-draft","status":"NEEDS_REVIEW","revision":1}'

    async def list_todos():
        events.append("list_todos")
        return '[{"title":"Synthetic task"}]'

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (prepare_meeting, list_todos)]
    model = ScriptedModel(AIMessage(content="", tool_calls=[
        call("prepare_meeting", 0, event_id="event-1"),
        call("list_todos", 1)]), AIMessage(content="会议草稿待审核；Synthetic task 已列出。"))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "准备会议并列出待办",
        thread_id="web:ordinary:synthetic-user:p1-meeting", turn_id="web-request:p1-meeting",
        origin="web", return_state=True)
    assert events == ["prepare_meeting", "list_todos"]
    assert [item.tool_call_id for item in state["messages"] if isinstance(item, ToolMessage)] == [
        "call-0", "call-1"]
    assert "待审核" in reply
    assert "Synthetic task" in reply
    assert not state.get("meeting_evidence_pending")
    assert not state.get("validated_meeting_response")


@pytest.mark.asyncio
async def test_meeting_completion_allows_next_model_round_to_call_tool(monkeypatch):
    from app.agent import graph
    events = []

    async def prepare_meeting(event_id: str):
        events.append("prepare_meeting")
        return '{"draft_id":"synthetic-draft","status":"NEEDS_REVIEW","revision":1}'

    async def list_todos():
        events.append("list_todos")
        return '[{"title":"Synthetic task"}]'

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (prepare_meeting, list_todos)]
    model = ScriptedModel(
        AIMessage(content="", tool_calls=[call("prepare_meeting", 0, event_id="event-1")]),
        AIMessage(content="", tool_calls=[call("list_todos", 1)]),
        AIMessage(content="会议草稿待审核；Synthetic task 已列出。"))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "准备会议，然后列出待办",
        thread_id="web:ordinary:synthetic-user:p1-next", turn_id="web-request:p1-next",
        origin="web", return_state=True)
    assert events == ["prepare_meeting", "list_todos"]
    assert model.calls == 3
    assert "待审核" in reply
    assert "Synthetic task" in reply
    assert [item.tool_call_id for item in state["messages"] if isinstance(item, ToolMessage)] == [
        "call-0", "call-1"]


@pytest.mark.asyncio
async def test_eleven_batched_calls_execute_at_most_ten_and_all_have_tool_results(monkeypatch):
    from app.agent import graph
    events = []

    async def list_todos():
        events.append("list_todos")
        return "Synthetic result"

    tools = [StructuredTool.from_function(coroutine=list_todos, name="list_todos",
                                          description="synthetic")]
    model = ScriptedModel(AIMessage(content="", tool_calls=[call("list_todos", i)
                                                       for i in range(11)]))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "列出待办",
        thread_id="web:ordinary:synthetic-user:p1-budget", turn_id="web-request:p1-budget",
        origin="web", return_state=True)
    assert len(events) == 10
    tool_messages = [item for item in state["messages"] if isinstance(item, ToolMessage)]
    assert [item.tool_call_id for item in tool_messages] == [f"call-{i}" for i in range(11)]
    assert "not executed" in tool_messages[-1].content.lower()
    assert "未执行" in reply
    assert state["unified_tool_rounds"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_count", [0, 9, 10, 11])
async def test_budget_boundaries_in_full_graph(monkeypatch, tool_count):
    from app.agent import graph
    events = []

    async def list_todos():
        events.append("read")
        return "synthetic result"

    tools = [StructuredTool.from_function(coroutine=list_todos, name="list_todos",
                                          description="synthetic")]
    first = (AIMessage(content="done") if tool_count == 0 else
             AIMessage(content="", tool_calls=[call("list_todos", i)
                                                    for i in range(tool_count)]))
    model = ScriptedModel(first, AIMessage(content="done"))
    compiled = await setup(monkeypatch, tools, model)
    _, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "列出待办",
        thread_id=f"web:ordinary:synthetic-user:p1-boundary-{tool_count}",
        turn_id=f"web-request:p1-boundary-{tool_count}", origin="web", return_state=True)
    assert len(events) == min(tool_count, 10)
    assert state["unified_tool_rounds"] == min(tool_count, 10)
    assert state.get("budget_blocked_tool_calls", 0) == max(0, tool_count - 10)
    assert len([item for item in state["messages"] if isinstance(item, ToolMessage)]) == tool_count


@pytest.mark.asyncio
async def test_budget_reply_names_the_blocked_todo_without_executing_it(monkeypatch):
    from app.agent import graph
    created = []

    async def create_todo(title: str):
        created.append(title)
        return f"Created {title}"

    tools = [StructuredTool.from_function(coroutine=create_todo, name="create_todo",
                                          description="synthetic")]
    titles = [f"任务{i}" for i in range(10)] + ["准备发布说明"]
    model = ScriptedModel(AIMessage(content="", tool_calls=[
        call("create_todo", i, title=title) for i, title in enumerate(titles)]))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "请分别创建这 11 条待办：" + "、".join(titles),
        thread_id="web:ordinary:synthetic-user:p1-budget-title",
        turn_id="web-request:p1-budget-title", origin="web", return_state=True)
    assert created == titles[:10]
    assert state["budget_blocked_tool_calls"] == 1
    assert "准备发布说明" in reply and "未执行" in reply
    assert [item.tool_call_id for item in state["messages"] if isinstance(item, ToolMessage)] == [
        f"call-{i}" for i in range(11)]


def test_budget_summary_uses_only_paired_calls_in_current_user_turn():
    from app.agent.unified_loop import _blocked_call_summary, _BUDGET_BLOCKED_MESSAGE

    messages = [
        HumanMessage(content="previous request"),
        AIMessage(content="", tool_calls=[call("create_todo", 0, title="旧请求标题")]),
        ToolMessage(content=_BUDGET_BLOCKED_MESSAGE, tool_call_id="call-0"),
        HumanMessage(content="current request"),
        AIMessage(content="", tool_calls=[call("create_todo", 1, title="当前标题")]),
        ToolMessage(content=_BUDGET_BLOCKED_MESSAGE, tool_call_id="unpaired"),
        ToolMessage(content=_BUDGET_BLOCKED_MESSAGE, tool_call_id="call-1"),
    ]
    summary = _blocked_call_summary(messages)
    assert "当前标题" in summary
    assert "旧请求标题" not in summary
    assert "unpaired" not in summary


@pytest.mark.asyncio
async def test_over_budget_guarded_proposal_never_runs_or_interrupts(monkeypatch):
    from app.agent import graph
    reads, proposals = [], []

    async def list_todos():
        reads.append("read")
        return "synthetic result"

    async def propose_todo_delete(target: str):
        proposals.append(target)
        raise AssertionError("Over-budget proposal must never be invoked")

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (list_todos, propose_todo_delete)]
    model = ScriptedModel(AIMessage(content="", tool_calls=[
        *[call("list_todos", i) for i in range(10)],
        call("propose_todo_delete", 10, target="Synthetic task")]))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "查询并删除任务",
        thread_id="web:ordinary:synthetic-user:p1-guarded-block",
        turn_id="web-request:p1-guarded-block", origin="web", return_state=True)
    assert len(reads) == 10
    assert proposals == []
    assert not state.get("pending_agent_proposal")
    assert "__interrupt__" not in state
    assert state["budget_blocked_tool_calls"] == 1
    assert "未执行" in reply
    assert len([item for item in state["messages"] if isinstance(item, ToolMessage)]) == 11


@pytest.mark.asyncio
async def test_approval_resume_consumes_one_budget_slot_and_blocks_later_call(monkeypatch):
    from app.agent import graph, workflow_runtime
    reads, proposals = [], []
    proposal = ActionProposal(
        action_id="synthetic-budget-action", domain="todo", operation="delete_todo",
        args={"user_id": "synthetic-user", "todo_id": "synthetic-todo",
              "approved_todo_version": 1}, human_readable_preview="delete synthetic task",
        reason="synthetic", risk_level=RiskLevel.DESTRUCTIVE)

    async def list_todos():
        reads.append("read")
        return "synthetic result"

    async def propose_todo_delete(target: str):
        proposals.append(target)
        return {"approval_proposal": proposal.model_dump(), "response": "await approval"}

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (list_todos, propose_todo_delete)]
    calls = [*[call("list_todos", i) for i in range(9)],
             call("propose_todo_delete", 9, target="Synthetic task"),
             call("list_todos", 10)]
    model = ScriptedModel(AIMessage(content="", tool_calls=calls))
    compiled = await setup(monkeypatch, tools, model)
    executor = AsyncMock()
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    reply, pending = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "查询、删除、再查询",
        thread_id="web:ordinary:synthetic-user:p1-resume-budget",
        turn_id="web-request:p1-resume-budget", origin="web", return_state=True)
    assert pending["__interrupt__"][0].value["action_id"] == proposal.action_id
    assert pending["unified_tool_rounds"] == 10
    assert len(reads) == 9 and len(proposals) == 1
    result = await compiled.ainvoke(Command(resume={"decision": "reject"}),
                                    {"configurable": {"thread_id":
                                      "web:ordinary:synthetic-user:p1-resume-budget"}},
                                    context={"user_id": "synthetic-user", "telegram_id": "owner",
                                             "chat_id": "", "origin": "web"})
    assert len(reads) == 9 and len(proposals) == 1
    assert result["budget_blocked_tool_calls"] == 1
    assert result["unified_tool_rounds"] == 10
    assert [item.tool_call_id for item in result["messages"]
            if isinstance(item, ToolMessage)] == [f"call-{i}" for i in range(11)]
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_meeting_validation_failure_does_not_skip_unrelated_read(monkeypatch):
    from app.agent import graph
    from app.agent.meeting_prep_full import GroundingError
    events = []

    async def prepare_meeting(event_id: str):
        events.append("prepare_meeting")
        raise GroundingError("Factual claim lacks evidence")

    async def list_todos():
        events.append("list_todos")
        return '[{"title":"Synthetic task"}]'

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (prepare_meeting, list_todos)]
    model = ScriptedModel(AIMessage(content="", tool_calls=[
        call("prepare_meeting", 0, event_id="event-1"),
        call("list_todos", 1)]), AIMessage(content="会议草稿未生成，来源校验失败；Synthetic task 已列出。"))
    compiled = await setup(monkeypatch, tools, model)
    reply, state = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "准备会议并列出待办",
        thread_id="web:ordinary:synthetic-user:p1-meeting-fail",
        turn_id="web-request:p1-meeting-fail", origin="web", return_state=True)
    assert events == ["prepare_meeting", "list_todos"]
    assert "Synthetic task" in reply
    assert "来源校验失败" in reply
    assert "Meeting draft not generated" in next(
        item.content for item in state["messages"]
        if isinstance(item, ToolMessage) and item.tool_call_id == "call-0")
    assert not state.get("validated_meeting_response")


@pytest.mark.asyncio
async def test_duplicate_turn_does_not_reexecute_budgeted_tool_calls(monkeypatch):
    from app.agent import graph
    events = []

    async def list_todos():
        events.append("read")
        return "synthetic result"

    tools = [StructuredTool.from_function(coroutine=list_todos, name="list_todos",
                                          description="synthetic")]
    model = ScriptedModel(AIMessage(content="", tool_calls=[call("list_todos", i)
                                                       for i in range(11)]))
    compiled = await setup(monkeypatch, tools, model)
    kwargs = {"thread_id": "web:ordinary:synthetic-user:p1-duplicate",
              "turn_id": "web-request:p1-duplicate", "origin": "web", "return_state": True}
    _, first = await graph.run_agent(compiled, "synthetic-user", "owner", "", "查询", **kwargs)
    _, second = await graph.run_agent(compiled, "synthetic-user", "owner", "", "查询", **kwargs)
    assert len(events) == 10
    assert first["budget_blocked_tool_calls"] == second["budget_blocked_tool_calls"] == 1
    assert model.calls == 1


@pytest.mark.asyncio
async def test_meeting_followed_by_todo_write_keeps_both_results_in_final_reply(monkeypatch):
    from app.agent import graph
    writes = []

    async def prepare_meeting(event_id: str):
        return '{"draft_id":"synthetic-draft","status":"NEEDS_REVIEW","revision":1}'

    async def create_todo(title: str):
        writes.append(title)
        return f"✅ 已创建待办：{title}"

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (prepare_meeting, create_todo)]
    model = ScriptedModel(AIMessage(content="", tool_calls=[
        call("prepare_meeting", 0, event_id="event-1"),
        call("create_todo", 1, title="Synthetic followup")]),
        AIMessage(content="会议草稿待审核；已创建待办 Synthetic followup。"))
    compiled = await setup(monkeypatch, tools, model)
    reply, _ = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "准备会议，然后创建待办",
        thread_id="web:ordinary:synthetic-user:p2-write-summary",
        turn_id="web-request:p2-write-summary", origin="web", return_state=True)
    assert writes == ["Synthetic followup"]
    assert "待审核" in reply
    assert "已创建待办" in reply
    assert "Synthetic followup" in reply
    observed = [item.content for item in model.observed[-1] if isinstance(item, ToolMessage)]
    assert any("synthetic-draft" in str(item) for item in observed)
    assert any("Synthetic followup" in str(item) for item in observed)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision, expected_text", [
    ("reject", "已拒绝"), ("approve", "已执行")])
async def test_meeting_then_guarded_decision_reports_status_and_remaining_read(
        monkeypatch, decision, expected_text):
    from app.agent import graph, workflow_runtime
    proposal = ActionProposal(
        action_id="synthetic-meeting-delete", domain="todo", operation="delete_todo",
        args={"user_id": "synthetic-user", "todo_id": "synthetic-todo",
              "approved_todo_version": 1}, human_readable_preview="delete synthetic task",
        reason="synthetic", risk_level=RiskLevel.DESTRUCTIVE)
    events = []

    async def prepare_meeting(event_id: str):
        events.append("prepare_meeting")
        return '{"draft_id":"synthetic-draft","status":"NEEDS_REVIEW","revision":1}'

    async def propose_todo_delete(target: str):
        events.append("propose_todo_delete")
        return {"approval_proposal": proposal.model_dump(), "response": "await approval"}

    async def list_todos():
        events.append("list_todos")
        return '[{"title":"Synthetic remaining task"}]'

    tools = [StructuredTool.from_function(coroutine=fn, name=fn.__name__,
                                          description="synthetic") for fn in
             (prepare_meeting, propose_todo_delete, list_todos)]
    model = ScriptedModel(AIMessage(content="", tool_calls=[
        call("prepare_meeting", 0, event_id="event-1"),
        call("propose_todo_delete", 1, target="Synthetic task"),
        call("list_todos", 2)]), AIMessage(content=f"会议草稿待审核；删除{expected_text}；Synthetic remaining task 已列出。"))
    compiled = await setup(monkeypatch, tools, model)
    executor = AsyncMock()
    executor.execute.return_value = ActionResult(
        action_id=proposal.action_id, status=ActionStatus.EXECUTED,
        message="synthetic execution")
    monkeypatch.setattr(workflow_runtime, "get_action_executor", lambda: executor)
    _, pending = await graph.run_agent(
        compiled, "synthetic-user", "owner", "", "准备会议、删任务、列任务",
        thread_id="web:ordinary:synthetic-user:p2-meeting-approval",
        turn_id="web-request:p2-meeting-approval", origin="web", return_state=True)
    assert pending["__interrupt__"][0].value["action_id"] == proposal.action_id
    assert events == ["prepare_meeting", "propose_todo_delete"]
    result = await compiled.ainvoke(Command(resume={"decision": decision}),
                                    {"configurable": {"thread_id":
                                      "web:ordinary:synthetic-user:p2-meeting-approval"}},
                                    context={"user_id": "synthetic-user", "telegram_id": "owner",
                                             "chat_id": "", "origin": "web"})
    reply = result["messages"][-1].content
    assert events == ["prepare_meeting", "propose_todo_delete",
                      "list_todos"]
    assert "待审核" in reply
    assert "Synthetic remaining task" in reply
    assert expected_text in reply
    assert [item.tool_call_id for item in result["messages"] if isinstance(item, ToolMessage)] == [
        "call-0", "call-1", "call-2"]
    if decision == "approve":
        executor.execute.assert_awaited_once()
    else:
        executor.execute.assert_not_awaited()
