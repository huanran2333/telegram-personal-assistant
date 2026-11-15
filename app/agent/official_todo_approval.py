"""Official HITL with project-owned Todo preflight and execution contracts."""
from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from app.agent.actions import ActionProposal, ActionStatus
from app.agent.agent_approval import execute_approved_proposal
from app.agent.state import require_agent_runtime_identity
from app.agent.unified_tools import prepare_todo_delete


class ApprovalState(AgentState[Any]):
    user_id: str
    telegram_id: str
    chat_id: str
    request_origin: str
    prepared_actions: NotRequired[dict[str, dict[str, Any]]]


def _turn_id(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.id or message.content)
    raise PermissionError("AgentActiveRequestMissing")


def _prepared(state: dict, call_id: str) -> dict:
    item = (state.get("prepared_actions") or {}).get(call_id)
    if not isinstance(item, dict):
        raise PermissionError("ApprovalPreflightMissing")
    return item


def _already_answered(state: dict, call_id: str) -> bool:
    """A budget/permission middleware may have paired this call already."""
    return any(isinstance(message, ToolMessage) and message.tool_call_id == call_id
               for message in state.get("messages", []))


def needs_todo_review(request) -> bool:
    """Official HITL `when`: invalid/repeated calls never ask for approval."""
    if _already_answered(request.state, request.tool_call["id"]):
        return False
    return bool(_prepared(request.state, request.tool_call["id"]).get("proposal"))


def todo_review_description(call, state, _runtime) -> str:
    item = _prepared(state, call["id"])
    proposal = ActionProposal.model_validate(item["proposal"])
    return f"{proposal.human_readable_preview} (action_id={proposal.action_id})"


class TodoApprovalPreflight(AgentMiddleware[ApprovalState, dict, Any]):
    """Only exact owner/version binding and same-turn rejection are custom.

    The official middleware performs interrupt/resume; ToolNode executes the
    approved call. `prepared_actions` is the one checkpointed handoff value
    needed between those two framework phases, replacing the old cursor and
    approval-decision graph fields.
    """

    state_schema = ApprovalState

    def __init__(self, session_factory):
        super().__init__()
        self.session_factory = session_factory

    async def aafter_model(self, state: ApprovalState, runtime) -> dict | None:
        last = next((message for message in reversed(state["messages"])
                     if isinstance(message, AIMessage)), None)
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return None
        owner = require_agent_runtime_identity(state, runtime)["user_id"]
        turn = _turn_id(state["messages"])
        old = {call_id: item for call_id, item in
               (state.get("prepared_actions") or {}).items()
               if isinstance(item, dict) and item.get("turn_id") == turn}
        rejected_ids = {message.tool_call_id for message in state["messages"]
                        if isinstance(message, ToolMessage) and message.status == "error"
                        and "User rejected the tool call" in str(message.content)}
        rejected_targets = {item.get("target_key") for call_id, item in old.items()
                            if call_id in rejected_ids}
        proposed_targets: set[str] = set()
        for call in last.tool_calls:
            if call["name"] != "propose_todo_delete" or _already_answered(state, call["id"]):
                continue
            try:
                proposal = await prepare_todo_delete(
                    self.session_factory, user_id=owner,
                    target=str((call.get("args") or {}).get("target", "")))
                target_key = f"delete_todo:{proposal.args['todo_id']}"
                if target_key in rejected_targets or target_key in proposed_targets:
                    old[call["id"]] = {"turn_id": turn, "target_key": target_key,
                                       "error": "This exact Todo deletion was already proposed "
                                                "in this request; do not request another approval."}
                else:
                    old[call["id"]] = {"turn_id": turn, "target_key": target_key,
                                       "proposal": proposal.model_dump(mode="json")}
                    proposed_targets.add(target_key)
            except ValueError:
                old[call["id"]] = {"turn_id": turn,
                                   "error": "Todo target is missing or not unique; nothing executed."}
        return {"prepared_actions": old}

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] != "propose_todo_delete":
            return await handler(request)
        item = _prepared(request.state, request.tool_call["id"])
        if item.get("error"):
            return ToolMessage(content=item["error"],
                               tool_call_id=request.tool_call["id"], status="error")
        return await handler(request)


def make_approved_todo_delete_tool():
    @tool
    async def propose_todo_delete(target: str, runtime: ToolRuntime[dict, ApprovalState]) -> ToolMessage:
        """Start the formal approval for deleting one uniquely resolved owned Todo.

        Use this when the user asks to delete but says to confirm with them first.
        A text-only question is not an approval; this tool pauses before deletion
        and executes only after the official approval is granted.
        """
        item = _prepared(runtime.state, runtime.tool_call_id)
        proposal = ActionProposal.model_validate(item["proposal"])
        identity = require_agent_runtime_identity(runtime.state, runtime)
        if item.get("turn_id") != _turn_id(runtime.state["messages"]):
            raise PermissionError("AgentApprovalTurnMismatch")
        result = await execute_approved_proposal(proposal, identity)
        if result.action_id != proposal.action_id:
            raise RuntimeError("ActionExecutionIdentityMismatch")
        if result.status == ActionStatus.EXECUTED:
            return ToolMessage(
                content=(f"Approved action executed: action_id={proposal.action_id}; "
                         f"Todo={proposal.args['title']}; status=EXECUTED. "
                         "It is no longer waiting for approval."),
                tool_call_id=runtime.tool_call_id)
        category = (result.result or {}).get("failure_category")
        detail = ("审批后待办目标已变化或不再存在；本次删除未执行。"
                  "请重新查询后发起新审批。" if category == "todo_target_changed"
                  else "Do not claim deletion succeeded; verify the outcome before retrying.")
        return ToolMessage(
            content=(f"Approved action outcome not confirmed: action_id={proposal.action_id}; "
                     f"status={result.status.value.upper()}. {detail}"),
            tool_call_id=runtime.tool_call_id, status="error")

    return propose_todo_delete
