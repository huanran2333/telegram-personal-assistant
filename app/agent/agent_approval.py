"""Approval nodes inside the ordinary Agent graph.

The checkpointed Agent owns the pending tool call.  Approval pauses that same
graph; ActionExecution and the existing executor remain the business authority.
Legacy productivity checkpoints still use app.agent.workflow unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent.actions import ActionProposal, ActionResult, ActionStatus, requires_approval
from app.agent.business_clock import business_now
from app.agent.calendar import CalendarError
from app.agent.scheduling import verified_free_interval
from app.config import get_settings
from app.localization import tr
from app.agent.state import AgentRuntimeContext, require_agent_runtime_identity


def rejected_todo_key(proposal: ActionProposal) -> str:
    """Identify the resolved Todo operation, independent of a model's title/ID wording."""
    if proposal.domain != "todo":
        return ""
    if proposal.operation == "delete_todo":
        todo_id = proposal.args.get("todo_id")
        return f"delete_todo:{todo_id}" if todo_id else ""
    if proposal.operation == "delete_todos_by_tag":
        tag = proposal.args.get("tag")
        return f"delete_todos_by_tag:{str(tag).casefold()}" if tag else ""
    return ""


def _pending(state: dict, runtime: Runtime[AgentRuntimeContext]) -> ActionProposal:
    identity = require_agent_runtime_identity(state, runtime)
    proposal = ActionProposal.model_validate(state.get("pending_agent_proposal"))
    if proposal.args.get("user_id") != identity["user_id"]:
        raise PermissionError("AgentApprovalOwnerMismatch")
    if not requires_approval(proposal) or not state.get("pending_agent_call_id"):
        raise PermissionError("AgentApprovalNotRequiredOrUnbound")
    return proposal


def _tool_result(state: dict, proposal: ActionProposal, content: str) -> dict:
    outcomes = dict(state.get("agent_action_outcomes") or {})
    entry = {
        "status": proposal.status.value,
        "from_web": bool(state.get("approval_web_request_id")),
    }
    if proposal.status == ActionStatus.REJECTED:
        entry["turn_id"] = state.get("active_turn_id")
        entry["rejected_todo_key"] = rejected_todo_key(proposal)
    outcomes[proposal.action_id] = entry
    # Telegram UI reconciliation scans at most 30 days of delivered buttons;
    # bound the Saver map so a long-lived Agent thread cannot grow forever.
    if len(outcomes) > 256:
        outcomes = dict(list(outcomes.items())[-256:])
    return {
        "messages": [ToolMessage(content=content, tool_call_id=state["pending_agent_call_id"],
                                 status="error" if proposal.status == ActionStatus.REJECTED
                                 else "success")],
        "pending_agent_proposal": {}, "pending_agent_call_id": "",
        "pending_tool_index": int(state.get("pending_tool_index") or 0) + 1,
        "proposal": proposal.model_dump(),
        "agent_action_outcomes": outcomes,
        "last_resolved_action_id": proposal.action_id,
    }


def approval_gate(state: dict, runtime: Runtime[AgentRuntimeContext]) -> dict:
    proposal = _pending(state, runtime)
    resume = interrupt({
        "kind": "action_proposal", "action_id": proposal.action_id,
        "preview": proposal.human_readable_preview,
        "risk_level": proposal.risk_level.value,
        "allowed": ["approve", "edit", "reject"],
    })
    decision = (resume or {}).get("decision", "reject") if isinstance(resume, dict) else str(resume)
    if decision not in {"approve", "edit", "reject"}:
        raise ValueError("InvalidAgentApprovalDecision")
    return {
        "approval_decision": decision,
        "approval_web_request_id": (resume or {}).get("web_request_id")
        if isinstance(resume, dict) else None,
    }


def reject_action(state: dict, runtime: Runtime[AgentRuntimeContext]) -> dict:
    proposal = _pending(state, runtime).model_copy(update={"status": ActionStatus.REJECTED})
    feedback = ("User rejected this exact proposed action: " +
                proposal.human_readable_preview[:240] + ". Nothing was changed. "
                "Do not propose the same operation on the same target again in this "
                "user request. Continue other independent tasks. A new explicit user "
                "request may authorize a new proposal.")
    return _tool_result(state, proposal, feedback)


def request_edit(state: dict, runtime: Runtime[AgentRuntimeContext]) -> dict:
    proposal = _pending(state, runtime)
    instruction = interrupt({
        "kind": "edit_request", "action_id": proposal.action_id,
        "message": tr("edit_request"),
    })
    return {
        "agent_edit_instruction": instruction.get("instruction", "")
        if isinstance(instruction, dict) else str(instruction),
        "edit_web_request_id": instruction.get("web_request_id")
        if isinstance(instruction, dict) else None,
    }


async def apply_agent_edit(state: dict, runtime: Runtime[AgentRuntimeContext]) -> dict:
    proposal = _pending(state, runtime)
    from app.agent.planner import apply_edit
    from app.agent.workflow import bind_approved_calendar_version
    from app.agent.workflow_runtime import get_calendar_adapter
    try:
        edited = await apply_edit(
            proposal, state.get("agent_edit_instruction") or "",
            get_settings().default_timezone)
        if edited.args.get("user_id") != runtime.context["user_id"]:
            raise PermissionError("EditedActionOwnerMismatch")
        await bind_approved_calendar_version(get_calendar_adapter(), edited)
        if edited.operation == "create_event":
            available = await get_calendar_adapter().availability(
                edited.args["start"], edited.args["end"],
                get_settings().default_timezone)
            if not verified_free_interval(
                    datetime.fromisoformat(edited.args["start"]),
                    datetime.fromisoformat(edited.args["end"]), available):
                raise CalendarError("EditedCalendarSlotNotFree")
    except (ValueError, CalendarError, PermissionError):
        failed = proposal.model_copy(update={"status": ActionStatus.FAILED})
        return _tool_result(state, failed, "The requested edit could not be safely applied; no action was executed.")
    outcomes = dict(state.get("agent_action_outcomes") or {})
    outcomes[proposal.action_id] = {"status": "replaced", "from_web": bool(
        state.get("edit_web_request_id"))}
    if len(outcomes) > 256:
        outcomes = dict(list(outcomes.items())[-256:])
    return {
        "pending_agent_proposal": edited.model_dump(), "proposal": edited.model_dump(),
        "approval_decision": "", "agent_edit_instruction": "",
        "agent_action_outcomes": outcomes,
    }


async def execute_approved_proposal(proposal: ActionProposal,
                                    identity: AgentRuntimeContext) -> ActionResult:
    """Preserve business checks after HITL, regardless of the Agent graph used.

    The caller must establish a real approval bound to this proposal; neither
    the model's tool arguments nor this helper create approval authority.
    """
    if proposal.args.get("user_id") != identity["user_id"] or not requires_approval(proposal):
        raise PermissionError("AgentApprovalOwnerMismatch")
    settings = get_settings()
    if proposal.domain == "calendar":
        if (not settings.google_calendar_enabled or
                identity["telegram_id"] != settings.google_calendar_owner_telegram_id):
            raise PermissionError("CalendarOwnerMismatch")
    if proposal.operation == "create_todo_from_email":
        if (not settings.gmail_enabled or
                identity["telegram_id"] != settings.gmail_owner_telegram_id):
            raise PermissionError("GmailOwnerMismatch")
    from app.agent.workflow_runtime import get_action_executor, get_calendar_adapter
    executor = get_action_executor()
    if proposal.operation == "create_event" and not await executor.calendar_create_prepared(proposal.action_id):
        start = datetime.fromisoformat(proposal.args["start"])
        end = datetime.fromisoformat(proposal.args["end"])
        if (start.tzinfo is None or end.tzinfo is None or end <= start or
                end <= business_now(timezone.utc) or
                proposal.args.get("timeZone") != settings.default_timezone):
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED,
                                message="Calendar slot expired or changed; no action was executed.",
                                result={"preflight_rejected": True})
        try:
            available = await get_calendar_adapter().availability(
                proposal.args["start"], proposal.args["end"], settings.default_timezone)
        except CalendarError:
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED,
                                message="日历核验失败，本次写入尚未发送。请稍后重新查询并审批。",
                                result={"preflight_rejected": True,
                                        "failure_category": "calendar_recheck_read_failed"})
        if not verified_free_interval(start, end, available):
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED,
                                message="Calendar slot is no longer free; no action was executed.",
                                result={"preflight_rejected": True})
    return await executor.execute(proposal)


async def execute_action(state: dict, runtime: Runtime[AgentRuntimeContext]) -> dict:
    proposal = _pending(state, runtime)
    if state.get("approval_decision") != "approve":
        raise PermissionError("AgentActionNotApproved")
    result = await execute_approved_proposal(proposal, runtime.context)
    if result.result.get("preflight_rejected"):
        failed = proposal.model_copy(update={"status": ActionStatus.FAILED})
        return _tool_result(state, failed, result.message)
    finished = proposal.model_copy(update={"status": result.status})
    # The resumed model sees this as the original guarded tool's result. A
    # generic "Action completed" leaves the earlier proposal/interrupt text
    # more salient and can make the model report that approval is still due.
    # State only the executor-confirmed outcome, never a model inference.
    identity = (f"action_id={proposal.action_id}; tool_call_id="
                f"{state['pending_agent_call_id']}; operation={proposal.operation}; "
                f"target={proposal.human_readable_preview[:240]}; "
                f"status={result.status.value.upper()}.")
    if result.status == ActionStatus.EXECUTED:
        message = ("Approved action result: " + identity +
                   " The approved operation has executed and is no longer awaiting approval. "
                   "Continue any other independent goals in this user request; do not ask "
                   "for approval of this same completed action again.")
    elif result.status == ActionStatus.EXECUTING:
        message = ("Approved action result: " + identity +
                   " The remote outcome remains unconfirmed or requires review; "
                   "no success is claimed. Continue other independent goals when safe.")
    else:
        message = ("Approved action result: " + identity +
                   " The approved action did not complete; no success is claimed. "
                   "Continue other independent goals when safe.")
    return _tool_result(state, finished, message)
