"""Framework-managed HITL candidate using shared, explicitly scoped preflight.

No legacy tool invocation or ContextVar is needed for proposal preparation.
The seven actions remain project-owned business contracts; official HITL owns
interrupt/resume, and ToolNode executes approved calls.
"""
from __future__ import annotations

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from app.agent.actions import ActionProposal, ActionStatus
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError
from app.agent.agent_approval import execute_approved_proposal, rejected_todo_key
from app.agent.state import require_agent_runtime_identity
from app.agent.unified_tools import (
    GUARDED_NAMES, prepare_calendar_create, prepare_calendar_delete,
    prepare_calendar_update, prepare_email_calendar, prepare_email_todo,
    prepare_todo_tag_delete)
from app.agent.official_todo_approval import (
    ApprovalState, TodoApprovalPreflight, _already_answered, _prepared, _turn_id)


_PREFLIGHT_REASONS = {
    "CalendarStartInPast": "日程开始时间已经过去，请提供未来时间。",
    "CalendarSlotNotVerifiedFree": "所选时段未被确认空闲，请重新查询。",
    "CalendarUpdateFieldsInvalid": "日程修改内容不完整，请明确要修改的字段。",
    "TodoTagRequired": "缺少待办标签，请明确删除目标。",
    "TodoTagDeleteCountInvalid": "无法确定要删除的待办集合，请先核对目标。",
    "EmailSourceNotVerified": "邮件内容与指定来源未能匹配，请重新查询。",
    "GmailUnavailable": "邮件来源不可用，请检查授权。",
}


def _preflight_rejection(exc: PermissionError | ValueError) -> str:
    reason = _PREFLIGHT_REASONS.get(str(exc))
    if reason is None:
        reason = "无法安全确认操作目标或参数，请重新查询并明确目标。"
    return f"审批预检拒绝：{reason}本次写入未执行。"


def _preflight_read_failure(exc: CalendarError | GmailError) -> str:
    """Report only typed read failures; provider exception text may contain secrets."""
    if isinstance(exc, CalendarError) and str(exc) == "ManagedTodoCalendarEvent":
        return ("该 Google 日历时间块由待办同步管理，不能作为独立事件修改或删除；"
                "未进入审批，本次写入未执行。请修改或删除对应待办。")
    source = "日历" if isinstance(exc, CalendarError) else "邮件"
    reason = "具体原因无法安全确认"
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = (current.status_code if isinstance(current, CalendarProviderError)
                  else current.response.status_code if isinstance(current, httpx.HTTPStatusError)
                  else None)
        if status in {401, 403}:
            reason = "认证或访问权限不足"
            break
        if status == 429 or (status is not None and status >= 500):
            reason = "服务暂时不可用"
            break
        if isinstance(current, (httpx.TimeoutException, TimeoutError)):
            reason = "读取连接超时"
            break
        current = current.__cause__
    return (f"{source}预检读取失败（{reason}），未获得可靠结果；"
            "未进入审批，本次写入未执行。请核对服务授权或稍后重试。")


class AllGuardedPreflight(TodoApprovalPreflight):
    def __init__(self, session_factory):
        super().__init__(session_factory)

    async def aafter_model(self, state: ApprovalState, runtime) -> dict | None:
        update = await super().aafter_model(state, runtime)
        last = next((message for message in reversed(state["messages"])
                     if isinstance(message, AIMessage)), None)
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return update
        identity = require_agent_runtime_identity(state, runtime)
        turn = _turn_id(state["messages"])
        prepared = dict((update or {}).get("prepared_actions") or
                        state.get("prepared_actions") or {})
        for call in last.tool_calls:
            if (call["name"] not in GUARDED_NAMES or
                    call["name"] == "propose_todo_delete" or
                    _already_answered(state, call["id"])):
                continue
            try:
                args = call.get("args") or {}
                if call["name"] == "propose_calendar_create":
                    proposal = await prepare_calendar_create(
                        user_id=identity["user_id"], telegram_id=identity["telegram_id"],
                        title=str(args.get("title", "")), start=str(args.get("start", "")),
                        end=str(args.get("end", "")))
                elif call["name"] == "propose_calendar_delete":
                    proposal = await prepare_calendar_delete(
                        user_id=identity["user_id"], telegram_id=identity["telegram_id"],
                        event_id=str(args.get("event_id", "")))
                elif call["name"] == "propose_calendar_update":
                    proposal = await prepare_calendar_update(
                        user_id=identity["user_id"], telegram_id=identity["telegram_id"],
                        event_id=str(args.get("event_id", "")),
                        title=str(args.get("title", "")), start=str(args.get("start", "")),
                        end=str(args.get("end", "")))
                elif call["name"] == "propose_todo_tag_delete":
                    proposal = await prepare_todo_tag_delete(
                        self.session_factory, user_id=identity["user_id"],
                        tag=str(args.get("tag", "")))
                elif call["name"] == "propose_email_todo":
                    proposal = await prepare_email_todo(
                        user_id=identity["user_id"], telegram_id=identity["telegram_id"],
                        thread_id=str(args.get("thread_id", "")),
                        message_id=str(args.get("message_id", "")),
                        evidence_quote=str(args.get("evidence_quote", "")),
                        title=str(args.get("title", "")))
                else:
                    proposal = await prepare_email_calendar(
                        user_id=identity["user_id"], telegram_id=identity["telegram_id"],
                        thread_id=str(args.get("thread_id", "")),
                        message_id=str(args.get("message_id", "")),
                        evidence_quote=str(args.get("evidence_quote", "")),
                        title=str(args.get("title", "")),
                        start=str(args.get("start", "")), end=str(args.get("end", "")))
                if proposal.args.get("user_id") != identity["user_id"]:
                    raise PermissionError("ActionProposalOwnerMismatch")
                key = rejected_todo_key(proposal)
                if key and any(item.get("target_key") == key and item.get("turn_id") == turn
                               for call_id, item in prepared.items() if call_id != call["id"]):
                    prepared[call["id"]] = {
                        "turn_id": turn, "target_key": key,
                        "error": "This exact Todo operation was already proposed in this request."}
                else:
                    prepared[call["id"]] = {
                        "turn_id": turn, "target_key": key,
                        "proposal": proposal.model_dump(mode="json")}
            except (PermissionError, ValueError) as exc:
                prepared[call["id"]] = {
                    "turn_id": turn,
                    "error": _preflight_rejection(exc)}
            except (CalendarError, GmailError) as exc:
                # These proposal builders only read Calendar availability/event
                # versions or Gmail evidence. No approved write has started.
                if not ((isinstance(exc, CalendarError) and call["name"] in {
                    "propose_calendar_create", "propose_calendar_update",
                    "propose_calendar_delete", "propose_email_calendar"}) or
                    (isinstance(exc, GmailError) and call["name"] in {
                        "propose_email_todo", "propose_email_calendar"})):
                    raise
                prepared[call["id"]] = {
                    "turn_id": turn,
                    "error": _preflight_read_failure(exc)}
        return {"prepared_actions": prepared}

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in GUARDED_NAMES:
            return await handler(request)
        item = _prepared(request.state, request.tool_call["id"])
        if item.get("error"):
            return ToolMessage(content=item["error"],
                               tool_call_id=request.tool_call["id"], status="error")
        return await handler(request)


async def _execute(runtime: ToolRuntime[dict, ApprovalState], operation: str) -> ToolMessage:
    item = _prepared(runtime.state, runtime.tool_call_id)
    proposal = ActionProposal.model_validate(item["proposal"])
    if proposal.operation != operation or item.get("turn_id") != _turn_id(runtime.state["messages"]):
        raise PermissionError("ApprovedActionMismatch")
    identity = require_agent_runtime_identity(runtime.state, runtime)
    result = await execute_approved_proposal(proposal, identity)
    if result.action_id != proposal.action_id:
        raise RuntimeError("ActionExecutionIdentityMismatch")
    confirmed = result.status == ActionStatus.EXECUTED
    if confirmed:
        feedback = "The operation executed and is no longer awaiting approval."
    elif result.status == ActionStatus.EXECUTING:
        # A timeout or concurrent execution may leave a side effect unresolved.
        # Its existing action_id, not another model tool call, is the recovery key.
        feedback = ("outcome=UNCONFIRMED. The action may already have executed. "
                    "Do not retry or propose a replacement for this action. "
                    "Tell the user the outcome is unconfirmed pending resolution or review.")
    else:
        category = (result.result or {}).get("failure_category")
        if category == "todo_target_changed":
            detail = ("审批后待办目标已变化或不再存在；本次删除未执行。"
                      "请重新查询后发起新审批。")
        elif category == "calendar_recheck_read_failed":
            detail = "日历核验失败，本次写入尚未发送。请稍后重新查询并审批。"
        elif category == "provider_permission_denied":
            detail = "日历服务拒绝访问，本次写入未执行；请检查日历授权后重新发起操作。"
        else:
            detail = "Do not claim the operation succeeded."
        feedback = f"outcome=FAILED. {detail}"
    return ToolMessage(
        content=(f"Approved action result: action_id={proposal.action_id}; "
                 f"operation={proposal.operation}; "
                 f"target={proposal.human_readable_preview[:240]}; "
                 f"status={result.status.value.upper()}. " + feedback),
        tool_call_id=runtime.tool_call_id,
        status="success" if confirmed else "error")


def make_official_guarded_tools() -> list:
    @tool
    async def propose_todo_tag_delete(tag: str, runtime: ToolRuntime[dict, ApprovalState]) -> ToolMessage:
        """Delete an exact tagged set of owned Todos only after human approval."""
        return await _execute(runtime, "delete_todos_by_tag")

    @tool
    async def propose_calendar_create(title: str, start: str, end: str,
                                      runtime: ToolRuntime[dict, ApprovalState]) -> ToolMessage:
        """Create a Google Calendar event in a verified slot only after human approval."""
        return await _execute(runtime, "create_event")

    @tool
    async def propose_calendar_update(event_id: str, runtime: ToolRuntime[dict, ApprovalState],
                                      title: str = "", start: str = "", end: str = "") -> ToolMessage:
        """Update one exact Calendar event and approved ETag only after human approval."""
        return await _execute(runtime, "update_event")

    @tool
    async def propose_calendar_delete(event_id: str,
                                      runtime: ToolRuntime[dict, ApprovalState]) -> ToolMessage:
        """Delete one exact Calendar event and approved ETag only after human approval."""
        return await _execute(runtime, "delete_event")

    @tool
    async def propose_email_todo(thread_id: str, message_id: str, evidence_quote: str,
                                 title: str, runtime: ToolRuntime[dict, ApprovalState]) -> ToolMessage:
        """Create a Todo from a verified Gmail source only after human approval."""
        return await _execute(runtime, "create_todo_from_email")

    @tool
    async def propose_email_calendar(thread_id: str, message_id: str, evidence_quote: str,
                                     title: str, start: str, end: str,
                                     runtime: ToolRuntime[dict, ApprovalState]) -> ToolMessage:
        """Create an event from verified Gmail evidence only after human approval."""
        return await _execute(runtime, "create_event")

    return [propose_todo_tag_delete, propose_calendar_create, propose_calendar_update,
            propose_calendar_delete, propose_email_todo, propose_email_calendar]
