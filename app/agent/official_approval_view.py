"""Read-only projection of official per-call HITL interrupts for approval UIs.

This does not authorize or resume anything. The Saver interrupt ID and the
business ActionProposal/action_id are deliberately distinct identities.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import ToolMessage

from app.agent.actions import ActionProposal


@dataclass(frozen=True)
class PendingToolApproval:
    interrupt_id: str
    tool_call_id: str
    proposal: ActionProposal


def pending_official_approvals(snapshot, *, user_id: str) -> tuple[PendingToolApproval, ...]:
    values = getattr(snapshot, "values", None) or {}
    if not isinstance(values, dict) or values.get("user_id") != user_id:
        raise PermissionError("OfficialApprovalOwnerMismatch")
    prepared = values.get("prepared_actions") or {}
    # LangGraph may retain a resolved per-call interrupt in a partially resumed
    # snapshot. A paired ToolMessage is the durable evidence that this call has
    # already received a decision/result and must not be offered again in UI.
    completed_call_ids = {
        message.tool_call_id for message in values.get("messages", ())
        if isinstance(message, ToolMessage)
    }
    result = []
    seen = set()
    for interruption in getattr(snapshot, "interrupts", ()):
        value = getattr(interruption, "value", None)
        if not isinstance(value, dict) or value.get("type") != "tool_approval":
            raise ValueError("OfficialApprovalInterruptInvalid")
        call_id = value.get("tool_call_id")
        if call_id in completed_call_ids:
            continue
        item = prepared.get(call_id) if isinstance(prepared, dict) else None
        if (not isinstance(call_id, str) or not call_id or
                not isinstance(item, dict) or not item.get("proposal") or
                not getattr(interruption, "id", None)):
            raise ValueError("OfficialApprovalPreflightMissing")
        proposal = ActionProposal.model_validate(item["proposal"])
        if proposal.args.get("user_id") != user_id:
            raise PermissionError("OfficialApprovalProposalOwnerMismatch")
        if (interruption.id in seen or call_id in seen or
                proposal.action_id in seen):
            raise ValueError("OfficialApprovalIdentityDuplicate")
        seen.update((interruption.id, call_id, proposal.action_id))
        result.append(PendingToolApproval(interruption.id, call_id, proposal))
    return tuple(result)


def official_decision_for_action(snapshot, *, user_id: str, action_id: str,
                                 decision: str, checkpoint_id: str | None = None) -> dict:
    """Map one authenticated business action to one official interrupt decision.

    This is deliberately not an authorization to execute the business action:
    the guarded tool still checks owner, target version and ActionExecution.
    """
    if decision not in {"approve", "reject"}:
        raise ValueError("OfficialApprovalDecisionInvalid")
    current = ((getattr(snapshot, "config", None) or {}).get("configurable") or {}).get(
        "checkpoint_id")
    if checkpoint_id is not None and str(current) != str(checkpoint_id):
        raise ValueError("OfficialApprovalVersionConflict")
    matches = [item for item in pending_official_approvals(snapshot, user_id=user_id)
               if item.proposal.action_id == action_id]
    if len(matches) != 1:
        raise ValueError("OfficialApprovalNotPending")
    return {matches[0].interrupt_id: {"type": decision}}


async def committed_official_web_decision(graph, thread_id: str, *, user_id: str,
                                          action_id: str, decision: str,
                                          request_id: str, checkpoint_id: str):
    """Find a completed resume *attributed* to one durable WebRequest.

    A changed checkpoint or an ActionExecution row alone cannot establish who
    made the decision: Telegram may have resolved the same approval while a
    queued WebRequest was waiting. LangGraph persists invocation metadata in
    Saver history. Require the original pending action, this request's marker,
    and a paired ToolMessage before treating a crashed worker as completed.
    Return None on any missing proof, so the caller fails closed.
    """
    if not request_id or not checkpoint_id:
        return None
    config = {"configurable": {"thread_id": thread_id}}
    original = await graph.aget_state({"configurable": {
        "thread_id": thread_id, "checkpoint_id": checkpoint_id}})
    try:
        decision_map = official_decision_for_action(
            original, user_id=user_id, action_id=action_id,
            decision=decision, checkpoint_id=checkpoint_id)
    except (PermissionError, ValueError, AttributeError, TypeError):
        return None
    interrupt_id = next(iter(decision_map))
    original_call = next((item.tool_call_id for item in
                          pending_official_approvals(original, user_id=user_id)
                          if item.interrupt_id == interrupt_id), None)
    if not original_call:
        return None
    current = await graph.aget_state(config)
    if not current or (current.values or {}).get("user_id") != user_id:
        return None
    if not any(isinstance(message, ToolMessage) and
               message.tool_call_id == original_call
               for message in current.values.get("messages", ())):
        return None
    async for saved in graph.aget_state_history(config, limit=128):
        if ((saved.metadata or {}).get("web_request_id") == request_id and
                saved.metadata.get("approval_action_id") == action_id and
                saved.metadata.get("approval_decision") == decision and
                (saved.values or {}).get("user_id") == user_id and
                any(isinstance(message, ToolMessage) and
                    message.tool_call_id == original_call
                    for message in saved.values.get("messages", ()))):
            # An unrelated later WebRequest can inherit an *earlier* result
            # in its message history. Prove this exact ToolMessage first
            # appeared in a checkpoint attributed to the queried request.
            if not saved.parent_config:
                continue
            parent = await graph.aget_state(saved.parent_config)
            if (not parent or (parent.values or {}).get("user_id") != user_id or
                    any(isinstance(message, ToolMessage) and
                        message.tool_call_id == original_call
                        for message in parent.values.get("messages", ()))):
                continue
            return saved
    return None
