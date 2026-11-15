"""
Telegram webhook handler.
Receives updates from Telegram and routes them to the LangGraph agent.
"""
import hashlib
import hmac
import logging
import re
import secrets

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from telegram import Update

from app.config import get_settings
from app.db.session import get_db, AsyncSessionLocal
from app.db.crud import get_or_create_user
from app.telegram.sender import (send_message, send_typing, send_action_proposal,
                                 send_official_action_proposals,
                                 send_slot_choices, send_workbench_launch)
from app.agent.official_runtime import get_graph, run_agent
from app.agent.workflow import resume_workflow, workflow_waits_for_edit
from app.agent.workflow_runtime import (get_productivity_graph, get_calendar_create_coordinator,
                                        get_calendar_modify_coordinator)
from app.trace import begin_trace, current_trace_id, diagnostic_error, end_trace, set_correlation, set_user, emit_trace

import httpx
import tempfile
import os
from app.telegram.sender import get_bot
from app.localization import (ResponsePreferences, current_preferences, language_request,
                              load_preferences, reset_preferences, save_language,
                              tr, use_preferences)
from app.db.crud import get_user_by_telegram_id

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

def webhook_secret_token() -> str:
    """Stable Telegram webhook secret without another credential to provision."""
    return hashlib.sha256(settings.telegram_bot_token.encode()).hexdigest()


def _workflow_thread(telegram_id: str) -> str:
    return f"workflow_user_{telegram_id}"


def _terminal_approval_notice(values: dict, action_id: str | None = None) -> str:
    """Describe the Saver outcome, never the stale button's requested decision."""
    outcome = (values.get("agent_action_outcomes") or {}).get(action_id) if action_id else None
    proposal = values.get("proposal") or {}
    status = (outcome or {}).get("status") or proposal.get("status")
    if status == "rejected":
        return tr("approval_web_rejected" if ((outcome or {}).get("from_web") if outcome
                                                else values.get("approval_web_request_id"))
                  else "approval_already_rejected")
    if status == "executed":
        return tr("approval_already_completed")
    if status == "failed":
        return tr("approval_already_failed")
    if status == "replaced":
        return tr("approval_replaced")
    if status in {"approved", "executing"}:
        return tr("action_unconfirmed")
    return tr("processed_action")


async def _answer_callback_best_effort(callback, *args, **kwargs) -> None:
    """Ephemeral UI acknowledgement must not suppress a durable HITL result."""
    try:
        await callback.answer(*args, **kwargs)
    except Exception as exc:
        logger.warning("callback_ui_ack stage=send category=%s", type(exc).__name__)


async def _durable_callback_reply(callback, text: str) -> None:
    from app.telegram.processing_context import current_run
    if current_run() is not None and getattr(callback, "message", None) is not None:
        await send_message(callback.message.chat_id, text)


def _record_no_business(reason: str) -> None:
    """Only call before an Agent/Store mutation has begun."""
    from app.telegram.processing_context import current_run
    run = current_run()
    if run is not None:
        run.record_outcome("no_business", reason)


def _prior_checkpoint_id(snapshot) -> str | None:
    from app.telegram.processing_context import current_run
    from app.telegram.durable import DurableSafetyError
    if current_run() is None:
        return None
    checkpoint_id = ((snapshot.config or {}).get("configurable") or {}).get(
        "checkpoint_id") if snapshot else None
    if not checkpoint_id:
        raise DurableSafetyError("MissingPendingCheckpoint")
    return str(checkpoint_id)


def _official_callback_tool_result(snapshot, action_id: str):
    """Return the exact persisted result of one official approval call, if any.

    With per-call HITL, PostgreSQL Saver can persist a completed task as a
    pending write while another call in the same ToolNode batch is interrupted.
    In that state the checkpoint ID has not advanced, but ``aget_state``
    includes the completed ToolMessage. Neither an ActionExecution row nor a
    model reply alone proves that this exact callback completed.
    """
    from langchain_core.messages import ToolMessage
    values = (snapshot.values or {}) if snapshot else {}
    prepared = values.get("prepared_actions") or {}
    calls = [call_id for call_id, item in prepared.items()
             if isinstance(item, dict) and
             (item.get("proposal") or {}).get("action_id") == action_id]
    if len(calls) != 1:
        return None
    return next((message for message in values.get("messages", ())
                 if isinstance(message, ToolMessage) and
                 message.tool_call_id == calls[0]), None)


async def _record_checkpoint_outcome(graph, thread_id: str, *, ordinary: bool = False,
                                     allow_pending: bool = False,
                                     prior_checkpoint_id: str | None = None,
                                     replies: int = 1) -> None:
    """Verify the Saver's committed result before declaring a reply outcome."""
    from app.telegram.processing_context import current_run
    from app.telegram.durable import DurableSafetyError
    from langchain_core.messages import HumanMessage
    run = current_run()
    if run is None:
        return
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    checkpoint_id = ((snapshot.config or {}).get("configurable") or {}).get(
        "checkpoint_id") if snapshot else None
    if not checkpoint_id:
        raise DurableSafetyError("MissingCommittedCheckpoint")
    if prior_checkpoint_id and checkpoint_id == prior_checkpoint_id:
        from app.db.models import TelegramInboundReceipt
        async with AsyncSessionLocal() as db:
            receipt = await db.get(TelegramInboundReceipt, run.receipt_id)
        callback_data = ((receipt.envelope.get("callback_query") or {}).get("data", "")
                         if receipt and receipt.envelope else "")
        if callback_data.startswith("act:"):
            _, action_id, _ = callback_data.split(":", 2)
            message = _official_callback_tool_result(snapshot, action_id)
            if message is not None:
                run.record_outcome("tool_result",
                                   f"{checkpoint_id}:{message.tool_call_id}")
                return
        raise DurableSafetyError("MissingCommittedCheckpoint")
    if ordinary:
        incoming_id = f"telegram-receipt:{run.receipt_id}"
        valid = (not snapshot.next or (allow_pending and any(
            node in snapshot.next for node in
            ("agent_approval_gate", "agent_request_edit"))) or
            (allow_pending and bool(snapshot.interrupts) and
             bool(snapshot.values.get("prepared_actions")))) and any(
            isinstance(item, HumanMessage) and item.id == incoming_id
            for item in snapshot.values.get("messages", []))
    elif prior_checkpoint_id is None:
        valid = snapshot.values.get("transcript_turn_id") == f"telegram-receipt:{run.receipt_id}"
    else:
        # The callback/edit handler first verified the pending action and
        # passed the checkpoint that existed before Command(resume).
        valid = bool(snapshot.values.get("user_id"))
        from app.db.models import TelegramInboundReceipt
        async with AsyncSessionLocal() as db:
            receipt = await db.get(TelegramInboundReceipt, run.receipt_id)
            callback_data = ((receipt.envelope.get("callback_query") or {}).get("data", "")
                             if receipt else "")
        if callback_data.startswith("act:"):
            _, action_id, decision = callback_data.split(":", 2)
            if snapshot.values.get("prepared_actions"):
                from langchain_core.messages import ToolMessage
                prepared = snapshot.values["prepared_actions"]
                calls = [call_id for call_id, item in prepared.items()
                         if (item.get("proposal") or {}).get("action_id") == action_id]
                valid = (valid and len(calls) == 1 and any(
                    isinstance(message, ToolMessage) and message.tool_call_id == calls[0]
                    for message in snapshot.values.get("messages", ())))
            else:
                valid = (valid and snapshot.values.get("approval_decision") == decision and
                         ((snapshot.values.get("proposal") or {}).get("action_id") == action_id or
                          snapshot.values.get("last_resolved_action_id") == action_id))
        elif callback_data.startswith("slot:"):
            _, selection_id, raw_index = callback_data.split(":", 2)
            candidates = snapshot.values.get("slot_candidates") or []
            index = int(raw_index)
            valid = (valid and snapshot.values.get("slot_selection_id") == selection_id and
                     0 <= index < len(candidates) and
                     snapshot.values.get("selected_slot") == candidates[index])
    if not valid:
        raise DurableSafetyError("CheckpointOutcomeMismatch")
    run.record_outcome("ordinary_checkpoint" if ordinary else "workflow_checkpoint",
                       str(checkpoint_id), replies=replies)


async def _resume_with_origin(graph, thread_id: str, payload, snapshot,
                              *, agent_context: dict | None = None):
    from app.telegram.processing_context import current_run
    prior_id = _prior_checkpoint_id(snapshot)
    run = current_run()
    if run is not None:
        from app.telegram.durable import record_resume_origin
        await record_resume_origin(AsyncSessionLocal, run.receipt_id,
                                   run.fence_token, prior_id)
    if agent_context is not None:
        return await resume_workflow(graph, thread_id, payload,
                                     context=agent_context), prior_id
    return await resume_workflow(graph, thread_id, payload), prior_id


async def _replay_completed_resume(graph, thread_id: str, snapshot, chat_id: int,
                                   *, agent: bool = False,
                                   agent_context: dict | None = None) -> bool:
    """After a crash, never issue Command(resume) twice for one receipt.

    Same-thread admission blocks newer receipts until this one terminates, so
    a changed Saver cursor after our persisted pre-resume cursor belongs to
    this receipt. If the resumed graph has more runnable work, continue it
    without re-supplying the human decision.
    """
    from app.telegram.processing_context import current_run
    from app.telegram.durable import DurableSafetyError
    from app.db.models import TelegramInboundReceipt
    run = current_run()
    if run is None:
        return False
    async with AsyncSessionLocal() as db:
        receipt = await db.get(TelegramInboundReceipt, run.receipt_id)
        origin = receipt.resume_from_checkpoint_id if receipt else None
    if not origin:
        return False
    current_id = _prior_checkpoint_id(snapshot)
    if origin == current_id:
        if agent:
            callback_data = ((receipt.envelope.get("callback_query") or {}).get("data", "")
                             if receipt and receipt.envelope else "")
            if callback_data.startswith("act:"):
                _, action_id, _ = callback_data.split(":", 2)
                if _official_callback_tool_result(snapshot, action_id) is not None:
                    result = {**snapshot.values, "__interrupt__": snapshot.interrupts}
                    await _send_agent_result(chat_id, thread_id, result,
                                             prior_checkpoint_id=origin)
                    return True
        return False
    interrupts = [item for task in snapshot.tasks
                  for item in getattr(task, "interrupts", ())]
    if interrupts:
        # A pending interrupt after this receipt's resume is itself a durable
        # result (e.g. Edit -> request_edit or slot -> action proposal).
        result = {"__interrupt__": interrupts, **snapshot.values}
    elif snapshot.next:
        kwargs = {"context": agent_context} if agent else {}
        result = await graph.ainvoke(None, {"configurable": {"thread_id": thread_id}},
                                    **kwargs)
    else:
        result = snapshot.values
    if not result:
        raise DurableSafetyError("MissingResumedWorkflowResult")
    if agent:
        await _send_agent_result(chat_id, thread_id, result,
                                 prior_checkpoint_id=origin)
    else:
        await _send_workflow_result(chat_id, thread_id, result,
                                    prior_checkpoint_id=origin)
    return True


async def _send_agent_result(chat_id: int, thread_id: str, result: dict,
                             *, prior_checkpoint_id: str | None = None) -> None:
    """Render the same Agent checkpoint that owns the interrupted tool call."""
    graph = await get_graph()
    interrupts = result.get("__interrupt__", [])
    payload = getattr(interrupts[0], "value", {}) if interrupts else {}
    pending = None
    snapshot = None
    partial = False
    replies = 1
    if payload.get("type") == "tool_approval":
        from app.agent.official_approval_view import pending_official_approvals
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        pending = pending_official_approvals(
            snapshot, user_id=snapshot.values.get("user_id"))
        if not pending:
            raise RuntimeError("OfficialApprovalInterruptWithoutPendingAction")
        current_id = ((snapshot.config or {}).get("configurable") or {}).get("checkpoint_id")
        partial = bool(prior_checkpoint_id and current_id == prior_checkpoint_id)
        if not partial:
            replies = len(pending)
    await _record_checkpoint_outcome(
        graph, thread_id, ordinary=prior_checkpoint_id is None,
        allow_pending=True, prior_checkpoint_id=prior_checkpoint_id, replies=replies)
    if interrupts:
        if payload.get("type") == "tool_approval":
            if partial:
                # Only the resolved card loses its buttons. Other independent
                # proposal messages remain actionable until their own result.
                await send_message(
                    chat_id, f"本项审批决定已处理；还有 {len(pending)} 项待确认。"
                    "请在对应的审批消息继续选择，完成后会给出完整结果。")
            else:
                await send_official_action_proposals(
                    chat_id, [item.proposal.model_dump(mode="json")
                              for item in pending])
        elif payload.get("kind") == "action_proposal":
            snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            proposal = snapshot.values.get("proposal") if snapshot else None
            if not proposal or proposal.get("action_id") != payload.get("action_id"):
                raise RuntimeError("AgentApprovalCheckpointMismatch")
            await send_action_proposal(chat_id, proposal,
                                       completed_summary=snapshot.values.get("handoff_summary") or "")
        elif payload.get("kind") == "edit_request":
            await send_message(chat_id, tr("edit_request"))
        return
    from langchain_core.messages import AIMessage
    response = next((str(item.content) for item in reversed(result.get("messages", []))
                     if isinstance(item, AIMessage) and not item.tool_calls and item.content),
                    tr("done"))
    await send_message(chat_id, response)


async def _send_workflow_result(chat_id: int, thread_id: str, result: dict,
                                *, prior_checkpoint_id: str | None = None,
                                completed_summary: str = "") -> None:
    if any(getattr(item, "value", {}).get("kind") == "calendar_confirmation"
           for item in result.get("__interrupt__", [])):
        from app.agent.workflow import bind_continuation_checkpoint
        graph = await get_productivity_graph()
        await bind_continuation_checkpoint(graph, thread_id, result)
    from app.telegram.processing_context import current_run
    if current_run() is not None:
        graph = await get_productivity_graph()
        await _record_checkpoint_outcome(graph, thread_id,
                                         prior_checkpoint_id=prior_checkpoint_id)
    async def project(response: str = "") -> None:
        try:
            graph = await get_productivity_graph()
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            values = state.values if state else {}
            if not values.get("transcript_turn_id") or not values.get("user_id"):
                return
            from app.agent.transcript import sync_workflow_turn
            async with AsyncSessionLocal() as projection_db:
                await sync_workflow_turn(projection_db, user_id=values["user_id"],
                                         thread_id=thread_id, turn_id=values["transcript_turn_id"],
                                         request=values["request"],
                                         response=response if values.get("response") == response else "",
                                         occurred_at=values.get("transcript_occurred_at"))
                await projection_db.commit()
            emit_trace("transcript_projection", operation="workflow_sync", success=True)
        except Exception as exc:
            logger.warning("transcript_projection stage=workflow_sync category=%s", type(exc).__name__)
            emit_trace("transcript_projection", operation="workflow_sync", success=False,
                       error_category=type(exc).__name__)

    await project()
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        if payload.get("kind") == "action_proposal":
            # The interrupt contains the stable action id; the checkpoint state
            # holds the full proposal so callback_data never needs raw args.
            graph = await get_productivity_graph()
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            proposal_data = state.values.get("proposal") if state else None
            if proposal_data:
                await send_action_proposal(chat_id, proposal_data,
                                           completed_summary=completed_summary)
            else:
                await send_message(chat_id, tr("plan_failed"))
        elif payload.get("kind") == "edit_request":
            await send_message(chat_id, tr("edit_request"))
        elif payload.get("kind") == "slot_selection":
            await send_slot_choices(chat_id, payload["selection_id"], payload["candidates"])
        elif payload.get("kind") == "calendar_confirmation":
            await send_message(chat_id, payload["message"])
        return
    response = result.get("response", tr("done"))
    await send_message(chat_id, response)
    await project(response)


async def _handle_action_callback(callback) -> None:
    """Resume exactly the pending interrupt for this Telegram user."""
    try:
        _, action_id, decision = callback.data.split(":", 2)
        if decision not in {"approve", "edit", "reject"}:
            raise ValueError("unknown decision")
    except (AttributeError, ValueError):
        await _answer_callback_best_effort(callback, tr("invalid_button"), show_alert=True)
        _record_no_business("invalid_callback")
        await _durable_callback_reply(callback, tr("invalid_button"))
        return
    telegram_id = str(callback.from_user.id)
    chat_id = callback.message.chat_id
    agent_thread = f"user_{telegram_id}"
    agent_graph = await get_graph()
    official_thread = f"official-v1:{agent_thread}"
    official_state = await agent_graph.aget_state(
        {"configurable": {"thread_id": official_thread}})
    official_prepared = ((official_state.values or {}).get("prepared_actions")
                         if official_state else None) or {}
    official = any((item.get("proposal") or {}).get("action_id") == action_id
                   for item in official_prepared.values())
    # Old Agent approval callbacks must never invoke their previous graph
    # after the official-v1 cutover. Read raw Saver data only to recognize
    # the old action; do not treat the new graph's filtered State as evidence.
    saver = getattr(agent_graph, "checkpointer", None)
    old_tuple = (await saver.aget_tuple({"configurable": {"thread_id": agent_thread}})
                 if saver is not None else None)
    old_values = ((old_tuple.checkpoint or {}).get("channel_values") or {}
                  if old_tuple else {})
    old_actions = (old_values.get("proposal") or {},
                   old_values.get("pending_agent_proposal") or {})
    old_action = any(item.get("action_id") == action_id
                     for item in old_actions if isinstance(item, dict))
    if old_action and not official:
        async with AsyncSessionLocal() as db:
            owner = await get_user_by_telegram_id(db, telegram_id)
        if (owner is None or old_values.get("user_id") != owner.id or
                str(old_values.get("chat_id")) != str(chat_id)):
            raise PermissionError("LegacyApprovalTelegramOwnerMismatch")
        notice = "旧版审批按钮已停用，未据此执行操作。请查看当前业务状态后重新发送明确请求。"
        await _answer_callback_best_effort(callback, notice, show_alert=True)
        _record_no_business("legacy_agent_approval_disabled")
        await _durable_callback_reply(callback, notice)
        return
    agent_state = await agent_graph.aget_state(
        {"configurable": {"thread_id": agent_thread}})
    agent_proposal = agent_state.values.get("proposal") if agent_state else None
    agent = bool(agent_state and (
        (agent_proposal and agent_proposal.get("action_id") == action_id) or
        action_id in (agent_state.values.get("agent_action_outcomes") or {})))
    authenticated_user = None
    if agent or official:
        async with AsyncSessionLocal() as db:
            authenticated_user = await get_user_by_telegram_id(db, telegram_id)
        if authenticated_user is None:
            raise PermissionError("AgentApprovalUserMissing")
        owner_state = official_state if official else agent_state
        if (owner_state.values.get("user_id") != authenticated_user.id or
                owner_state.values.get("telegram_id") != telegram_id or
                str(owner_state.values.get("chat_id")) != str(chat_id)):
            raise PermissionError("AgentApprovalTelegramOwnerMismatch")
        graph, thread_id, state = (agent_graph,
                                   official_thread if official else agent_thread,
                                   owner_state)
        agent = True
    else:
        thread_id = _workflow_thread(telegram_id)
        graph = await get_productivity_graph()
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    proposal = state.values.get("proposal") if state else None
    agent_context = ({"user_id": authenticated_user.id,
                      "telegram_id": telegram_id, "chat_id": str(chat_id),
                      "origin": "telegram"} if agent else None)
    if state:
        set_correlation(origin_trace_id=state.values.get("trace_origin_id"), thread_id=thread_id)
        set_user(state.values.get("user_id"))
    if await _replay_completed_resume(graph, thread_id, state, chat_id,
                                      agent=agent, agent_context=agent_context):
        return
    if official:
        from app.agent.official_approval_view import official_decision_for_action
        try:
            payload = official_decision_for_action(
                state, user_id=authenticated_user.id, action_id=action_id,
                decision=decision)
        except ValueError:
            await _answer_callback_best_effort(callback, tr("expired_action"), show_alert=True)
            _record_no_business("stale_official_action")
            await _durable_callback_reply(callback, tr("expired_action"))
            return
        await _answer_callback_best_effort(callback)
        result, prior_checkpoint_id = await _resume_with_origin(
            graph, thread_id, payload, state, agent_context=agent_context)
        await _send_agent_result(chat_id, thread_id, result,
                                 prior_checkpoint_id=prior_checkpoint_id)
        return
    if agent and action_id in (state.values.get("agent_action_outcomes") or {}):
        notice = _terminal_approval_notice(state.values, action_id)
        await _answer_callback_best_effort(callback, notice, show_alert=True)
        _record_no_business("already_processed_action")
        await _durable_callback_reply(callback, notice)
        return
    if not proposal or proposal.get("action_id") != action_id:
        emit_trace("callback_stale", action_id=action_id, success=False)
        notice = (tr("approval_replaced") if proposal and proposal.get("action_id") != action_id
                  else tr("expired_action"))
        await _answer_callback_best_effort(callback, notice, show_alert=True)
        _record_no_business("stale_action")
        await _durable_callback_reply(callback, notice)
        return
    if "wait_for_calendar" in state.next:
        # The approval already happened. A repeated button is neither a new
        # authorization nor the external reconciliation signal.
        notice = tr("action_unconfirmed")
        coordinator = get_calendar_create_coordinator()
        if coordinator is not None:
            evidence, _ = await coordinator.store.load_create(action_id)
            if evidence is not None and evidence.outcome == "PARKED_REVIEW":
                notice = tr("action_review_required")
        await _answer_callback_best_effort(callback, notice, show_alert=True)
        _record_no_business("calendar_action_waiting")
        await _durable_callback_reply(callback, notice)
        return
    if not state.next:
        coordinators = (get_calendar_create_coordinator(), get_calendar_modify_coordinator())
        for coordinator in coordinators:
            if coordinator is None:
                continue
            evidence, ledger = await coordinator.store.load_create(action_id)
            if (evidence is not None and evidence.operation in
                    {"create_event", "update_event", "delete_event"} and
                    not (evidence.outcome == "CONFIRMED" and ledger is not None and
                         ledger.status == "executed")):
                # A new callback is not a second authorization or resume. The
                # durable recovery loop reconciles this same action; report its
                # unresolved outcome rather than implying the remote write failed.
                notice = tr("action_review_required" if evidence.outcome == "PARKED_REVIEW"
                            else "action_unconfirmed")
                await _answer_callback_best_effort(callback, notice, show_alert=True)
                _record_no_business("calendar_action_unconfirmed")
                await _durable_callback_reply(callback, notice)
                return
            if evidence is not None:
                break
        emit_trace("callback_duplicate", action_id=action_id, success=True)
        # A Web decision may have resolved the original Telegram Saver thread.
        # Read its actual terminal text (including rejection/failure), not a
        # generic success claim. No HTTP handler sends a Telegram message.
        notice = _terminal_approval_notice(state.values)
        await _answer_callback_best_effort(callback, notice, show_alert=True)
        _record_no_business("already_processed_action")
        await _durable_callback_reply(callback, notice)
        return
    if ("agent_approval_gate" if agent else "approval_gate") not in state.next:
        await _answer_callback_best_effort(callback, tr("expired_action"), show_alert=True)
        _record_no_business("non_approval_interrupt")
        await _durable_callback_reply(callback, tr("expired_action"))
        return
    await _answer_callback_best_effort(callback)
    emit_trace("resume", operation=decision, action_id=action_id, success=True)
    result, prior_checkpoint_id = await _resume_with_origin(
        graph, thread_id, {"decision": decision}, state,
        agent_context=agent_context)
    sender = _send_agent_result if agent else _send_workflow_result
    if prior_checkpoint_id is None:
        await sender(chat_id, thread_id, result)
    else:
        await sender(chat_id, thread_id, result,
                     prior_checkpoint_id=prior_checkpoint_id)


async def _handle_slot_callback(callback) -> None:
    try:
        _, selection_id, raw_index = callback.data.split(":", 2)
        index = int(raw_index)
    except (AttributeError, ValueError):
        await _answer_callback_best_effort(callback, tr("invalid_button"), show_alert=True)
        _record_no_business("invalid_slot_callback")
        await _durable_callback_reply(callback, tr("invalid_button"))
        return
    thread_id = _workflow_thread(str(callback.from_user.id))
    graph = await get_productivity_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if state:
        set_correlation(origin_trace_id=state.values.get("trace_origin_id"), thread_id=thread_id)
    if await _replay_completed_resume(graph, thread_id, state, callback.message.chat_id):
        return
    if (not state or not state.next or "choose_slot" not in state.next or
            state.values.get("slot_selection_id") != selection_id or
            not 0 <= index < len(state.values.get("slot_candidates", []))):
        await _answer_callback_best_effort(callback, tr("slot_expired"), show_alert=True)
        _record_no_business("stale_slot")
        await _durable_callback_reply(callback, tr("slot_expired"))
        return
    await _answer_callback_best_effort(callback)
    set_user(state.values.get("user_id"))
    emit_trace("resume", operation="choose_slot", success=True)
    result, prior_checkpoint_id = await _resume_with_origin(
        graph, thread_id, {"selection_index": index}, state)
    if prior_checkpoint_id is None:
        await _send_workflow_result(callback.message.chat_id, thread_id, result)
    else:
        await _send_workflow_result(callback.message.chat_id, thread_id, result,
                                    prior_checkpoint_id=prior_checkpoint_id)


@router.get("/webhook-info")
async def webhook_info(request: Request, response: Response):
    """Operator-only diagnostics; never expose Bot configuration publicly."""
    from app.web.auth import SESSION_COOKIE, no_store
    from app.web.api import _authenticated
    if not request.cookies.get(SESSION_COOKIE):
        raise HTTPException(401, "AuthenticationRequired")
    _, session = await _authenticated(request)
    if session.auth_source != "password":
        raise HTTPException(403, "OperatorSessionRequired")
    no_store(response)
    try:
        bot = get_bot()
        
        if not settings.telegram_bot_token:
            return {"error": "telegram_bot_token not configured"}
        if not settings.telegram_webhook_url:
            return {"error": "telegram_webhook_url not configured"}
        
        webhook_info = await bot.get_webhook_info()
        return {
            "status": "ok",
            "bot_token_configured": bool(settings.telegram_bot_token),
            "webhook_url_configured": settings.telegram_webhook_url,
            "webhook_info": {
                "url": webhook_info.url,
                "has_custom_certificate": webhook_info.has_custom_certificate,
                "pending_update_count": webhook_info.pending_update_count,
                "last_error_date": webhook_info.last_error_date,
                "last_error_message": webhook_info.last_error_message,
            }
        }
    except Exception as e:
        logger.error("Failed to get webhook info category=%s", diagnostic_error(e))
        return {"error": "Telegram webhook information unavailable"}


@router.post("/webhook")
async def telegram_webhook(request: Request):
    """ACK only after authenticated update has committed replayable custody."""
    if settings.telegram_bot_token and not hmac.compare_digest(
        request.headers.get("X-Telegram-Bot-Api-Secret-Token", ""),
        webhook_secret_token(),
    ):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    # A previously registered webhook can still deliver while this process is
    # starting or degraded. Reject before parsing/claiming the update so
    # Telegram may retry it after the operator restarts a healthy runtime.
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.ready:
        raise HTTPException(status_code=503, detail="Assistant runtime is not ready",
                            headers={"Retry-After": "5"})
    durable = getattr(request.app.state, "telegram_durable", None)
    if durable is None or durable.bot_identity is None or not durable.is_running:
        raise HTTPException(status_code=503, detail="Durable ingress is unavailable",
                            headers={"Retry-After": "5"})
    from app.telegram.durable import admit_receipt, admit_terminal_receipt, normalize_update
    try:
        data = await request.json()
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid Telegram JSON") from None
    # Serialize admission with lifecycle quiesce. A request that passed the
    # first readiness check before shutdown must recheck under this gate;
    # otherwise it could commit a new receipt after shutdown begins.
    async with durable.admission_lock:
        if not runtime.ready or not durable.is_running:
            raise HTTPException(status_code=503, detail="Durable ingress is shutting down",
                                headers={"Retry-After": "5"})
        try:
            update_id, envelope, kind, _ = normalize_update(data)
            actor = (envelope["callback_query"]["from"] if kind == "callback"
                     else envelope["message"]["from"])
            thread_id = f"telegram_user_{actor['id']}"
        except (ValueError, KeyError, TypeError) as exc:
            try:
                await admit_terminal_receipt(
                    AsyncSessionLocal, bot_identity=durable.bot_identity, data=data,
                    category=str(exc) if str(exc).isidentifier() else "InvalidEnvelope")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Telegram update") from None
            except Exception as custody_error:
                logger.warning("telegram_admission stage=terminal category=%s",
                               type(custody_error).__name__)
                raise HTTPException(status_code=503, detail="Durable receipt unavailable",
                                    headers={"Retry-After": "5"}) from None
            return {"ok": True}
        if settings.telegram_allowed_ingress_ids.strip():
            allowed = {part.strip() for part in
                       settings.telegram_allowed_ingress_ids.split(",")}
            if str(actor["id"]) not in allowed:
                # Do not ACK or persist an unapproved chat while the attended
                # test is active. Telegram may retry this Update later.
                raise HTTPException(status_code=503,
                                    detail="This Bot session is limited to a test chat",
                                    headers={"Retry-After": "60"})
        admission_trace = begin_trace()
        try:
            try:
                receipt = await admit_receipt(
                    AsyncSessionLocal, bot_identity=durable.bot_identity, data=data,
                    graph_kind="telegram_session", thread_id=thread_id,
                    origin_trace_id=current_trace_id())
            except ValueError:
                raise HTTPException(status_code=409, detail="Telegram update identity conflict") from None
            except Exception as exc:
                logger.warning("telegram_admission stage=commit category=%s", type(exc).__name__)
                raise HTTPException(status_code=503, detail="Durable receipt unavailable",
                                    headers={"Retry-After": "5"}) from None
            try:
                durable.kick(receipt.id)
            except RuntimeError:
                # The receipt is committed; restart recovery can claim it.
                raise HTTPException(status_code=503, detail="Ingress recovery unavailable",
                                    headers={"Retry-After": "5"}) from None
            return {"ok": True}
        finally:
            end_trace(admission_trace)


async def process_admitted_update(data: dict):
    """Process a previously committed typed envelope under a receipt lease."""
    from app.telegram.processing_context import current_run
    update_id = data["update_id"]

    trace_token = begin_trace()
    locale_tokens = []
    try:
        run = current_run()
        if run is not None:
            from app.db.models import TelegramInboundReceipt
            async with AsyncSessionLocal() as db:
                receipt = await db.get(TelegramInboundReceipt, run.receipt_id)
                if receipt is not None:
                    set_correlation(origin_trace_id=receipt.origin_trace_id,
                                    thread_id=receipt.thread_id)
        emit_trace("telegram_update", update_id=update_id, success=True)
        try:
            # CallbackQuery.answer() needs the Bot bound to the parsed update.
            update = Update.de_json(data, get_bot())
        except Exception as e:
            logger.error("Failed to parse Telegram update category=%s", diagnostic_error(e))
            raise HTTPException(status_code=400, detail="Invalid update")

        sender = update.callback_query.from_user if update.callback_query else (
            update.message.from_user if update.message else None)
        preferences = ResponsePreferences()
        if sender is not None:
            try:
                async with AsyncSessionLocal() as db:
                    known_user = await get_user_by_telegram_id(db, str(sender.id))
                if known_user is not None:
                    preferences = await load_preferences(known_user.id, strict=True)
            except Exception:
                logger.warning("Response preferences unavailable for Telegram update")
                raise HTTPException(status_code=503, detail="Profile Store is unavailable",
                                    headers={"Retry-After": "5"}) from None
        locale_tokens.append(use_preferences(preferences))

        if update.callback_query:
            if (update.callback_query.data or "").startswith("act:"):
                await _handle_action_callback(update.callback_query)
            elif (update.callback_query.data or "").startswith("slot:"):
                await _handle_slot_callback(update.callback_query)
            else:
                await _answer_callback_best_effort(update.callback_query, tr("invalid_button"), show_alert=True)
                _record_no_business("unsupported_callback")
                await _durable_callback_reply(update.callback_query, tr("invalid_button"))
            return {"ok": True}

        # Only handle text messages
        # if not update.message or not update.message.text:
        #     return {"ok": True}
        if not update.message or (not update.message.text and not update.message.voice):
            raise ValueError("MissingSupportedMessage")

        message = update.message
        telegram_user = message.from_user
        chat_id = message.chat_id

        if message.voice:
            file_path = None
            try:
                await send_typing(chat_id)
                file_path = await download_voice(message.voice.file_id)
                text = await transcribe_voice(file_path)
                logger.info("Voice transcribed for Telegram update")
            except Exception as e:
                logger.error("Voice transcription failed category=%s", diagnostic_error(e))
                _record_no_business("voice_transcription_unavailable")
                await send_message(chat_id, tr("voice_error"))
                return {"ok": True}
            finally:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)

        # Handle text message
        elif message.text:
            text = message.text.strip()
        else:
            raise ValueError("MissingSupportedMessage")
            
        # Ignore commands for now (can extend later)
        if text.startswith("/start"):
            logger.info("Received /start")
            _record_no_business("start_command")
            await send_message(chat_id, tr("start"))
            return {"ok": True}

        if text.startswith("/help"):
            logger.info("Received /help")
            help_text = tr("help")
            _record_no_business("help_command")
            await send_message(chat_id, help_text)
            return {"ok": True}

        if text and text.split(maxsplit=1)[0].split("@", 1)[0] == "/workbench":
            from urllib.parse import urlencode
            from app.web.telegram_miniapp import issue_ticket
            from app.db.models import TelegramInboundReceipt
            from app.web.auth import validate_web_configuration
            config = get_settings()
            if (message.chat.type != "private" or telegram_user is None
                    or chat_id != telegram_user.id or known_user is None):
                _record_no_business("workbench_not_authorized")
                await send_message(chat_id, "请先在 Bot 私聊中建立个人账户，再打开工作台。")
                return {"ok": True}
            try:
                validate_web_configuration(config)
            except ValueError:
                _record_no_business("workbench_not_configured")
                await send_message(chat_id, "工作台暂不可用，请稍后再试。")
                return {"ok": True}
            if run is None:
                raise RuntimeError("DurableReceiptRequired")
            async with AsyncSessionLocal() as db:
                receipt = await db.get(TelegramInboundReceipt, run.receipt_id)
                if receipt is None or receipt.bot_identity <= 0:
                    raise RuntimeError("VerifiedBotIdentityRequired")
                ticket = await issue_ticket(
                    db, bot_identity=receipt.bot_identity,
                    telegram_user_id=str(telegram_user.id))
            # Fragment is not sent to the HTTPS server in normal requests.
            # A non-secret view nonce avoids Telegram WebView reusing an older
            # cached HTML shell. The one-use credential remains fragment-only.
            url = (config.web_public_origin.rstrip("/") +
                   "/workbench/?view=" + secrets.token_urlsafe(8) + "#" +
                   urlencode({"launchTicket": ticket}))
            _record_no_business("workbench_launch_issued")
            await send_workbench_launch(chat_id, url)
            return {"ok": True}

        # Show typing indicator
        try:
            await send_typing(chat_id)
        except Exception as e:
            logger.warning("Failed to send typing indicator category=%s", diagnostic_error(e))

        # A failure after invoking a graph or writing Profile is ambiguous;
        # retry from its durable authority instead of enqueueing a misleading
        # generic error reply that could suppress recovery.
        async with AsyncSessionLocal() as db:
            user = await get_or_create_user(
                db,
                telegram_id=str(telegram_user.id),
                username=telegram_user.username,
                first_name=telegram_user.first_name,
            )
            await db.commit()
            user_id = user.id
            set_user(user_id)

        language_change = language_request(text)
        if language_change is not None:
            operation, requested = language_change
            if operation == "query":
                _record_no_business("language_query")
                await send_message(chat_id, tr("language_query", language=current_preferences().language))
            elif requested not in {"en", "zh-CN"}:
                _record_no_business("language_invalid")
                await send_message(chat_id, tr("language_invalid"))
            else:
                updated = await save_language(user_id, requested)
                locale_tokens.append(use_preferences(updated))
                run = current_run()
                if run is not None:
                    run.record_outcome("profile_store", f"language:{user_id}:{requested}")
                await send_message(chat_id, tr("language_set"))
            return {"ok": True}

        # Run agent
        workflow_thread = _workflow_thread(str(telegram_user.id))
        set_correlation(thread_id=workflow_thread)
        workflow_graph = None
        try:
            workflow_graph = await get_productivity_graph()
        except RuntimeError:
            # The legacy Todo/chat path can operate without Calendar.
            pass
        prior = (await workflow_graph.aget_state({"configurable": {"thread_id": workflow_thread}})
                 if workflow_graph is not None else None)
        if workflow_graph and await _replay_completed_resume(
                workflow_graph, workflow_thread, prior, chat_id):
            return {"ok": True}
        if workflow_graph and await workflow_waits_for_edit(workflow_graph, workflow_thread):
            result, prior_id = await _resume_with_origin(
                workflow_graph, workflow_thread, text, prior)
            if prior_id is None:
                await _send_workflow_result(chat_id, workflow_thread, result)
            else:
                await _send_workflow_result(chat_id, workflow_thread, result,
                                            prior_checkpoint_id=prior_id)
        else:
            # A new turn enters the single Agent loop. Legacy pending approval
            # checkpoints above still resume on their original workflow graph.
            graph = await get_graph()
            agent_thread = f"official-v1:user_{telegram_user.id}"
            agent_prior = await graph.aget_state(
                {"configurable": {"thread_id": agent_thread}})
            if (agent_prior and agent_prior.values.get("user_id") not in {None, user_id}):
                raise PermissionError("AgentApprovalOwnerMismatch")
            agent_context = {"user_id": user_id,
                             "telegram_id": str(telegram_user.id),
                             "chat_id": str(chat_id), "origin": "telegram"}
            if agent_prior and await _replay_completed_resume(
                    graph, agent_thread, agent_prior, chat_id, agent=True,
                    agent_context=agent_context):
                return {"ok": True}
            if agent_prior and agent_prior.interrupts:
                from app.telegram.processing_context import current_run
                from langchain_core.messages import HumanMessage
                receipt = current_run()
                incoming_id = f"telegram-receipt:{receipt.receipt_id}" if receipt else ""
                if not any(isinstance(item, HumanMessage) and item.id == incoming_id
                           for item in agent_prior.values.get("messages", [])):
                    _record_no_business("approval_requires_decision")
                    await send_message(chat_id, "这项操作仍在等待审批，请先使用原审批按钮或工作台处理。")
                    return {"ok": True}
            response, agent_state = await run_agent(
                graph=graph, user_id=user_id, telegram_id=str(telegram_user.id),
                chat_id=str(chat_id), user_message=text,
                projection_session_factory=AsyncSessionLocal,
                return_state=True,
            )
            if agent_state.get("__interrupt__"):
                await _send_agent_result(chat_id, agent_thread, agent_state)
            else:
                await _record_checkpoint_outcome(graph, agent_thread, ordinary=True)
                await send_message(chat_id, response)

        return {"ok": True}
    finally:
        for token in reversed(locale_tokens):
            reset_preferences(token)
        end_trace(trace_token)

    

async def transcribe_voice(file_path: str) -> str:
    """Transcribe Telegram voice in its spoken language via Groq Whisper."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    async with httpx.AsyncClient(timeout=45.0) as client:
        with open(file_path, "rb") as f:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                data={"model": "whisper-large-v3"},
                files={"file": ("voice.ogg", f, "audio/ogg")},
            )
        response.raise_for_status()
        return response.json()["text"]


async def download_voice(file_id: str) -> str:
    """Download voice file from Telegram and return ldocal path."""
    
    
    bot = get_bot()
    
    # Get file path from Telegram
    file = await bot.get_file(file_id)
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await file.download_to_memory(tmp)
        return tmp.name
