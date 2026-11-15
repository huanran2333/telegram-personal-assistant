"""Retired Web Agent custody, retained for historical tests and records.

Current production routes never submit a WebRequest and app startup never
starts WebRunWorker. Telegram Receipt owns all Agent execution and resume.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import User, WebRequest
from app.telegram.durable import thread_invocation_guard
from app.web.processing_context import web_run_scope

logger = logging.getLogger(__name__)


class WebRequestRejected(ValueError):
    """Input/approval mismatch proven before any new business side effect."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


async def submit_run(factory, *, user_id: str, operation: str,
                     idempotency_key: str, body: dict,
                     thread_id: str | None = None) -> tuple[WebRequest, bool]:
    # Historical WebRequest rows remain loadable for custody/recovery; their
    # chat/form execution paths are retired below and no public route creates them.
    if operation not in {"chat", "approval", "edit_instruction", "form_proposal"}:
        raise ValueError("UnsupportedWebOperation")
    if not idempotency_key or len(idempotency_key) > 96:
        raise ValueError("InvalidIdempotencyKey")
    identifier = str(uuid4())
    thread = thread_id or f"web:entry:{user_id}"
    async with factory() as db:
        inserted = await db.scalar(pg_insert(WebRequest).values(
            id=identifier, user_id=user_id, operation=operation,
            idempotency_key=idempotency_key, request_hash=_hash(body),
            request_body=body, thread_id=thread,
            turn_id=f"web-request:{identifier}", state="ACCEPTED",
        ).on_conflict_do_nothing(constraint="uq_web_request_key").returning(WebRequest.id))
        row = await db.scalar(select(WebRequest).where(
            WebRequest.user_id == user_id, WebRequest.operation == operation,
            WebRequest.idempotency_key == idempotency_key))
        if (row is None or row.request_hash != _hash(body)
                or (thread_id is not None and row.thread_id != thread)):
            raise ValueError("IdempotencyConflict")
        await db.commit()
        return row, inserted is not None


async def claim_one(factory, *, owner: str, lease_seconds: int) -> tuple[str, str] | None:
    """One due request per user at a time, with PostgreSQL row fencing."""
    async with factory() as db:
        candidates = (await db.scalars(select(WebRequest).where(
            (WebRequest.state == "ACCEPTED") |
            ((WebRequest.state == "RUNNING") & (WebRequest.lease_until < _now())) |
            ((WebRequest.state == "WAITING_EXTERNAL") &
             (WebRequest.updated_at < _now() - timedelta(seconds=10)))
        ).order_by(WebRequest.created_at, WebRequest.id).with_for_update(skip_locked=True)
          .limit(20))).all()
        for row in candidates:
            earlier = await db.scalar(select(WebRequest.id).where(
                WebRequest.user_id == row.user_id, WebRequest.id != row.id,
                WebRequest.state.in_(("ACCEPTED", "RUNNING")),
                ((WebRequest.created_at < row.created_at) |
                 ((WebRequest.created_at == row.created_at) & (WebRequest.id < row.id)))
            ).limit(1))
            if earlier is not None:
                continue
            token = str(uuid4())
            row.state = "RUNNING"
            row.lease_owner = owner
            row.fence_token = token
            row.lease_until = _now() + timedelta(seconds=lease_seconds)
            row.attempts += 1
            row.updated_at = _now()
            await db.commit()
            return row.id, token
        await db.rollback()
    return None


async def finish_run(factory, run_id: str, fence: str, *, state: str,
                     result: dict | None = None, safe_error: str | None = None) -> bool:
    if state not in {"COMPLETED", "FAILED", "WAITING_APPROVAL", "WAITING_EDIT", "WAITING_EXTERNAL"}:
        raise ValueError("InvalidWebResultState")
    async with factory() as db:
        row = await db.scalar(select(WebRequest).where(WebRequest.id == run_id).with_for_update())
        if row is None or row.state != "RUNNING" or row.fence_token != fence:
            return False
        row.state, row.result, row.safe_error = state, result, safe_error
        row.lease_owner = row.fence_token = row.lease_until = None
        row.updated_at = _now()
        if (row.operation in {"approval", "edit_instruction"} and state != "FAILED"
                and (row.request_body or {}).get("origin") != "telegram"):
            parent_id = (row.request_body or {}).get("parent_run_id")
            parent = await db.scalar(select(WebRequest).where(
                WebRequest.id == parent_id,
                WebRequest.user_id == row.user_id).with_for_update())
            if parent and parent.state in {
                    "WAITING_APPROVAL", "WAITING_EDIT", "WAITING_EXTERNAL"}:
                # The decision and the user-visible parent run outcome must
                # survive or roll back together. Saver/ActionExecution remain
                # the approval/business authorities; this is only Web status.
                parent.state, parent.result = state, result
                parent.updated_at = _now()
        await db.commit()
        return True


async def _workflow_result(graph, thread_id: str, result: dict, factory,
                           *, completed_summary: str = "") -> tuple[str, dict]:
    from app.agent.workflow import bind_continuation_checkpoint
    from app.agent.transcript import sync_workflow_turn
    await bind_continuation_checkpoint(graph, thread_id, result)
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values if snapshot else {}
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = getattr(interrupts[0], "value", {})
        kind = payload.get("kind") if isinstance(payload, dict) else None
        if kind == "action_proposal":
            proposal = values.get("proposal") or {}
            response = {"pending_approval": {
                "action_id": proposal.get("action_id"),
                "preview": proposal.get("human_readable_preview", ""),
                "risk": proposal.get("risk_level"),
                "checkpoint_version": (snapshot.config.get("configurable") or {}).get("checkpoint_id"),
            }}
            if completed_summary:
                response["pending_approval"]["completed_summary"] = completed_summary[:2800]
            state = "WAITING_APPROVAL"
        elif kind == "edit_request":
            response, state = {"pending_edit": True}, "WAITING_EDIT"
        elif kind == "calendar_confirmation":
            response, state = {"response": str(payload.get("message", "尚未确认"))}, "WAITING_EXTERNAL"
        else:
            response, state = {"response": "需要进一步确认"}, "WAITING_APPROVAL"
    else:
        response, state = {"response": str(result.get("response", "已处理"))}, "COMPLETED"
    if values.get("transcript_turn_id") and values.get("user_id") and values.get("request"):
        try:
            async with factory() as db:
                await sync_workflow_turn(
                    db, user_id=values["user_id"], thread_id=thread_id,
                    turn_id=values["transcript_turn_id"], request=values["request"],
                    response=response.get("response", "") if state == "COMPLETED" else "",
                    occurred_at=values.get("transcript_occurred_at"))
                await db.commit()
        except Exception as exc:
            logger.warning("web_projection stage=sync category=%s", type(exc).__name__)
    return state, response


async def _agent_approval_result(graph, thread_id: str, result: dict) -> tuple[str, dict]:
    """Project an interrupt from the original Agent thread, not a second workflow."""
    from langchain_core.messages import AIMessage
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values if snapshot else {}
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = getattr(interrupts[0], "value", {})
        if payload.get("type") == "tool_approval":
            from app.agent.official_approval_view import pending_official_approvals
            pending = pending_official_approvals(snapshot, user_id=values.get("user_id"))
            if not pending:
                # A per-call Saver snapshot may retain resolved interrupt
                # objects. Paired ToolMessages, not the stale objects, decide
                # whether another approval is still outstanding.
                if snapshot.next:
                    raise RuntimeError("OfficialApprovalInterruptWithoutPendingAction")
            else:
                checkpoint = (snapshot.config.get("configurable") or {}).get("checkpoint_id")
                return "WAITING_APPROVAL", {"pending_approvals": [{
                    "action_id": item.proposal.action_id,
                    "preview": item.proposal.human_readable_preview,
                    "risk": item.proposal.risk_level.value,
                    "checkpoint_version": checkpoint,
                } for item in pending]}
        elif payload.get("kind") == "action_proposal":
            proposal = values.get("proposal") or {}
            if proposal.get("action_id") != payload.get("action_id"):
                raise RuntimeError("AgentApprovalCheckpointMismatch")
            return "WAITING_APPROVAL", {"pending_approval": {
                "action_id": proposal["action_id"],
                "preview": proposal.get("human_readable_preview", ""),
                "risk": proposal.get("risk_level"),
                "checkpoint_version": (snapshot.config.get("configurable") or {}).get("checkpoint_id"),
                "completed_summary": values.get("handoff_summary") or "",
            }}
        elif payload.get("kind") == "edit_request":
            return "WAITING_EDIT", {"pending_edit": True}
        elif payload.get("type") != "tool_approval":
            raise RuntimeError("UnexpectedAgentInterrupt")
    answer = next((str(item.content) for item in reversed(values.get("messages", []))
                   if isinstance(item, AIMessage) and not item.tool_calls and item.content),
                  "已处理")
    return "COMPLETED", {"response": answer}


async def _execute(factory, row: WebRequest, fence: str) -> tuple[str, dict]:
    from app.agent.official_runtime import get_graph
    from app.agent.workflow import resume_workflow
    from app.agent.workflow_runtime import get_productivity_graph

    async with factory() as db:
        user = await db.get(User, row.user_id)
        if user is None:
            raise RuntimeError("WebOwnerUnavailable")
        telegram_id = user.telegram_id
    if row.operation not in {"approval", "edit_instruction"}:
        raise WebRequestRejected("WebAgentRemoved")
    origin = (row.request_body or {}).get("origin")
    if origin != "telegram":
        raise WebRequestRejected("ApprovalOriginInvalid")
    workflow_thread = f"workflow_user_{telegram_id}"
    official_approval = bool(row.request_body.get("official_approval"))
    expected_approval_thread = (
        f"official-v1:user_{telegram_id}" if official_approval else
        f"user_{telegram_id}" if row.request_body.get("agent_approval") else workflow_thread)
    if (row.operation in {"approval", "edit_instruction"} and origin == "telegram"
            and (not telegram_id or row.thread_id != expected_approval_thread)):
        raise WebRequestRejected("ApprovalThreadMismatch")
    if row.operation in {"approval", "edit_instruction"}:
        agent_approval = bool(row.request_body.get("agent_approval"))
        approval_thread = (f"official-v1:user_{telegram_id}" if official_approval else
                           f"user_{telegram_id}") if agent_approval else workflow_thread
        graph = await (get_graph() if agent_approval else get_productivity_graph())
        snapshot = await graph.aget_state({"configurable": {"thread_id": approval_thread}})
        if (not snapshot or snapshot.values.get("user_id") != row.user_id or
                (origin == "telegram" and snapshot.values.get("telegram_id") != telegram_id)):
            raise WebRequestRejected("ApprovalUnavailable")
        expected = row.request_body.get("checkpoint_version")
        current = (snapshot.config.get("configurable") or {}).get("checkpoint_id")
        if current != expected:
            if official_approval:
                from app.agent.official_approval_view import committed_official_web_decision
                committed = await committed_official_web_decision(
                    graph, approval_thread, user_id=row.user_id,
                    action_id=row.request_body.get("action_id"),
                    decision=row.request_body.get("decision"),
                    request_id=row.id, checkpoint_id=expected)
                if committed is None:
                    # An unrelated entry might have advanced this thread.
                    # Do not replay or infer approval from a changed version.
                    raise WebRequestRejected("ApprovalVersionConflict")
                committed_turn = committed.values.get("active_turn_id")
                if (committed_turn and snapshot.values.get("active_turn_id") ==
                        committed_turn):
                    return await _agent_approval_result(
                        graph, approval_thread,
                        {"__interrupt__": snapshot.interrupts})
                return "COMPLETED", {
                    "response": "该审批已由本次请求处理；对话后来又有更新，请查看当前业务状态。",
                    "action_id": row.request_body.get("action_id"),
                    "reconciled_from_saver": True,
                }
            # The same decision may already have reached Saver before the Web
            # result row was finalized. Never issue Command(resume) twice.
            marker = ("approval_web_request_id" if row.operation == "approval"
                      else "edit_web_request_id")
            if snapshot.values.get(marker) != row.id:
                raise WebRequestRejected("ApprovalVersionConflict")
            if snapshot.next:
                interrupts = [item for task in snapshot.tasks for item in getattr(task, "interrupts", ())]
                result = {"__interrupt__": interrupts, **snapshot.values}
            else:
                result = snapshot.values
            return (await _agent_approval_result(graph, approval_thread, result)
                    if agent_approval else
                    await _workflow_result(graph, approval_thread, result, factory))
        proposal = snapshot.values.get("proposal") or {}
        if official_approval:
            from app.agent.official_approval_view import official_decision_for_action
            from langgraph.types import Command
            if row.operation != "approval":
                raise WebRequestRejected("OfficialApprovalEditRequiresNewRequest")
            try:
                payload = official_decision_for_action(
                    snapshot, user_id=row.user_id,
                    action_id=row.request_body.get("action_id"),
                    decision=row.request_body.get("decision"),
                    checkpoint_id=expected)
            except ValueError as exc:
                raise WebRequestRejected(str(exc)) from None
            agent_context = {"user_id": row.user_id,
                             "telegram_id": str(snapshot.values.get("telegram_id") or ""),
                             "chat_id": str(snapshot.values.get("chat_id") or ""),
                             "origin": str(snapshot.values.get("request_origin") or "web")}
            # Saver history retains this server-owned origin marker. If the
            # worker crashes after Saver commit but before finish_run, its
            # retry can prove attribution without replaying the decision or
            # adding another field to AgentState.
            result = await graph.ainvoke(
                Command(resume=payload),
                config={"configurable": {"thread_id": approval_thread},
                        "metadata": {"web_request_id": row.id,
                                     "approval_action_id": row.request_body["action_id"],
                                     "approval_decision": row.request_body["decision"]}},
                context=agent_context)
            return await _agent_approval_result(graph, approval_thread, result)
        if row.operation == "approval":
            if (("agent_approval_gate" if agent_approval else "approval_gate") not in snapshot.next or
                    proposal.get("action_id") != row.request_body.get("action_id")):
                raise WebRequestRejected("ApprovalUnavailable")
            payload = {"decision": row.request_body["decision"], "web_request_id": row.id}
        else:
            if ("agent_request_edit" if agent_approval else "request_edit") not in snapshot.next:
                raise WebRequestRejected("EditUnavailable")
            payload = {"instruction": row.request_body["instruction"],
                       "web_request_id": row.id}
        agent_context = ({"user_id": row.user_id, "telegram_id": telegram_id,
                          "chat_id": str(snapshot.values.get("chat_id") or ""),
                          "origin": str(snapshot.values.get("request_origin") or "web")}
                         if agent_approval else None)
        result = await resume_workflow(graph, approval_thread, payload,
                                       context=agent_context)
        return (await _agent_approval_result(graph, approval_thread, result)
                if agent_approval else
                await _workflow_result(graph, approval_thread, result, factory))

class WebRunWorker:
    def __init__(self, session_factory, database_url: str, *, lease_seconds: int = 120):
        self.factory = session_factory
        self.database_url = database_url
        self.lease_seconds = lease_seconds
        self.owner = str(uuid4())
        self._stopping = asyncio.Event()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._active: set[asyncio.Task] = set()

    @property
    def accepting(self) -> bool:
        """A dead or draining recovery loop cannot accept new Web custody."""
        return (self._task is not None and not self._task.done()
                and not self._stopping.is_set())

    async def preflight(self) -> None:
        from sqlalchemy import text
        async with self.factory() as db:
            await db.execute(text("SELECT 1 FROM web_requests LIMIT 0"))
            await db.execute(text("SELECT 1 FROM web_sessions LIMIT 0"))

    async def start(self) -> None:
        await self.preflight()
        self._task = asyncio.create_task(self._loop())

    def wake(self) -> None:
        self._wakeup.set()

    async def quiesce(self) -> None:
        self._stopping.set()
        self._wakeup.set()

    async def drain(self, deadline: float) -> None:
        await self.quiesce()
        tasks = {task for task in self._active if not task.done()}
        if self._task is not None:
            tasks.add(self._task)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=max(0, deadline - asyncio.get_running_loop().time()))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                claimed = await claim_one(self.factory, owner=self.owner,
                                          lease_seconds=self.lease_seconds)
                if claimed is not None and not self._stopping.is_set():
                    task = asyncio.create_task(self.process(*claimed))
                    self._active.add(task)
                    task.add_done_callback(self._active.discard)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("web_worker stage=sweep category=%s", type(exc).__name__)
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self._wakeup.clear()

    async def _heartbeat(self, run_id: str, fence: str) -> None:
        while True:
            await asyncio.sleep(max(1.0, self.lease_seconds / 3))
            async with self.factory() as db:
                row = await db.scalar(select(WebRequest).where(
                    WebRequest.id == run_id).with_for_update())
                if row is None or row.state != "RUNNING" or row.fence_token != fence:
                    return
                row.lease_until = _now() + timedelta(seconds=self.lease_seconds)
                await db.commit()

    async def process(self, run_id: str, fence: str) -> None:
        async with self.factory() as db:
            row = await db.get(WebRequest, run_id)
        if row is None:
            return
        heartbeat = asyncio.create_task(self._heartbeat(run_id, fence))
        try:
            # R5 background continuation uses the workflow advisory lock.
            # Share that same lock even before routing so a Web approval/read
            # cannot race a late Calendar confirmation on the Saver thread.
            # Ordinary Web requests are serialized per user as well; distinct
            # users retain independent lock keys.
            target_thread = (row.thread_id if row.operation in {"approval", "edit_instruction"}
                             and (row.request_body or {}).get("origin") == "telegram"
                             else f"web:workflow:{row.user_id}")
            async with thread_invocation_guard(self.database_url, "workflow", target_thread):
                with web_run_scope(run_id, row.user_id, fence):
                    state, result = await _execute(self.factory, row, fence)
            await finish_run(self.factory, run_id, fence, state=state, result=result)
        except asyncio.CancelledError:
            # Leave the armed lease for restart/expiry; no fake business result.
            raise
        except (PermissionError, WebRequestRejected) as exc:
            category = (str(exc) if isinstance(exc, WebRequestRejected) and str(exc) in {
                "ApprovalVersionConflict", "ApprovalUnavailable", "EditUnavailable",
                "ApprovalThreadMismatch", "ApprovalOriginInvalid", "AgentApprovalPending"}
                else type(exc).__name__)
            await finish_run(self.factory, run_id, fence, state="FAILED",
                             safe_error=category)
        except Exception as exc:
            # Saver/ActionExecution may already have committed. Lease expiry
            # drives replay/readback of the same stable turn; do not fail it.
            logger.warning("web_worker stage=execute category=%s", type(exc).__name__)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
