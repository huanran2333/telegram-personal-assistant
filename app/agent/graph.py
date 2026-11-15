"""
Build and compile the LangGraph agent graph with PostgreSQL checkpointing.
"""
import uuid

from langgraph.graph import StateGraph, END

from app.agent.state import AgentRuntimeContext, AgentState
from app.agent.tools import make_todo_tools
from app.agent.unified_tools import make_unified_tools
from app.agent.unified_loop import make_unified_nodes
from app.agent.agent_approval import (approval_gate as agent_approval_gate,
                                      reject_action as agent_reject_action,
                                      request_edit as agent_request_edit,
                                      apply_agent_edit, execute_action as agent_execute_action)
from app.config import get_settings
from app.localization import tr
import psycopg
import logging
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

settings = get_settings()

_compiled_graph = None


def clear_graph_runtime() -> None:
    global _compiled_graph
    _compiled_graph = None


class DurableGraphInitializationError(RuntimeError):
    """Safe stage/category metadata; never include connection exception text."""

    def __init__(self, stage: str, category: str):
        self.stage = stage
        self.category = category
        super().__init__(f"Durable agent initialization failed at {stage} ({category})")


async def build_graph(db_session_factory, store=None, *, durable: bool = True,
                      memory_orchestrator=None, memory_runtime=None,
                      test_checkpointer=None):
    """
    Build and compile the agent graph.
    Call once at startup; reuse the returned graph for all requests.
    """
    global _compiled_graph
    _compiled_graph = None
    if memory_runtime is not None:
        memory_orchestrator = memory_runtime.orchestrator

    if not durable and settings.app_env.lower() not in {"development", "test"}:
        raise ValueError("Non-durable graph requires explicit development or test environment")
    if durable and test_checkpointer is not None:
        raise ValueError("Test checkpointer cannot replace the durable PostgreSQL Saver")

    todo_tools = make_todo_tools(db_session_factory)
    unified_tools = make_unified_tools(
        todo_tools, db_session_factory, memory_orchestrator=memory_orchestrator)
    agent_model, agent_tools = make_unified_nodes(
        unified_tools, memory_orchestrator=memory_orchestrator,
        memory_token_budget=getattr(settings, "memory_context_token_budget", 8192))
 
    builder = StateGraph(AgentState, context_schema=AgentRuntimeContext)
    builder.add_node("agent_model", agent_model)
    builder.add_node("agent_tools", agent_tools)
    builder.add_node("agent_approval_gate", agent_approval_gate)
    builder.add_node("agent_request_edit", agent_request_edit)
    builder.add_node("agent_apply_edit", apply_agent_edit)
    builder.add_node("agent_execute_action", agent_execute_action)
    builder.add_node("agent_reject_action", agent_reject_action)
 
    builder.set_entry_point("agent_model")
    def after_agent_model(state):
        return "execute_tools" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    builder.add_conditional_edges(
        "agent_model", after_agent_model,
        {"execute_tools": "agent_tools", "end": END})
    def after_agent_tool(state):
        if state.get("pending_agent_proposal"):
            return "approval"
        if int(state.get("pending_tool_index") or 0) < len(state.get("pending_tool_calls") or []):
            return "more"
        # A completed tool call settles only its own operation. Return the
        # result to the model so it can handle other requests in this turn.
        return "continue"

    def after_approved_tool(state):
        return ("more" if int(state.get("pending_tool_index") or 0) <
                len(state.get("pending_tool_calls") or []) else "continue")

    builder.add_conditional_edges("agent_tools", after_agent_tool,
                                  {"approval": "agent_approval_gate", "more": "agent_tools",
                                   "continue": "agent_model"})
    builder.add_conditional_edges("agent_approval_gate", lambda state: state["approval_decision"],
                                  {"approve": "agent_execute_action",
                                   "edit": "agent_request_edit", "reject": "agent_reject_action"})
    builder.add_edge("agent_request_edit", "agent_apply_edit")
    builder.add_conditional_edges("agent_apply_edit", lambda state: (
        "approval" if state.get("pending_agent_proposal") else "done"),
        {"approval": "agent_approval_gate", "done": "agent_model"})
    for completed in ("agent_execute_action", "agent_reject_action"):
        builder.add_conditional_edges(completed, after_approved_tool,
                                      {"more": "agent_tools", "continue": "agent_model"})
 
    logger.info(f"✅ Nodes and edges added to builder")
    logger.info(f"DEBUG nodes in builder: {list(builder.nodes.keys())}")

    # Non-persistent evaluation is an explicit caller choice. A production
    # dependency failure must never turn into this branch.
    if not durable:
        options = {"store": store} if store is not None else {}
        if test_checkpointer is not None:
            options["checkpointer"] = test_checkpointer
        _compiled_graph = builder.compile(**options)
        logger.info("Graph compiled without checkpointer for explicit non-production evaluation")
        return _compiled_graph

    conn = None
    stage = "saver_connect"
    try:
        conn_string = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")
        conn = await psycopg.AsyncConnection.connect(conn_string, autocommit=True)
        stage = "saver_construct"
        checkpointer = AsyncPostgresSaver(conn)
        stage = "saver_setup"
        await checkpointer.setup()
        compile_kwargs = {"checkpointer": checkpointer}
        if store is not None:
            compile_kwargs["store"] = store
        stage = "graph_compile"
        compiled = builder.compile(**compile_kwargs)
        if compiled.checkpointer is None:
            raise RuntimeError("Durable graph compiled without a checkpointer")
        _compiled_graph = compiled
        logger.info("Graph compiled with PostgreSQL checkpointer")
    except Exception as exc:
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                logger.warning("Failed to close unavailable Saver connection category=%s",
                               type(exc).__name__)
        raise DurableGraphInitializationError(stage, type(exc).__name__) from exc
    return _compiled_graph


async def get_graph():
    """Return the compiled graph (must call build_graph first)."""
    if _compiled_graph is None:
        raise RuntimeError("Graph not built. Call build_graph() at startup.")
    return _compiled_graph


async def run_agent(graph, user_id: str, telegram_id: str, chat_id: str, user_message: str,
                    projection_session_factory=None, *, thread_id: str | None = None,
                    turn_id: str | None = None, origin: str = "telegram",
                    memory_history_threads: tuple[str, ...] = (),
                    return_state: bool = False):
    """
    Invoke the agent for a user message and return the assistant's text response.
    thread_id is per-user to maintain conversation history.
    """
    from langchain_core.messages import HumanMessage
    from datetime import datetime, timezone

    logger.info("Running agent user_id=%s message_chars=%s", user_id, len(user_message))

    selected_thread = thread_id or f"user_{telegram_id}"
    if origin not in {"telegram", "web"} or (origin == "web" and not turn_id):
        raise ValueError("Web Agent requires an explicit stable turn identity")
    config = {
        "configurable": {
            "thread_id": selected_thread,
        },
        
        "metadata": {
            "user_id": user_id,
            "origin": origin,
        },
        "tags": [origin, "personal-assistant"],
    }

    existing = await graph.aget_state(config)
    has_existing = bool(existing and existing.values)
    if has_existing and existing.values.get("user_id") != user_id:
        raise PermissionError("CheckpointOwnerMismatch")
    from app.telegram.processing_context import current_run
    receipt_run = current_run()
    incoming_id = (f"telegram-receipt:{receipt_run.receipt_id}" if receipt_run is not None
                   else turn_id or str(uuid.uuid4()))
    # A committed Saver state can be ahead of the receipt/outbox transaction.
    # On replay, never inject the same Telegram update as a fresh user turn.
    already_in_checkpoint = bool(has_existing and (receipt_run is not None or turn_id) and any(
        isinstance(message, HumanMessage) and message.id == incoming_id
        for message in existing.values.get("messages", [])))
    if (has_existing and not already_in_checkpoint and
            any(node in existing.next for node in
                ("agent_approval_gate", "agent_request_edit"))):
        raise RuntimeError("AgentApprovalPending")
    incoming = HumanMessage(id=incoming_id, content=user_message,
                            additional_kwargs={"occurred_at_utc": datetime.now(timezone.utc).isoformat()})
    runtime_context: AgentRuntimeContext = {
        "user_id": user_id, "telegram_id": telegram_id,
        "chat_id": chat_id, "origin": origin,
    }

    if has_existing:
        # Resume from checkpoint while resetting only per-turn execution fields.
        input_state = {
            "messages": [incoming],
            "active_request": user_message,
            "active_turn_id": incoming_id,
            "request_origin": origin,
            "completed_read_summaries": [],
            "handoff_summary": "",
            "meeting_evidence_pending": False,
            "validated_meeting_response": "",
            "meeting_submission_attempts": 0,
            "meeting_validation_exhausted": False,
            "offered_tools": [],
            "pending_tool_calls": [],
            "pending_tool_index": 0,
            "pending_agent_proposal": {},
            "pending_agent_call_id": "",
            "proposal": {},
            "approval_decision": "",
            "approval_web_request_id": None,
            "edit_web_request_id": None,
            "agent_edit_instruction": "",
            "last_resolved_action_id": "",
            "unified_tool_rounds": 0,
            "budget_blocked_tool_calls": 0,
            "capability_failed": False,
            "read_tool_rounds": 0,
            "read_sources_seen": [],
            "memory_context_payload": "",
            "memory_context_prepared": False,
            "memory_search_rounds": 0,
        }
        if origin == "web":
            # Keep server-derived memory scopes current even when the Web
            # checkpoint predates an added authorized conversation thread.
            input_state["conversation_thread_id"] = selected_thread
            input_state["memory_history_threads"] = list(memory_history_threads)
    else:
        input_state = {
            "messages": [incoming],
            "user_id": user_id,
            "telegram_id": telegram_id,
            "chat_id": chat_id,
            "conversation_thread_id": selected_thread,
            "memory_history_threads": list(memory_history_threads),
            "active_request": user_message,
            "active_turn_id": incoming_id,
            "request_origin": origin,
            "completed_read_summaries": [],
            "handoff_summary": "",
            "meeting_evidence_pending": False,
            "validated_meeting_response": "",
            "meeting_submission_attempts": 0,
            "meeting_validation_exhausted": False,
            "offered_tools": [],
            "pending_tool_calls": [],
            "pending_tool_index": 0,
            "pending_agent_proposal": {},
            "pending_agent_call_id": "",
            "proposal": {},
            "approval_decision": "",
            "approval_web_request_id": None,
            "edit_web_request_id": None,
            "agent_edit_instruction": "",
            "agent_action_outcomes": {},
            "last_resolved_action_id": "",
            "unified_tool_rounds": 0,
            "budget_blocked_tool_calls": 0,
            "capability_failed": False,
            "read_tool_rounds": 0,
            "read_sources_seen": [],
            "recent_gmail_thread_id": None,
            "recent_gmail_owner_telegram_id": None,
            "last_action": "",
            "memory_context_payload": "",
            "memory_context_prepared": False,
        }

    if already_in_checkpoint:
        result = (await graph.ainvoke(None, config=config, context=runtime_context) if existing.next
                  else existing.values)
    else:
        result = await graph.ainvoke(input_state, config=config, context=runtime_context)

    if projection_session_factory is not None:
        # Saver has already committed. Projection is derived and must not turn
        # a successful user action into a retryable Telegram failure.
        try:
            from app.agent.transcript import sync_checkpoint_messages
            from app.trace import emit_trace
            recent = result.get("messages", [])
            for index in range(len(recent) - 1, -1, -1):
                if isinstance(recent[index], HumanMessage):
                    recent = recent[index:]
                    break
            async with projection_session_factory() as projection_db:
                await sync_checkpoint_messages(projection_db, user_id=user_id,
                                               thread_id=selected_thread, messages=recent)
                await projection_db.commit()
            emit_trace("transcript_projection", operation="ordinary_sync", success=True)
        except Exception as exc:
            logger.warning("transcript_projection stage=sync category=%s", type(exc).__name__)
            from app.trace import emit_trace
            emit_trace("transcript_projection", operation="ordinary_sync", success=False,
                       error_category=type(exc).__name__)

    # Extract the last AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
            content = msg.content
            if isinstance(content, list):
                # Handle structured content blocks
                text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
                response = " ".join(text_parts).strip() or tr("done")
                return (response, result) if return_state else response
            response = str(content).strip() or tr("done")
            return (response, result) if return_state else response

    response = tr("processed")
    return (response, result) if return_state else response
