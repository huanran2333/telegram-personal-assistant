"""Current unified Agent entry using LangChain's ToolNode and per-call HITL.

Old checkpoints are never read as this graph's state. A one-time, owner-checked
history migration must initialize the versioned thread before existing users
are switched to this entry.
"""
from __future__ import annotations

import logging
import uuid
from datetime import timezone

import psycopg
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.graph import DurableGraphInitializationError
from app.agent.business_clock import business_now
from app.agent.nodes import _build_chat_llm
from app.agent.official_graph import build_official_agent
from app.agent.official_model_failover import configured_failover
from app.config import get_settings
from app.localization import tr

logger = logging.getLogger(__name__)
_compiled_graph = None
_legacy_reader = None
_migration_memory_orchestrator = None


def clear_graph_runtime() -> None:
    global _compiled_graph, _legacy_reader, _migration_memory_orchestrator
    _compiled_graph = None
    _legacy_reader = None
    _migration_memory_orchestrator = None


async def close_legacy_reader() -> None:
    """Release the temporary read-only Saver connection at app shutdown."""
    global _legacy_reader
    reader = _legacy_reader
    _legacy_reader = None
    saver = getattr(reader, "checkpointer", None)
    connection = getattr(saver, "conn", None)
    if connection is not None:
        await connection.close()


async def build_graph(db_session_factory, store=None, *, durable=True,
                      memory_orchestrator=None, memory_runtime=None,
                      test_checkpointer=None, model=None):
    """Build the production official Agent; never silently downgrade Saver."""
    global _compiled_graph, _legacy_reader, _migration_memory_orchestrator
    _compiled_graph = None
    settings = get_settings()
    if memory_runtime is not None:
        memory_orchestrator = memory_runtime.orchestrator
    _migration_memory_orchestrator = memory_orchestrator
    if not durable and settings.app_env.lower() not in {"development", "test"}:
        raise ValueError("Non-durable graph requires development or test")
    if durable and test_checkpointer is not None:
        raise ValueError("Test checkpointer cannot replace PostgreSQL Saver")

    async def construct(checkpointer):
        failover = configured_failover(settings) if model is None else None
        primary = model or _build_chat_llm(
            settings.llm_provider, settings.llm_model,
            disable_qwen_thinking=True,
            request_timeout=(settings.llm_primary_timeout_seconds
                             if failover is not None else None))
        graph, _ = build_official_agent(
            model=primary,
            session_factory=db_session_factory, settings=settings,
            checkpointer=checkpointer, store=store,
            memory_orchestrator=memory_orchestrator,
            interrupt_mode="per_call", model_failover=failover)
        return graph

    if not durable:
        _compiled_graph = await construct(test_checkpointer)
        return _compiled_graph
    conn = None
    stage = "saver_connect"
    try:
        conn_string = settings.database_url_sync.replace(
            "postgresql+psycopg://", "postgresql://")
        conn = await psycopg.AsyncConnection.connect(conn_string, autocommit=True)
        stage = "saver_construct"
        saver = AsyncPostgresSaver(conn)
        stage = "saver_setup"
        await saver.setup()
        # Transitional read-only reader: old Saver channels are not visible
        # through the official graph's narrower State schema. Never invoke
        # this graph for a new user turn.
        stage = "legacy_reader_construct"
        from app.agent import graph as legacy
        _legacy_reader = await legacy.build_graph(
            db_session_factory, store=store, durable=True,
            memory_orchestrator=memory_orchestrator,
            memory_runtime=memory_runtime)
        stage = "graph_compile"
        _compiled_graph = await construct(saver)
        if _compiled_graph.checkpointer is None:
            raise RuntimeError("MissingCheckpointer")
        return _compiled_graph
    except Exception as exc:
        if conn is not None:
            await conn.close()
        if _legacy_reader is not None and _legacy_reader.checkpointer is not None:
            await _legacy_reader.checkpointer.conn.close()
            _legacy_reader = None
        raise DurableGraphInitializationError(stage, type(exc).__name__) from exc


async def get_graph():
    if _compiled_graph is None:
        raise RuntimeError("Official Agent graph has not been built")
    return _compiled_graph


async def assert_legacy_approvals_cleared(session_factory) -> None:
    """Fail production readiness before an old pending action becomes orphaned.

    This is read-only. Cancellation and user notification require a separate
    audited operation; this function never calls an old approval executor.
    """
    if _legacy_reader is None:
        raise RuntimeError("LegacyApprovalReaderUnavailable")
    from sqlalchemy import select
    from app.db.models import User, WebRequest
    async with session_factory() as db:
        owners = (await db.execute(select(User.id, User.telegram_id))).all()
        old_waiting = await db.scalar(select(WebRequest.id).where(
            WebRequest.state.in_(("WAITING_APPROVAL", "WAITING_EDIT")),
            WebRequest.graph_kind == "ordinary",
            WebRequest.thread_id.like("web:ordinary:%")).limit(1))
    if old_waiting is not None:
        raise RuntimeError("LegacyWebApprovalsRequireCancellation")
    for user_id, telegram_id in owners:
        for thread in (f"user_{telegram_id}", f"web:ordinary:{user_id}"):
            if not await _legacy_reader.checkpointer.aget_tuple({
                    "configurable": {"thread_id": thread}}):
                continue
            snapshot = await _legacy_reader.aget_state({
                "configurable": {"thread_id": thread}})
            if snapshot.values.get("user_id") != user_id:
                raise PermissionError("LegacyCheckpointOwnerMismatch")
            if snapshot.next or snapshot.interrupts:
                raise RuntimeError("LegacyApprovalsRequireCancellation")


def official_thread_id(thread_id: str) -> str:
    return thread_id if thread_id.startswith("official-v1:") else f"official-v1:{thread_id}"


async def run_agent(graph, user_id: str, telegram_id: str, chat_id: str,
                    user_message: str, projection_session_factory=None, *,
                    thread_id: str | None = None, turn_id: str | None = None,
                    origin: str = "telegram", memory_history_threads=(),
                    return_state: bool = False):
    """Invoke one authenticated turn on a versioned official Saver thread."""
    if origin not in {"telegram", "web"} or (origin == "web" and not turn_id):
        raise ValueError("Web Agent requires a stable turn identity")
    from app.telegram.processing_context import current_run
    receipt = current_run()
    incoming_id = (f"telegram-receipt:{receipt.receipt_id}" if receipt
                   else turn_id or str(uuid.uuid4()))
    selected_thread = official_thread_id(thread_id or f"user_{telegram_id}")
    config = {"configurable": {"thread_id": selected_thread},
              "metadata": {"user_id": user_id, "origin": origin},
              "tags": [origin, "personal-assistant", "official-v1"],
              "recursion_limit": 50}
    snapshot = await graph.aget_state(config)
    existing = bool(snapshot and snapshot.values)
    if existing and snapshot.values.get("user_id") != user_id:
        raise PermissionError("CheckpointOwnerMismatch")
    replay = existing and any(
        isinstance(message, HumanMessage) and message.id == incoming_id
        for message in snapshot.values.get("messages", ()))
    if existing and snapshot.interrupts and not replay:
        raise RuntimeError("AgentApprovalPending")
    if existing and snapshot.next and not snapshot.interrupts and not replay:
        # A failed tool superstep may have committed a Todo whose result is
        # still pending in the Saver. Only its original durable ingress may
        # resume that checkpoint; a new Receipt must wait for that turn.
        raise RuntimeError("IncompleteAgentTurnRequiresOriginalIngress")
    context = {"user_id": user_id, "telegram_id": telegram_id,
               "chat_id": chat_id, "origin": origin}
    if replay:
        # The Saver is ahead of the ingress acknowledgment. A pending decision
        # is handled by the separate callback; never inject this turn twice.
        result = snapshot.values
        if snapshot.interrupts:
            result = {**result, "__interrupt__": snapshot.interrupts}
        elif snapshot.next:
            # A tool batch may have committed a local side effect before a
            # sibling tool or feedback failed. Resume the checkpointed node
            # only under the original durable ingress identity: Todo replay
            # proof then binds the same call ID and argument digest.
            from app.web.processing_context import current_web_run
            web_run = current_web_run()
            if receipt is None and (web_run is None or
                                    web_run.user_id != user_id or
                                    incoming_id != f"web-request:{web_run.request_id}"):
                raise RuntimeError("IncompleteAgentTurnRequiresDurableIngress")
            result = await graph.ainvoke(None, config=config, context=context)
    else:
        incoming = HumanMessage(
            id=incoming_id, content=user_message,
            additional_kwargs={"occurred_at_utc": business_now(timezone.utc).isoformat()})
        migrated_messages = []
        if not existing:
            old_thread = selected_thread.removeprefix("official-v1:")
            # The new graph's State schema filters old-only owner fields when
            # reading an old checkpoint. Presence must be checked through the
            # raw Saver; migration requires the OLD graph schema as reader.
            if graph.checkpointer is not None and await graph.checkpointer.aget_tuple(
                    {"configurable": {"thread_id": old_thread}}):
                if _legacy_reader is None or _migration_memory_orchestrator is None:
                    raise RuntimeError("HistoryMigrationRequired")
                from app.agent.recent_history_migration import select_recent_history
                old_snapshot = await _legacy_reader.aget_state({
                    "configurable": {"thread_id": old_thread}})
                suppressed = await _migration_memory_orchestrator.suppressed_message_ids(
                    user_id=user_id, thread_id=old_thread)
                terms = await _migration_memory_orchestrator.suppressed_terms(
                    user_id=user_id)
                migrated = select_recent_history(
                    old_snapshot, user_id=user_id, thread_id=old_thread,
                    telegram_id=telegram_id if origin == "telegram" else None,
                    suppressed_message_ids=frozenset(suppressed),
                    suppressed_terms=terms)
                migrated_messages = list(migrated.messages)
                logger.info("official_history_migration retained_turns=%s dropped_turns=%s",
                            migrated.retained_turns, migrated.dropped_turns)
        update = {"messages": [*migrated_messages, incoming], "active_request": user_message,
                  "active_turn_id": incoming_id, "request_origin": origin,
                  "conversation_thread_id": selected_thread,
                  "memory_history_threads": list(memory_history_threads),
                  "memory_context_payload": "", "memory_context_prepared": False}
        if not existing:
            update.update(user_id=user_id, telegram_id=telegram_id, chat_id=chat_id)
        result = await graph.ainvoke(update, config=config, context=context)

    if projection_session_factory is not None and not result.get("__interrupt__"):
        try:
            from app.agent.transcript import sync_checkpoint_messages
            recent = result.get("messages", [])
            for index in range(len(recent) - 1, -1, -1):
                if isinstance(recent[index], HumanMessage):
                    recent = recent[index:]
                    break
            async with projection_session_factory() as db:
                await sync_checkpoint_messages(
                    db, user_id=user_id, thread_id=selected_thread, messages=recent)
                await db.commit()
        except Exception as exc:
            logger.warning("transcript_projection stage=sync category=%s",
                           type(exc).__name__)
    answer = next((str(message.content).strip()
                   for message in reversed(result.get("messages", ()))
                   if isinstance(message, AIMessage) and
                   not message.tool_calls and str(message.content).strip()),
                  tr("processed"))
    return (answer, result) if return_state else answer
