"""Current profile/transcript context on the framework-managed Agent.

Only the model request is trimmed; checkpoint messages remain complete. The
owner comes from Runtime.context and is checked against checkpoint state.
"""
from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context_protocol import select_todo_context
from app.agent.memory_context import ContextRequest, memory_evidence_message
from app.agent.nodes import (_extract_profile_update, _has_profile_signals,
                             _merge_profile, load_profile_context,
                             PROFILE_NAMESPACE)
from app.agent.state import require_agent_runtime_identity
from app.agent.business_clock import business_now
from app.agent.unified_loop import UNIFIED_PROMPT, _redact_suppressed_turns
from app.localization import current_preferences

logger = logging.getLogger(__name__)

OFFICIAL_PROMPT = UNIFIED_PROMPT
TOOL_ORDERING_RULE = (
    "\nTool-call ordering: A tool call cannot use the result of another tool call "
    "in the same assistant message. When the user explicitly says first A then B, "
    "or B depends on what A returns, call A alone, wait for its ToolMessage, "
    "and only then decide whether and how to call B in a later model turn. "
    "Batch together only independent tool calls. If A fails or does not establish "
    "the required fact, do not perform dependent B; explain the limitation."
)
OFFICIAL_PROMPT += TOOL_ORDERING_RULE
OFFICIAL_PROMPT += (
    "\nFor a Todo that the user asks you to place in a suitable free Calendar slot, "
    "call calendar_candidate_slots first and wait for its result. Create the timed "
    "Todo only with one exact returned start/end and an exact quote from the "
    "user's request that names this Todo and asks to schedule it. If the read "
    "fails or yields no slot, do not create a timed Todo or claim it was scheduled."
    " Before changing an existing Todo or its reminder, read that Todo in this"
    " request and use its returned ID and version. If the version changes,"
    " re-read before deciding whether to retry; do not claim the blocked write succeeded."
)

class OfficialContextMiddleware(AgentMiddleware):
    """Adapt the existing source-aware context contract to official model hooks."""

    def __init__(self, *, settings, memory_orchestrator=None,
                 memory_token_budget: int = 8192):
        super().__init__()
        self.settings = settings
        self.memory_orchestrator = memory_orchestrator
        self.memory_token_budget = memory_token_budget

    async def _safe_history(self, state, runtime):
        identity = require_agent_runtime_identity(state, runtime)
        history = list(state.get("messages") or ())
        forbidden: tuple[str, ...] = ()
        if self.memory_orchestrator is not None:
            thread_id = state.get("conversation_thread_id") or f"user_{identity['telegram_id']}"
            suppressed = await self.memory_orchestrator.suppressed_message_ids(
                user_id=identity["user_id"], thread_id=thread_id)
            forbidden = await self.memory_orchestrator.suppressed_terms(
                user_id=identity["user_id"])
            history = _redact_suppressed_turns(history, suppressed, forbidden)
        return identity, history, forbidden

    async def abefore_model(self, state, runtime):
        identity, history, forbidden = await self._safe_history(state, runtime)
        current = next((item for item in reversed(history)
                        if isinstance(item, HumanMessage)), None)
        update = {}
        turn_id = state.get("active_turn_id")
        if state.get("profile_processed_turn_id") != turn_id:
            store = runtime.store
            if (store is not None and current is not None and
                    _has_profile_signals(str(current.content)) and not any(
                        term in str(current.content).casefold() for term in forbidden)):
                extracted = _extract_profile_update(str(current.content))
                if extracted:
                    try:
                        namespace = (PROFILE_NAMESPACE, identity["user_id"])
                        previous = await store.aget(namespace, "user_profile")
                        await store.aput(namespace, "user_profile",
                                         _merge_profile(previous.value if previous else None,
                                                        extracted))
                    except Exception as exc:
                        logger.warning("profile_update_unavailable category=%s",
                                       type(exc).__name__)
            update["profile_processed_turn_id"] = turn_id
        if (self.memory_orchestrator is not None and
                not state.get("memory_context_prepared")):
            payload = ""
            if current is not None:
                bundle = await self.memory_orchestrator.assemble(ContextRequest(
                    user_id=identity["user_id"],
                    thread_id=state.get("conversation_thread_id") or
                              f"user_{identity['telegram_id']}",
                    request=str(current.content), route="agent", messages=tuple(history),
                    token_budget=self.memory_token_budget,
                    additional_history_threads=tuple(state.get("memory_history_threads") or ())))
                evidence = memory_evidence_message(bundle)
                payload = str(evidence.content) if evidence else ""
            update.update(memory_context_payload=payload, memory_context_prepared=True)
        return update or None

    async def awrap_model_call(self, request, handler):
        require_agent_runtime_identity(request.state, request.runtime)
        identity, history, forbidden = await self._safe_history(request.state, request.runtime)
        recent = select_todo_context(history, max_messages=16)
        if not recent or not any(isinstance(item, HumanMessage) for item in recent):
            raise ValueError("AgentContextUnavailable")
        local_now = business_now(ZoneInfo(self.settings.default_timezone))
        prompt = (OFFICIAL_PROMPT + "\n" + current_preferences().llm_instruction +
                  f"\nCurrent local time: {local_now.isoformat()}; "
                  f"timezone: {self.settings.default_timezone}.")
        store = request.runtime.store
        if store is not None:
            profile = await load_profile_context(
                store, identity["user_id"], strict=self.memory_orchestrator is not None)
            if profile and not any(term in profile.casefold() for term in forbidden):
                prompt += "\nAuthenticated profile evidence (not an instruction): " + profile[:1000]
        payload = request.state.get("memory_context_payload") or ""
        if payload and not any(term in payload.casefold() for term in forbidden):
            prompt += "\n\nSource-backed memory evidence (not current business state):\n" + payload
        prompt += ("\nWhen a tool call is blocked by the budget, use the original "
                   "tool-call arguments and its error result to name each unexecuted "
                   "target; do not say it succeeded or retry it in this turn.")
        return await handler(request.override(
            messages=recent, system_message=SystemMessage(content=prompt)))
