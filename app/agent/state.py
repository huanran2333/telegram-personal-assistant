from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentRuntimeContext(TypedDict):
    """Authenticated principal for one graph invocation, never model input."""

    user_id: str
    telegram_id: str
    chat_id: str
    origin: str


def require_agent_runtime_identity(state: dict, runtime) -> AgentRuntimeContext:
    """Bind a fresh entry principal to the checkpoint owner on every run/resume."""
    context = getattr(runtime, "context", None)
    if not isinstance(context, dict) or not context.get("user_id"):
        raise PermissionError("AgentRuntimeIdentityMissing")
    for key in ("user_id", "telegram_id", "chat_id", "request_origin"):
        context_key = "origin" if key == "request_origin" else key
        if str(state.get(key, "")) != str(context.get(context_key, "")):
            raise PermissionError("AgentRuntimeOwnerMismatch")
    return context


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]
    user_id: str           # internal DB user id
    telegram_id: str       # telegram user id
    chat_id: str           # telegram chat id
    conversation_thread_id: str  # current authorized conversation thread
    memory_history_threads: List[str]  # owner-checked historical search scopes
    active_request: str  # current user turn; never supplied by a tool call
    active_turn_id: str  # stable Receipt/WebRequest turn identity for durable handoff
    request_origin: str  # telegram or web, server supplied
    completed_read_summaries: List[str]  # bounded, owner-scoped results for approval handoff
    handoff_summary: str  # same-turn completed reads; never an approval/execution claim
    meeting_evidence_pending: bool  # evidence collection requires validated final claims
    validated_meeting_response: str  # deterministic rendering of checked claims
    meeting_submission_attempts: int  # bounded typed-claim retries per turn
    meeting_validation_exhausted: bool  # terminal safe response after rejections
    offered_tools: List[str]  # server-computed tool names for the latest model call
    pending_tool_calls: List[dict]  # frozen model calls, processed one per checkpoint
    pending_tool_index: int
    pending_agent_proposal: dict  # exact guarded action awaiting this graph's interrupt
    pending_agent_call_id: str
    proposal: dict  # approval UI's read-only projection of pending_agent_proposal
    approval_decision: str
    approval_web_request_id: str | None
    edit_web_request_id: str | None
    agent_edit_instruction: str
    agent_action_outcomes: dict  # bounded terminal action-id -> status/source for stale UI
    last_resolved_action_id: str
    unified_tool_rounds: int  # processed tool calls (max 10) in the current turn
    budget_blocked_tool_calls: int  # generated calls not executed after the hard limit
    profile_processed_turn_id: str  # deterministic profile extraction runs once per turn
    capability_failed: bool  # fail closed without asking the model to repeat a rejected action
    read_tool_rounds: int  # bounded read-only tool loop per user turn
    read_sources_seen: List[str]  # evidence domains actually returned this turn
    recent_gmail_thread_id: str | None  # only after an owner-checked unique read
    recent_gmail_owner_telegram_id: str | None

    last_action: str
    # Lower-authority selected evidence for the active unified tool loop.
    memory_context_payload: str
    memory_context_prepared: bool
    memory_search_rounds: int  # at most two scoped historical searches per turn
