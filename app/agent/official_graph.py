"""Framework-managed Agent graph with official ToolNode and per-call HITL."""
from __future__ import annotations

from typing import NotRequired

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolErrorMiddleware
from langchain.agents.middleware.types import AgentState
from requests.exceptions import (ConnectionError as RequestsConnectionError,
                                 RequestException, Timeout as RequestsTimeout)
from tavily.errors import (BadRequestError as TavilyBadRequestError,
                           ForbiddenError as TavilyForbiddenError,
                           InvalidAPIKeyError as TavilyInvalidAPIKeyError,
                           TimeoutError as TavilyTimeoutError,
                           UsageLimitExceededError)

from app.agent.official_permissions import OfficialToolAuthorization
from app.agent.official_context import OfficialContextMiddleware, OFFICIAL_PROMPT
from app.agent.official_read_tools import make_official_read_tools
from app.agent.state import AgentRuntimeContext
from app.agent.official_guarded_tools import (
    AllGuardedPreflight, make_official_guarded_tools)
from app.agent.official_todo_approval import (
    needs_todo_review, todo_review_description, make_approved_todo_delete_tool)
from app.agent.official_todo_tools import make_official_todo_tools
from app.agent.calendar import CalendarError
from app.agent.calendar_conditional import CalendarProviderError
from app.agent.gmail import GmailError, GmailUnavailable


class OfficialAgentState(AgentState):
    user_id: str
    telegram_id: str
    chat_id: str
    request_origin: str
    active_request: str
    active_turn_id: str
    conversation_thread_id: str
    memory_history_threads: NotRequired[list[str]]
    prepared_actions: NotRequired[dict]
    profile_processed_turn_id: NotRequired[str]
    memory_context_payload: NotRequired[str]
    memory_context_prepared: NotRequired[bool]


GUARDED = ("propose_todo_delete", "propose_todo_tag_delete",
           "propose_calendar_create", "propose_calendar_update",
           "propose_calendar_delete", "propose_email_todo",
           "propose_email_calendar")


_CALENDAR_READ_TOOLS = ("calendar_events", "calendar_availability",
                        "calendar_candidate_slots", "calendar_event")
_GMAIL_READ_TOOLS = ("gmail_search", "gmail_unread", "gmail_thread")
_WEB_READ_ERRORS = (TavilyBadRequestError, TavilyForbiddenError,
                    TavilyInvalidAPIKeyError, TavilyTimeoutError,
                    UsageLimitExceededError, RequestException)
_CALENDAR_READ_FAILURE = "本次日历读取失败，未获得可靠结果；不要将其当作空日程或成功读取。"


def _read_error_reason(exc: Exception) -> tuple[str, str]:
    """Classify only typed provider evidence; never serialize exception text."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = (current.status_code if isinstance(current, CalendarProviderError)
                  else current.response.status_code if isinstance(current, httpx.HTTPStatusError)
                  else None)
        if status is not None:
            if status in {401, 403}:
                return "permission_denied", "认证或访问权限不足"
            if status in {404, 410}:
                return "resource_unavailable", "资源不存在或当前账号无权查看"
            if status == 429:
                return "rate_limited", "服务请求限流"
            if status == 408:
                return "connection_timeout", "读取连接超时"
            if status >= 500:
                return "service_unavailable", "服务暂时不可用"
        if isinstance(current, CalendarProviderError):
            if current.category in {"oauth_unavailable", "oauth_refresh_failed",
                                    "calendar_scope_unavailable"}:
                return "permission_denied", "授权或日历访问范围不可用"
            if current.category in {"not_found_or_inaccessible", "gone"}:
                return "resource_unavailable", "资源不存在或当前账号无权查看"
            if current.category == "provider_unavailable":
                return "service_unavailable", "服务暂时不可用"
        if isinstance(current, (httpx.TimeoutException, RequestsTimeout, TimeoutError)):
            return "connection_timeout", "读取连接超时"
        if isinstance(current, (httpx.TransportError, RequestsConnectionError, ConnectionError)):
            return "service_unavailable", "服务连接暂时不可用"
        current = current.__cause__
    if isinstance(exc, TavilyBadRequestError):
        return "invalid_request", "搜索请求被服务拒绝"
    if isinstance(exc, (TavilyForbiddenError, TavilyInvalidAPIKeyError)):
        return "permission_denied", "搜索服务授权不足"
    if isinstance(exc, UsageLimitExceededError):
        return "rate_limited", "搜索服务额度或频率受限"
    if isinstance(exc, TavilyTimeoutError):
        return "connection_timeout", "搜索服务响应超时"
    if isinstance(exc, GmailUnavailable) and str(exc) in {
        "Gmail OAuth token is not configured", "Gmail OAuth token lacks gmail.readonly scope",
        "Gmail OAuth token needs reauthorization", "Gmail OAuth authorization failed",
    }:
        return "permission_denied", "邮件授权不可用或需要重新授权"
    if isinstance(exc, CalendarError) and str(exc) == "CalendarAvailabilityUnverified":
        return "invalid_response", "日历空闲状态未得到可靠验证"
    return "read_failed", "具体原因无法安全确认"


def _read_tool_error(exc, request):
    """Only known read failures become paired results; writes remain uncertain."""
    name = request.tool_call["name"]
    if name in _CALENDAR_READ_TOOLS and isinstance(exc, CalendarError):
        suffix = "不要据此判断没有日程。"
    elif name in _GMAIL_READ_TOOLS and isinstance(exc, GmailError):
        suffix = "不要据此判断没有邮件。"
    elif name == "web_search" and isinstance(exc, _WEB_READ_ERRORS):
        suffix = "不要编造网页来源或搜索结论。"
    else:
        return None
    category, reason = _read_error_reason(exc)
    return f"{name} 读取失败（{category}：{reason}）；未获得可靠结果。{suffix}"


def _calendar_read_error(exc, _request):
    """Compatibility for the original OA11 Calendar-only regression test."""
    if isinstance(exc, CalendarError):
        return _CALENDAR_READ_FAILURE
    return None


def build_official_agent(*, model, session_factory, settings, todo_tools=None,
                         checkpointer=None, store=None, memory_orchestrator=None,
                         interrupt_mode="per_call", model_failover=None):
    tools = (make_official_todo_tools(session_factory) +
             make_official_read_tools(session_factory,
                                      memory_orchestrator=memory_orchestrator) +
             [make_approved_todo_delete_tool()] + make_official_guarded_tools())
    if len({item.name for item in tools}) != len(tools):
        raise ValueError("DuplicateOfficialToolName")
    review = {name: {"allowed_decisions": ["approve", "reject"],
                     "when": needs_todo_review,
                     "description": todo_review_description}
              for name in GUARDED}
    middleware = [
        OfficialToolAuthorization(tools, settings=settings, session_factory=session_factory),
        OfficialContextMiddleware(settings=settings,
                                  memory_orchestrator=memory_orchestrator),
    ]
    if model_failover is not None:
        middleware.append(model_failover)
    middleware.extend([
        HumanInTheLoopMiddleware(interrupt_on=review,
                                 **({"interrupt_mode": interrupt_mode}
                                    if interrupt_mode != "batched" else {})),
        ToolErrorMiddleware(
            on_error=_read_tool_error,
            tools=[*_CALENDAR_READ_TOOLS, *_GMAIL_READ_TOOLS, "web_search"]),
        AllGuardedPreflight(session_factory),
    ])
    graph = create_agent(
        model=model, tools=tools, system_prompt=OFFICIAL_PROMPT,
        state_schema=OfficialAgentState, context_schema=AgentRuntimeContext,
        middleware=middleware, checkpointer=checkpointer, store=store)
    return graph, tools


# Frozen migration experiments import their original names.
CandidateState = OfficialAgentState
build_candidate = build_official_agent
