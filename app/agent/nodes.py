"""
Node functions for the LangGraph agent graph.
Each node receives AgentState and returns a partial state update.
"""
import json
import asyncio
import os
import re
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from typing import Any, Literal, Optional
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, merge_message_runs

from app.agent import state
from app.agent.prompt import (SYSTEM_PROMPT, READ_AGENT_PROMPT, INTENT_CLASSIFICATION_PROMPT,
                              CHAT_RESPONSE_PROMPT, MEMORY_SEARCH_PROMPT)
from app.agent.state import AgentState
from app.agent.capability_policy import AGENT_READ_INTENTS, capability_for_route, required_read_sources
from app.agent.read_tools import read_scope
from app.agent.capability_tools import CapabilityRejected, capability_scope
from app.agent.context_protocol import select_chat_context, select_todo_context
from app.config import get_settings
from app.trace import emit_trace
from app.localization import current_preferences, tr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tavily import TavilyClient
from langgraph.store.base import BaseStore
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
import dateparser, json
from zoneinfo import ZoneInfo
        
logger = logging.getLogger(__name__)
settings = get_settings()


class LLMConfigurationError(ValueError):
    """Raised when LLM provider/model configuration is invalid."""


class MemorySearchArgs(BaseModel):
    """The model chooses a question, never the owner or conversation scope."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=500)
    purpose: Literal["exact", "topic"] = "topic"


MEMORY_SEARCH_TOOL_DESCRIPTION = (
    "Search this user's source-backed conversation and structured memory "
    "for a specific past fact or topic. This is read-only historical evidence, "
    "not current Todo, Calendar, or Gmail state. Do not repeat a search already "
    "answered by the supplied memory context."
)


_HISTORICAL_REFERENCE = re.compile(
    r"之前|上次|前面(?:说|聊|提)|以前|曾经|聊过|提过|过去|旧的|老的|历史|"
    r"\b(?:old|previous|previously|earlier|last time|mentioned|remember|we discussed|i told you)\b",
    re.IGNORECASE,
)


def memory_search_relevant(state: AgentState) -> bool:
    """Cheap disclosure filter; actual access is still owner-scoped at execution."""
    if (state.get("entities") or {}).get("memory_search_query"):
        return True
    current = next((item for item in reversed(state.get("messages") or [])
                    if isinstance(item, HumanMessage) and isinstance(item.content, str)), None)
    return bool(current and _HISTORICAL_REFERENCE.search(current.content[:1000]))


def _build_chat_llm(provider: str, model_name: str, routing: bool = False,
                    meeting_structured: bool = False,
                    request_timeout: float | None = None,
                    disable_qwen_thinking: bool = False):
    provider_name = (provider or "").strip().lower()
    kwargs = {"model": model_name, "temperature": 0, "max_tokens": 4096}

    try:
        if provider_name == "anthropic":
            from langchain_anthropic import ChatAnthropic

            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
            return ChatAnthropic(**kwargs)

        if provider_name == "openai":
            from langchain_openai import ChatOpenAI
            from app.agent.provider_clients import current_provider_owner

            # Bounded structured extraction (including meeting drafts) must
            # leave output tokens for the JSON result. Keep ordinary Agent
            # tool-calling and user-facing generation unchanged.
            if (routing or meeting_structured) and "qwen" in model_name.lower():
                kwargs["reasoning_effort"] = "none"
            sync_http, async_http = current_provider_owner().openai_http_clients()
            return ChatOpenAI(
                **kwargs,
                **({"max_retries": 0, "request_timeout": request_timeout}
                   if request_timeout is not None else {}),
                **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                   if disable_qwen_thinking and "qwen" in model_name.lower() else {}),
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                http_client=sync_http,
                http_async_client=async_http,
            )

        if provider_name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key)
            return ChatGoogleGenerativeAI(**kwargs)

        if provider_name == "nvidia":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            os.environ.setdefault("NVIDIA_API_KEY", settings.nvidia_api_key)
            if settings.nvidia_api_endpoint:
                os.environ.setdefault("NVIDIA_API_ENDPOINT", settings.nvidia_api_endpoint)
            return ChatNVIDIA(**kwargs)

        raise LLMConfigurationError(
            f"Unsupported llm_provider '{provider_name}'. Expected one of: nvidia, anthropic, openai, gemini"
        )
    except LLMConfigurationError:
        raise
    except ImportError as exc:
        raise LLMConfigurationError(
            f"Missing dependency for provider '{provider_name}'. Install required LangChain package."
        ) from exc
    except Exception as exc:
        raise LLMConfigurationError(
            f"Failed to initialize provider '{provider_name}' with model '{model_name}': {exc}"
        ) from exc

ALLOWED_INTENTS = {"clarify", "create", "list", "update", "delete", "search", "chat", "memory", "web_search", "update_profile",
                   "meeting_prep", "calendar_create", "calendar_update", "calendar_delete", "cross_domain",
                   "smart_schedule", "email_to_todo", "email_to_calendar", *AGENT_READ_INTENTS}
CONFIRM_POSITIVE = {"yes", "y", "yeah", "yep", "ok", "sure", "confirm"}
CONFIRM_NEGATIVE = {"no", "n", "nope", "cancel", "stop"}
CONFIRM_REPLIES = CONFIRM_POSITIVE | CONFIRM_NEGATIVE
TODO_TOOL_NAMES = {
    "create_todo",
    "list_todos",
    "update_todo",
    "delete_todo",
    "search_todos",
    "set_reminder",
    "cancel_reminder",
}
CALENDAR_READ_ONLY_TOOL_NAMES = {
    "list-calendars",
    "list-events",
    "search-events",
    "get-event",
    "list-colors",
    "get-freebusy",
    "get-current-time",
}
CALENDAR_TIMEZONE = settings.default_timezone
CALENDAR_TZ = ZoneInfo(CALENDAR_TIMEZONE)
PROFILE_NAMESPACE = "profile"
PROFILE_SIGNAL_PATTERNS = [
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi'm\b",
    r"\bim\b",
    r"\bi live in\b",
    r"\bi work as\b",
    r"\bi work in\b",
    r"\bi like\b",
    r"\bi love\b",
    r"\bi enjoy\b",
    r"\bi prefer\b",
    r"\bfavorite\b",
    r"\bmy hobby\b",
    r"\bmy hobbies\b",
    r"\bi'm into\b",
    r"我(?:更)?(?:喜欢|偏好|希望)",
    r"以后(?:请|尽量)",
]

class Profile(BaseModel):
    """Structured profile memory for the user."""

    name: Optional[str] = Field(default=None, description="The user's name")
    location: Optional[str] = Field(default=None, description="The user's location")
    job: Optional[str] = Field(default=None, description="The user's job")
    connections: list[str] = Field(default_factory=list, description="Family, friends, or coworkers")
    interests: list[str] = Field(default_factory=list, description="Interests or hobbies")
    preferences: list[str] = Field(default_factory=list, description="User preferences")
    assistant_language: Optional[str] = Field(default=None, description="Explicit Bot response language: en or zh-CN")

def _has_profile_signals(message: str) -> bool:
    lowered = (message or "").lower()
    return any(re.search(pattern, lowered) for pattern in PROFILE_SIGNAL_PATTERNS)

def _format_profile_context(profile_items) -> str:
    if not profile_items:
        return ""

    lines = []
    for item in profile_items:
        value = item.value
        if isinstance(value, dict):
            parts = []
            for key in ("name", "location", "job", "connections", "interests", "preferences", "assistant_language"):
                if value.get(key):
                    parts.append(f"{key}: {value[key]}")
            lines.append("; ".join(parts) if parts else json.dumps(value, ensure_ascii=False))
        else:
            lines.append(str(value))
    return "\n".join(lines)

def _first_clause(value: str) -> str:
    return re.split(r"[.?!。？！]", value, maxsplit=1)[0].strip()

def _normalize_list_values(raw_value: str) -> list[str]:
    values = []
    for piece in re.split(r",|，|、|\band\b|&|/", raw_value, flags=re.IGNORECASE):
        cleaned = piece.strip().strip(".?!;:")
        cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values

def _extract_profile_update(user_message: str) -> dict:
    message = (user_message or "").strip()
    if not message:
        return {}

    lowered = message.lower()
    update: dict[str, Any] = {}

    name_match = re.search(r"\bmy name is\s+(?P<value>.+)", lowered)
    if name_match:
        update["name"] = _first_clause(name_match.group("value")).title()

    location_match = re.search(r"\bi live in\s+(?P<value>.+)", lowered)
    if location_match:
        update["location"] = _first_clause(location_match.group("value")).title()

    job_match = re.search(r"\bi work as\s+(?P<value>.+)", lowered)
    if not job_match:
        job_match = re.search(r"\bi work in\s+(?P<value>.+)", lowered)
    if job_match:
        update["job"] = _first_clause(job_match.group("value")).strip()

    hobby_patterns = [
        r"\bmy hobby is\s+(?P<value>.+)",
        r"\bmy hobbies are\s+(?P<value>.+)",
        r"\bi like\s+(?P<value>.+)",
        r"\bi love\s+(?P<value>.+)",
        r"\bi enjoy\s+(?P<value>.+)",
        r"\bi(?:'|)m into\s+(?P<value>.+)",
    ]
    hobby_values: list[str] = []
    for pattern in hobby_patterns:
        match = re.search(pattern, lowered)
        if match:
            hobby_values.extend(_normalize_list_values(_first_clause(match.group("value"))))
    if hobby_values:
        update["interests"] = hobby_values

    preference_patterns = [
        r"\bi prefer\s+(?P<value>.+)",
        r"\bmy favorite (?P<label>.+?) is\s+(?P<value>.+)",
        r"\bmy favourite (?P<label>.+?) is\s+(?P<value>.+)",
    ]
    preference_values: list[str] = []
    for pattern in preference_patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        if "label" in match.groupdict():
            label = _first_clause(match.group("label")).strip()
            value = _first_clause(match.group("value")).strip()
            if label and value:
                preference_values.append(f"{label}: {value}")
        else:
            preference_values.extend(_normalize_list_values(_first_clause(match.group("value"))))
    if preference_values:
        update["preferences"] = preference_values

    chinese_preference = re.search(r"(?:我(?:更)?(?:偏好|希望)|以后请?(?:尽量)?)\s*(?P<value>.+)", message)
    if chinese_preference:
        update["preferences"] = _normalize_list_values(_first_clause(chinese_preference.group("value")))
    chinese_like = re.search(r"我(?:更)?喜欢\s*(?P<value>.+)", message)
    if chinese_like:
        value = _first_clause(chinese_like.group("value"))
        destination = ("preferences" if re.search(r"回答|回复|提醒|安排|交流|表达|简洁|简短", value)
                       else "interests")
        update[destination] = _normalize_list_values(value)

    return update

def _merge_profile(existing_profile: dict | None, profile_update: dict) -> dict:
    merged = dict(existing_profile or {})
    for key, value in profile_update.items():
        if value in (None, "", []):
            continue
        existing_value = merged.get(key)
        if isinstance(existing_value, list) and isinstance(value, list):
            merged[key] = existing_value + [item for item in value if item not in existing_value]
        else:
            merged[key] = value
    return merged

def _fallback_intent_response() -> dict:
    return {
        "intent": "chat",
        "confidence": 0.3,
        "entities": {},
        "ambiguous_fields": ["all"],
        "clarification_needed": True,
        "clarification_question": "I'm not sure what you meant. Can you rephrase?",
    }

def _validate_intent_payload(intent_data: dict[str, Any]) -> dict:
    intent = intent_data.get("intent")
    confidence = intent_data.get("confidence")

    if not isinstance(intent, str) or intent not in ALLOWED_INTENTS:
        raise ValueError(f"Invalid intent: {intent}")

    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Invalid confidence type: {type(confidence)}")
    confidence = max(0.0, min(1.0, float(confidence)))

    entities = intent_data.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}

    # Enforce minimum required field for create intent.
    if intent == "create" and not entities.get("title"):
        raise ValueError("Missing required entity for create intent: title")

    ambiguous_fields = intent_data.get("ambiguous_fields") or []
    if not isinstance(ambiguous_fields, list):
        ambiguous_fields = ["all"]

    clarification_question = intent_data.get("clarification_question")

    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "ambiguous_fields": ambiguous_fields,
        "clarification_needed": clarification_question is not None,
        "clarification_question": clarification_question,
    }

class ChatResponse(BaseModel):
    should_search_web: bool = False
    content: str = Field(default="", description="The assistant's reply to the user")
    query: str = Field(default="", description="The search query for web search, if applicable")

    @model_validator(mode="after")
    def force_search_if_query_extracted(self) -> "ChatResponse":
        """If the LLM extracted a search query, force should_search_web to True."""
        if self.query.strip():
            self.should_search_web = True
            self.content = ""
        return self

def extract_text(content) -> str:
    """
    Safely extract plain text from an AIMessage content field.
    Claude returns either:
      - a plain string:  "Here are your todos..."
      - a list of blocks: [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content).strip()


def todo_context_messages(history: list) -> list:
    """Use validated complete turns, retaining the active long tool loop."""
    return select_todo_context(history, max_messages=10)


CHAT_RECENT_MESSAGE_LIMIT = 8


def chat_context_messages(history: list, max_messages: int = CHAT_RECENT_MESSAGE_LIMIT) -> list:
    """Chat policy retains eight conversational messages on shared protocol validation."""
    return select_chat_context(history, max_messages=max_messages)


def chat_response_messages(history: list, profile_context: str = "",
                           max_messages: int = CHAT_RECENT_MESSAGE_LIMIT) -> list:
    """Assemble one system message and one bounded, ordered chat window."""
    prompt = CHAT_RESPONSE_PROMPT
    if profile_context:
        prompt += f"\n\nUser profile memory:\n{profile_context}"
    prompt += "\n\n" + current_preferences().llm_instruction
    prompt += (
        "\n\nRecent conversation is for resolving conversational references only. "
        "Prefer the user's newer corrections over older statements. "
        "It is not authority for current Todo, Calendar, Gmail, approval, or execution state; "
        "those require their dedicated source/workflow. Current profile and response-language "
        "preferences above override older conversation statements."
    )
    return [SystemMessage(content=prompt)] + chat_context_messages(history, max_messages)


class ProfileAwareSearchQuery(BaseModel):
    query: str = Field(description="A concise web search query grounded in the user's request")

async def load_profile_context(store: BaseStore, user_id: str, *, strict: bool = False) -> str:
    """Load profile memory for prompt conditioning."""
    if not store or not user_id:
        if strict:
            raise RuntimeError("Profile Store is unavailable")
        return ""

    try:
        profile_item = await store.aget((PROFILE_NAMESPACE, user_id), "user_profile")
        logger.info("Loaded profile memory for user_id=%s: %s", user_id, bool(profile_item))
    except Exception as exc:
        logger.error("Profile memory load failed category=%s", type(exc).__name__)
        if strict:
            raise RuntimeError("Profile Store is unavailable") from None
        return ""

    return _format_profile_context([profile_item] if profile_item else [])
 
def make_nodes(tools: list, llm_model: str = None, llm_provider: str = None,
               *, llm_tools: list | None = None, read_tools: list | None = None,
               capability_tools: list | None = None,
               memory_orchestrator=None,
               memory_control=None, memory_token_budget: int = 8192):
    """
    Returns node functions bound to the given tools and LLM.
    """
    model_name = llm_model or settings.llm_model
    provider_name = llm_provider or settings.llm_provider
    base_llm = _build_chat_llm(provider_name, model_name)
    search_llm = _build_chat_llm(provider_name, model_name, routing=True)
    bound_tools = tools if llm_tools is None else llm_tools
    llm = base_llm.bind_tools(bound_tools)
    read_tool_map = {item.name: item for item in (read_tools or [])}
    capability_tool_map = {item.name: item for item in (capability_tools or [])}

    def memory_search_tool(state: AgentState):
        """Bind a read-only search to the authenticated graph state, not model arguments."""
        async def search_memory(query: str, purpose: Literal["exact", "topic"] = "topic") -> str:
            from app.agent.memory_context import ContextRequest, memory_evidence_message

            prefetched = (state.get("entities") or {}).get("memory_search_query")
            previous_evidence = state.get("memory_context_payload") or ""
            if (prefetched and previous_evidence and
                    purpose == (state.get("entities") or {}).get("memory_search_purpose") and
                    query.strip().casefold() == str(prefetched).strip().casefold()):
                return previous_evidence
            history = state.get("messages") or []
            current = next((item for item in reversed(history)
                            if isinstance(item, HumanMessage) and isinstance(item.content, str)), None)
            if current is None or not state.get("user_id"):
                return "Historical memory is unavailable for this request."
            bundle = await memory_orchestrator.assemble(ContextRequest(
                user_id=state["user_id"],
                thread_id=state.get("conversation_thread_id") or f"user_{state['telegram_id']}",
                request=current.content, route="todo", messages=(current,),
                token_budget=min(memory_token_budget, 4096),
                search_query=query.strip(), search_purpose=purpose,
                allow_summary_update=False,
                additional_history_threads=tuple(state.get("memory_history_threads") or ()),
            ))
            evidence = memory_evidence_message(bundle)
            return (str(evidence.content) if evidence else
                    "No matching source-backed historical evidence was found.")

        return StructuredTool.from_function(
            coroutine=search_memory, name="search_memory", args_schema=MemorySearchArgs,
            description=MEMORY_SEARCH_TOOL_DESCRIPTION,
        )

    def scoped_read_tools(state: AgentState) -> list:
        """Hard-filter capabilities before model binding; tools recheck owner on call."""
        from app.agent.planner import RoutedRequest
        payload = state.get("precomputed_route")
        if not payload:
            return []
        route = RoutedRequest.model_validate(payload)
        if route.intent not in AGENT_READ_INTENTS:
            return []
        available = []
        calendar_owner = (settings.google_calendar_enabled and
                          state.get("telegram_id") == settings.google_calendar_owner_telegram_id and
                          bool(settings.google_calendar_owner_telegram_id))
        gmail_owner = (settings.gmail_enabled and
                       state.get("telegram_id") == settings.gmail_owner_telegram_id and
                       bool(settings.gmail_owner_telegram_id))
        if (route.intent == "calendar_read" or
                (route.intent == "read_cross_domain" and route.date_reference) or
                (route.intent == "email_cross_domain" and route.email_calendar_check)) and calendar_owner:
            for name in ("read_calendar_events", "read_calendar_availability"):
                if name in read_tool_map:
                    available.append(read_tool_map[name])
        if (route.intent in {"email_search", "email_unread", "email_thread", "email_draft",
                             "email_cross_domain", "read_cross_domain"} and gmail_owner and
                (route.intent != "read_cross_domain" or route.email_query or route.email_thread_id)):
            if "read_gmail_evidence" in read_tool_map:
                available.append(read_tool_map["read_gmail_evidence"])
        if route.intent == "read_cross_domain":
            available.extend(item for item in bound_tools
                             if item.name in {"list_todos", "search_todos"})
        return available

    def selected_tools(state: AgentState) -> list:
        from app.agent.planner import RoutedRequest
        payload = state.get("precomputed_route")
        route = RoutedRequest.model_validate(payload) if payload else None
        if route is None and state.get("intent") in (AGENT_READ_INTENTS | {
                "delete", "calendar_create", "calendar_update", "calendar_delete",
                "cross_domain", "smart_schedule", "meeting_prep", "email_to_todo",
                "email_to_calendar"}):
            return []
        capability = capability_for_route(route) if route else None
        if capability:
            tool = capability_tool_map.get(capability)
            return [tool] if tool is not None else []
        if route and route.intent in AGENT_READ_INTENTS:
            return scoped_read_tools(state)
        if (memory_orchestrator is not None and memory_search_relevant(state) and
                state.get("memory_search_rounds", 0) < 2):
            return [*bound_tools, memory_search_tool(state)]
        return bound_tools

    async def selected_memory(state: AgentState, route: str):
        if memory_orchestrator is None:
            return None
        from app.agent.memory_context import ContextRequest
        history = list(state.get("messages", []))
        current = next((item for item in reversed(history)
                        if isinstance(item, HumanMessage) and isinstance(item.content, str)), None)
        if current is None:
            return None
        entities = state.get("entities") or {}
        query = entities.get("memory_search_query")
        purpose = entities.get("memory_search_purpose", "none") if query else "none"
        return await memory_orchestrator.assemble(ContextRequest(
            user_id=state["user_id"], thread_id=state.get("conversation_thread_id") or f"user_{state['telegram_id']}",
            request=current.content, route=route, messages=tuple(history),
            token_budget=memory_token_budget,
            search_query=query, search_purpose=purpose,
            additional_history_threads=tuple(state.get("memory_history_threads") or ()),
        ))
    async def classify_intent(state: AgentState, store: BaseStore) -> dict:
        """
        Explicitly classify the user's intent and extract entities.
        Returns confidence score and detects ambiguity early.
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        awaiting = state.get("awaiting_conflict_confirmation", False)
        normalized_message = user_message.strip().lower()
        keep_confirmation_mode = awaiting and normalized_message in CONFIRM_REPLIES

        user_id = state.get("user_id", "")
        profile_context = await load_profile_context(
            store, user_id, strict=memory_orchestrator is not None)

        
        logger.info("Classifying intent for user_id=%s message_chars=%s", user_id, len(user_message))
        
        try:
            from app.agent.planner import (RouteModelError, RoutedRequest,
                                           classify_core_intent, core_personal_time_block)
            precomputed = state.get("precomputed_route")
            if precomputed and (precomputed.get("intent") in AGENT_READ_INTENTS or
                                capability_for_route(RoutedRequest.model_validate(precomputed))):
                route = RoutedRequest.model_validate(precomputed)
            else:
                route = await classify_core_intent(user_message, base_llm)
                if precomputed and precomputed.get("intent") == "todo":
                    route = core_personal_time_block(route, user_message, precomputed)
                    expected_operation = precomputed.get("todo_operation")
                    if route.intent != "todo" or route.todo_operation != expected_operation:
                        raise ValueError("TodoRouteChangedBetweenClassifiers")
            intent = (route.todo_operation or "search") if route.intent == "todo" else {
                "profile": "update_profile",
                "web_search": "web_search",
            }.get(route.intent, route.intent)
            entities = {}
            if route.title:
                entities["title"] = route.title
            if route.query:
                entities["query"] = route.query
            if route.todo_tag:
                entities["tags"] = route.todo_tag.strip().lstrip("#")
            if getattr(route, "due_date_str", None):
                entities["due_date_str"] = route.due_date_str
            if getattr(route, "planned_start_str", None):
                entities["planned_start_str"] = route.planned_start_str
            if getattr(route, "planned_end_str", None):
                entities["planned_end_str"] = route.planned_end_str
            if precomputed and precomputed.get("intent") == "todo" and route.intent == "todo":
                # Preserve the outer router's bounded schedule extraction only
                # after the independent classifier agrees on the same operation.
                for field in ("planned_start_str", "planned_end_str"):
                    if field not in entities and precomputed.get(field):
                        entities[field] = precomputed[field]
            if getattr(route, "reminder_at_str", None):
                entities["reminder_at_str"] = route.reminder_at_str
            if route.memory_search_query and route.memory_search_purpose != "none":
                entities["memory_search_query"] = route.memory_search_query
                entities["memory_search_purpose"] = route.memory_search_purpose
            if route.memory_operation:
                entities["memory_operation"] = route.memory_operation
                if route.memory_target_query:
                    entities["memory_target_query"] = route.memory_target_query
            validated = _validate_intent_payload({
                "intent": intent, "confidence": route.confidence,
                "entities": entities, "ambiguous_fields": [],
                "clarification_question": None,
            })
            
            logger.info(
                f"Intent classified: {validated['intent']}, "
                f"confidence: {validated['confidence']:.2f}"
            )

            return {
                **validated,
                "classification_failure_kind": "",
                "awaiting_conflict_confirmation": keep_confirmation_mode,
                "memory_context_payload": "", "memory_context_prepared": False,
                "memory_search_rounds": 0,
            }
        except RouteModelError as e:
            logger.warning("Intent model classification failed category=%s",
                           type(e.__cause__).__name__ if e.__cause__ else type(e).__name__)
            return {
                **_fallback_intent_response(),
                "classification_failure_kind": "model",
                "awaiting_conflict_confirmation": keep_confirmation_mode,
                "memory_context_payload": "", "memory_context_prepared": False,
                "memory_search_rounds": 0,
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Intent classification failed category=%s", type(e).__name__)
            # An uncertain result asks for clarification; it never silently
            # reclassifies a failed Calendar request as ordinary chat.
            return {
                **_fallback_intent_response(),
                "classification_failure_kind": "validation",
                "awaiting_conflict_confirmation": keep_confirmation_mode,
                "memory_context_payload": "", "memory_context_prepared": False,
                "memory_search_rounds": 0,
            }
 
    async def ask_clarification(state: AgentState) -> dict:
        """
        When intent is ambiguous, generate clarification options
        and ask the user to choose.
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        if state.get("classification_failure_kind") == "model":
            return {"messages": [AIMessage(content=tr("model_unavailable"))],
                    "last_action": "model_classification_failed",
                    "awaiting_conflict_confirmation": False}
        intent = state.get("intent", "")
        question = state.get("clarification_question") or tr("request_unclear")
        if current_preferences().language == "zh-CN":
            question = tr("request_unclear")
        
        logger.info(f"Asking for clarification (intent={intent})")
        
        # Build a friendly clarification message
        response_text = tr("clarification", message=user_message, question=question)
        
        return {
            "messages": [AIMessage(content=response_text)],
            "last_action": "clarification_asked",
            "awaiting_conflict_confirmation": False,

        }
 
    async def todo_llm(state: AgentState, store: BaseStore = None) -> dict:
        """
        Main LLM reasoning node — uses tools to fulfill the request.
        Only called when intent is clear (confidence >= threshold).
        """
        user_id = state.get("user_id", "")
        read_mode = state.get("intent") in AGENT_READ_INTENTS
        capability_mode = state.get("intent") in {
            "calendar_create", "calendar_update", "calendar_delete", "cross_domain",
            "smart_schedule", "meeting_prep", "email_to_todo", "email_to_calendar",
        } or state.get("intent") == "delete"
        selected = selected_tools(state)
        if (read_mode or capability_mode) and not selected:
            return {"messages": [AIMessage(content=tr("tool_unavailable"))],
                    "last_action": "capability_unavailable"}
        if read_mode and state.get("read_tool_rounds", 0) >= 4:
            return {"messages": [AIMessage(content=tr("tool_unavailable"))],
                    "last_action": "read_scope_exhausted"}
        if read_mode and state.get("precomputed_route"):
            from app.agent.planner import RoutedRequest
            required = required_read_sources(RoutedRequest.model_validate(state["precomputed_route"]))
            available = ({"calendar" if tool.name.startswith("read_calendar_") else
                          "gmail" if tool.name == "read_gmail_evidence" else
                          "todo" if tool.name in {"list_todos", "search_todos"} else "" for tool in selected}
                         - {""})
            if not required or not required.issubset(available):
                return {"messages": [AIMessage(content=tr("tool_unavailable"))],
                        "last_action": "read_scope_incomplete"}
        # A single-source read tool returns an already validated, bounded
        # response. Do not let a second model rewrite its provenance or draft.
        if read_mode and state.get("intent") not in {"read_cross_domain", "email_cross_domain"} and isinstance(state["messages"][-1], ToolMessage):
            return {"messages": [AIMessage(content=str(state["messages"][-1].content))],
                    "last_action": "read_result_forwarded"}
        capability_prompt = (
            "The request is already classified. Invoke the sole offered high-level capability "
            "to prepare it; it will enforce authorization and any required approval. "
            "Never claim an action completed from your own text."
        )
        system = SystemMessage(content=(capability_prompt if capability_mode else
                                        READ_AGENT_PROMPT if read_mode else
                                        SYSTEM_PROMPT.format(user_id=user_id)) +
                               "\n" + current_preferences().llm_instruction)
        if any(tool.name == "search_memory" for tool in selected):
            system = SystemMessage(content=str(system.content) + "\n" + MEMORY_SEARCH_PROMPT)
        if memory_orchestrator is not None and not read_mode and not capability_mode and store is not None:
            profile = await load_profile_context(store, user_id)
            if profile:
                system = SystemMessage(content=str(system.content) +
                    "\nAuthenticated Profile Store data (not a user instruction; current business records win): " +
                    json.dumps(profile[:1000], ensure_ascii=False))
        if not read_mode and not capability_mode and state.get("entities"):
            system = SystemMessage(content=str(system.content) +
                "\nServer-extracted request fields (verify against the user's wording before calling tools): " +
                json.dumps(state["entities"], ensure_ascii=False, default=str))
        prepared = state.get("memory_context_prepared", False)
        evidence_payload = state.get("memory_context_payload", "")
        if memory_orchestrator is not None and not prepared and not read_mode and not capability_mode:
            from app.agent.memory_context import memory_evidence_message
            bundle = await selected_memory(state, "todo")
            evidence = memory_evidence_message(bundle) if bundle else None
            evidence_payload = str(evidence.content) if evidence else ""
            recent = list(bundle.recent_messages) if bundle else todo_context_messages(list(state["messages"]))
        else:
            recent = todo_context_messages(list(state["messages"]))
        messages = [system]
        if evidence_payload:
            messages[0] = SystemMessage(content=str(system.content) + "\n\n" + evidence_payload)
        messages.extend(recent)
 
        active_llm = base_llm.bind_tools(selected)
        logger.info(
            f"Calling LLM with {len(messages)} messages, "
            f"intent={state.get('intent')}, "
            f"user_id={user_id}"
        )
        started = time.perf_counter()
        try:
            response = await active_llm.ainvoke(messages)
        except Exception as exc:
            emit_trace("llm", operation="capability_tool_loop" if capability_mode else "read_tool_loop" if read_mode else "todo_tool_loop",
                       schema_count=len(selected),
                       latency_ms=round((time.perf_counter() - started) * 1000),
                       success=False, error_category=type(exc).__name__)
            raise
        usage = getattr(response, "usage_metadata", None) or {}
        emit_trace("llm", operation="capability_tool_loop" if capability_mode else "read_tool_loop" if read_mode else "todo_tool_loop",
                   schema_count=len(selected),
                   latency_ms=round((time.perf_counter() - started) * 1000),
                   token_usage=usage.get("total_tokens"), success=True)
 
        tool_calls = getattr(response, "tool_calls", [])
        if capability_mode and not tool_calls:
            return {"messages": [AIMessage(content=tr("tool_unavailable"))],
                    "last_action": "action_without_handoff_blocked"}
        if read_mode and not tool_calls:
            active_turn = []
            for item in reversed(state.get("messages", [])):
                active_turn.append(item)
                if isinstance(item, HumanMessage):
                    break
            if not any(isinstance(item, ToolMessage) for item in active_turn):
                return {"messages": [AIMessage(content=tr("tool_unavailable"))],
                        "last_action": "read_without_evidence_blocked"}
            if read_mode and state.get("precomputed_route"):
                from app.agent.planner import RoutedRequest
                required = required_read_sources(RoutedRequest.model_validate(state["precomputed_route"]))
                if not required.issubset(set(state.get("read_sources_seen") or [])):
                    return {"messages": [AIMessage(content=tr("tool_unavailable"))],
                            "last_action": "read_missing_source_blocked"}
        text = extract_text(response.content)
        logger.info("LLM response intent=%s tool_call_count=%s", state.get("intent"),
                    len(tool_calls))
 
        return {
            "messages": [response],
            "last_action": "llm_called",
            "memory_context_payload": evidence_payload,
            "memory_context_prepared": True,
        }
    
    async def web_search_node(state: AgentState, store: BaseStore = None) -> dict:
        """
        Perform a web search using Tavily API for non-todo queries.
        Returns formatted search results.
        """
        
        user_message = state["messages"][-1].content
        query = state.get("entities", {}).get("query", user_message)
        profile_context = await load_profile_context(
            store, state.get("user_id", ""), strict=memory_orchestrator is not None)
        if profile_context:
            try:
                rewritten = await search_llm.with_structured_output(ProfileAwareSearchQuery).ainvoke([
                    SystemMessage(content=(
                        "Rewrite a web search query using only profile details relevant to the user's question. "
                        "Do not add unrelated interests or change the user's intent."
                    )),
                    HumanMessage(content=f"Question: {user_message}\nProfile: {profile_context}"),
                ])
                query = ProfileAwareSearchQuery.model_validate(rewritten).query.strip() or query
            except Exception as exc:
                logger.warning("Profile-aware search query rewrite failed: %s", type(exc).__name__)
        
        logger.info("Web search query prepared for user_id=%s query_chars=%s",
                    state.get("user_id", ""), len(query))
        
        try:
            client = TavilyClient(api_key=settings.tavily_api_key)
            
            results = await asyncio.to_thread(
                client.search, query=query, max_results=5, include_answer=True,
            )
            sources = results.get("results", []) if isinstance(results, dict) else []
            if not sources:
                response_text = tr("web_none")
            else:
                context = [{"title": item.get("title", ""), "url": item.get("url", ""),
                            "content": item.get("content", "")[:1200]} for item in sources[:5]]
                answer = await search_llm.ainvoke([
                    SystemMessage(content=(
                        "Answer the user's question using only the provided search results. "
                        "Do not invent facts. Include source URLs for substantive claims. "
                        "If the sources do not answer the question, say so clearly. "
                        f"Follow relevant user response preferences if any: {profile_context or 'none'}. "
                        + current_preferences().llm_instruction
                    )),
                    HumanMessage(content=f"Question: {user_message}\nSearch results: {json.dumps(context, ensure_ascii=False)}"),
                ])
                response_text = extract_text(answer.content) or tr("web_summarize_error")
            logger.info("Web search completed: %s results", len(sources))
            
        except Exception as e:
            logger.error("Web search failed: %s", type(e).__name__)
            response_text = tr("web_unavailable")
        
        return {
            "messages": [AIMessage(content=response_text)],
            "last_action": "web_search_completed",
            "search_results": response_text,
        }
 
    async def chat_response_node(state: AgentState, store: BaseStore) -> dict:
        """
        Generate a response for general chat messages (non-todo).
        """
        user_message = state["messages"][-1].content if state["messages"] else ""
        user_id = state.get("user_id", "")

        if state.get("intent") == "memory":
            if memory_control is None:
                return {"messages": [AIMessage(content=tr("memory_unavailable"))],
                        "last_action": "memory_unavailable", "should_search_web": False,
                        "entities": {}}
            current = next((item for item in reversed(state.get("messages", []))
                            if isinstance(item, HumanMessage)), None)
            if (current is None or not current.id or
                    (state.get("entities") or {}).get("memory_operation") not in {"correct", "forget"}):
                return {"messages": [AIMessage(content=tr("memory_target_missing"))],
                        "last_action": "memory_target_missing", "should_search_web": False,
                        "entities": {}}
            try:
                result = await memory_control.apply_control(
                    user_id=user_id, thread_id=state.get("conversation_thread_id") or f"user_{state['telegram_id']}",
                    operation=state["entities"]["memory_operation"],
                    target_query=state["entities"].get("memory_target_query", ""),
                    request=str(current.content), current_message_id=str(current.id),
                    history=list(state.get("messages", [])),
                    additional_history_threads=tuple(state.get("memory_history_threads") or ()))
            except Exception as exc:
                logger.warning("memory_control stage=apply category=%s", type(exc).__name__)
                return {"messages": [AIMessage(content=tr("memory_unavailable"))],
                        "last_action": "memory_control_failed", "should_search_web": False,
                        "entities": {}}
            messages = []
            if result.supersedes_message_id:
                messages.append(HumanMessage(
                    id=current.id, content=current.content,
                    additional_kwargs={**current.additional_kwargs,
                                       "supersedes_message_id": result.supersedes_message_id,
                                       "supersedes_thread_id": result.supersedes_thread_id}))
            messages.append(AIMessage(content=result.response))
            return {"messages": messages,
                    "last_action": "memory_control_applied" if result.changed else "memory_control_clarified",
                    "should_search_web": False, "entities": {}}

        profile_context = await load_profile_context(
            store, user_id, strict=memory_orchestrator is not None)
        messages = chat_response_messages(list(state.get("messages", [])), profile_context)
        if memory_orchestrator is not None:
            from app.agent.memory_context import memory_evidence_message
            bundle = await selected_memory(state, "chat")
            if bundle is not None:
                evidence = memory_evidence_message(bundle)
                prompt = str(messages[0].content)
                if evidence:
                    prompt += "\n\n" + str(evidence.content)
                messages = [SystemMessage(content=prompt)] + list(bundle.recent_messages)
        if not messages[1:] or not isinstance(messages[-1], HumanMessage):
            return {"messages": [AIMessage(content=tr("chat_fallback"))],
                    "last_action": "chat_response_fallback", "should_search_web": False,
                    "entities": {}}
        
        logger.info("Generating chat response for user_id=%s message_chars=%s",
                    user_id, len(user_message))

        structured_llm = llm.with_structured_output(ChatResponse)
        try:
            response = await structured_llm.ainvoke(messages)
        except Exception as e:
            logger.error("Structured chat response failed category=%s", type(e).__name__)
            fallback_content = tr("chat_fallback")
            return {
                "messages": [AIMessage(content=fallback_content)],
                "last_action": "chat_response_fallback",
                "should_search_web": False,
                "entities": {},
            }


        return {
            "messages": [AIMessage(content=response.content)],
            "last_action": "chat_response_generated",
            "should_search_web": response.should_search_web,
            "entities": {"query": response.query} if response.should_search_web else {},
            "awaiting_conflict_confirmation": False,
        }

    async def update_profile_node(state: AgentState, store: BaseStore) -> dict:
        """Update profile memory from self-introductions and preference statements."""
        user_message = state["messages"][-1].content if state.get("messages") else ""
        user_id = state.get("user_id", "")

        if not user_message or not user_id:
            return {"last_action": "profile_memory_skipped"}

        if not _has_profile_signals(user_message):
            return {"last_action": "profile_memory_skipped"}

        profile_update = _extract_profile_update(user_message)
        if not profile_update:
            return {"last_action": "profile_memory_skipped"}

        namespace = (PROFILE_NAMESPACE, user_id)

        try:
            existing_item = await store.aget(namespace, "user_profile")
            existing_profile = existing_item.value if existing_item else None
        except Exception as exc:
            logger.error("Profile memory load failed category=%s", type(exc).__name__)
            existing_profile = None

        merged_profile = _merge_profile(existing_profile, profile_update)

        try:
            await store.aput(namespace, "user_profile", merged_profile)
        except Exception as exc:
            logger.warning("Profile memory update failed category=%s", type(exc).__name__)
            return {"last_action": "profile_memory_failed"}

        logger.info(f"Profile memory updated for user_id={user_id}: {list(profile_update.keys())}")
        return {"last_action": "profile_memory_updated"}
    
    async def execute_tools(state: AgentState) -> dict:
        """Execute any tool calls requested by the LLM."""
        
        last_message = state["messages"][-1]
        tool_results = []
 
        read_mode = state.get("intent") in AGENT_READ_INTENTS
        selected = selected_tools(state)
        tool_map = {t.name: t for t in selected}
        recent_gmail_thread_id = state.get("recent_gmail_thread_id")
        recent_gmail_owner = state.get("recent_gmail_owner_telegram_id")
        workflow_handoff = {}
        capability_failed = False
        read_sources_seen = set(state.get("read_sources_seen") or [])
        memory_search_rounds = state.get("memory_search_rounds", 0)
 
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = dict(tool_call["args"])

            if workflow_handoff or capability_failed:
                tool_results.append(ToolMessage(
                    content="A capability was already handled for this turn.",
                    tool_call_id=tool_call["id"]))
                continue

            if tool_name not in tool_map:
                logger.warning("Unknown tool call blocked")
                result = ("Memory search limit reached for this turn; use existing evidence or state uncertainty."
                          if tool_name == "search_memory" and memory_search_rounds >= 2
                          else tr("tool_unavailable"))
                if tool_name in capability_tool_map or state.get("intent") in {
                        "delete", "calendar_create", "calendar_update", "calendar_delete",
                        "cross_domain", "smart_schedule", "meeting_prep", "email_to_todo",
                        "email_to_calendar"}:
                    capability_failed = True
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if tool_name == "search_memory":
                if memory_search_rounds >= 2:
                    tool_results.append(ToolMessage(
                        content="Memory search limit reached for this turn; use existing evidence or state uncertainty.",
                        tool_call_id=tool_call["id"]))
                    continue
                memory_search_rounds += 1

            is_calendar_read_tool = tool_name in CALENDAR_READ_ONLY_TOOL_NAMES
            is_todo_tool = tool_name in TODO_TOOL_NAMES
            # Any MCP calendar tool outside the read-only list is treated as write/restricted.
            is_calendar_write_tool = ("-" in tool_name) and not is_calendar_read_tool

            if is_calendar_write_tool:
                logger.warning("Blocked restricted calendar tool call")
                result = tr("calendar_write_blocked")
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            # The MCP server is authenticated to one Google account. A Todo
            # user's own Telegram identity must match that account's owner
            # before any legacy graph tool can read its private calendar.
            if is_calendar_read_tool and (
                not settings.google_calendar_owner_telegram_id
                or str(state.get("telegram_id", "")) != settings.google_calendar_owner_telegram_id
            ):
                tool_results.append(ToolMessage(
                    content=tr("calendar_user_unavailable"),
                    tool_call_id=tool_call["id"],
                ))
                continue

            if tool_name == "delete_todo":
                logger.warning("Blocked unapproved Todo delete tool call")
                tool_results.append(ToolMessage(
                    content=tr("todo_delete_blocked"),
                    tool_call_id=tool_call["id"],
                ))
                continue

            if state.get("intent") == "create" and not (is_todo_tool or is_calendar_read_tool or
                                                       tool_name == "search_memory"):
                logger.warning("Blocked non-todo tool for create intent")
                result = tr("tool_blocked")
                tool_results.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
 
            # Inject user_id if the tool needs it
            if tool_name in TODO_TOOL_NAMES:
                tool_args["user_id"] = state.get("user_id", "")
            if (read_mode and tool_name in read_tool_map) or tool_name in capability_tool_map:
                # The model cannot choose identity, dates or Gmail query.
                # Those come from the validated route/current server state.
                tool_args = {}
 
            try:
                from app.telegram.processing_context import tool_call_scope
                scope = (capability_scope(
                            route=state.get("precomputed_route") or {},
                            request=state.get("active_request", ""),
                            user_id=state.get("user_id", ""),
                            telegram_id=state.get("telegram_id", ""),
                            turn_id=state.get("active_turn_id", ""),
                            origin=state.get("request_origin", ""),
                            memory_history_threads=tuple(state.get("memory_history_threads") or ()),
                            recent_context=tuple({
                                "source": str(item.id or "turn"), "text": item.content}
                                for item in state.get("messages", [])[-20:]
                                if isinstance(item, HumanMessage) and isinstance(item.content, str)))
                         if tool_name in capability_tool_map else
                         read_scope(route=state.get("precomputed_route") or {},
                                    request=state.get("active_request", ""),
                                    user_id=state.get("user_id", ""),
                                    telegram_id=state.get("telegram_id", ""),
                                    recent_gmail_thread_id=(recent_gmail_thread_id
                                        if recent_gmail_owner == state.get("telegram_id") else None))
                         if read_mode and tool_name in read_tool_map else nullcontext())
                with tool_call_scope(tool_call["id"]), scope as read_context:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                if read_mode and tool_name == "read_gmail_evidence":
                    recent_gmail_thread_id = read_context.selected_gmail_thread_id
                    recent_gmail_owner = (state.get("telegram_id") if recent_gmail_thread_id else None)
                if read_mode:
                    if tool_name.startswith("read_calendar_"):
                        read_sources_seen.add("calendar")
                    elif tool_name == "read_gmail_evidence":
                        read_sources_seen.add("gmail")
                    elif tool_name in {"list_todos", "search_todos"}:
                        read_sources_seen.add("todo")
                if tool_name in capability_tool_map and isinstance(result, dict) and result.get("handoff"):
                    workflow_handoff = {"thread_id": result["thread_id"], "turn_id": result["turn_id"]}
            except Exception as e:
                from app.telegram.durable import DurableSafetyError
                if isinstance(e, DurableSafetyError):
                    raise
                from app.telegram.processing_context import current_run
                from app.web.processing_context import current_web_run
                if tool_name in capability_tool_map and isinstance(e, CapabilityRejected):
                    capability_failed = True
                    result = tr("tool_unavailable")
                    tool_results.append(ToolMessage(
                        content=result, tool_call_id=tool_call["id"]))
                    continue
                if (current_run() is not None or current_web_run() is not None) and (
                        is_todo_tool or tool_name in capability_tool_map):
                    raise DurableSafetyError("ToolOutcomeUncertain") from e
                logger.error("Tool call failed category=%s", type(e).__name__)
                result = tr("tool_error", error=type(e).__name__)
                if tool_name in capability_tool_map:
                    capability_failed = True
 
            tool_results.append(
                ToolMessage(
                    content=str(result.get("response", "")) if isinstance(result, dict) else str(result),
                    tool_call_id=tool_call["id"],
                )
            )
 
        if capability_failed:
            tool_results.append(AIMessage(content=tr("tool_unavailable")))
        return {
            "messages": tool_results,
            "last_action": "tools_executed",
            "read_tool_rounds": state.get("read_tool_rounds", 0) + 1 if read_mode else state.get("read_tool_rounds", 0),
            "read_sources_seen": sorted(read_sources_seen),
            "memory_search_rounds": memory_search_rounds,
            "recent_gmail_thread_id": recent_gmail_thread_id,
            "recent_gmail_owner_telegram_id": recent_gmail_owner,
            "workflow_handoff": workflow_handoff,
            "capability_failed": capability_failed,
        }
 
    async def check_calendar_node(state: AgentState) -> dict:
        intent = state.get("intent", "")
        entities = state.get("entities", {})
        due_date_str = entities.get("due_date_str", "")

        if not settings.google_calendar_owner_telegram_id or (
            str(state.get("telegram_id", "")) != settings.google_calendar_owner_telegram_id
        ):
            logger.info("Calendar conflict check skipped: user has no Calendar authorization")
            return {
                "has_calendar_conflict": False,
                "calendar_conflicts": [],
                "calendar_check_done": True,
            }

        if intent != "create" or not due_date_str:
            return {
                "has_calendar_conflict": False,
                "calendar_conflicts": [],
                "calendar_check_done": True,
            }

        parsed_dt = dateparser.parse(
            due_date_str,
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": CALENDAR_TIMEZONE,
                "RETURN_AS_TIMEZONE_AWARE": True,
            }
        )
        if not parsed_dt:
            logger.warning("Could not parse calendar date expression")
            return {
                "has_calendar_conflict": False,
                "calendar_conflicts": [],
                "calendar_check_done": True,
            }

        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=CALENDAR_TZ)
        else:
            parsed_dt = parsed_dt.astimezone(CALENDAR_TZ)

        start_iso = parsed_dt.isoformat()
        end_iso = (parsed_dt + timedelta(hours=1)).isoformat()

        logger.info("Checking calendar availability")

        freebusy_tool = next((t for t in tools if t.name == "get-freebusy"), None)
        list_events_tool = next((t for t in tools if t.name == "list-events"), None)

        has_conflict = False
        conflict_events = []

        # ── Step 1: get-freebusy (catches timed events) ──────────────
        if freebusy_tool:
            try:
                result = await freebusy_tool.ainvoke({
                    "calendars": [{"id": "primary"}],
                    "timeMin": start_iso,
                    "timeMax": end_iso,
                    "timeZone": CALENDAR_TIMEZONE,
                })

                text = next(
                    (b["text"] for b in result
                    if isinstance(b, dict) and b.get("type") == "text"),
                    ""
                ) if isinstance(result, list) else str(result)

                data = json.loads(text)
                for cal_id, cal_data in data.get("calendars", {}).items():
                    busy = cal_data.get("busy", [])
                    if busy:
                        has_conflict = True
                        conflict_events = [
                            {"title": "Busy", "start": b["start"], "end": b["end"], "all_day": False}
                            for b in busy
                        ]
                        break

                logger.info(f"get-freebusy: conflict={has_conflict}")

            except Exception as e:
                logger.error("get-freebusy failed category=%s", type(e).__name__)

        # ── Step 2: list-events detail enrichment (adds titles/all-day info) ────
        if list_events_tool:
            try:
                result = await list_events_tool.ainvoke({
                    "calendarId": "primary",
                    "timeMin": start_iso,
                    "timeMax": end_iso,
                    "timeZone": CALENDAR_TIMEZONE,
                })

                text = next(
                    (b["text"] for b in result
                    if isinstance(b, dict) and b.get("type") == "text"),
                    ""
                ) if isinstance(result, list) else str(result)

                data = json.loads(text)
                events = data if isinstance(data, list) else data.get("events", [])

                if events:
                    has_conflict = True
                    conflict_events = [
                        {
                            "title": e.get("summary", "Busy"),
                            "start": e.get("start", {}).get("dateTime")
                                    or e.get("start", {}).get("date", ""),
                            "end": e.get("end", {}).get("dateTime")
                                or e.get("end", {}).get("date", ""),
                            "all_day": "date" in e.get("start", {}),
                        }
                        for e in events
                    ]
                    logger.info(
                        f"list-events: found {len(events)} event(s) — using detailed conflict events"
                    )
                else:
                    logger.info("list-events: no events found — keeping freebusy conflict details")

            except Exception as e:
                logger.error("list-events failed category=%s", type(e).__name__)

        logger.info("Calendar conflict check complete conflict=%s event_count=%s",
                    has_conflict, len(conflict_events))

        return {
            "has_calendar_conflict": has_conflict,
            "calendar_conflicts": conflict_events,
            "calendar_check_done": True,
            "awaiting_conflict_confirmation": False,

        }

    async def ask_calendar_confirmation(state: AgentState) -> dict:
        """
        Inform user about conflict and ask if they want to proceed.
        """
        from langchain_core.messages import AIMessage

        entities = state.get("entities", {})
        conflicts = state.get("calendar_conflicts", [])
        title = entities.get("title", "the task")
        due_date = entities.get("due_date_str", "the requested time")

        conflict_detail = ""
        if conflicts:
            lines = []
            for conflict in conflicts[:3]:
                conflict_title = conflict.get("title", "Busy")
                start = conflict.get("start", "")
                end = conflict.get("end", "")
                if conflict.get("all_day"):
                    lines.append(f"- {conflict_title} (all day: {start} to {end})")
                elif start and end:
                    lines.append(f"- {conflict_title} ({start} to {end})")
                else:
                    lines.append(f"- {conflict_title}")

            if lines:
                conflict_detail = "\n" + tr("calendar_conflict_heading") + "\n" + "\n".join(lines)

        message = tr("calendar_conflict_prompt", title=title, time=due_date,
                     details=conflict_detail)

        return {
            "messages": [AIMessage(content=message)],
            "last_action": "calendar_confirmation_asked",
            "awaiting_conflict_confirmation": True,
        }
    return classify_intent, ask_clarification, todo_llm, execute_tools, web_search_node, chat_response_node, update_profile_node, check_calendar_node, ask_calendar_confirmation
 
def messages_router(state: AgentState) -> str:
    """
    Route after classify_intent:
    - If web_search intent → web_search_node
    - If very low confidence → ask_clarification
    - Otherwise → todo_llm
    """
    intent = state.get("intent", "")
    confidence = state.get("confidence", 0.0)

    awaiting = state.get("awaiting_conflict_confirmation", False)
    logger.info(f"Routing after intent classification: intent={intent}, awaiting_confirmation={awaiting}")

    # User is confirming/cancelling a conflict warning
    if awaiting:
        last_msg = state["messages"][-1].content.strip().lower()
        if last_msg in CONFIRM_POSITIVE:
            logger.info("User confirmed calendar conflict — proceeding to create")
            return "todo_llm"

        if last_msg in CONFIRM_NEGATIVE:
            logger.info("User declined calendar conflict — switching to chat response")
            return "chat_response_node"

        logger.info("Stale/irrelevant confirmation state detected — routing by intent")
    
    if confidence < 0.65:
        logger.info(f"Low confidence ({confidence:.2f}) - asking for clarification")
        return "ask_clarification"

    if intent == "clarify":
        return "ask_clarification"
    
    if intent == "create":
        # A Todo deadline is not a Calendar reservation. Checking a one-hour
        # slot at that deadline caused false conflicts (and silently treated
        # MCP errors as free). Calendar scheduling belongs to the controlled
        # productivity workflow, not this Todo fast path.
        return "todo_llm"

    if intent in {"calendar_create", "calendar_update", "calendar_delete", "cross_domain",
                  "smart_schedule", "meeting_prep", "email_to_todo", "email_to_calendar",
                  "delete"}:
        return "todo_llm"
    
    if intent in AGENT_READ_INTENTS:
        return "todo_llm"  # stable node name retained for existing Saver checkpoints

    if intent in {"chat", "memory"}:
        logger.info("Chat intent detected")
        return "chat_response_node"

    if intent == "update_profile":
        logger.info("Profile update intent detected")
        return "update_profile_node"
    
    if intent == "web_search":
        logger.info("Web search intent detected")
        return "web_search"
    
    logger.info(f"Adequate confidence ({confidence:.2f}) - proceeding to LLM")
    return "todo_llm"
 
def should_execute_tools(state: AgentState) -> str:
    """Route after todo_llm: if tool_calls → execute, else → end."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "end"

def should_search_web_after_chat(state: AgentState) -> str:
    """After a chat response, check if we should do a web search."""
    if state.get("should_search_web"):
        logger.info("Routing to web search")
        return "web_search"
    return "end" 

def should_continue_after_calendar(state: AgentState) -> str:
    """Route after calendar check: conflict → confirm, else → todo_llm."""
    if state.get("has_calendar_conflict", False):
        return "ask_calendar_confirmation"
    return "todo_llm"

def format_tavily_results(query: str, results: dict) -> str:
    """
    Format Tavily search results into readable text.
    
    Tavily returns:
    {
        "answer": "AI-generated answer",
        "results": [
            {
                "title": "Result title",
                "url": "https://...",
                "content": "Result content",
                "score": 0.95
            },
            ...
        ]
    }
    """
    lines = [f"🔍 Search results for: **{query}**\n"]
    
    # Include AI answer if available
    if results.get("answer"):
        lines.append(f"📝 Summary: {results['answer']}\n")
    
    # Include top results
    if results.get("results"):
        lines.append("📋 Sources:")
        for i, result in enumerate(results["results"][:3], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "No content")[:150]
            
            lines.append(f"\n{i}. **{title}**")
            if url:
                lines.append(f"   🔗 {url}")
            lines.append(f"   {content}...")
    else:
        lines.append("No results found.")
    
    return "\n".join(lines)
