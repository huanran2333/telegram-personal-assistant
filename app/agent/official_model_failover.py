"""Bounded model-only failover for the current create_agent graph.

This hook wraps model requests, not ToolNode execution.  It never retries a
Todo/Calendar write or resumes an approval.  Permanent provider errors remain
visible rather than being hidden by another model.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import openai
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.provider_clients import current_provider_owner


logger = logging.getLogger(__name__)


def _model_name(model: BaseChatModel) -> str:
    return str(getattr(model, "model_name", None) or model._llm_type)


def _request_round(request) -> int:
    """One-based model turn since the last incoming user message."""
    messages = list(request.state.get("messages") or ())
    last_user = max((i for i, message in enumerate(messages)
                     if isinstance(message, HumanMessage)), default=-1)
    return 1 + sum(isinstance(message, AIMessage) for message in messages[last_user + 1:])


def _retryable_model_error(exc: Exception) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                        httpx.TransportError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return False


class OfficialModelFailover(AgentMiddleware):
    """Two bounded primary attempts, then one attempt per configured backup."""

    def __init__(self, backups: list[BaseChatModel], *, primary_attempts: int = 2):
        super().__init__()
        if not backups or not 1 <= primary_attempts <= 3:
            raise ValueError("InvalidModelFailoverConfiguration")
        self.backups = backups
        self.primary_attempts = primary_attempts

    async def awrap_model_call(self, request, handler):
        last_error: Exception | None = None
        model_round = _request_round(request)
        for attempt in range(self.primary_attempts):
            try:
                result = await handler(request)
                logger.info("agent_model_response round=%s backend=primary model=%s",
                            model_round, _model_name(request.model))
                return result
            except Exception as exc:
                if not _retryable_model_error(exc):
                    raise
                last_error = exc
                logger.warning("agent_model_primary_failed round=%s attempt=%s model=%s category=%s",
                               model_round, attempt + 1, _model_name(request.model),
                               type(exc).__name__)
                if attempt + 1 < self.primary_attempts:
                    await asyncio.sleep(0.25)
        for index, backup in enumerate(self.backups, 1):
            try:
                result = await handler(request.override(model=backup))
                logger.warning("agent_model_fallback_used round=%s index=%s model=%s reason=%s",
                               model_round, index, _model_name(backup),
                               type(last_error).__name__ if last_error is not None else "unknown")
                return result
            except Exception as exc:
                if not _retryable_model_error(exc):
                    raise
                last_error = exc
                logger.warning("agent_model_fallback_failed round=%s index=%s model=%s category=%s",
                               model_round, index, _model_name(backup), type(exc).__name__)
        assert last_error is not None
        raise last_error


def configured_failover(settings) -> OfficialModelFailover | None:
    """Construct backup clients under the application's provider-client owner."""
    configs = [
        (getattr(settings, f"llm_fallback_{index}_base_url"),
         getattr(settings, f"llm_fallback_{index}_model"),
         getattr(settings, f"llm_fallback_{index}_api_key"))
        for index in (1, 2)
    ]
    configs = [(url, model, key) for url, model, key in configs if url]
    if not configs:
        return None
    if settings.llm_provider != "openai":
        raise ValueError("ModelFallbackRequiresOpenAICompatibleProvider")
    from langchain_openai import ChatOpenAI

    sync_http, async_http = current_provider_owner().openai_http_clients()
    backups = [ChatOpenAI(
        model=model, base_url=url, api_key=key,
        temperature=0, max_tokens=4096, max_retries=0,
        request_timeout=settings.llm_fallback_timeout_seconds,
        http_client=sync_http, http_async_client=async_http,
        **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
           if "qwen" in model.lower() else {}))
        for url, model, key in configs]
    return OfficialModelFailover(backups,
                                 primary_attempts=settings.llm_primary_max_attempts)
