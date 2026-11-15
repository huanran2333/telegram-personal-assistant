"""Small privacy-preserving, per-turn structured trace.

The execution ledger remains the authority for mutations. These JSON log
events are diagnostic only and deliberately exclude user text, tool arguments,
OAuth data, and message bodies.
"""
from __future__ import annotations

import json
import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

logger = logging.getLogger("app.turn_trace")


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    user_id: str | None
    started: float
    origin_trace_id: str | None = None
    thread_ref: str | None = None


_context: ContextVar[TraceContext | None] = ContextVar("turn_trace", default=None)
_ALLOWED = {"route", "operation", "tool", "success", "latency_ms", "token_usage",
            "external_id", "error_category", "update_id", "action_id", "selection_id",
            "schema_count", "model_calls", "context_tokens", "source_count"}
_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SECRETISH = re.compile(r"(?i)(bearer|secret|token|api.?key|sk-|eyJ[A-Za-z0-9_-]{12,})")


def diagnostic_error(exc: BaseException) -> str:
    """Exception class only: remote exception text can contain private payloads."""
    return type(exc).__name__


def _safe_field(key: str, value):
    if key == "success":
        return value if isinstance(value, bool) else None
    if key in {"latency_ms", "token_usage", "update_id", "schema_count",
               "model_calls", "context_tokens", "source_count"}:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    if key in _ALLOWED:
        return (value if isinstance(value, str) and _LABEL.fullmatch(value)
                and not _SECRETISH.search(value) else None)
    return None


def begin_trace(user_id: str | None = None):
    return _context.set(TraceContext(uuid4().hex, user_id, time.perf_counter()))


def current_trace_id() -> str | None:
    current = _context.get()
    return current.trace_id if current else None


def set_user(user_id: str):
    current = _context.get()
    if current:
        _context.set(replace(current, user_id=user_id))


def set_correlation(*, origin_trace_id: str | None = None, thread_id: str | None = None) -> None:
    """Diagnostic linkage only; never used to authorize or execute an action.

    The thread reference in logs is a digest, not the raw Telegram-derived ID.
    """
    current = _context.get()
    if current is None:
        return
    origin = origin_trace_id if isinstance(origin_trace_id, str) and _TRACE_ID.fullmatch(origin_trace_id) else None
    thread_ref = sha256(thread_id.encode("utf-8")).hexdigest()[:16] if isinstance(thread_id, str) else None
    _context.set(replace(current, origin_trace_id=origin, thread_ref=thread_ref))


def end_trace(token) -> None:
    current = _context.get()
    if current:
        emit_trace("request_end", latency_ms=round((time.perf_counter() - current.started) * 1000))
    _context.reset(token)


def emit_trace(stage: str, **fields) -> None:
    current = _context.get()
    if current is None:
        return
    if not isinstance(stage, str) or not _LABEL.fullmatch(stage):
        return
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "trace_id": current.trace_id, "stage": stage}
    if current.user_id and _LABEL.fullmatch(current.user_id) and not _SECRETISH.search(current.user_id):
        payload["user_id"] = current.user_id
    if current.origin_trace_id:
        payload["origin_trace_id"] = current.origin_trace_id
    if current.thread_ref:
        payload["thread_ref"] = current.thread_ref
    for key, value in fields.items():
        if key in _ALLOWED:
            sanitized = _safe_field(key, value)
            if sanitized is not None:
                payload[key] = sanitized
    logger.info(json.dumps(payload, ensure_ascii=False))
