"""Stable Web request identity for checkpointed Todo tool-call replay."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class WebRun:
    request_id: str
    user_id: str
    fence_token: str


_run: ContextVar[WebRun | None] = ContextVar("web_request_run", default=None)


def current_web_run() -> WebRun | None:
    return _run.get()


@contextmanager
def web_run_scope(request_id: str, user_id: str, fence_token: str):
    token = _run.set(WebRun(request_id, user_id, fence_token))
    try:
        yield
    finally:
        _run.reset(token)
