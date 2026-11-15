"""Request-local receipt and checkpointed tool-call correlation.

The receipt ID is *not* an authorization source. Tool calls are already
authorized by the graph and user-scoped tool policy; this only provides a
replay key for their local database side effects.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class ReceiptRun:
    receipt_id: str
    fence_token: str
    reply_ordinal: int = 0
    outcome_kind: str | None = None
    outcome_ref: str | None = None
    required_reply_count: int | None = None

    def record_outcome(self, kind: str, reference: str, *, replies: int = 1) -> None:
        """Declare verified business evidence before its reply can be queued.

        The declaration is not durable until enqueue_reply commits it and the
        immutable reply intent in the same transaction.
        """
        if (kind not in {"no_business", "ordinary_checkpoint", "workflow_checkpoint",
                         "tool_result", "profile_store"} or not reference or len(reference) > 256 or
                not 1 <= replies <= 10 or
                (replies > 1 and kind != "ordinary_checkpoint") or self.reply_ordinal):
            raise ValueError("InvalidBusinessOutcome")
        if self.outcome_kind is not None and (kind, reference, replies) != (
                self.outcome_kind, self.outcome_ref, self.required_reply_count):
            raise ValueError("ConflictingBusinessOutcome")
        self.outcome_kind = kind
        self.outcome_ref = reference
        self.required_reply_count = replies

    def next_reply(self) -> int:
        if self.required_reply_count is None or self.reply_ordinal >= self.required_reply_count:
            raise ValueError("ReplyCountExceeded")
        value = self.reply_ordinal
        self.reply_ordinal += 1
        return value


_run: ContextVar[ReceiptRun | None] = ContextVar("telegram_receipt_run", default=None)
_tool_call_id: ContextVar[str | None] = ContextVar("telegram_tool_call_id", default=None)


def current_run() -> ReceiptRun | None:
    return _run.get()


def current_tool_call_id() -> str | None:
    return _tool_call_id.get()


@contextmanager
def receipt_scope(receipt_id: str, fence_token: str):
    token = _run.set(ReceiptRun(receipt_id, fence_token))
    try:
        yield _run.get()
    finally:
        _run.reset(token)


@contextmanager
def tool_call_scope(tool_call_id: str):
    token = _tool_call_id.set(tool_call_id)
    try:
        yield
    finally:
        _tool_call_id.reset(token)
