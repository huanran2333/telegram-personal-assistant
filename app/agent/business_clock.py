"""Task-local business time for isolated Agent evaluation.

Only the evaluation runner binds a fixed instant. Production requests never
accept a clock value, so the default remains the system clock. Lease, network,
and database audit times deliberately do not use this boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone, tzinfo
from typing import Iterator
from zoneinfo import ZoneInfo


_evaluation_now: ContextVar[datetime | None] = ContextVar(
    "agent_evaluation_business_now", default=None)


def business_now(zone: tzinfo) -> datetime:
    fixed = _evaluation_now.get()
    return fixed.astimezone(zone) if fixed is not None else datetime.now(zone)


@contextmanager
def isolated_evaluation_time(now: datetime, zone_name: str, *, app_env: str) -> Iterator[None]:
    """Bind an aware, zone-consistent instant only for an isolated test run."""
    if app_env.lower() != "test":
        raise PermissionError("EvaluationClockRequiresTestEnvironment")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("EvaluationClockMustBeTimezoneAware")
    ZoneInfo(zone_name)
    token = _evaluation_now.set(now.astimezone(timezone.utc))
    try:
        yield
    finally:
        _evaluation_now.reset(token)
