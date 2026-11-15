"""Typed, policy-controlled actions shared by planner and executor."""
from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    INTERNAL_LOW_RISK = "internal_low_risk"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"


class PlanStep(BaseModel):
    step_id: str
    domain: str
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None
    mutation: bool = False


class RequestAnalysis(BaseModel):
    mode: str = "fast_path"
    requires_plan: bool = False
    actions: list[PlanStep] = Field(default_factory=list)
    explanation: str = ""


class ApprovedCalendarBaseline(BaseModel):
    """The exact remote version shown for update/delete HITL approval."""

    calendar_id: str
    event_id: str
    etag: str
    status: str
    summary: str | None = None
    description: str | None = None
    location: str | None = None
    start: dict[str, Any] = Field(default_factory=dict)
    end: dict[str, Any] = Field(default_factory=dict)


class ActionProposal(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid4().hex)
    domain: str
    operation: str
    args: dict[str, Any]
    human_readable_preview: str
    reason: str
    risk_level: RiskLevel
    status: ActionStatus = ActionStatus.PROPOSED
    approved_calendar_baseline: ApprovedCalendarBaseline | None = None


class ActionResult(BaseModel):
    action_id: str
    status: ActionStatus
    message: str
    result: dict[str, Any] = Field(default_factory=dict)


def requires_approval(proposal: ActionProposal) -> bool:
    """Operation-owned policy; a model-supplied risk label cannot downgrade writes."""
    mandatory = {
        ("calendar", "create_event"), ("calendar", "update_event"),
        ("calendar", "delete_event"), ("todo", "delete_todo"),
        ("todo", "delete_todos_by_tag"),
        ("todo", "create_todo_from_email"),
    }
    return ((proposal.domain, proposal.operation) in mandatory or
            proposal.risk_level in {RiskLevel.EXTERNAL_WRITE, RiskLevel.DESTRUCTIVE})
