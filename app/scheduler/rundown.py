"""Source-backed facts and deterministic priorities for the daily rundown.

Inspired by @context's daily-rundown backbone/best-effort source design and
bmitioglov's collect → judge → deliver separation. No Agno runtime is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from app.localization import safe_error, tr


@dataclass
class FactSnapshot:
    user_id: str
    local_date: date
    timezone: str
    calendar_events: list[dict[str, Any]] = field(default_factory=list)
    due_today: list[Any] = field(default_factory=list)
    overdue: list[Any] = field(default_factory=list)
    due_soon: list[Any] = field(default_factory=list)
    reminders: list[Any] = field(default_factory=list)
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    source_errors: dict[str, str] = field(default_factory=dict)


class PriorityItem(BaseModel):
    source_id: str
    title: str
    time_text: str = ""
    reason: str
    suggestion: str
    score: int


class RundownSynthesis(BaseModel):
    """The LLM may select evidence IDs, never supply facts or free-form claims."""
    ordered_source_ids: list[str] = Field(default_factory=list)


def _local(value: datetime | None, zone: ZoneInfo) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(zone) if value.tzinfo else value.replace(tzinfo=zone)


def build_snapshot(user_id: str, today: date, timezone: str, events: list[dict[str, Any]],
                   todos: list[Any], reminders: list[Any], source_errors: dict[str, str] | None = None) -> FactSnapshot:
    zone = ZoneInfo(timezone)
    snapshot = FactSnapshot(user_id=user_id, local_date=today, timezone=timezone,
                            calendar_events=events, reminders=reminders,
                            source_errors=source_errors or {})
    for todo in todos:
        due = _local(todo.due_date, zone)
        if due is None:
            continue
        if due.date() < today:
            snapshot.overdue.append(todo)
        elif due.date() == today:
            snapshot.due_today.append(todo)
        elif due.date() == today + timedelta(days=1):
            snapshot.due_soon.append(todo)
    timed = []
    for event in events:
        try:
            start = _local(datetime.fromisoformat(event["start"]["dateTime"]), zone)
            end = _local(datetime.fromisoformat(event["end"]["dateTime"]), zone)
            if end > start:
                timed.append((start, end, str(event.get("id") or event.get("summary") or "unknown")))
        except (KeyError, TypeError, ValueError):
            continue
    timed.sort()
    for index, (start, end, source_id) in enumerate(timed):
        for other_start, _, other_id in timed[index + 1:]:
            if other_start >= end:
                break
            snapshot.conflicts.append((source_id, other_id))
    return snapshot


def prioritize(snapshot: FactSnapshot, now: datetime | None = None) -> list[PriorityItem]:
    from app.agent.meeting_prep import related_to

    zone = ZoneInfo(snapshot.timezone)
    current = (now or datetime.now(zone)).astimezone(zone)
    items: list[PriorityItem] = []
    for todo, label, base in (
        *[(item, tr("overdue"), 90) for item in snapshot.overdue],
        *[(item, tr("due_today"), 75) for item in snapshot.due_today],
        *[(item, tr("due_tomorrow"), 55) for item in snapshot.due_soon],
    ):
        priority = getattr(todo.priority, "value", str(todo.priority))
        score = base + (15 if priority == "high" else 5 if priority == "medium" else 0)
        due = _local(todo.due_date, zone)
        items.append(PriorityItem(source_id=f"todo:{todo.id}", title=todo.title,
                                  time_text=due.strftime("%m-%d %H:%M") if due else "",
                                  reason=tr("rundown_todo_reason", priority=tr("priority_" + priority), due=label),
                                  suggestion=tr("rundown_todo_step", title=todo.title), score=score))
    for event in snapshot.calendar_events:
        title = str(event.get("summary") or tr("untitled"))
        raw = event.get("start") or {}
        try:
            start = _local(datetime.fromisoformat(raw["dateTime"]), zone)
        except (KeyError, TypeError, ValueError):
            start = None
        related = [todo for todo in snapshot.due_today + snapshot.due_soon + snapshot.overdue
                   if related_to(title, todo.title)]
        preparation_gap = bool(start and current <= start <= current + timedelta(days=1) and related)
        conflicting = any(str(event.get("id")) in pair for pair in snapshot.conflicts)
        reason_parts = [tr("rundown_meeting")]
        if preparation_gap:
            reason_parts.append(tr("rundown_related", title=related[0].title, id=related[0].id))
        if conflicting:
            reason_parts.append(tr("rundown_overlap"))
        score = 65 + (20 if preparation_gap else 0) + (20 if conflicting else 0)
        if start and timedelta(0) <= start - current <= timedelta(hours=3):
            score += 10
            reason_parts.append(tr("rundown_soon"))
        source_id = f"event:{event.get('id') or title}"
        suggestion = (tr("rundown_prepare", todo=related[0].title, event=title) if preparation_gap
                      else tr("rundown_agenda", title=title))
        if conflicting:
            suggestion = tr("rundown_review_conflict", title=title)
        items.append(PriorityItem(source_id=source_id, title=title,
                                  time_text=start.strftime("%H:%M") if start else tr("all_day"),
                                  reason="; ".join(reason_parts), suggestion=suggestion, score=score))
    items.sort(key=lambda item: (-item.score, item.time_text, item.source_id))
    return items


async def synthesize_order(items: list[PriorityItem], llm=None) -> list[str]:
    """LLM ranks already-grounded IDs; errors or invalid IDs use deterministic rank."""
    if not items:
        return []
    if llm is None:
        from app.agent.nodes import _build_chat_llm
        from app.config import get_settings
        settings = get_settings()
        llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    from langchain_core.messages import HumanMessage, SystemMessage
    summary = [{"id": item.source_id, "reason": item.reason, "score": item.score} for item in items[:10]]
    try:
        output = await llm.with_structured_output(RundownSynthesis).ainvoke([
            SystemMessage(content="Rank up to three source IDs by today's urgency. Return only existing IDs. Do not add facts."),
            HumanMessage(content=str(summary)),
        ])
        selected = RundownSynthesis.model_validate(output).ordered_source_ids
        allowed = {item.source_id for item in items}
        if not selected or len(set(selected)) != len(selected) or any(item not in allowed for item in selected):
            raise ValueError("invalid source IDs")
        return selected[:3]
    except Exception:
        return [item.source_id for item in items[:3]]


def format_rundown(snapshot: FactSnapshot, items: list[PriorityItem],
                   selected_ids: list[str] | None = None) -> str:
    chosen = selected_ids or [item.source_id for item in items[:3]]
    by_id = {item.source_id: item for item in items}
    lines = [tr("rundown_greeting", date=snapshot.local_date, timezone=snapshot.timezone),
             "", tr("top_priorities")]
    for index, source_id in enumerate(chosen[:3], start=1):
        item = by_id.get(source_id)
        if item is None:
            continue
        lines.append(f"{index}. {item.time_text} {item.title} — {item.reason} [{item.source_id}]")
        lines.append("   " + tr("next_step", step=item.suggestion))
    if not items:
        lines.append("- " + tr("rundown_empty"))
    if snapshot.reminders:
        lines.append("\n" + tr("reminders_heading"))
        for todo in snapshot.reminders[:5]:
            lines.append(f"- {todo.title} [{todo.id}]")
    if snapshot.conflicts:
        lines.append("\n" + tr("conflicts_heading"))
        for first, second in snapshot.conflicts[:3]:
            lines.append("- " + tr("conflict_line", first=first, second=second))
    for source, error in snapshot.source_errors.items():
        source_key = "source_calendar" if source == "Calendar" else "source_todo" if source == "Todo" else None
        lines.append("\n" + tr("source_unavailable", source=tr(source_key) if source_key else source,
                                error=safe_error(error)))
    return "\n".join(lines)
