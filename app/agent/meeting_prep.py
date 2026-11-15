"""Read-only, source-attributed preparation for one unambiguous meeting."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo
from app.localization import current_preferences, profile_preferences, tr


_GENERIC = {"会议", "讨论", "安排", "任务", "今天", "明天", "meeting", "task"}


@dataclass(frozen=True)
class MeetingPrepSnapshot:
    """Read-only, per-user evidence passed to the presentation boundary."""
    user_id: str
    event: dict[str, Any]
    todos: list[dict[str, Any]]
    profile: dict[str, Any] | None
    recent_turns: list[dict[str, str]]
    timezone: str
    profile_error: str | None = None


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        words.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return words - _GENERIC


def related_to(meeting_title: str, item_title: str) -> bool:
    return bool(_terms(meeting_title) & _terms(item_title))


def select_meeting(events: list[dict[str, Any]], timezone: str,
                   hour: int | None = None, minute: int = 0,
                   query: str = "") -> tuple[dict[str, Any] | None, str | None]:
    zone = ZoneInfo(timezone)
    matches = []
    for event in events:
        attendees = event.get("attendees") or []
        attendee_names = " ".join(
            str(person.get("displayName") or person.get("email") or "")
            for person in attendees if isinstance(person, dict)
        )
        searchable = f"{event.get('summary') or ''} {attendee_names}"
        if query and not related_to(query, searchable):
            continue
        raw_start = event.get("start", {})
        stamp = raw_start.get("dateTime") if isinstance(raw_start, dict) else None
        if hour is not None:
            if not stamp:
                continue
            try:
                local = datetime.fromisoformat(stamp).astimezone(zone)
            except ValueError:
                continue
            if (local.hour, local.minute) != (hour, minute):
                continue
        matches.append(event)
    if not matches:
        return None, "No matching Calendar meeting was found; I did not invent an agenda."
    if len(matches) > 1:
        choices = ", ".join(str(event.get("summary", event.get("id", "Untitled"))) for event in matches[:5])
        return None, f"Multiple meetings match ({choices}). Please give a title and local time."
    return matches[0], None


def format_meeting_prep(snapshot: MeetingPrepSnapshot) -> str:
    event, todos, profile = snapshot.event, snapshot.todos, snapshot.profile
    recent_context, timezone = snapshot.recent_turns, snapshot.timezone
    zone = ZoneInfo(timezone)
    preferences = profile_preferences(profile) if profile else current_preferences()
    text = lambda key, **kwargs: tr(key, preferences=preferences, **kwargs)
    title = str(event.get("summary") or text("untitled"))
    event_id = str(event.get("id") or "unknown")
    start = event.get("start", {})
    stamp = start.get("dateTime") if isinstance(start, dict) else None
    try:
        time_text = datetime.fromisoformat(stamp).astimezone(zone).strftime("%Y-%m-%d %H:%M %Z") if stamp else str(start.get("date", text("time_unavailable")))
    except (ValueError, TypeError):
        time_text = text("time_unavailable")
    attendees = event.get("attendees") or []
    names = [person.get("displayName") or person.get("email") for person in attendees if isinstance(person, dict)]
    concise = preferences.concise
    lines = [text("meeting_title", title=title), text("meeting_when", time=time_text),
             text("meeting_source", id=event_id)]
    if names:
        lines.append(text("meeting_attendees", names=", ".join(name for name in names if name)))
    related = [todo for todo in todos if related_to(title, todo["title"])]
    lines.append("\n" + text("meeting_related"))
    for todo in related[:2 if concise else 5]:
        due = datetime.fromisoformat(todo["due_date"]) if todo.get("due_date") else None
        due_text = due.astimezone(zone).strftime("%Y-%m-%d") if due and due.tzinfo else (due.strftime("%Y-%m-%d") if due else text("no_deadline"))
        status = text("status_" + ("done" if todo["status"] in {"done", "completed"} else "pending"))
        priority = text("priority_" + todo["priority"])
        lines.append("- " + text("meeting_todo_line", title=todo["title"], id=todo["id"],
                                  status=status, priority=priority, due=due_text))
    if not related:
        lines.append("- " + text("meeting_no_todo"))
    relevant_turns = [turn for turn in recent_context if related_to(title, turn.get("text", ""))]
    if relevant_turns or not concise:
        lines.append("\n" + text("meeting_context"))
    for turn in relevant_turns[-1 if concise else -2:]:
        lines.append("- " + text("meeting_user_said", source=turn.get("source", "recent turn"),
                                  text=turn["text"][:100 if concise else 180]))
    if not relevant_turns and not concise:
        lines.append("- " + text("meeting_no_history"))
    lines.append("\n" + text("meeting_points"))
    lines.append("- " + text("meeting_confirm", title=title, id=event_id))
    for todo in related[:1 if concise else 2]:
        lines.append("- " + text("meeting_progress", title=todo["title"], id=todo["id"]))
    if not related:
        lines.append("- " + text("meeting_followup", id=event_id))
    lines.append("\n" + text("meeting_missing", details=text(
        "meeting_no_older" if relevant_turns else "meeting_no_prior")))
    if snapshot.profile_error:
        lines.append(text("meeting_profile_unavailable"))
    return "\n".join(lines)
