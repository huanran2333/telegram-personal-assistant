"""Deterministic daily brief: Todo first, Calendar when available."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, and_

from app.agent.calendar import CalendarAdapter, CalendarError, UnavailableCalendarAdapter
from app.config import get_settings
from app.db import crud
from app.db.models import User, Todo, TodoStatus
from app.db.session import AsyncSessionLocal
from app.scheduler.rundown import build_snapshot, format_rundown, prioritize, synthesize_order
from app.trace import begin_trace, end_trace, emit_trace
from app.localization import load_preferences, reset_preferences, safe_error, tr, use_preferences

logger = logging.getLogger(__name__)


def format_daily_brief(events: list[dict], todos, calendar_error: str | None = None,
                       timezone_name: str | None = None, today=None) -> str:
    zone = ZoneInfo(timezone_name or get_settings().default_timezone)
    today = today or datetime.now(zone).date()
    lines = [tr("daily_greeting"), "", tr("schedule_heading")]
    if calendar_error:
        lines.append("- " + tr("calendar_tasks_only", error=safe_error(calendar_error)))
    elif events:
        for event in events:
            start = event.get("start", {})
            if isinstance(start, dict) and start.get("dateTime"):
                try:
                    local = datetime.fromisoformat(start["dateTime"]).astimezone(zone)
                    time_text = local.strftime("%H:%M")
                except ValueError:
                    time_text = tr("time_unavailable")
            elif isinstance(start, dict) and start.get("date"):
                time_text = tr("all_day")
            else:
                time_text = str(start or event.get("startTime", tr("time_unavailable")))
            lines.append(f"- {time_text} {event.get('summary', tr('calendar_busy'))}")
    else:
        lines.append("- " + tr("no_calendar_events"))
    lines.extend(["", tr("tasks_heading")])
    if todos:
        for todo in todos:
            due = todo.due_date
            due_day = (due.astimezone(zone).date() if due and due.tzinfo else
                       due.date() if due else None)
            label = tr("overdue" if due_day and due_day < today else "due_today")
            lines.append(f"- [{tr('priority_' + todo.priority.value)}] {todo.title} ({label})")
    else:
        lines.append("- " + tr("no_due_tasks"))
    return "\n".join(lines)


async def render_daily_occurrence(occurrence, *, calendar: CalendarAdapter | None = None,
                                  session_factory=AsyncSessionLocal, synthesizer=None,
                                  calendar_owner_telegram_id: str | None = None) -> tuple[str, str]:
    """Render one occurrence; caller durably records immutable text before send."""
    settings = get_settings()
    timezone = ZoneInfo(occurrence.timezone_name)
    from app.scheduler.proactive import as_utc
    planned_local = as_utc(occurrence.planned_at).astimezone(timezone)
    day_start = planned_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    calendar = calendar or UnavailableCalendarAdapter()
    async with session_factory() as db:
        user = await db.get(User, occurrence.user_id)
    if user is None:
        raise RuntimeError("MissingRundownUser")
    trace_token = begin_trace(user.id)
    preferences = await load_preferences(user.id)
    preference_token = use_preferences(preferences)
    try:
        events, source_errors = [], {}
        if calendar_owner_telegram_id is not None and user.telegram_id != calendar_owner_telegram_id:
            source_errors["Calendar"] = "not connected for this user"
        else:
            try:
                events = await calendar.search_events("", day_start.isoformat(), day_end.isoformat())
            except CalendarError:
                source_errors["Calendar"] = "read failed"
            except Exception as exc:
                logger.error("Daily rundown Calendar read failed user_id=%s category=%s",
                             user.id, type(exc).__name__)
                source_errors["Calendar"] = "unexpected read error"
        emit_trace("read", operation="daily_rundown", tool="calendar.list-events",
                   success="Calendar" not in source_errors,
                   error_category="calendar_unavailable" if "Calendar" in source_errors else None)
        async with session_factory() as db:
            todos = await crud.list_todos(db, user.id, due_before=day_end + timedelta(days=1),
                                          exclude_done=True, limit=None)
            reminder_query = select(Todo).where(and_(
                Todo.user_id == user.id, Todo.reminder_at >= day_start,
                Todo.reminder_at < day_end, Todo.reminder_sent_at.is_(None),
                Todo.status != TodoStatus.done))
            reminders = (await db.execute(reminder_query)).scalars().all()
        snapshot = build_snapshot(user.id, planned_local.date(), occurrence.timezone_name,
                                  events, todos, reminders, source_errors)
        items = prioritize(snapshot, datetime.now(timezone))
        selected = await synthesizer(items) if synthesizer else None
        return format_rundown(snapshot, items, selected), preferences.language
    finally:
        reset_preferences(preference_token)
        end_trace(trace_token)


async def send_daily_briefs(calendar=None, session_factory=AsyncSessionLocal,
                            sender=None, target_user_id=None, synthesizer=None,
                            calendar_owner_telegram_id=None):
    """Explicit injected-sender compatibility harness; uses the R4.3 substrate.

    Production's scheduled entrypoint below never calls this. A missing sender
    fails closed rather than silently reverting to the pre-R4.3 direct send.
    """
    if sender is None:
        raise RuntimeError("InjectedTestSenderRequired")
    from types import SimpleNamespace
    from sqlalchemy import select
    from app.db.models import ScheduledOccurrence
    from app.scheduler.proactive import ProactiveRuntime, upsert_daily

    settings = get_settings()
    local = datetime.now(ZoneInfo(settings.default_timezone))
    keys = await upsert_daily(session_factory, bot_identity=1,
                              timezone_name=settings.default_timezone,
                              hour=local.hour, local_date=local.date(),
                              target_user_id=target_user_id)

    class InjectedBot:
        serial = 0

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            await sender(chat_id, text)
            self.serial += 1
            return SimpleNamespace(chat_id=chat_id, message_id=self.serial)

    runtime = ProactiveRuntime(
        session_factory, InjectedBot(), bot_identity=1,
        timezone_name=settings.default_timezone, daily_hour=local.hour,
        daily_renderer=lambda occurrence: render_daily_occurrence(
            occurrence, calendar=calendar, session_factory=session_factory,
            synthesizer=synthesizer,
            calendar_owner_telegram_id=calendar_owner_telegram_id),
        reminder_renderer=None)
    async with session_factory() as db:
        ids = (await db.scalars(select(ScheduledOccurrence.id).where(
            ScheduledOccurrence.occurrence_key.in_(keys)))).all()
    for occurrence_id in ids:
        await runtime.process(occurrence_id)


async def configured_daily_briefs():
    from app.scheduler.reminders import get_proactive_runtime
    runtime = get_proactive_runtime()
    if runtime is None:
        raise RuntimeError("ProactiveRuntimeUnavailable")
    await runtime.tick()


def register_daily_brief(scheduler) -> None:
    settings = get_settings()
    if scheduler.get_job("daily_brief") is None:
        scheduler.add_job(
            configured_daily_briefs,
            CronTrigger(hour=settings.daily_brief_hour, timezone=ZoneInfo(settings.default_timezone)),
            id="daily_brief",
            replace_existing=True,
            max_instances=1,
        )
