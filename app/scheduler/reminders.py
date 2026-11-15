"""
APScheduler-based reminder system.
Polls the database every minute for todos that are due and haven't been notified.
"""
import logging
import asyncio
from calendar import month_abbr
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.db.session import AsyncSessionLocal
from app.db.models import Todo, User
from app.config import get_settings
from app.localization import load_preferences, reset_preferences, tr, use_preferences
from app.scheduler.proactive import observe_detached_tick

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()
_proactive_runtime = None


def bind_proactive_runtime(runtime) -> None:
    global _proactive_runtime
    _proactive_runtime = runtime


def get_proactive_runtime():
    return _proactive_runtime


async def render_reminder_occurrence(occurrence, *, session_factory=AsyncSessionLocal) -> tuple[str, str]:
    """Render before the immutable shared outbox commit; never send inline."""
    async with session_factory() as db:
        todo = await db.get(Todo, occurrence.todo_id)
        user = await db.get(User, occurrence.user_id)
    if todo is None or user is None:
        raise RuntimeError("ReminderSourceUnavailable")
    preferences = await load_preferences(user.id)
    token = use_preferences(preferences)
    try:
        if todo.due_date:
            zone = ZoneInfo(occurrence.timezone_name)
            due = todo.due_date if todo.due_date.tzinfo else todo.due_date.replace(tzinfo=timezone.utc)
            local = due.astimezone(zone)
            due_str = tr("reminder_date", month=local.month,
                         month_name=month_abbr[local.month], day=local.day,
                         hour=local.hour, minute=local.minute, timezone=local.tzname())
        else:
            due_str = tr("no_deadline")
        text = (f"⏰ {tr('reminder_heading')}\n\n📌 {todo.title}\n"
                f"📅 {tr('due_label')} {due_str}\n"
                f"{tr('priority_label')} {tr('priority_' + todo.priority.value)}")
        if todo.description:
            text += f"\n📝 {todo.description}"
        return text, preferences.language
    finally:
        reset_preferences(token)


async def check_and_send_reminders():
    """APScheduler wakeup; durable occurrence and dispatcher own all work."""
    runtime = get_proactive_runtime()
    if runtime is None:
        raise RuntimeError("ProactiveRuntimeUnavailable")
    # APScheduler cancels running coroutine jobs on shutdown even with
    # shutdown(wait=True). The runtime owns the active tick and drains it.
    work = asyncio.create_task(runtime.tick())
    work.add_done_callback(observe_detached_tick)
    await asyncio.shield(work)


def start_scheduler():
    """Start the APScheduler background scheduler."""
    if scheduler.get_job("reminder_checker") is None:
        scheduler.add_job(
            check_and_send_reminders,
            trigger=IntervalTrigger(minutes=1),
            id="reminder_checker",
            replace_existing=True,
            max_instances=1,
        )
    if not scheduler.running:
        scheduler.start()
        logger.info("Reminder scheduler started.")


def stop_scheduler():
    """Close APScheduler after owned active jobs have drained."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        # logger.info("Reminder scheduler stopped.")


def pause_scheduler():
    """Stop new wakeups without cancelling an already executing job."""
    if scheduler.running:
        scheduler.pause()
