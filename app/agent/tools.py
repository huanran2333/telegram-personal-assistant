"""Todo operations and compatibility tools bound to an async DB factory.

The production Agent calls the plain operations with an explicit tool-call ID.
``make_todo_tools`` keeps the older tool interface for its other callers.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import re
import dateparser
from langchain_core.tools import tool
from zoneinfo import ZoneInfo
from app.localization import tr
from app.db.models import TodoStatus
from app.agent.business_clock import business_now


class TodoResult(str):
    """Human-readable result with server-owned version evidence for the ToolMessage."""

    def __new__(cls, content: str, todos):
        value = super().__new__(cls, content)
        value.todo_versions = {todo.id: todo.version for todo in todos}
        return value


def _versioned(content: str, todos) -> TodoResult:
    return TodoResult(content, todos)


_CJK_ABSOLUTE_CLOCK = re.compile(
    r"^\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日?\s*"
    r"(早上|上午|中午|下午|晚上)?\s*(\d{1,2})[点时]"
    r"(?:(\d{1,2})分?)?\s*$"
)


def parse_cjk_absolute_clock(value: str, timezone_name: str) -> Optional[datetime]:
    """Bounded Chinese calendar-clock fallback; reject invalid or DST-unsafe times."""
    match = _CJK_ABSOLUTE_CLOCK.fullmatch(value)
    if match is None:
        return None
    year, month, day, day_part, hour, minute = match.groups()
    hour_number = int(hour)
    minute_number = int(minute or 0)
    if hour_number > 23 or minute_number > 59:
        return None
    if day_part:
        if not 1 <= hour_number <= 12:
            return None
        if day_part in {"早上", "上午"}:
            hour_number %= 12
        elif day_part == "中午":
            if hour_number not in {11, 12, 1}:
                return None
            hour_number = 12 if hour_number == 12 else (13 if hour_number == 1 else 11)
        else:
            hour_number = 12 if hour_number == 12 else hour_number + 12
    try:
        zone = ZoneInfo(timezone_name)
        result = datetime(int(year), int(month), int(day), hour_number, minute_number, tzinfo=zone)
    except (ValueError, KeyError):
        return None
    # Avoid inventing an instant in a DST gap or silently picking one side of
    # a repeated hour. The user must clarify such an ambiguous local time.
    if result.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != result.replace(tzinfo=None):
        return None
    if result.replace(fold=1).utcoffset() != result.utcoffset():
        return None
    return result


def _format_local(value: datetime, pattern: str) -> str:
    from app.config import get_settings
    zone = ZoneInfo(get_settings().default_timezone)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(zone).strftime(pattern)


def make_todo_operations(db_session_factory):
    """
    Return plain business callables; no nested LangChain tool invocation is needed.
    """
    def parse_datetime(value: str):
        """Parse user time in the configured default timezone as an aware value."""
        from app.config import get_settings
        settings = get_settings()
        try:
            exact = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if exact.tzinfo is not None:
                return exact
        except ValueError:
            pass
        parsed = dateparser.parse(
            value,
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": settings.default_timezone,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "RELATIVE_BASE": business_now(ZoneInfo(settings.default_timezone)),
            },
        )
        if parsed is not None:
            return parsed
        parsed = parse_cjk_absolute_clock(value, settings.default_timezone)
        if parsed is not None:
            return parsed
        # dateparser does not parse combinations such as 周四晚上8点. Reuse
        # the bounded Calendar date/clock normalization for this case.
        from app.agent.planner import RoutedRequest, _explicit_clock, resolve_day
        weekdays = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        day_ref = None
        weekday = None
        if "后天" in value:
            day_ref = "day_after_tomorrow"
        elif "明天" in value:
            day_ref = "tomorrow"
        elif "今天" in value or "今晚" in value or "今夜" in value:
            day_ref = "today"
        else:
            match = re.search(r"(?:周|星期)([一二三四五六日天])", value)
            if match:
                day_ref = "weekday"
                weekday = weekdays[match.group(1)]
        if day_ref is None:
            return None
        clock = _explicit_clock(value)
        if clock is None:
            colon = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", value)
            if colon:
                clock = {"hour": int(colon.group(1)), "minute": int(colon.group(2)),
                         "day_part": "evening" if "今晚" in value or "晚上" in value else None}
        route = RoutedRequest(intent="calendar_read", date_reference=day_ref, weekday=weekday)
        day = resolve_day(route, settings.default_timezone,
                          now=business_now(ZoneInfo(settings.default_timezone)))
        if clock is None:
            return day
        hour = clock["hour"]
        if hour < 12 and clock["day_part"] in {"afternoon", "evening"}:
            hour += 12
        return day.replace(hour=hour, minute=clock["minute"])

    async def create_todo(
        title: str,
        user_id: str,
        description: str = "",
        due_date_str: str = "",
        planned_start_str: str = "",
        planned_end_str: str = "",
        reminder_at_str: str = "",
        priority: str = "medium",
        tags: str = "",
        tool_call_id: str | None = None,
    ) -> str:
        """Create one item. planned_start_str/end_str are its calendar window (end defaults to one hour); due_date_str is a deadline and reminder_at_str is a separate notification."""
        from app.db import crud
        due_date = None
        if due_date_str:
            due_date = parse_datetime(due_date_str)
            if due_date is None:
                return tr("todo_bad_due", value=due_date_str)
            if not re.search(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|[点时]", due_date_str, re.IGNORECASE):
                due_date = due_date.replace(hour=23, minute=59, second=59, microsecond=0)
        reminder_at = None
        if reminder_at_str:
            reminder_at = parse_datetime(reminder_at_str)
            if reminder_at is None:
                return tr("todo_bad_reminder", value=reminder_at_str)
            from app.config import get_settings
            if reminder_at <= business_now(ZoneInfo(get_settings().default_timezone)):
                return tr("todo_past_reminder")

        planned_start = parse_datetime(planned_start_str) if planned_start_str else None
        planned_end = parse_datetime(planned_end_str) if planned_end_str else None
        if (planned_start and planned_end and planned_end <= planned_start and planned_end_str
                and not re.search(r"\d{4}|今天|明天|后天|周|星期|today|tomorrow", planned_end_str, re.I)):
            planned_end = planned_start.replace(hour=planned_end.hour, minute=planned_end.minute,
                                                second=0, microsecond=0)
        if planned_start_str and planned_start is None:
            return tr("todo_bad_due", value=planned_start_str)
        if planned_end_str and planned_end is None:
            return tr("todo_bad_due", value=planned_end_str)
        # Any explicitly timed personal item is visible on the local calendar.
        # A notification-only request still describes a scheduled item, while
        # a date-only deadline remains an unscheduled Todo.
        if planned_start is None and due_date_str and re.search(
                r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|[点时]", due_date_str, re.IGNORECASE):
            planned_start = due_date
        if planned_start is None and reminder_at is not None:
            planned_start = reminder_at
        if planned_end and (planned_start is None or planned_end <= planned_start):
            return tr("todo_bad_due", value=planned_end_str)

        tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else []

        async with db_session_factory() as db:
            from app.telegram.durable import commit_todo_tool_operation
            async def perform():
                todo = await crud.create_todo(
                    db, user_id=user_id, title=title, description=description or None,
                    due_date=due_date, planned_start_at=planned_start,
                    planned_end_at=planned_end, reminder_at=reminder_at,
                    priority=priority, tags=tag_list)
                await db.flush()
                due_str = _format_local(todo.due_date, "%Y-%m-%d %H:%M") if todo.due_date else tr("todo_no_due")
                reminder_str = _format_local(todo.reminder_at, "%Y-%m-%d %H:%M") if todo.reminder_at else tr("none")
                return _versioned(
                    tr("todo_created", id=todo.id[:8], title=todo.title,
                       priority=tr("priority_" + todo.priority.value), due=due_str,
                       reminder=reminder_str, tags=todo.tags) +
                    f" | version: {todo.version}", [todo])
            return await commit_todo_tool_operation(
                db, name="create_todo", tool_call_id=tool_call_id,
                args={"user_id": user_id, "title": title, "description": description,
                      "due_date": due_date, "reminder_at": reminder_at,
                      "planned_start_at": planned_start, "planned_end_at": planned_end,
                      "priority": priority, "tags": tag_list}, perform=perform)

    async def list_todos(
        user_id: str,
        status: str = "",
        priority: str = "",
        tags: str = "",
        period: str = "",
        due_date_str: str = "",
    ) -> str:
        """List todos for a user. period must be one of: 'today', 'tomorrow', 'week', 'all'. Use 'tomorrow' when user asks about tomorrow's tasks. Use 'today' for today. Use 'week' for next 7 days. Use 'all' to list everything."""
        from app.db import crud
        due_before = None
        due_after = None
        from app.config import get_settings
        now = business_now(ZoneInfo(get_settings().default_timezone))

        if due_date_str:
            parsed_date = parse_datetime(due_date_str)
            if parsed_date:
                due_after = parsed_date.replace(hour=0, minute=0, second=0)
                due_before = parsed_date.replace(hour=23, minute=59, second=59)
 
        elif period == "today":
            due_after = now.replace(hour=0, minute=0, second=0)
            due_before = now.replace(hour=23, minute=59, second=59)
        elif period == "tomorrow":
            tomorrow = now + timedelta(days=1)
            due_after = tomorrow.replace(hour=0, minute=0, second=0)
            due_before = tomorrow.replace(hour=23, minute=59, second=59)
        elif period == "week":
            due_after = now
            due_before = now + timedelta(days=7)

        tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else None

        async with db_session_factory() as db:
            todos = await crud.list_todos(
                db,
                user_id=user_id,
                status=status or None,
                priority=priority or None,
                tags=tag_list,
                due_before=due_before,
                due_after=due_after,
            )

        if not todos:
            return tr("todo_empty")

        lines = [tr("todo_heading")]
        for t in todos:
            due_str = _format_local(t.due_date, "%b %d %H:%M") if t.due_date else "—"
            tag_str = " ".join(f"#{tag}" for tag in (t.tags or []))
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "done": "✅"}.get(t.status.value, "•")
            priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(t.priority.value, "•")
            lines.append(tr("todo_line", icon=status_emoji, id=t.id[:8], title=t.title,
                            status=tr("status_" + t.status.value),
                            priority=tr("priority_" + t.priority.value), due=due_str, tags=tag_str) +
                         f" | version: {t.version}")

        return _versioned("\n".join(lines), todos)

    async def update_todo(
        user_id: str,
        todo_id: str,
        title: str = "",
        status: str = "",
        priority: str = "",
        due_date_str: str = "",
        planned_start_str: str = "",
        planned_end_str: str = "",
        tags: str = "",
        tool_call_id: str | None = None,
        expected_version: int | None = None,
    ) -> str:
        """Update or complete a todo by ID prefix or unique title. Provide the fields to change; status=done completes it."""
        from app.db import crud
        due_date = None
        if due_date_str:
            due_date = parse_datetime(due_date_str)

        planned_start = parse_datetime(planned_start_str) if planned_start_str else None
        planned_end = parse_datetime(planned_end_str) if planned_end_str else None
        if planned_start_str and planned_start is None:
            return tr("todo_bad_due", value=planned_start_str)
        if planned_end_str and planned_end is None:
            return tr("todo_bad_due", value=planned_end_str)
        if planned_end is not None and planned_start is not None and planned_end <= planned_start:
            return tr("todo_bad_due", value=planned_end_str)
        tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()] if tags else None

        async with db_session_factory() as db:
            from app.telegram.durable import commit_todo_tool_operation
            async def perform():
                todos = await crud.list_todos(db, user_id=user_id, limit=100)
                matched = [t for t in todos if t.id.startswith(todo_id)]
                if not matched:
                    matched = [t for t in todos if todo_id.lower() in t.title.lower()]
                if not matched:
                    return tr("todo_not_found", id=todo_id)
                if len(matched) > 1:
                    return tr("todo_many")
                current = await crud.get_todo(db, matched[0].id, user_id, lock=True)
                if current is None:
                    return tr("todo_not_found", id=todo_id)
                if expected_version is not None and current.version != expected_version:
                    return tr("todo_version_conflict")
                updates = {"title": title or None, "status": status or None,
                           "priority": priority or None, "due_date": due_date, "tags": tag_list}
                if planned_start is not None:
                    updates["planned_start_at"] = planned_start
                    updates["planned_end_at"] = planned_end
                elif planned_end is not None:
                    if (current.planned_start_at is None or
                            planned_end <= current.planned_start_at):
                        return tr("todo_bad_due", value=planned_end_str)
                    updates["planned_end_at"] = planned_end
                updated = await crud.update_todo(
                    db, todo_id=matched[0].id, user_id=user_id,
                    expected_version=expected_version, **updates)
                await db.flush()
                if updated is None:
                    return tr("todo_update_error", id=todo_id)
                return _versioned(
                    tr("todo_updated", id=updated.id[:8], title=updated.title,
                       status=tr("status_" + updated.status.value)) +
                    f" | version: {updated.version}", [updated])
            return await commit_todo_tool_operation(
                db, name="update_todo", tool_call_id=tool_call_id,
                args={"user_id": user_id, "todo_id": todo_id, "title": title,
                      "status": status, "priority": priority,
                      "due_date": due_date, "planned_start_at": planned_start,
                      "planned_end_at": planned_end, "tags": tag_list,
                      "expected_version": expected_version}, perform=perform)

    async def delete_todo(
        user_id: str,
        todo_id: str = "",
        tag: str = "",
        tool_call_id: str | None = None,
    ) -> str:
        """Delete a todo by ID prefix or unique title, or all todos with a specific tag."""
        from app.db import crud
        async with db_session_factory() as db:
            from app.telegram.durable import commit_todo_tool_operation

            async def perform():
                if tag:
                    tag_clean = tag.lstrip("#")
                    count = await crud.delete_todos_by_tag(db, user_id=user_id, tag=tag_clean)
                    return tr("todo_deleted_tag", count=count, tag=tag_clean)
                if todo_id:
                    todos = await crud.list_todos(db, user_id=user_id, limit=100)
                    matched = [t for t in todos if t.id.startswith(todo_id)]
                    if not matched:
                        matched = [t for t in todos if todo_id.lower() in t.title.lower()]
                    if not matched:
                        return tr("todo_not_found", id=todo_id)
                    if len(matched) > 1:
                        return tr("todo_many")
                    title = matched[0].title
                    ok = await crud.delete_todo(db, todo_id=matched[0].id, user_id=user_id)
                    return tr("todo_deleted", title=title) if ok else tr("todo_delete_failed")
                return tr("todo_delete_args")

            return await commit_todo_tool_operation(
                db, name="delete_todo", tool_call_id=tool_call_id,
                args={"user_id": user_id, "todo_id": todo_id, "tag": tag},
                perform=perform)

    async def search_todos(user_id: str, query: str) -> str:
        """Search todos by keyword in title or description."""
        from app.db import crud
        async with db_session_factory() as db:
            todos = await crud.search_todos(db, user_id=user_id, query=query)

        if not todos:
            return tr("todo_search_empty", query=query)

        lines = [tr("todo_search_heading", count=len(todos), query=query)]
        for t in todos:
            due_str = _format_local(t.due_date, "%b %d %H:%M") if t.due_date else "—"
            lines.append(tr("todo_search_line", id=t.id[:8], title=t.title,
                            status=tr("status_" + t.status.value), due=due_str) +
                         f" | version: {t.version}")
        return _versioned("\n".join(lines), todos)

    async def set_reminder(
        user_id: str,
        todo_id: str,
        remind_at_str: str,
        tool_call_id: str | None = None,
        expected_version: int | None = None,
    ) -> str:
        """Set or update a reminder for a todo. remind_at_str accepts natural language."""
        from app.db import crud
        remind_at = parse_datetime(remind_at_str)
        if not remind_at:
            return tr("todo_bad_reminder", value=remind_at_str)
        from app.config import get_settings
        if remind_at <= business_now(ZoneInfo(get_settings().default_timezone)):
            return tr("todo_past_reminder")

        async with db_session_factory() as db:
            from app.telegram.durable import commit_todo_tool_operation
            async def perform():
                todos = await crud.list_todos(db, user_id=user_id, limit=100)
                matched = [t for t in todos if t.id.startswith(todo_id) or t.id == todo_id]
                if not matched:
                    return tr("todo_not_found", id=todo_id)
                if len(matched) > 1:
                    return tr("todo_many")
                current = await crud.get_todo(db, matched[0].id, user_id, lock=True)
                if current is None:
                    return tr("todo_not_found", id=todo_id)
                if expected_version is not None and current.version != expected_version:
                    return tr("todo_version_conflict")
                if current.status == TodoStatus.done:
                    return tr("todo_completed_reminder")
                updated = await crud.update_todo(
                    db, todo_id=matched[0].id, user_id=user_id,
                    reminder_at=remind_at, reminder_sent=False,
                    reminder_sent_at=None, reminder_claimed_at=None,
                    derive_plan_from_reminder=False,
                    expected_version=expected_version)
                await db.flush()
                return _versioned(
                    f"[{updated.id[:8]}] " + tr("todo_reminder_set", title=updated.title,
                           time=_format_local(remind_at, "%Y-%m-%d %H:%M")) +
                    f" | version: {updated.version}", [updated])
            return await commit_todo_tool_operation(
                db, name="set_reminder", tool_call_id=tool_call_id,
                args={"user_id": user_id, "todo_id": todo_id, "remind_at": remind_at,
                      "expected_version": expected_version},
                perform=perform)

    async def cancel_reminder(user_id: str, todo_id: str,
                              tool_call_id: str | None = None,
                              expected_version: int | None = None) -> str:
        """Cancel the future reminder on a uniquely identified Todo; keep the Todo and its planned calendar window."""
        from app.db import crud
        async with db_session_factory() as db:
            from app.telegram.durable import commit_todo_tool_operation
            async def perform():
                todos = await crud.list_todos(db, user_id=user_id, limit=100)
                matched = [t for t in todos if t.id.startswith(todo_id) or todo_id.lower() in t.title.lower()]
                if not matched:
                    return tr("todo_not_found", id=todo_id)
                if len(matched) > 1:
                    return tr("todo_many")
                current = await crud.get_todo(db, matched[0].id, user_id, lock=True)
                if current is None:
                    return tr("todo_not_found", id=todo_id)
                if expected_version is not None and current.version != expected_version:
                    return tr("todo_version_conflict")
                updated = await crud.update_todo(
                    db, matched[0].id, user_id, reminder_at=None,
                    expected_version=expected_version)
                return _versioned(
                    f"[{updated.id[:8]}] " + tr("todo_reminder_cancelled", title=updated.title) +
                    f" | version: {updated.version}", [updated])
            return await commit_todo_tool_operation(
                db, name="cancel_reminder", tool_call_id=tool_call_id,
                args={"user_id": user_id, "todo_id": todo_id,
                      "expected_version": expected_version}, perform=perform)

    return {operation.__name__: operation for operation in
            (create_todo, list_todos, update_todo, delete_todo, search_todos,
             set_reminder, cancel_reminder)}


def make_todo_tools(db_session_factory):
    """Compatibility tool interface for callers outside the production Agent.

    The production Agent uses ``make_todo_operations`` directly. These tools
    retain their existing schema and forward the already-scoped call ID.
    """
    operations = make_todo_operations(db_session_factory)

    def call_id():
        from app.telegram.processing_context import current_tool_call_id
        return current_tool_call_id()

    @tool
    async def create_todo(title: str, user_id: str, description: str = "",
                          due_date_str: str = "", planned_start_str: str = "",
                          planned_end_str: str = "", reminder_at_str: str = "",
                          priority: str = "medium", tags: str = "") -> str:
        """Create one item. Planned time, deadline, and reminder are separate fields."""
        return await operations["create_todo"](
            title=title, user_id=user_id, description=description,
            due_date_str=due_date_str, planned_start_str=planned_start_str,
            planned_end_str=planned_end_str, reminder_at_str=reminder_at_str,
            priority=priority, tags=tags, tool_call_id=call_id())

    @tool
    async def list_todos(user_id: str, status: str = "", priority: str = "",
                         tags: str = "", period: str = "",
                         due_date_str: str = "") -> str:
        """List todos for a user, optionally filtered by status, priority, tags, or period."""
        return await operations["list_todos"](
            user_id=user_id, status=status, priority=priority, tags=tags,
            period=period, due_date_str=due_date_str)

    @tool
    async def update_todo(user_id: str, todo_id: str, title: str = "",
                          status: str = "", priority: str = "",
                          due_date_str: str = "", planned_start_str: str = "",
                          planned_end_str: str = "", tags: str = "") -> str:
        """Update or complete a todo by ID prefix or unique title."""
        return await operations["update_todo"](
            user_id=user_id, todo_id=todo_id, title=title, status=status,
            priority=priority, due_date_str=due_date_str,
            planned_start_str=planned_start_str, planned_end_str=planned_end_str,
            tags=tags, tool_call_id=call_id())

    @tool
    async def delete_todo(user_id: str, todo_id: str = "", tag: str = "") -> str:
        """Delete a todo by ID prefix or unique title, or all todos with a tag."""
        return await operations["delete_todo"](
            user_id=user_id, todo_id=todo_id, tag=tag, tool_call_id=call_id())

    @tool
    async def search_todos(user_id: str, query: str) -> str:
        """Search this user's todos by title or description."""
        return await operations["search_todos"](user_id=user_id, query=query)

    @tool
    async def set_reminder(user_id: str, todo_id: str,
                           remind_at_str: str) -> str:
        """Set or update a reminder for a todo using natural-language time."""
        return await operations["set_reminder"](
            user_id=user_id, todo_id=todo_id, remind_at_str=remind_at_str,
            tool_call_id=call_id())

    @tool
    async def cancel_reminder(user_id: str, todo_id: str) -> str:
        """Cancel a future reminder, preserving the Todo and planned window."""
        return await operations["cancel_reminder"](
            user_id=user_id, todo_id=todo_id, tool_call_id=call_id())

    return [create_todo, list_todos, update_todo, delete_todo, search_todos,
            set_reminder, cancel_reminder]

