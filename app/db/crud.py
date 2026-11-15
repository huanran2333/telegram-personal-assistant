import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, delete
from sqlalchemy.orm import selectinload
from app.db.models import DailyBriefDelivery, User, Todo, Priority, TodoStatus

logger = logging.getLogger(__name__)


class TodoVersionConflict(Exception):
    """The Todo changed after its version was shown to the caller."""

# ── Users ──────────────────────────────────────────────────────────────────

async def get_or_create_user(db: AsyncSession, telegram_id: str, username: str = None, first_name: str = None) -> User:
    query = select(User).where(User.telegram_id == telegram_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if user is None:
        user = User(telegram_id=telegram_id, username=username, first_name=first_name)
        db.add(user)
        await db.flush()
    return user

async def get_user_by_telegram_id(db: AsyncSession, telegram_id: str) -> Optional[User]:
    query = select(User).where(User.telegram_id == telegram_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    return user

# ── Todos ──────────────────────────────────────────────────────────────────

async def create_todo(
    db: AsyncSession,
    user_id: str,
    title: str,
    description: str = None,
    due_date: datetime = None,
    planned_start_at: datetime = None,
    planned_end_at: datetime = None,
    reminder_at: datetime = None,
    priority: str = "medium",
    tags: list = None,
) -> Todo:
    # Every newly created item with a notification time has a visible local
    # time block, regardless of whether the caller is Agent, Web, or Workflow.
    # The notification remains independently editable/cancellable.
    if planned_start_at is None and reminder_at is not None:
        planned_start_at = reminder_at
    if planned_start_at is not None:
        planned_end_at = planned_end_at or planned_start_at + timedelta(hours=1)
        if planned_start_at.tzinfo is None or planned_end_at.tzinfo is None or planned_end_at <= planned_start_at:
            raise ValueError("InvalidPlannedWindow")
    elif planned_end_at is not None:
        raise ValueError("PlannedStartRequired")
    todo = Todo(
        user_id=user_id,
        title=title,
        description=description,
        due_date=due_date,
        planned_start_at=planned_start_at,
        planned_end_at=planned_end_at,
        reminder_at=reminder_at,
        reminder_state="ACTIVE" if reminder_at is not None else "NONE",
        priority=Priority(priority),
        tags=tags or [],
    )
    db.add(todo)
    await db.flush()
    await db.refresh(todo)
    return todo

async def list_todos(
    db: AsyncSession,
    user_id: str,
    status: str = None,
    priority: str = None,
    tags: list = None,
    due_before: datetime = None,
    due_after: datetime = None,
    limit: int | None = 20,
    exclude_done: bool = False,
) -> List[Todo]:
    conditions = [Todo.user_id == user_id]

    if status:
        conditions.append(Todo.status == TodoStatus(status))
    elif exclude_done:
        conditions.append(Todo.status != TodoStatus.done)
    if priority:
        conditions.append(Todo.priority == Priority(priority))
    if due_before and due_after:
        conditions.append(or_(and_(Todo.due_date >= due_after, Todo.due_date <= due_before),
                              and_(Todo.planned_start_at >= due_after,
                                   Todo.planned_start_at <= due_before)))
    elif due_before:
        conditions.append(or_(Todo.due_date <= due_before, Todo.planned_start_at <= due_before))
    elif due_after:
        conditions.append(or_(Todo.due_date >= due_after, Todo.planned_start_at >= due_after))
    
    logger.debug("Listing todos filter_count=%s", len(conditions))

    query = select(Todo).where(and_(*conditions)).order_by(Todo.due_date.asc().nulls_last(), Todo.created_at.desc())
    if limit is not None and not tags:
        query = query.limit(limit)
    result = await db.execute(query)
    todos = result.scalars().all()
    logger.info("Query result in crud: %s todos", len(todos))

    # Filter by tags if needed
    if tags:
        todos = [t for t in todos if any(tag in (t.tags or []) for tag in tags)]

    # Tags live in a JSON list, so apply the requested limit only after the
    # tag predicate. Otherwise unrelated earlier rows hide valid matches.
    return todos[:limit] if limit is not None else todos

async def get_todo(db: AsyncSession, todo_id: str, user_id: str,
                   *, lock: bool = False) -> Optional[Todo]:
    query = select(Todo).where(and_(Todo.id == todo_id, Todo.user_id == user_id))
    if lock:
        # A model tool may have resolved the target earlier in this session.
        # Refresh it after waiting for a Web writer's row lock.
        query = query.with_for_update().execution_options(populate_existing=True)
    result = await db.execute(query)
    todo = result.scalar_one_or_none()
    return todo

async def update_todo(
    db: AsyncSession,
    todo_id: str,
    user_id: str,
    derive_plan_from_reminder: bool = True,
    expected_version: int | None = None,
    **kwargs,
) -> Optional[Todo]:
    todo = await get_todo(db, todo_id, user_id, lock=True)
    if todo is None:
        return None
    if expected_version is not None and todo.version != expected_version:
        raise TodoVersionConflict

    allowed_fields = {
        "title", "description", "due_date", "priority", "status", "tags",
        "planned_start_at", "planned_end_at",
        "reminder_at", "reminder_sent", "reminder_claimed_at", "reminder_sent_at",
    }
    for key, value in kwargs.items():
        if key in allowed_fields and (value is not None or key in {"planned_start_at", "planned_end_at", "reminder_at"}):
            if key == "priority":
                value = Priority(value)
            elif key == "status":
                value = TodoStatus(value)
            setattr(todo, key, value)

    if ("reminder_at" in kwargs and kwargs["reminder_at"] is not None and
            todo.planned_start_at is None and derive_plan_from_reminder):
        todo.planned_start_at = kwargs["reminder_at"]
        todo.planned_end_at = todo.planned_start_at + timedelta(hours=1)

    if "planned_start_at" in kwargs:
        if todo.planned_start_at is None:
            todo.planned_end_at = None
        elif "planned_end_at" not in kwargs or todo.planned_end_at is None:
            todo.planned_end_at = todo.planned_start_at + timedelta(hours=1)
    if todo.planned_start_at is None and todo.planned_end_at is not None:
        raise ValueError("PlannedStartRequired")
    if todo.planned_start_at is not None and (todo.planned_start_at.tzinfo is None or
            todo.planned_end_at is None or todo.planned_end_at.tzinfo is None or
            todo.planned_end_at <= todo.planned_start_at):
        raise ValueError("InvalidPlannedWindow")
    if "reminder_at" in kwargs:
        todo.reminder_state = "ACTIVE" if kwargs["reminder_at"] is not None else "CANCELLED"
        todo.reminder_version += 1
        todo.reminder_sent = False
        todo.reminder_sent_at = None
        todo.reminder_claimed_at = None
    if todo.status == TodoStatus.done and todo.reminder_state in {"ACTIVE", "PAUSED"}:
        todo.reminder_at = None
        todo.reminder_state = "CANCELLED"
        todo.reminder_version += 1
        todo.reminder_sent = False
        todo.reminder_sent_at = None
        todo.reminder_claimed_at = None
    todo.version += 1

    todo.updated_at = datetime.now()
    await db.flush()
    await db.refresh(todo)
    return todo


async def delete_todo(db: AsyncSession, todo_id: str, user_id: str) -> bool:
    todo = await get_todo(db, todo_id, user_id, lock=True)
    if todo is None:
        return False
    await db.delete(todo)
    await db.flush()
    return True

async def delete_todos_by_tag(db: AsyncSession, user_id: str, tag: str) -> int:
    query = select(Todo).where(Todo.user_id == user_id)
    result = await db.execute(query)
    todos = result.scalars().all()
    logger.debug("Query result: %s todos", len(todos))
    count = 0
    for todo in todos:
        if tag in (todo.tags or []):
            await db.delete(todo)
            count += 1
    await db.flush()
    return count

async def search_todos(db: AsyncSession, user_id: str, query: str) -> List[Todo]:
    q = f"%{query.lower()}%"
    query_stmt = select(Todo).where(
            and_(
                Todo.user_id == user_id,
                or_(
                    Todo.title.ilike(q),
                    Todo.description.ilike(q),
                )
            )
        ).limit(10)
    result = await db.execute(query_stmt)
    todos = result.scalars().all()
    logger.debug("Query result: %s todos", len(todos))
    return todos

async def get_due_reminders(db: AsyncSession, before: datetime) -> List[Todo]:
    """Legacy read helper; dispatch uses ``claim_due_reminders`` below."""
    query = select(Todo).where(
            and_(
                Todo.reminder_at <= before,
                Todo.reminder_state == "ACTIVE",
                Todo.reminder_sent_at.is_(None),
                Todo.status != TodoStatus.done,
            )
        )
    result = await db.execute(query)
    todos = result.scalars().all()
    logger.debug("Query result: %s todos", len(todos))
    return todos


async def claim_due_reminders(
    db: AsyncSession,
    before: datetime,
    claim_timeout: timedelta = timedelta(minutes=5),
    allowed_telegram_ids: frozenset[str] | None = None,
) -> List[Todo]:
    """Claim due reminders transactionally before calling Telegram.

    A claim is persisted before sending, so concurrent scheduler instances
    cannot normally send the same reminder.  Expired claims are reclaimable
    after a crashed worker; failed sends are explicitly released.
    """
    now = datetime.now(timezone.utc)
    stale_before = now - claim_timeout
    query = (
        select(Todo)
        .options(selectinload(Todo.user))
        .where(
            and_(
                Todo.reminder_at.is_not(None),
                Todo.reminder_state == "ACTIVE",
                Todo.reminder_at <= before,
                Todo.reminder_sent_at.is_(None),
                Todo.status != TodoStatus.done,
                or_(Todo.reminder_claimed_at.is_(None), Todo.reminder_claimed_at < stale_before),
            )
        )
        .with_for_update(skip_locked=True)
    )
    if allowed_telegram_ids is not None:
        query = query.join(User, Todo.user_id == User.id).where(
            User.telegram_id.in_(allowed_telegram_ids))
    result = await db.execute(query)
    todos = result.scalars().all()
    for todo in todos:
        todo.reminder_claimed_at = now
    await db.flush()
    return todos


async def mark_reminder_sent(db: AsyncSession, todo_id: str) -> bool:
    todo = await db.get(Todo, todo_id)
    if todo is None or todo.reminder_sent_at is not None:
        return False
    todo.reminder_sent_at = datetime.now(timezone.utc)
    todo.reminder_sent = True
    todo.reminder_claimed_at = None
    await db.flush()
    return True


async def release_reminder_claim(db: AsyncSession, todo_id: str) -> None:
    todo = await db.get(Todo, todo_id)
    if todo is not None and todo.reminder_sent_at is None:
        todo.reminder_claimed_at = None
        await db.flush()


async def claim_daily_brief(db: AsyncSession, user_id: str, brief_date: str) -> bool:
    """Atomically reserve one user/date brief; false means it was already sent/claimed."""
    row = await db.get(DailyBriefDelivery, {"user_id": user_id, "brief_date": brief_date})
    if row is not None:
        return False
    db.add(DailyBriefDelivery(user_id=user_id, brief_date=brief_date, claimed_at=datetime.now(timezone.utc)))
    try:
        await db.flush()
        return True
    except Exception:
        # A concurrent primary-key insert loses the race.  The caller rolls
        # back/commits its surrounding unit without sending another brief.
        return False


async def mark_daily_brief_sent(db: AsyncSession, user_id: str, brief_date: str) -> None:
    row = await db.get(DailyBriefDelivery, {"user_id": user_id, "brief_date": brief_date})
    if row is not None:
        row.sent_at = datetime.now(timezone.utc)
        await db.flush()


async def release_daily_brief_claim(db: AsyncSession, user_id: str, brief_date: str) -> None:
    row = await db.get(DailyBriefDelivery, {"user_id": user_id, "brief_date": brief_date})
    if row is not None and row.sent_at is None:
        await db.delete(row)
        await db.flush()
