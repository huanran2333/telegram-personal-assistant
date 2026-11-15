"""Idempotent mutation executor; this is the only calendar write boundary."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.agent.actions import ActionProposal, ActionResult, ActionStatus
from app.agent.calendar import CalendarAdapter
from app.db import crud
from app.db.models import ActionExecution, Todo

logger = logging.getLogger(__name__)


class ActionLedger(Protocol):
    async def begin(self, proposal: ActionProposal) -> ActionResult | None: ...
    async def complete(self, proposal: ActionProposal, result: dict[str, Any]) -> None: ...
    async def fail(self, proposal: ActionProposal, message: str) -> None: ...


class SqlAlchemyActionLedger:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def begin(self, proposal: ActionProposal) -> ActionResult | None:
        async with self.session_factory() as db:
            existing = await db.get(ActionExecution, proposal.action_id)
            if existing is not None:
                return ActionResult(action_id=proposal.action_id, status=ActionStatus(existing.status),
                                    message="Action was already processed.", result=existing.result or {})
            db.add(ActionExecution(action_id=proposal.action_id, user_id=proposal.args["user_id"],
                                   domain=proposal.domain, operation=proposal.operation,
                                   status=ActionStatus.EXECUTING.value))
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is not None:
                    return ActionResult(action_id=proposal.action_id, status=ActionStatus(existing.status),
                                        message="Action was already processed.", result=existing.result or {})
                raise
        return None

    async def complete(self, proposal: ActionProposal, result: dict[str, Any]) -> None:
        async with self.session_factory() as db:
            row = await db.get(ActionExecution, proposal.action_id)
            if row is None:
                raise RuntimeError("Execution ledger row disappeared")
            row.status = ActionStatus.EXECUTED.value
            row.result = result
            row.external_id = str(result.get("id") or result.get("eventId") or "") or None
            row.executed_at = datetime.now(timezone.utc)
            await db.commit()

    async def fail(self, proposal: ActionProposal, message: str) -> None:
        async with self.session_factory() as db:
            row = await db.get(ActionExecution, proposal.action_id)
            if row is not None:
                row.status = ActionStatus.FAILED.value
                row.result = {"error": message}
                await db.commit()

    async def create_todo_atomically(self, proposal: ActionProposal) -> ActionResult:
        """Commit the local Todo and its child business proof in one PG transaction.

        The action ID is the concurrency key. A loser of the unique insert reads
        the winner's terminal row; there is no committed EXECUTING-only window.
        """
        if proposal.domain != "todo" or proposal.operation not in {"create_todo", "create_todo_from_email"}:
            raise ValueError("Expected a Todo create")
        try:
            async with self.session_factory() as db:
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is not None:
                    return ActionResult(action_id=proposal.action_id,
                                        status=ActionStatus(existing.status),
                                        message="Action was already processed.",
                                        result=existing.result or {})
                row = ActionExecution(action_id=proposal.action_id,
                                      user_id=proposal.args["user_id"],
                                      domain="todo", operation=proposal.operation,
                                      status=ActionStatus.EXECUTING.value)
                db.add(row)
                await db.flush()  # UNIQUE(action_id) serializes competing workers.
                due_date = proposal.args.get("due_date")
                reminder_at = proposal.args.get("reminder_at")
                if isinstance(due_date, str):
                    due_date = datetime.fromisoformat(due_date)
                if isinstance(reminder_at, str):
                    reminder_at = datetime.fromisoformat(reminder_at)
                todo = await crud.create_todo(
                    db, user_id=proposal.args["user_id"], title=proposal.args["title"],
                    due_date=due_date, reminder_at=reminder_at,
                    description=proposal.args.get("description"))
                result = {"id": todo.id, "title": todo.title}
                if proposal.args.get("source_message_id"):
                    result.update(source_message_id=proposal.args["source_message_id"],
                                  source_thread_id=proposal.args.get("source_thread_id"))
                row.status = ActionStatus.EXECUTED.value
                row.result = result
                row.external_id = todo.id
                row.executed_at = datetime.now(timezone.utc)
                await db.commit()
                return ActionResult(action_id=proposal.action_id,
                                    status=ActionStatus.EXECUTED,
                                    message="Action completed.", result=result)
        except IntegrityError:
            async with self.session_factory() as db:
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is None:
                    raise
                return ActionResult(action_id=proposal.action_id,
                                    status=ActionStatus(existing.status),
                                    message="Action was already processed.",
                                    result=existing.result or {})

    async def delete_todo_atomically(self, proposal: ActionProposal) -> ActionResult:
        """One commit for approved local deletion and its ActionExecution proof."""
        if proposal.domain != "todo" or proposal.operation != "delete_todo":
            raise ValueError("Expected exact Todo deletion")
        try:
            async with self.session_factory() as db:
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is not None:
                    return ActionResult(action_id=proposal.action_id,
                                        status=ActionStatus(existing.status),
                                        message="Action was already processed.", result=existing.result or {})
                row = ActionExecution(action_id=proposal.action_id,
                                      user_id=proposal.args["user_id"], domain="todo",
                                      operation="delete_todo", status=ActionStatus.EXECUTING.value)
                db.add(row)
                await db.flush()
                todo = await db.scalar(select(Todo).where(
                    Todo.id == proposal.args["todo_id"],
                    Todo.user_id == proposal.args["user_id"]).with_for_update())
                approved_version = proposal.args.get("approved_todo_version")
                if todo is None or (approved_version is not None and todo.version != approved_version):
                    row.status = ActionStatus.FAILED.value
                    row.result = {"failure_category": "todo_target_changed"}
                    await db.commit()
                    return ActionResult(action_id=proposal.action_id,
                                        status=ActionStatus.FAILED,
                                        message="Todo changed since approval; request a new proposal.",
                                        result=row.result)
                result = {"id": todo.id, "title": todo.title}
                await db.delete(todo)
                row.status = ActionStatus.EXECUTED.value
                row.result = result
                row.external_id = todo.id
                row.executed_at = datetime.now(timezone.utc)
                await db.commit()
                return ActionResult(action_id=proposal.action_id,
                                    status=ActionStatus.EXECUTED,
                                    message="Action completed.", result=result)
        except IntegrityError:
            async with self.session_factory() as db:
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is None:
                    raise
                return ActionResult(action_id=proposal.action_id,
                                    status=ActionStatus(existing.status),
                                    message="Action was already processed.", result=existing.result or {})


    async def delete_todos_by_tag_atomically(self, proposal: ActionProposal) -> ActionResult:
        """Commit the approved exact ID set and its ledger proof together."""
        if proposal.domain != "todo" or proposal.operation != "delete_todos_by_tag":
            raise ValueError("Expected tagged Todo deletion")
        ids = proposal.args["todo_ids"]
        tag = proposal.args["tag"]
        approved_versions = proposal.args.get("approved_todo_versions") or {}
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Bulk Todo approval contained no unique task IDs")
        try:
            async with self.session_factory() as db:
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is not None:
                    return ActionResult(action_id=proposal.action_id,
                                        status=ActionStatus(existing.status),
                                        message="Action was already processed.",
                                        result=existing.result or {})
                row = ActionExecution(action_id=proposal.action_id,
                                      user_id=proposal.args["user_id"], domain="todo",
                                      operation="delete_todos_by_tag",
                                      status=ActionStatus.EXECUTING.value)
                db.add(row)
                await db.flush()  # UNIQUE(action_id) serializes competing workers.
                todos = (await db.scalars(select(Todo).where(
                    Todo.id.in_(ids), Todo.user_id == proposal.args["user_id"]
                ).order_by(Todo.id).with_for_update())).all()
                if len(todos) != len(ids) or any(
                    tag not in (todo.tags or []) or
                    (approved_versions.get(todo.id) is not None and
                     todo.version != approved_versions[todo.id])
                    for todo in todos
                ):
                    row.status = ActionStatus.FAILED.value
                    row.result = {"failure_category": "todo_target_changed"}
                    await db.commit()
                    return ActionResult(action_id=proposal.action_id,
                                        status=ActionStatus.FAILED,
                                        message="A tagged Todo changed since approval; request a new proposal",
                                        result=row.result)
                for todo in todos:
                    await db.delete(todo)
                result = {"count": len(todos), "ids": ids, "tag": tag}
                row.status = ActionStatus.EXECUTED.value
                row.result = result
                row.executed_at = datetime.now(timezone.utc)
                await db.commit()
                return ActionResult(action_id=proposal.action_id,
                                    status=ActionStatus.EXECUTED,
                                    message="Action completed.", result=result)
        except IntegrityError:
            async with self.session_factory() as db:
                existing = await db.get(ActionExecution, proposal.action_id)
                if existing is None:
                    raise
                return ActionResult(action_id=proposal.action_id,
                                    status=ActionStatus(existing.status),
                                    message="Action was already processed.",
                                    result=existing.result or {})


class ActionExecutor:
    def __init__(self, calendar: CalendarAdapter, ledger: ActionLedger, session_factory,
                 calendar_create_coordinator=None, calendar_modify_coordinator=None):
        self.calendar = calendar
        self.ledger = ledger
        self.session_factory = session_factory
        self.calendar_create_coordinator = calendar_create_coordinator
        self.calendar_modify_coordinator = calendar_modify_coordinator

    async def calendar_create_prepared(self, action_id: str) -> bool:
        if self.calendar_create_coordinator is None:
            return False
        evidence, _ = await self.calendar_create_coordinator.store.load_create(action_id)
        return evidence is not None

    async def calendar_modification_prepared(self, action_id: str) -> bool:
        if self.calendar_modify_coordinator is None:
            return False
        evidence, _ = await self.calendar_modify_coordinator.store.load_create(action_id)
        return evidence is not None

    async def execute_child_todo(self, proposal: ActionProposal) -> ActionResult:
        if isinstance(self.ledger, SqlAlchemyActionLedger):
            return await self.ledger.create_todo_atomically(proposal)
        return await self.execute(proposal)

    async def execute(self, proposal: ActionProposal) -> ActionResult:
        if (proposal.domain == "todo" and proposal.operation in
                {"create_todo", "create_todo_from_email"} and
                isinstance(self.ledger, SqlAlchemyActionLedger)):
            return await self.ledger.create_todo_atomically(proposal)
        if (proposal.domain == "todo" and proposal.operation == "delete_todo" and
                isinstance(self.ledger, SqlAlchemyActionLedger)):
            return await self.ledger.delete_todo_atomically(proposal)
        if (proposal.domain == "todo" and proposal.operation == "delete_todos_by_tag" and
                isinstance(self.ledger, SqlAlchemyActionLedger)):
            return await self.ledger.delete_todos_by_tag_atomically(proposal)
        if (proposal.domain == "calendar" and proposal.operation == "create_event" and
                self.calendar_create_coordinator is not None):
            # Production create retains the sealed R5.2 coordinator.
            return await self.calendar_create_coordinator.execute(proposal)
        if (proposal.domain == "calendar" and proposal.operation in
                {"update_event", "delete_event"} and self.calendar_modify_coordinator is not None):
            return await self.calendar_modify_coordinator.execute(proposal)
        already_processed = await self.ledger.begin(proposal)
        if already_processed is not None:
            logger.info("action_id=%s duplicate execution prevented", proposal.action_id)
            return already_processed
        try:
            if proposal.domain == "calendar":
                operation = getattr(self.calendar, proposal.operation, None)
                if operation is None:
                    raise ValueError(f"Unsupported calendar operation {proposal.operation}")
                result = await operation(proposal.args)
            elif proposal.domain == "todo" and proposal.operation in {"create_todo", "create_todo_from_email"}:
                async with self.session_factory() as db:
                    due_date = proposal.args.get("due_date")
                    reminder_at = proposal.args.get("reminder_at")
                    if isinstance(due_date, str):
                        due_date = datetime.fromisoformat(due_date)
                    if isinstance(reminder_at, str):
                        reminder_at = datetime.fromisoformat(reminder_at)
                    todo = await crud.create_todo(
                        db, user_id=proposal.args["user_id"], title=proposal.args["title"],
                        due_date=due_date, reminder_at=reminder_at,
                        description=proposal.args.get("description"),
                    )
                    await db.commit()
                    result = {"id": todo.id, "title": todo.title}
            elif proposal.domain == "todo" and proposal.operation == "delete_todo":
                async with self.session_factory() as db:
                    deleted = await crud.delete_todo(db, proposal.args["todo_id"], proposal.args["user_id"])
                    if not deleted:
                        raise ValueError("Todo no longer exists or belongs to another user")
                    await db.commit()
                    result = {"id": proposal.args["todo_id"], "title": proposal.args["title"]}
            elif proposal.domain == "todo" and proposal.operation == "delete_todos_by_tag":
                ids = proposal.args["todo_ids"]
                tag = proposal.args["tag"]
                if not ids or len(ids) != len(set(ids)):
                    raise ValueError("Bulk Todo approval contained no unique task IDs")
                async with self.session_factory() as db:
                    todos = []
                    for todo_id in ids:
                        todo = await crud.get_todo(db, todo_id, proposal.args["user_id"])
                        if todo is None or tag not in (todo.tags or []):
                            raise ValueError("A tagged Todo changed since approval; request a new proposal")
                        todos.append(todo)
                    for todo in todos:
                        await db.delete(todo)
                    await db.commit()
                    result = {"count": len(todos), "ids": ids, "tag": tag}
            else:
                raise ValueError(f"Unsupported action {proposal.domain}.{proposal.operation}")
            if proposal.args.get("source_message_id"):
                result = {**result, "source_message_id": proposal.args["source_message_id"],
                          "source_thread_id": proposal.args.get("source_thread_id")}
            await self.ledger.complete(proposal, result)
            logger.info("action_id=%s executed %s.%s", proposal.action_id, proposal.domain, proposal.operation)
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.EXECUTED, message="Action completed.", result=result)
        except Exception as exc:
            await self.ledger.fail(proposal, str(exc))
            logger.error("action_id=%s execution failed category=%s", proposal.action_id,
                         type(exc).__name__)
            return ActionResult(action_id=proposal.action_id, status=ActionStatus.FAILED, message=f"Action failed: {exc}")
