"""Explicit migration of old Telegram Todo proofs into ActionExecution.

Never run from application startup.  In particular, do not discard an active
Receipt's old proof just because its envelope or result cannot be resolved.
Run retire() only during an approved, stopped-application deployment. The
backfill and DROP run in one caller-owned transaction, never on startup.
"""
from __future__ import annotations

from sqlalchemy import select, text

from app.db.models import (ActionExecution, TelegramBusinessOperation,
                           TelegramInboundReceipt, User)
from app.telegram.durable import telegram_todo_action_id


def backfill(connection) -> dict[str, int]:
    copied = skipped_terminal = already_present = 0
    old = TelegramBusinessOperation.__table__.alias("old")
    receipt = TelegramInboundReceipt.__table__.alias("receipt")
    rows = connection.execute(select(
        old.c.receipt_id, old.c.operation_key, old.c.args_sha256,
        old.c.status.label("operation_status"), old.c.result.label("operation_result"),
        old.c.created_at, receipt.c.envelope, receipt.c.state.label("receipt_state")
    ).select_from(old.join(receipt, receipt.c.id == old.c.receipt_id))).mappings().all()
    for row in rows:
        # Redacted terminal payloads cannot be replayed, and contain no proof
        # that could safely be reconstructed in another ledger.
        if (row["operation_status"] != "done" or not row["args_sha256"] or
                not row["operation_result"] or "text" not in row["operation_result"]):
            if row["receipt_state"] not in {"PROCESSED", "DEAD"}:
                raise RuntimeError("ActiveTelegramProofCannotBeBackfilled")
            skipped_terminal += 1
            continue
        envelope = row["envelope"] or {}
        actor = ((envelope.get("callback_query") or envelope.get("message") or {}).get("from") or {})
        telegram_id = actor.get("id")
        user_id = (connection.scalar(select(User.id).where(
            User.telegram_id == str(telegram_id))) if telegram_id is not None else None)
        if user_id is None:
            if row["receipt_state"] not in {"PROCESSED", "DEAD"}:
                raise RuntimeError("ActiveTelegramProofOwnerUnknown")
            skipped_terminal += 1
            continue
        name, separator, call_id = row["operation_key"].partition(":")
        if not separator or not name or not call_id:
            if row["receipt_state"] not in {"PROCESSED", "DEAD"}:
                raise RuntimeError("ActiveTelegramProofCallUnknown")
            skipped_terminal += 1
            continue
        action_id = telegram_todo_action_id(row["receipt_id"], name, call_id)
        existing = connection.execute(select(ActionExecution.__table__).where(
            ActionExecution.action_id == action_id)).mappings().first()
        expected_result = {"args_sha256": row["args_sha256"],
                           "receipt_id": row["receipt_id"],
                           "text": row["operation_result"]["text"]}
        if existing is not None:
            if (existing["user_id"] != user_id or existing["domain"] != "todo" or
                    existing["operation"] != name or existing["status"] != "executed" or
                    existing["result"] != expected_result):
                raise RuntimeError("TelegramProofBackfillConflict")
            already_present += 1
            continue
        connection.execute(ActionExecution.__table__.insert().values(
            action_id=action_id, user_id=user_id, domain="todo", operation=name,
            status="executed", result=expected_result, created_at=row["created_at"],
            executed_at=row["created_at"]))
        copied += 1
    return {"copied": copied, "already_present": already_present,
            "skipped_terminal": skipped_terminal}


REVISION = "20261021_telegram_todo_execution_retired"


def retire(connection) -> dict[str, int | bool]:
    """Backfill then drop the legacy table atomically on an approved PG target.

    ACCESS EXCLUSIVE blocks a concurrent old writer. An unresolved active
    proof, a conflicting action, or a dependent object aborts the transaction.
    """
    if connection.dialect.name != "postgresql":
        raise RuntimeError("PostgreSQLRequiredForTelegramLedgerRetirement")
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS app_schema_migrations "
        "(revision VARCHAR(80) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
    exists = connection.scalar(text(
        "SELECT to_regclass('telegram_business_operations')")) is not None
    if not exists:
        connection.execute(text(
            "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
            "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
        return {"copied": 0, "already_present": 0, "skipped_terminal": 0,
                "dropped": False}
    connection.execute(text(
        "LOCK TABLE telegram_business_operations IN ACCESS EXCLUSIVE MODE"))
    counts = backfill(connection)
    connection.execute(text("DROP TABLE telegram_business_operations"))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
    return {**counts, "dropped": True}
