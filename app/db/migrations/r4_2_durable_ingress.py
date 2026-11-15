"""R4.2 PostgreSQL migration: receipts, thread admission and shared replies.

The project predates Alembic environment/revisions. This migration uses the
same SQLAlchemy model metadata as fresh installations and records one version
marker in the *same* startup transaction. It adds no scheduler occurrence.
"""
from sqlalchemy import text

from app.db.models import (Base, OutboundDelivery, OutboundDeliveryAttempt,
                           OutboundDeliveryChunk, TelegramBusinessOperation,
                           TelegramInboundReceipt, TelegramThreadLease)

REVISION = "20261006_r4_2_durable_ingress"
TABLES = (
    TelegramInboundReceipt.__table__, TelegramThreadLease.__table__,
    OutboundDelivery.__table__, OutboundDeliveryChunk.__table__,
    OutboundDeliveryAttempt.__table__, TelegramBusinessOperation.__table__,
)


def apply(sync_connection, *, include_legacy: bool = True) -> None:
    sync_connection.execute(text(
        "CREATE TABLE IF NOT EXISTS app_schema_migrations "
        "(revision VARCHAR(80) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
    # Preserve the historical revision for explicit upgrade tests. Current
    # startup passes False so a retired proof table is never recreated.
    tables = TABLES if include_legacy else tuple(
        table for table in TABLES if table is not TelegramBusinessOperation.__table__)
    Base.metadata.create_all(sync_connection, tables=tables, checkfirst=True)
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
