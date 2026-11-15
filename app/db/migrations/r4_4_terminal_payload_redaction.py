"""Frozen additive DDL for R4.4 terminal payload retirement.

The old R4.2/R4.3 revisions remain immutable. NULL means content has been
retired only when the corresponding payload_redacted_at marker is committed.
"""

from sqlalchemy import text

REVISION = "20261006_r4_4_terminal_payload_redaction"


def apply(sync_connection) -> None:
    for statement in (
        "ALTER TABLE telegram_inbound_receipts ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ",
        "ALTER TABLE telegram_inbound_receipts ADD COLUMN IF NOT EXISTS payload_redacted_at TIMESTAMPTZ",
        "ALTER TABLE outbound_deliveries ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ",
        "ALTER TABLE outbound_deliveries ADD COLUMN IF NOT EXISTS payload_redacted_at TIMESTAMPTZ",
        "ALTER TABLE telegram_inbound_receipts ALTER COLUMN envelope DROP NOT NULL",
        "ALTER TABLE telegram_inbound_receipts ALTER COLUMN envelope_sha256 DROP NOT NULL",
        "ALTER TABLE outbound_deliveries ALTER COLUMN text DROP NOT NULL",
        "ALTER TABLE outbound_deliveries ALTER COLUMN payload_sha256 DROP NOT NULL",
        "ALTER TABLE outbound_delivery_chunks ALTER COLUMN text DROP NOT NULL",
        "ALTER TABLE outbound_delivery_chunks ALTER COLUMN content_sha256 DROP NOT NULL",
        # Existing terminal rows lack a terminal transition timestamp. Use
        # known commit time where present; otherwise start the full retention
        # horizon at upgrade time rather than guessing from received_at.
        "UPDATE telegram_inbound_receipts SET terminal_at = COALESCE(processed_at, CURRENT_TIMESTAMP) "
        "WHERE state IN ('PROCESSED', 'DEAD') AND terminal_at IS NULL",
        "UPDATE outbound_deliveries SET terminal_at = COALESCE(sent_at, CURRENT_TIMESTAMP) "
        "WHERE state IN ('SENT', 'EXHAUSTED') AND terminal_at IS NULL",
    ):
        sync_connection.execute(text(statement))
    # Historical schemas still need the nullable payload upgrade; current
    # schemas have no legacy proof table to alter.
    if sync_connection.execute(text(
            "SELECT to_regclass('telegram_business_operations')")).scalar() is not None:
        sync_connection.execute(text(
            "ALTER TABLE telegram_business_operations ALTER COLUMN args_sha256 DROP NOT NULL"))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
