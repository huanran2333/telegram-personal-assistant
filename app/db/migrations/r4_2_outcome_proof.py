"""Add R4.2 receipt business-outcome proof without replacing the original migration."""

from sqlalchemy import text

REVISION = "20261006_r4_2_business_outcome_proof"


def apply(sync_connection) -> None:
    sync_connection.execute(text(
        "ALTER TABLE telegram_inbound_receipts "
        "ADD COLUMN IF NOT EXISTS outcome_ref VARCHAR(256)"))
    sync_connection.execute(text(
        "ALTER TABLE telegram_inbound_receipts "
        "ADD COLUMN IF NOT EXISTS required_reply_count INTEGER"))
    sync_connection.execute(text(
        "ALTER TABLE telegram_inbound_receipts "
        "ADD COLUMN IF NOT EXISTS resume_from_checkpoint_id VARCHAR(128)"))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
