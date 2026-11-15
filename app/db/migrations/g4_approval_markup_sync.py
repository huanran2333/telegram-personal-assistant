"""Additive R4 UI-projection custody; no approval/business state is copied."""
from sqlalchemy import text

REVISION = "20261011_g4_approval_markup_sync"


def apply(connection) -> None:
    connection.execute(text("""CREATE TABLE IF NOT EXISTS telegram_approval_markup_sync (
        delivery_id VARCHAR(36) PRIMARY KEY REFERENCES outbound_deliveries(id),
        action_id VARCHAR(64) NOT NULL,
        state VARCHAR(16) NOT NULL DEFAULT 'PENDING',
        attempts INTEGER NOT NULL DEFAULT 0,
        fence_token VARCHAR(36),
        lease_until TIMESTAMPTZ,
        next_attempt_at TIMESTAMPTZ,
        failure_category VARCHAR(80),
        completed_at TIMESTAMPTZ)"""))
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_telegram_approval_markup_sync_action_id "
                            "ON telegram_approval_markup_sync (action_id)"))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
