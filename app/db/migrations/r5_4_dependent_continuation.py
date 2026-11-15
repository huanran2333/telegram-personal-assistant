"""Frozen, additive locator for approved Calendar-dependent workflow continuation."""

from sqlalchemy import text

REVISION = "20261007_r5_4_dependent_continuation"


def apply(sync_connection) -> None:
    sync_connection.execute(text("""
        CREATE TABLE IF NOT EXISTS calendar_dependent_continuations (
            parent_action_id VARCHAR(64) PRIMARY KEY REFERENCES action_executions(action_id),
            id VARCHAR(36) NOT NULL UNIQUE,
            thread_id VARCHAR(160) NOT NULL,
            checkpoint_id VARCHAR(128),
            child_action_id VARCHAR(80) NOT NULL UNIQUE,
            state VARCHAR(24) NOT NULL DEFAULT 'WAITING',
            lease_owner VARCHAR(80),
            fence_token VARCHAR(36),
            lease_until TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMPTZ,
            CONSTRAINT ck_calendar_continuation_state
                CHECK (state IN ('WAITING','READY','COMPLETED','REVIEW_REQUIRED'))
        )
    """))
    sync_connection.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_calendar_continuation_recovery
        ON calendar_dependent_continuations (state, lease_until)
    """))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
