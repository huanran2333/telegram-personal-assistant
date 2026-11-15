"""Frozen additive V3.2 form idempotency and one-shot reminder definition DDL."""

from sqlalchemy import text

REVISION = "20261008_v3_2_workbench"


def apply(connection) -> None:
    for ddl in (
        "ALTER TABLE todos ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE todos ADD COLUMN IF NOT EXISTS reminder_version INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE todos ADD COLUMN IF NOT EXISTS reminder_state VARCHAR(12) NOT NULL DEFAULT 'NONE'",
        """UPDATE todos SET reminder_state = 'ACTIVE'
           WHERE reminder_state = 'NONE' AND reminder_at IS NOT NULL""",
        """ALTER TABLE scheduled_occurrences
           ADD COLUMN IF NOT EXISTS reminder_version_snapshot INTEGER""",
        """CREATE TABLE IF NOT EXISTS web_mutations (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL REFERENCES users(id),
            operation VARCHAR(48) NOT NULL,
            idempotency_key VARCHAR(96) NOT NULL,
            request_hash VARCHAR(64) NOT NULL,
            result JSON NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_web_mutation_key UNIQUE (user_id, operation, idempotency_key))""",
        "CREATE INDEX IF NOT EXISTS ix_web_mutations_user_id ON web_mutations (user_id)",
    ):
        connection.execute(text(ddl))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
