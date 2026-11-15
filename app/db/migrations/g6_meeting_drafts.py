"""Add owner-scoped meeting content drafts; migration is never run implicitly."""

from sqlalchemy import text

REVISION = "20261016_g6_meeting_drafts"


def apply(connection) -> None:
    connection.execute(text("""CREATE TABLE IF NOT EXISTS meeting_drafts (
        id VARCHAR(36) PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL REFERENCES users(id),
        turn_id VARCHAR(160) NOT NULL,
        event_id VARCHAR(256) NOT NULL,
        calendar_account VARCHAR(256) NOT NULL,
        calendar_id VARCHAR(256) NOT NULL,
        event_etag VARCHAR(256) NOT NULL,
        revision INTEGER NOT NULL DEFAULT 1,
        status VARCHAR(20) NOT NULL DEFAULT 'NEEDS_REVIEW',
        rendered TEXT NOT NULL,
        brief_payload JSON NOT NULL,
        evidence_payload JSON NOT NULL,
        revisions JSON NOT NULL DEFAULT '[]',
        last_decision JSON,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_meeting_draft_turn_event UNIQUE (user_id, turn_id, event_id),
        CONSTRAINT ck_meeting_draft_status
            CHECK (status IN ('NEEDS_REVIEW','ACCEPTED','CANCELED','STALE'))
    )"""))
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_meeting_draft_owner_updated "
                            "ON meeting_drafts (user_id, updated_at)"))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
