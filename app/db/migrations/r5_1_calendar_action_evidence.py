"""Frozen R5.1 action-evidence schema; no Calendar executor switch in this revision."""

from sqlalchemy import text

REVISION = "20261007_r5_1_calendar_action_evidence"


def apply(sync_connection) -> None:
    sync_connection.execute(text("""
        CREATE TABLE IF NOT EXISTS calendar_action_evidence (
            action_id VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL REFERENCES users(id),
            operation VARCHAR(32) NOT NULL,
            account_identity VARCHAR(128) NOT NULL,
            calendar_id VARCHAR(256) NOT NULL,
            event_id VARCHAR(1024) NOT NULL,
            approved_args_fingerprint VARCHAR(64) NOT NULL,
            approved_baseline_etag VARCHAR(256),
            approved_baseline JSON,
            desired_postcondition JSON NOT NULL,
            outcome VARCHAR(32) NOT NULL DEFAULT 'NOT_STARTED',
            owner VARCHAR(80),
            fence_token VARCHAR(36),
            lease_until TIMESTAMPTZ,
            attempt_no INTEGER NOT NULL DEFAULT 0,
            dispatch_started_at TIMESTAMPTZ,
            failure_category VARCHAR(80),
            confirmed_external_id VARCHAR(1024),
            confirmation_source VARCHAR(40),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT ck_calendar_evidence_operation CHECK
                (operation IN ('create_event','update_event','delete_event')),
            CONSTRAINT ck_calendar_evidence_outcome CHECK
                (outcome IN ('NOT_STARTED','IN_FLIGHT','CONFIRMED',
                             'FAILED_DEFINITE','UNCERTAIN','PARKED_REVIEW')),
            CONSTRAINT ck_calendar_evidence_approved_etag CHECK
                (operation = 'create_event' OR approved_baseline_etag IS NOT NULL)
        )
    """))
    sync_connection.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_calendar_evidence_recovery
        ON calendar_action_evidence (outcome, lease_until)
    """))
    sync_connection.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_calendar_action_evidence_user_id
        ON calendar_action_evidence (user_id)
    """))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
