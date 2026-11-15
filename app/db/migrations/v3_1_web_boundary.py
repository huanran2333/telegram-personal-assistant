"""Frozen additive Web custody/auth DDL; no Telegram or domain rows are rewritten."""

from sqlalchemy import text

REVISION = "20261008_v3_1_web_boundary"


def apply(connection) -> None:
    for ddl in (
        """CREATE TABLE IF NOT EXISTS web_credentials (
            user_id VARCHAR(36) PRIMARY KEY REFERENCES users(id),
            password_hash TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS web_sessions (
            token_hash VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL REFERENCES users(id),
            credential_version INTEGER NOT NULL,
            csrf_token VARCHAR(64) NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            revoked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        "CREATE INDEX IF NOT EXISTS ix_web_session_owner_expiry ON web_sessions (user_id, expires_at)",
        """CREATE TABLE IF NOT EXISTS web_login_nonces (
            token_hash VARCHAR(64) PRIMARY KEY, csrf_token VARCHAR(64) NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS web_login_throttles (
            client_hash VARCHAR(64) PRIMARY KEY,
            attempts INTEGER NOT NULL DEFAULT 0, blocked_until TIMESTAMPTZ,
            window_started_at TIMESTAMPTZ NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS web_requests (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL REFERENCES users(id),
            operation VARCHAR(32) NOT NULL,
            idempotency_key VARCHAR(96) NOT NULL,
            request_hash VARCHAR(64) NOT NULL,
            request_body JSON NOT NULL,
            thread_id VARCHAR(160) NOT NULL,
            turn_id VARCHAR(160) NOT NULL UNIQUE,
            route_payload JSON, graph_kind VARCHAR(32),
            state VARCHAR(24) NOT NULL DEFAULT 'ACCEPTED',
            result JSON, safe_error VARCHAR(80),
            lease_owner VARCHAR(80), fence_token VARCHAR(36), lease_until TIMESTAMPTZ,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_web_request_key UNIQUE (user_id, operation, idempotency_key),
            CONSTRAINT ck_web_request_state CHECK (state IN (
                'ACCEPTED','RUNNING','WAITING_APPROVAL','WAITING_EDIT',
                'WAITING_EXTERNAL','COMPLETED','FAILED')))""",
        "CREATE INDEX IF NOT EXISTS ix_web_request_recovery ON web_requests (state, lease_until)",
        "CREATE INDEX IF NOT EXISTS ix_web_requests_user_id ON web_requests (user_id)",
    ):
        connection.execute(text(ddl))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
