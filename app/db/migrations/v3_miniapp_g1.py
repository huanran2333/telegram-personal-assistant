"""Frozen additive Mini App launch/session DDL; V3.3 identities remain intact."""

from sqlalchemy import text

REVISION = "20261009_v3_miniapp_g1"


def apply(connection) -> None:
    for ddl in (
        "ALTER TABLE web_sessions ADD COLUMN IF NOT EXISTS auth_source VARCHAR(16) NOT NULL DEFAULT 'password'",
        "ALTER TABLE web_sessions ADD COLUMN IF NOT EXISTS telegram_bot_identity BIGINT",
        "ALTER TABLE web_sessions ADD COLUMN IF NOT EXISTS telegram_user_id VARCHAR(64)",
        """CREATE TABLE IF NOT EXISTS telegram_miniapp_launch_tickets (
            token_hash VARCHAR(64) PRIMARY KEY,
            bot_identity BIGINT NOT NULL,
            telegram_user_id VARCHAR(64) NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            consumed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        "CREATE INDEX IF NOT EXISTS ix_miniapp_ticket_expiry ON telegram_miniapp_launch_tickets (expires_at)",
    ):
        connection.execute(text(ddl))
    connection.execute(text("""DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint
                       WHERE conname = 'ck_web_session_auth_source'
                         AND conrelid = 'web_sessions'::regclass) THEN
            ALTER TABLE web_sessions ADD CONSTRAINT ck_web_session_auth_source
            CHECK ((auth_source = 'password' AND telegram_bot_identity IS NULL AND telegram_user_id IS NULL)
                OR (auth_source = 'telegram' AND telegram_bot_identity IS NOT NULL AND telegram_user_id IS NOT NULL));
        END IF;
    END $$"""))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
