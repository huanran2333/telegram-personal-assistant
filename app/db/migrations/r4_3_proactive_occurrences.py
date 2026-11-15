"""Frozen R4.3 PostgreSQL schema for proactive occurrences.

Do not derive a historical revision from the evolving ORM model. The frozen
table is also created before the current R4.2 fresh-install path, whose
OutboundDelivery metadata already carries the later occurrence foreign key.
"""

from sqlalchemy import text

REVISION = "20261006_r4_3_proactive_occurrences"


def create_occurrence_table(sync_connection) -> None:
    """Create the immutable R4.3 occurrence table snapshot, idempotently."""
    sync_connection.execute(text("""
        CREATE TABLE IF NOT EXISTS scheduled_occurrences (
            id VARCHAR(36) PRIMARY KEY,
            occurrence_key VARCHAR(64) NOT NULL,
            kind VARCHAR(32) NOT NULL,
            user_id VARCHAR(36) NOT NULL REFERENCES users(id),
            bot_identity BIGINT NOT NULL,
            chat_id VARCHAR(64) NOT NULL,
            local_date VARCHAR(10),
            timezone_name VARCHAR(80) NOT NULL,
            planned_at TIMESTAMPTZ NOT NULL,
            todo_id VARCHAR(36),
            reminder_at_snapshot TIMESTAMPTZ,
            state VARCHAR(20) NOT NULL,
            attempts INTEGER NOT NULL,
            lease_owner VARCHAR(80),
            fence_token VARCHAR(36),
            lease_until TIMESTAMPTZ,
            next_attempt_at TIMESTAMPTZ,
            failure_category VARCHAR(80),
            created_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            CONSTRAINT uq_scheduled_occurrence_key UNIQUE (occurrence_key)
        )
    """))
    sync_connection.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_scheduled_occurrence_recovery
        ON scheduled_occurrences (state, next_attempt_at, lease_until)
    """))
    sync_connection.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_scheduled_occurrence_user
        ON scheduled_occurrences (user_id, kind, planned_at)
    """))


def apply(sync_connection) -> None:
    sync_connection.execute(text(
        "CREATE TABLE IF NOT EXISTS app_schema_migrations "
        "(revision VARCHAR(80) PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
    create_occurrence_table(sync_connection)
    sync_connection.execute(text(
        "ALTER TABLE outbound_deliveries ALTER COLUMN receipt_id DROP NOT NULL"))
    sync_connection.execute(text(
        "ALTER TABLE outbound_deliveries ADD COLUMN IF NOT EXISTS occurrence_id VARCHAR(36)"))
    sync_connection.execute(text("""
        DO $$ BEGIN
          IF NOT EXISTS (
              SELECT 1 FROM pg_constraint
              WHERE conrelid = 'outbound_deliveries'::regclass
                AND conname = 'fk_outbound_occurrence'
          ) THEN
            ALTER TABLE outbound_deliveries ADD CONSTRAINT fk_outbound_occurrence
              FOREIGN KEY (occurrence_id) REFERENCES scheduled_occurrences(id);
          END IF;
          IF NOT EXISTS (
              SELECT 1 FROM pg_constraint
              WHERE conrelid = 'outbound_deliveries'::regclass
                AND conname = 'ck_outbound_one_origin'
          ) THEN
            ALTER TABLE outbound_deliveries ADD CONSTRAINT ck_outbound_one_origin
              CHECK ((receipt_id IS NULL) <> (occurrence_id IS NULL));
          END IF;
        END $$
    """))
    sync_connection.execute(text("""
        DO $$ BEGIN
          IF NOT EXISTS (
              SELECT 1
                FROM pg_index i
                JOIN pg_class idx ON idx.oid = i.indexrelid
                JOIN pg_attribute a ON a.attrelid = i.indrelid
                                   AND a.attnum = i.indkey[0]
               WHERE i.indrelid = 'outbound_deliveries'::regclass
                 AND idx.relname = 'uq_outbound_occurrence'
                 AND i.indisunique AND i.indnkeyatts = 1
                 AND a.attname = 'occurrence_id'
          ) THEN
            -- A same-named index on another relation must fail visibly here,
            -- not make IF NOT EXISTS silently skip the uniqueness invariant.
            CREATE UNIQUE INDEX uq_outbound_occurrence
              ON outbound_deliveries (occurrence_id);
          END IF;
        END $$
    """))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
