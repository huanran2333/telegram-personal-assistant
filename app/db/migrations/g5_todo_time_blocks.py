"""Add a local planned window and a transactionally queued Google projection.

The preference defaults on, but this migration never backfills existing Todos.
If an earlier revision already created this column, its false values remain
unchanged because the old default cannot be distinguished from an opt-out.
"""

from sqlalchemy import text

REVISION = "20261014_g5_todo_time_blocks"


def apply(connection) -> None:
    for ddl in (
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS calendar_time_block_sync BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE users ALTER COLUMN calendar_time_block_sync SET DEFAULT TRUE",
        "ALTER TABLE todos ADD COLUMN IF NOT EXISTS planned_start_at TIMESTAMPTZ",
        "ALTER TABLE todos ADD COLUMN IF NOT EXISTS planned_end_at TIMESTAMPTZ",
        """ALTER TABLE todos ADD CONSTRAINT ck_todo_planned_window CHECK
           ((planned_start_at IS NULL AND planned_end_at IS NULL) OR
            (planned_start_at IS NOT NULL AND planned_end_at > planned_start_at))""",
        """CREATE TABLE IF NOT EXISTS todo_calendar_mirrors (
            todo_id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL REFERENCES users(id),
            calendar_id VARCHAR(256) NOT NULL DEFAULT 'primary',
            account_identity VARCHAR(256), event_id VARCHAR(256), etag VARCHAR(512),
            desired_revision INTEGER NOT NULL DEFAULT 1,
            synced_revision INTEGER NOT NULL DEFAULT 0,
            desired_operation VARCHAR(8) NOT NULL DEFAULT 'UPSERT',
            desired_payload JSON, status VARCHAR(24) NOT NULL DEFAULT 'PENDING',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TIMESTAMPTZ, lease_token VARCHAR(64), lease_until TIMESTAMPTZ,
            last_error_category VARCHAR(48),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_todo_calendar_remote_identity
                UNIQUE (account_identity, calendar_id, event_id))""",
        """CREATE INDEX IF NOT EXISTS ix_todo_calendar_mirror_recovery
           ON todo_calendar_mirrors (status, next_attempt_at, lease_until)""",
        """CREATE INDEX IF NOT EXISTS ix_todo_calendar_mirrors_user_id
           ON todo_calendar_mirrors (user_id)""",
    ):
        if ddl.startswith("ALTER TABLE todos ADD CONSTRAINT"):
            exists = connection.scalar(text("""SELECT 1 FROM pg_constraint
                WHERE conname = 'ck_todo_planned_window'
                AND conrelid = 'todos'::regclass"""))
            if exists:
                continue
        connection.execute(text(ddl))

    # A BEFORE trigger centralizes the one-hour default for every write path,
    # including the Agent, Web forms and approved typed actions.
    connection.execute(text("""CREATE OR REPLACE FUNCTION pa_todo_planned_window()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.planned_start_at IS NOT NULL AND NEW.planned_end_at IS NULL THEN
                NEW.planned_end_at := NEW.planned_start_at + INTERVAL '1 hour';
            END IF;
            RETURN NEW;
        END $$"""))
    connection.execute(text("DROP TRIGGER IF EXISTS pa_todo_planned_window ON todos"))
    connection.execute(text("""CREATE TRIGGER pa_todo_planned_window
        BEFORE INSERT OR UPDATE OF planned_start_at, planned_end_at ON todos
        FOR EACH ROW EXECUTE FUNCTION pa_todo_planned_window()"""))

    # Queue the desired state in the *same transaction* as the Todo. No
    # provider call runs in the trigger. A tombstone survives hard deletion.
    connection.execute(text("""CREATE OR REPLACE FUNCTION pa_queue_todo_calendar_mirror()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE active_sync BOOLEAN;
        DECLARE payload JSON;
        DECLARE desired VARCHAR(8);
        BEGIN
            IF TG_OP = 'UPDATE' AND
               NEW.title IS NOT DISTINCT FROM OLD.title AND
               NEW.planned_start_at IS NOT DISTINCT FROM OLD.planned_start_at AND
               NEW.planned_end_at IS NOT DISTINCT FROM OLD.planned_end_at AND
               NEW.status IS NOT DISTINCT FROM OLD.status THEN
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                SELECT calendar_time_block_sync INTO active_sync FROM users WHERE id = OLD.user_id;
            ELSE
                SELECT calendar_time_block_sync INTO active_sync FROM users WHERE id = NEW.user_id;
            END IF;
            IF NOT active_sync THEN
                IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'INSERT' AND NEW.planned_start_at IS NULL THEN
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                desired := 'DELETE';
                payload := NULL;
            ELSIF NEW.planned_start_at IS NULL THEN
                desired := 'DELETE';
                payload := NULL;
            ELSE
                desired := 'UPSERT';
                payload := json_build_object(
                    'summary', NEW.title,
                    'start', NEW.planned_start_at,
                    'end', NEW.planned_end_at,
                    'completed', NEW.status::text = 'done');
            END IF;
            IF desired = 'DELETE' THEN
                UPDATE todo_calendar_mirrors SET
                    desired_revision = desired_revision + 1,
                    desired_operation = desired, desired_payload = NULL,
                    status = CASE WHEN status = 'REVIEW' THEN 'REVIEW' ELSE 'PENDING' END,
                    next_attempt_at = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE todo_id = OLD.id;
            ELSE
                INSERT INTO todo_calendar_mirrors
                    (todo_id, user_id, desired_operation, desired_payload)
                VALUES (NEW.id, NEW.user_id, desired, payload)
                ON CONFLICT (todo_id) DO UPDATE SET
                    desired_revision = todo_calendar_mirrors.desired_revision + 1,
                    desired_operation = EXCLUDED.desired_operation,
                    desired_payload = EXCLUDED.desired_payload,
                    status = CASE WHEN todo_calendar_mirrors.status = 'REVIEW'
                                  THEN 'REVIEW' ELSE 'PENDING' END,
                    next_attempt_at = NULL, updated_at = CURRENT_TIMESTAMP;
            END IF;
            IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
            RETURN NEW;
        END $$"""))
    connection.execute(text("DROP TRIGGER IF EXISTS pa_queue_todo_calendar_mirror ON todos"))
    connection.execute(text("""CREATE TRIGGER pa_queue_todo_calendar_mirror
        AFTER INSERT OR UPDATE OR DELETE ON todos
        FOR EACH ROW EXECUTE FUNCTION pa_queue_todo_calendar_mirror()"""))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
