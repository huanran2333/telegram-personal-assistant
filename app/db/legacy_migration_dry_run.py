"""Copy only the user-approved V2 business subset; archive executable history.

Runs against an attested read-only historical work copy. The same transaction
is used for staging and production. A repeat verifies equality without writes.
"""
import argparse
import os
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json, Jsonb

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIR = (REPO.parent / ".local-postgres-55432-work-20261010T081649Z").resolve()
TARGETS = {
    "staging": ((REPO / ".local-tools/prod_migration_stage_pg18_data").resolve(),
                "55648", "personal_assistant_stage", "pa_stage_app"),
    "staging_fresh": ((REPO / ".local-tools/prod_migration_stage_pg18_data").resolve(),
                      "55648", "personal_assistant_stage_clean", "pa_stage_app"),
    "staging_conflict": ((REPO / ".local-tools/prod_migration_stage_pg18_data").resolve(),
                         "55648", "personal_assistant_stage_conflict", "pa_stage_app"),
    "production": (Path(r"C:\Program Files\PostgreSQL\18\data").resolve(),
                   "5432", "personal_assistant", "personal_assistant_app"),
}
OWNER_TABLES = (
    "users", "todos", "transcript_projection", "transcript_suppressions",
    "conversation_summaries", "structured_memories", "memory_suppressions",
    "memory_consolidation_cursors",
)
EXCLUDED_TABLES = (
    "checkpoints", "checkpoint_blobs", "checkpoint_writes", "action_executions",
    "calendar_action_evidence", "calendar_dependent_continuations",
    "telegram_inbound_receipts", "outbound_deliveries",
    "outbound_delivery_chunks", "outbound_delivery_attempts",
    "scheduled_occurrences", "daily_brief_deliveries", "web_requests",
    "web_sessions", "telegram_miniapp_launch_tickets",
    "transcript_embeddings", "structured_memory_embeddings",
)


def _columns(conn, table):
    return {row["column_name"]: row["data_type"] for row in conn.execute(
        "SELECT column_name,data_type FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=%s", (table,)).fetchall()}


def _count(conn, table):
    return conn.execute(sql.SQL("SELECT count(*) n FROM {}").format(
        sql.Identifier(table))).fetchone()["n"]


def _source_rows(source, table, ids):
    if table == "users":
        return source.execute("SELECT * FROM users WHERE id=ANY(%s) ORDER BY id",
                              (ids,)).fetchall()
    return source.execute(sql.SQL("SELECT * FROM {} WHERE user_id=ANY(%s)").format(
        sql.Identifier(table)), (ids,)).fetchall()


def _map_row(table, original):
    row = dict(original)
    if table == "todos":
        if row.get("reminder_at") is not None:
            raise RuntimeError("Active legacy reminder requires separate review")
        row.update(version=1, reminder_version=0, reminder_state="NONE")
    return row


def _copy_or_verify(source, target, table, rows, repeat):
    source_cols, target_cols = _columns(source, table), _columns(target, table)
    if not source_cols or not target_cols:
        raise RuntimeError(f"Migration table missing: {table}")
    common = [name for name in target_cols if name in source_cols]
    if table == "todos":
        common += ["version", "reminder_version", "reminder_state"]
    pk = [r["attname"] for r in target.execute(
        "SELECT a.attname FROM pg_index i JOIN pg_attribute a "
        "ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey) "
        "WHERE i.indrelid=%s::regclass AND i.indisprimary ORDER BY a.attnum",
        (table,)).fetchall()]
    if not pk:
        raise RuntimeError(f"No primary key: {table}")
    for original in rows:
        row = _map_row(table, original)
        if not repeat:
            values = []
            for name in common:
                value = row[name]
                if value is not None and target_cols[name] in {"json", "jsonb"}:
                    value = Jsonb(value) if target_cols[name] == "jsonb" else Json(value)
                values.append(value)
            target.execute(sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table),
                sql.SQL(",").join(map(sql.Identifier, common)),
                sql.SQL(",").join(sql.Placeholder() for _ in common)), values)
        if any(name not in row for name in pk):
            raise RuntimeError(f"Unverifiable primary key: {table}")
        actual = target.execute(sql.SQL("SELECT * FROM {} WHERE {}").format(
            sql.Identifier(table),
            sql.SQL(" AND ").join(sql.SQL("{}=%s").format(sql.Identifier(name))
                                    for name in pk)),
            [row[name] for name in pk]).fetchone()
        if actual is None or any(actual[name] != row[name] for name in common):
            raise RuntimeError(f"Source/target mismatch: {table}")
    if _count(target, table) != len(rows):
        raise RuntimeError(f"Unexpected target count: {table}")
    return len(rows)


def migrate(mode, source_dsn, target_dsn, admin_dsn):
    if mode not in TARGETS or not all((source_dsn, target_dsn, admin_dsn)):
        raise RuntimeError("Explicit mode and source/target/admin DSNs required")
    directory, port, database, role = TARGETS[mode]
    with psycopg.connect(source_dsn, row_factory=dict_row) as source, \
            psycopg.connect(admin_dsn, row_factory=dict_row) as admin, \
            psycopg.connect(target_dsn, row_factory=dict_row) as target:
        source.execute("SET TRANSACTION READ ONLY")
        admin.execute("SET TRANSACTION READ ONLY")
        src = source.execute(
            "SELECT current_database() db,current_setting('data_directory') dir,"
            "current_setting('port') port,current_setting('default_transaction_read_only') ro,"
            "current_setting('server_version') version").fetchone()
        dst = admin.execute(
            "SELECT current_database() db,current_setting('data_directory') dir,"
            "current_setting('port') port,current_setting('server_version') version").fetchone()
        app = target.execute(
            "SELECT current_database() db,current_user username,"
            "current_setting('port') port").fetchone()
        if (Path(src["dir"]).resolve() != SOURCE_DIR or src["db"] != "assistant"
                or src["port"] != "55632" or src["ro"] != "on"
                or not src["version"].startswith("18.")
                or Path(dst["dir"]).resolve() != directory or dst["db"] != database
                or dst["port"] != port or not dst["version"].startswith("18.")
                or app != {"db": database, "username": role, "port": port}):
            raise RuntimeError("Source or target database identity mismatch")
        vector = target.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        if vector is None or vector["extversion"] != "0.8.6":
            raise RuntimeError("pgvector mismatch")
        owners = source.execute(
            "SELECT * FROM users WHERE telegram_id ~ '^[0-9]+$' ORDER BY id"
        ).fetchall()
        if len(owners) != 2 or len({r["telegram_id"] for r in owners}) != 2:
            raise RuntimeError("Expected exactly two distinct confirmed owners")
        ids = [r["id"] for r in owners]
        profiles = source.execute(
            "SELECT * FROM store WHERE prefix=ANY(%s) AND key='user_profile'",
            ([f"profile.{value}" for value in ids],)).fetchall()
        if len(profiles) != 1:
            raise RuntimeError("Unexpected profile ownership")
        existing = _count(target, "users")
        if existing not in (0, 2):
            raise RuntimeError("Partial or unrelated target users")
        repeat = existing == 2
        for table in EXCLUDED_TABLES:
            if _columns(target, table) and _count(target, table):
                raise RuntimeError(f"Executable/derived rows present: {table}")
        counts = {}
        for table in OWNER_TABLES:
            counts[table] = _copy_or_verify(
                source, target, table, _source_rows(source, table, ids), repeat)
        counts["store"] = _copy_or_verify(source, target, "store", profiles, repeat)
        if counts["todos"] != 2 or counts["transcript_projection"] != 4:
            raise RuntimeError("Historical source count changed")
        for table in OWNER_TABLES[1:]:
            if _count(target, table) and target.execute(sql.SQL(
                    "SELECT count(*) n FROM {} t LEFT JOIN users u "
                    "ON u.id=t.user_id WHERE u.id IS NULL").format(
                    sql.Identifier(table))).fetchone()["n"]:
                raise RuntimeError(f"Lost owner: {table}")
        return {**counts, "repeat_verified": int(repeat)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(TARGETS), required=True)
    args = parser.parse_args()
    counts = migrate(
        args.mode, os.environ.get("PPA_LEGACY_SOURCE_DSN", ""),
        os.environ.get("PPA_LEGACY_TARGET_DSN", ""),
        os.environ.get("PPA_LEGACY_TARGET_ADMIN_DSN", ""))
    print("SAFE_BUSINESS_MIGRATION " + " ".join(
        f"{name}={count}" for name, count in counts.items()))
    print("ARCHIVED: Saver, approvals, R4, R5, derived embeddings")


if __name__ == "__main__":
    main()
