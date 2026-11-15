"""Add the source thread to R3 correction lineage for Web/Telegram continuity.

Existing null values mean the superseded source is in the correction's own
thread, preserving the pre-V3 R3 contract and its historical rows.
"""

from sqlalchemy import text

REVISION = "20261008_v3_1_cross_thread_lineage"


def apply(connection) -> None:
    connection.execute(text(
        "ALTER TABLE transcript_projection "
        "ADD COLUMN IF NOT EXISTS supersedes_thread_id VARCHAR(160)"))
    connection.execute(text(
        "ALTER TABLE transcript_suppressions "
        "ADD COLUMN IF NOT EXISTS supersedes_thread_id VARCHAR(160)"))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_transcript_supersedes_source "
        "ON transcript_projection (user_id, "
        "(COALESCE(supersedes_thread_id, thread_id)), supersedes_message_id)"))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_transcript_suppression_supersedes_source "
        "ON transcript_suppressions (user_id, "
        "(COALESCE(supersedes_thread_id, thread_id)), supersedes_message_id)"))
    connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
