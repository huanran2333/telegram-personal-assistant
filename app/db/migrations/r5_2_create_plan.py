"""Frozen additive R5.2 provider-create payload evidence."""

from sqlalchemy import text

REVISION = "20261007_r5_2_create_plan"


def apply(sync_connection) -> None:
    sync_connection.execute(text(
        "ALTER TABLE calendar_action_evidence ADD COLUMN IF NOT EXISTS immutable_body JSON"))
    sync_connection.execute(text(
        "ALTER TABLE calendar_action_evidence ADD COLUMN IF NOT EXISTS private_action_marker VARCHAR(128)"))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
