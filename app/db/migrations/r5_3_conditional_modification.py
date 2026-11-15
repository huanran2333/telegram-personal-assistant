"""Frozen additive evidence field for conditional Calendar modification."""

from sqlalchemy import text

REVISION = "20261007_r5_3_conditional_modification"


def apply(sync_connection) -> None:
    sync_connection.execute(text(
        "ALTER TABLE calendar_action_evidence ADD COLUMN IF NOT EXISTS confirmed_etag VARCHAR(256)"))
    sync_connection.execute(text(
        "INSERT INTO app_schema_migrations (revision) VALUES (:revision) "
        "ON CONFLICT (revision) DO NOTHING"), {"revision": REVISION})
