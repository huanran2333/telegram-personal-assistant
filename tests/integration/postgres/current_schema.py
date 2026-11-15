"""Create the current ORM test schema without retired replay-ledger tables.

Historical migration tests construct the legacy shape explicitly. Ordinary
integration fixtures must not resurrect it in public just because its ORM
mapping remains available for backfill compatibility.
"""
from app.db.models import Base, TelegramBusinessOperation


def create_current_tables(connection, **kwargs):
    return Base.metadata.create_all(
        connection,
        tables=[table for table in Base.metadata.sorted_tables
                if table is not TelegramBusinessOperation.__table__],
        **kwargs,
    )
