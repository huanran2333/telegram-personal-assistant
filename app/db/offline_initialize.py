"""Operator-only schema setup with no FastAPI/Bot/provider runtime.

The caller must supply explicit database URLs and the exact PostgreSQL data
directory. This is not an application startup path and never starts workers.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db.session import async_engine, init_db


async def initialize(mode: str, expected_data_directory: Path,
                     allow_existing: bool = False) -> None:
    settings = get_settings()
    async_url = make_url(settings.database_url)
    sync_url = make_url(settings.database_url_sync)
    expected = expected_data_directory.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if (async_url.drivername != "postgresql+asyncpg"
            or sync_url.drivername != "postgresql"
            or async_url.host != "127.0.0.1"
            or sync_url.host != async_url.host
            or sync_url.port != async_url.port
            or sync_url.database != async_url.database
            or sync_url.username != async_url.username):
        raise RuntimeError("Offline database URL pair mismatch")
    if mode == "staging":
        allowed_root = (repository_root / ".local-tools").resolve()
        if (async_url.database != "personal_assistant_stage"
                or async_url.username != "pa_stage_app"
                or async_url.port in {None, 5432, 55432, 55632}
                or not expected.is_relative_to(allowed_root)):
            raise RuntimeError("Offline staging target rejected")
    elif mode == "production":
        if (async_url.database != "personal_assistant"
                or async_url.username != "personal_assistant_app"
                or async_url.port != 5432
                or expected != Path(r"C:\Program Files\PostgreSQL\18\data").resolve()):
            raise RuntimeError("Offline production target rejected")
    else:
        raise RuntimeError("Unknown offline target mode")

    # data_directory is intentionally hidden from the non-superuser app role.
    # Use a separately supplied, short-lived admin connection only to attest
    # server identity, never to run application migrations as superuser.
    admin_dsn = os.environ.get("PPA_OFFLINE_ADMIN_DATABASE_URL", "")
    if not admin_dsn:
        raise RuntimeError("Admin data-directory attestation is required")
    admin_url = make_url(admin_dsn)
    if (admin_url.drivername != "postgresql+asyncpg"
            or admin_url.host != async_url.host or admin_url.port != async_url.port
            or admin_url.database != async_url.database or admin_url.username != "postgres"):
        raise RuntimeError("Admin attestation endpoint mismatch")
    admin_engine = create_async_engine(admin_dsn, hide_parameters=True)
    try:
        async with admin_engine.connect() as admin_connection:
            actual = Path(await admin_connection.scalar(text("SHOW data_directory"))).resolve()
            admin_database = await admin_connection.scalar(text("SELECT current_database()"))
            admin_port = await admin_connection.scalar(text("SHOW port"))
    finally:
        await admin_engine.dispose()
    if (actual != expected or admin_database != async_url.database
            or int(admin_port) != async_url.port):
        raise RuntimeError("Admin-attested PostgreSQL directory mismatch")

    async with async_engine.connect() as connection:
        database = await connection.scalar(text("SELECT current_database()"))
        role = await connection.scalar(text("SELECT current_user"))
        port = await connection.scalar(text("SHOW port"))
        version = await connection.scalar(text("SHOW server_version"))
        table_count = await connection.scalar(text(
            "SELECT count(*) FROM pg_tables WHERE schemaname='public'"))
    if (database != async_url.database or role != async_url.username
            or int(port) != async_url.port or not str(version).startswith("18.")):
        raise RuntimeError("Connected PostgreSQL identity mismatch")
    if table_count and not allow_existing:
        raise RuntimeError("Offline initializer accepts an empty database only")

    # This is the current production migration sequence, without startup
    # retention, sender, scheduler, model, Calendar or Web workers.
    await init_db()
    async with AsyncPostgresStore.from_conn_string(settings.database_url_sync) as store:
        await store.setup()
    async with AsyncPostgresSaver.from_conn_string(settings.database_url_sync) as saver:
        await saver.setup()
    print("OFFLINE_SCHEMA_INITIALIZED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("staging", "production"), required=True)
    parser.add_argument("--expected-data-directory", type=Path, required=True)
    parser.add_argument("--allow-existing", action="store_true",
                        help="Explicit idempotent retry after inspecting existing schema")
    args = parser.parse_args()
    async def run() -> None:
        try:
            await initialize(args.mode, args.expected_data_directory, args.allow_existing)
        finally:
            await async_engine.dispose()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())


if __name__ == "__main__":
    main()
