"""Prepare one synthetic Web login in the explicitly verified isolated PG18.

This is not a production migration or a real-user binding. The caller supplies
the temporary password through its process environment; no credential is logged.
"""

import asyncio
import os
from pathlib import Path

from sqlalchemy import text, select

from app.db import crud
from app.db.isolated_test_target import (validate_isolated_pg18_configuration,
                                         validate_isolated_pg18_target)
from app.db.models import User, WebCredential
from app.db.session import AsyncSessionLocal, async_engine, init_db
from app.config import get_settings
from app.web.auth import bind_existing_user


async def main() -> None:
    expected = Path(os.environ["PPA_ISOLATED_PG18_DATA_DIR"]).resolve()
    settings = get_settings()
    repository_root = Path(__file__).resolve().parents[2]
    validate_isolated_pg18_configuration(
        settings.database_url, settings.database_url_sync, str(expected),
        repository_root=repository_root)
    async with async_engine.connect() as conn:
        version = await conn.scalar(text("SHOW server_version"))
        directory = await conn.scalar(text("SHOW data_directory"))
        database = await conn.scalar(text("SELECT current_database()"))
        port = await conn.scalar(text("SHOW port"))
    validate_isolated_pg18_target(
        settings.database_url, settings.database_url_sync, str(expected),
        directory, database, port, version, repository_root=repository_root)
    password = os.environ["V32_E2E_PASSWORD"]
    tag = os.environ["V32_E2E_TAG"]
    if not tag.isascii() or not tag.isalnum() or len(tag) > 24:
        raise ValueError("V32_E2E_TAG must be a short synthetic identifier")
    synthetic_identity = f"v32-browser-synthetic-{tag}"
    await init_db()
    async with AsyncSessionLocal() as db:
        existing = await db.scalar(select(User).where(User.telegram_id == synthetic_identity))
        if existing is None:
            existing = await crud.get_or_create_user(db, synthetic_identity)
            await db.commit()
        credential = await db.get(WebCredential, existing.id)
        if credential is None:
            await bind_existing_user(db, existing.id, password)
        elif os.environ.get("V32_E2E_RESET_SYNTHETIC_PASSWORD") == "true":
            # Dedicated browser clusters reuse one synthetic identity so the
            # bounded personal-login credential set does not grow every run.
            # The prefix above and exact PG18 data-dir check are mandatory.
            await bind_existing_user(db, existing.id, password, reset=True)
    print("SYNTHETIC_WEB_FIXTURE_READY")
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
