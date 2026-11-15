"""Short-lived public HTTPS acceptance backend; synthetic PG18 and zero real providers.

The operator must supply the exact Funnel hostname and isolated PG18 directory.
This entry point refuses to boot if a real Bot, scheduler, Calendar, Gmail,
Memory, or non-loopback model endpoint is configured. It is not a production
startup mode and must never be used with a personal business database.
"""

import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db.models import CalendarActionEvidence, User, WebRequest


async def verify_synthetic_boundary() -> None:
    settings = get_settings()
    expected = Path(os.environ["PPA_ISOLATED_PG18_DATA_DIR"]).resolve()
    public = urlsplit(settings.web_public_origin)
    if (os.environ.get("PPA_G4_SYNTHETIC_PUBLIC") != "YES"
            or expected.name != "g4_public_pg18_data" or not expected.is_dir()
            or public.scheme != "https" or not public.hostname
            or not public.hostname.endswith(".ts.net") or public.port is not None
            or public.path not in ("", "/") or public.query or public.fragment
            or not settings.web_enabled or len(settings.secret_key) < 32
            or settings.secret_key == "change_me_in_production"
            or settings.telegram_bot_token or settings.telegram_webhook_url
            or settings.scheduler_enabled or settings.google_calendar_enabled
            or settings.gmail_enabled or settings.memory_enabled
            or settings.llm_provider != "openai"
            or settings.openai_base_url.rstrip("/") != "http://127.0.0.1:1/v1"):
        raise RuntimeError("G4SyntheticBoundaryRejected")
    async_url = make_url(settings.database_url)
    sync_url = make_url(settings.database_url_sync)
    if (async_url.drivername != "postgresql+asyncpg"
            or sync_url.drivername != "postgresql"
            or any(url.host != "127.0.0.1" or url.port != 55588
                   or url.database != "assistant" or url.username != "postgres"
                   for url in (async_url, sync_url))):
        raise RuntimeError("G4SyntheticDatabaseRejected")
    engine = create_async_engine(async_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            version = await connection.scalar(text("SHOW server_version"))
            directory = Path(await connection.scalar(text("SHOW data_directory"))).resolve()
            database = await connection.scalar(text("SELECT current_database()"))
            users = sorted((await connection.execute(select(User.telegram_id))).scalars().all())
            evidence = await connection.scalar(select(CalendarActionEvidence.action_id).limit(1))
            pending = await connection.scalar(select(WebRequest.id).where(
                WebRequest.state.in_(("ACCEPTED", "RUNNING", "WAITING_APPROVAL",
                                      "WAITING_EDIT", "WAITING_EXTERNAL"))).limit(1))
        if (not str(version).startswith("18.") or directory != expected
                or database != "assistant"
                or users != ["v32-browser-synthetic-g4other",
                             "v32-browser-synthetic-g4public"]
                or evidence is not None or pending is not None):
            raise RuntimeError("G4SyntheticDataRejected")
    finally:
        await engine.dispose()


def main() -> None:
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_synthetic_boundary())
    # The TLS endpoint is a foreground Tailscale Funnel; never bind publicly.
    config = uvicorn.Config("app.main:app", host="127.0.0.1", port=8000,
                            proxy_headers=False, lifespan="on", log_level="warning")
    # Uvicorn's own asyncio.run() may select Proactor on Windows even after
    # setting the policy; psycopg's async Saver requires Selector explicitly.
    asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
