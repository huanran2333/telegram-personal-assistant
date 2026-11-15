import os
import asyncio
import sys
from pathlib import Path

from dotenv import dotenv_values

local_env = dotenv_values(".env") if Path(".env").is_file() else {}
for name, fallback in (
    ("LLM_PROVIDER", "openai"),
    ("LLM_MODEL", "openai-compatible-test-model"),
    ("OPENAI_API_KEY", "test-key"),
    ("DEFAULT_TIMEZONE", "UTC"),
):
    if not local_env.get(name):
        os.environ.setdefault(name, fallback)

import pytest_asyncio
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base
from app.config import get_settings
from app.db.isolated_test_target import (validate_isolated_pg18_configuration,
                                         validate_isolated_pg18_target)
from app.agent.provider_clients import (
    ProviderClientOwner, bind_process_owner, provider_client_scope,
)


@pytest.fixture(scope="session")
def event_loop_policy():
    # Windows ProactorEventLoop creates a socket pair even for pure unit tests;
    # the Selector policy is the project's supported isolated-test runtime.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.fixture
def proactive_test_runtime(monkeypatch):
    """Bind a fake Bot to the real occurrence/outbox pipeline, never direct-send."""
    from types import SimpleNamespace
    from app.scheduler import reminders
    from app.scheduler.proactive import ProactiveRuntime
    from app.config import get_settings

    def build(factory, sender):
        class Bot:
            serial = 0

            async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
                await sender(chat_id, text, parse_mode=parse_mode)
                self.serial += 1
                return SimpleNamespace(chat_id=chat_id, message_id=self.serial)

        async def render_reminder(row):
            return await reminders.render_reminder_occurrence(row, session_factory=factory)

        runtime = ProactiveRuntime(
            factory, Bot(), bot_identity=1,
            timezone_name=get_settings().default_timezone, daily_hour=8,
            daily_renderer=None, reminder_renderer=render_reminder)
        monkeypatch.setattr(reminders, "_proactive_runtime", runtime)
        return runtime

    return build


@pytest_asyncio.fixture(autouse=True)
async def real_provider_client_lifecycle(request):
    """Each real-provider test owns clients through its async teardown.

    pytest-asyncio may run fixture setup and test calls in distinct task
    contexts, so a ContextVar alone is not a reliable cross-task owner here.
    The process binding is test-local and removed before the next test loop.
    Application-lifespan tests own and close their own process client instead.
    """
    if request.node.get_closest_marker("provider_lifespan"):
        yield
        return
    real_markers = ("llm_integration", "calendar_integration",
                    "gmail_integration", "pgvector_integration")
    if any(request.node.get_closest_marker(name) for name in real_markers):
        owner = ProviderClientOwner()
        bind_process_owner(owner)
        try:
            yield
        finally:
            bind_process_owner(None)
            await owner.aclose()
        return
    async with provider_client_scope():
        yield


@pytest_asyncio.fixture
async def isolated_pg18_target():
    """One checked endpoint for all opt-in real PG18 tests.

    The URL chooses the current temporary port; an independent absolute data
    directory identity prevents accidentally using the system PG18 instance.
    """
    settings = get_settings()
    url = make_url(settings.database_url)
    expected = os.environ.get("PPA_ISOLATED_PG18_DATA_DIR")
    if not expected:
        pytest.fail("Set PPA_ISOLATED_PG18_DATA_DIR to the isolated cluster directory")
    expected_path = Path(expected).resolve()
    try:
        validate_isolated_pg18_configuration(
            settings.database_url, settings.database_url_sync, str(expected_path),
            repository_root=Path(__file__).resolve().parents[1],
        )
    except ValueError as exc:
        pytest.fail(str(exc))
    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            version = await connection.scalar(text("SHOW server_version"))
            actual = await connection.scalar(text("SHOW data_directory"))
            database = await connection.scalar(text("SELECT current_database()"))
            port = await connection.scalar(text("SHOW port"))
        try:
            validate_isolated_pg18_target(
                settings.database_url, settings.database_url_sync, str(expected_path),
                actual, database, port, version,
                repository_root=Path(__file__).resolve().parents[1],
            )
        except ValueError as exc:
            pytest.fail(str(exc))
        yield url
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()
