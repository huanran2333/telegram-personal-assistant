"""Proactive startup guard and isolated Fake Bot behavior, not live Telegram."""
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.config import Settings
from app.db.models import ScheduledOccurrence, User
from app.scheduler.proactive import ProactiveRuntime, upsert_daily


def test_proactive_settings_default_to_enabled_with_live_start_guard():
    assert Settings.model_fields["scheduler_enabled"].default is True
    assert Settings.model_fields["daily_rundown_enabled"].default is True
    assert Settings.model_fields["default_timezone"].default == "Asia/Shanghai"
    assert Settings.model_fields["telegram_live_start_preflight_ack"].default is False
    assert Settings.model_fields["memory_enabled"].default is True


def test_live_bot_ack_cannot_target_historical_or_test_postgres():
    with pytest.raises(ValueError, match="dedicated production PostgreSQL"):
        Settings(
            _env_file=None, llm_provider="openai", llm_model="synthetic",
            openai_api_key="synthetic-key", telegram_live_start_preflight_ack=True,
            app_env="production", secret_key="x" * 48,
            web_public_origin="https://synthetic.example",
            database_url="postgresql+asyncpg://role@127.0.0.1:55432/assistant",
            database_url_sync="postgresql://role@127.0.0.1:55432/assistant",
            memory_enabled=False,
        )


@pytest.mark.asyncio
async def test_tick_without_daily_opt_in_never_creates_proactive_intent(session_factory):
    async with session_factory() as db:
        db.add_all([User(telegram_id="synthetic-852"),
                    User(telegram_id="synthetic-529")])
        await db.commit()

    class FakeBot:
        def __init__(self):
            self.calls = []

        async def send_message(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("Unexpected proactive send")

    bot = FakeBot()
    runtime = ProactiveRuntime(
        session_factory, bot, bot_identity=42, timezone_name="UTC",
        daily_hour=8, daily_renderer=None, reminder_renderer=None,
        daily_enabled=False)
    await runtime.tick()
    await runtime.tick()  # restart-equivalent wakeup remains quiet
    async with session_factory() as db:
        assert (await db.scalars(select(ScheduledOccurrence.id))).all() == []
    assert bot.calls == []

    # Explicitly enabled R4 mechanism still creates a durable occurrence.
    now = datetime.now(timezone.utc)
    await upsert_daily(session_factory, bot_identity=42, timezone_name="UTC",
                       hour=now.hour, local_date=now.date(), now=now)
    async with session_factory() as db:
        assert len((await db.scalars(select(ScheduledOccurrence.id))).all()) == 2
