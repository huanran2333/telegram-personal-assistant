"""E4 real model + isolated PG; deliberately no Telegram network send."""
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app.agent.nodes import _build_chat_llm
from app.config import get_settings
from app.db import crud
from app.db.models import OutboundDelivery, ScheduledOccurrence
from app.scheduler.daily_brief import render_daily_occurrence
from app.scheduler.proactive import ProactiveRuntime, upsert_daily, utc_now
from app.scheduler.rundown import synthesize_order

pytestmark = pytest.mark.llm_integration


class EmptyCalendar:
    async def search_events(self, query, start=None, end=None):
        return []


@pytest.mark.asyncio
async def test_real_model_render_is_immutable_after_restart(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    now = utc_now()
    async with factory() as db:
        await crud.create_todo(db, user.id, "R4.3 synthetic rundown test task",
                               due_date=now.replace(hour=23, minute=0))
        await db.commit()
    await upsert_daily(factory, bot_identity=771, timezone_name="UTC",
                       hour=now.hour, local_date=now.date(), now=now,
                       target_user_id=user.id)
    async with factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))

    settings = get_settings()
    real_llm = _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)
    calls = []

    class SpyLLM:
        def with_structured_output(self, schema):
            runnable = real_llm.with_structured_output(schema)

            class SpyRunnable:
                async def ainvoke(self, messages):
                    result = await runnable.ainvoke(messages)
                    calls.append(type(result).__name__)
                    return result

            return SpyRunnable()

    async def render(occurrence):
        return await render_daily_occurrence(
            occurrence, calendar=EmptyCalendar(), session_factory=factory,
            synthesizer=lambda items: synthesize_order(items, llm=SpyLLM()))

    class Bot:
        sent = []

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            self.sent.append(text)
            return SimpleNamespace(chat_id=chat_id, message_id=len(self.sent))

    bot = Bot()
    runtime = ProactiveRuntime(factory, bot, bot_identity=771,
                               timezone_name="UTC", daily_hour=now.hour,
                               daily_renderer=render, reminder_renderer=None)
    await runtime.process(row.id)
    assert len(calls) == 1 and len(bot.sent) == 1
    async with factory() as db:
        delivery = await db.scalar(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == row.id))
        immutable_text = delivery.text
        assert delivery.state == "SENT"

    async def forbidden_render(_):
        raise AssertionError("retry must not regenerate model text")

    reconstructed = ProactiveRuntime(factory, bot, bot_identity=771,
                                     timezone_name="UTC", daily_hour=now.hour,
                                     daily_renderer=forbidden_render,
                                     reminder_renderer=None)
    await reconstructed.process(row.id)
    assert len(calls) == 1 and bot.sent == [immutable_text]
