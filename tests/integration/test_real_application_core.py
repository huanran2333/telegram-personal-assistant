"""Opt-in acceptance of the production lifespan with real Postgres and LLM."""
import asyncio
import sys
from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy import delete, select, text

from app.agent.graph import get_graph, run_agent
from app.agent.store import generate_store
from app.config import get_settings
from app.db import crud
from app.db.models import (Todo, User, ScheduledOccurrence, OutboundDelivery,
                           OutboundDeliveryChunk, OutboundDeliveryAttempt)
from app.db.session import AsyncSessionLocal, async_engine
from app.main import app, lifespan
from app.scheduler.reminders import scheduler


pytestmark = pytest.mark.llm_integration


@pytest.fixture(scope="session")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.mark.asyncio
@pytest.mark.provider_lifespan
async def test_real_application_core_lifecycle(isolated_pg18_target, monkeypatch):
    settings = get_settings()
    assert isolated_pg18_target.database == "assistant"
    assert settings.openai_api_key and settings.openai_api_key != "test-key"
    assert settings.llm_provider == "openai"

    # Real Saver/Store/LLM lifecycle without registering or changing a live
    # Telegram webhook. The local target was version/data-directory checked by
    # the shared PG18 fixture before any Todo/profile mutation.
    from app import main as app_main
    monkeypatch.setattr(app_main.settings, "telegram_webhook_url", "https://example.invalid/r3-core")
    monkeypatch.setattr(app_main.settings, "telegram_bot_token", "synthetic-test-token")

    class FakeBot:
        async def get_me(self, **_kwargs):
            return type("SyntheticIdentity", (), {"id": 123456789})()

        async def set_webhook(self, **_kwargs):
            return True

    monkeypatch.setattr("app.telegram.sender.get_bot", lambda: FakeBot())

    telegram_id = f"itest-core-{uuid4().hex}"
    thread_id = f"user_{telegram_id}"
    user_id = None
    try:
        async with lifespan(app):
            assert app.state.runtime.ready
            assert app.state.runtime.required_ready_except("none")
            # An opt-in semantic index can be degraded by older rows in this
            # shared isolated test database without disabling durable ingress.
            assert isinstance(app.state.runtime.full_ready, bool)
            graph = await get_graph()
            assert isinstance(graph.checkpointer, AsyncPostgresSaver)
            assert isinstance(graph.store, AsyncPostgresStore)
            assert scheduler.running
            reminder_job = scheduler.get_job("reminder_checker")
            assert reminder_job is not None
            assert int(reminder_job.trigger.interval.total_seconds()) == 60

            async with AsyncSessionLocal() as db:
                user = await crud.get_or_create_user(db, telegram_id)
                await db.commit()
                user_id = user.id

            first = await run_agent(
                graph, user_id, telegram_id, telegram_id,
                "创建一个周五截止的任务：修改论文。",
            )
            async with AsyncSessionLocal() as db:
                rows = (await db.scalars(select(Todo).where(Todo.user_id == user_id))).all()
                paper = next((row for row in rows if "修改论文" in row.title), None)
                assert paper is not None, f"Real LLM did not create Todo: {first}"
                assert paper.due_date is not None
            print("REAL_TODO_WRITE_PASS", paper.id)

            second = await run_agent(
                graph, user_id, telegram_id, telegram_id,
                "把刚才那个任务优先级改成高。",
            )
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            human_messages = [m.content for m in state.values["messages"] if m.type == "human"]
            assert any("修改论文" in str(message) for message in human_messages)
            assert any("优先级" in str(message) for message in human_messages)
            async with async_engine.connect() as connection:
                assert await connection.scalar(
                    text("SELECT count(*) FROM checkpoints WHERE thread_id = :thread_id"),
                    {"thread_id": thread_id},
                ) > 0
            print("REAL_CHECKPOINT_TWO_TURNS_PASS", len(human_messages), second[:80])

            # Use explicit future dates so the live provider cannot turn a
            # Friday request made on Friday into an impossible past-Thursday
            # reminder. Relative weekday parsing is covered separately.
            today = datetime.now(ZoneInfo(settings.default_timezone)).date()
            due_day = today + timedelta(days=(4 - today.weekday()) % 7 + 7)
            reminder_day = due_day - timedelta(days=1)
            reminder_thread = f"{telegram_id}-reminder"
            reminder_response = await run_agent(
                graph, user_id, reminder_thread, reminder_thread,
                f"{due_day.isoformat()} 交论文，{reminder_day.isoformat()} 晚上 8 点提醒我。",
            )
            async with AsyncSessionLocal() as db:
                rows = (await db.scalars(select(Todo).where(Todo.user_id == user_id))).all()
                reminded = next((row for row in rows if row.reminder_at is not None), None)
                assert reminded is not None, f"Real LLM did not persist reminder: {reminder_response}"
                due_date, reminder_at = reminded.due_date, reminded.reminder_at
                assert due_date is not None
                zone = ZoneInfo(settings.default_timezone)
                assert due_date.astimezone(zone).date() == due_day
                assert reminder_at.astimezone(zone).date() == reminder_day
                assert reminder_at.astimezone(ZoneInfo(settings.default_timezone)).hour == 20
            print("REAL_DUE_REMINDER_PASS", reminded.id, due_date.isoformat(), reminder_at.isoformat())

            profile_thread = f"{telegram_id}-profile"
            await run_agent(graph, user_id, profile_thread, profile_thread, "I prefer concise answers.")
            profile = await graph.store.aget(("profile", user_id), "user_profile")
            assert profile is not None and "concise" in str(profile.value).lower()
            print("REAL_PROFILE_WRITE_PASS", profile.value)

        await asyncio.sleep(0)
        assert not scheduler.running
        assert graph.checkpointer.conn.closed
        async with lifespan(app):
            graph = await get_graph()
            recovered = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            assert any("修改论文" in str(m.content) for m in recovered.values["messages"])
            assert isinstance(graph.checkpointer, AsyncPostgresSaver)
            assert (await graph.store.aget(("profile", user_id), "user_profile")) is not None
            print("REAL_RESTART_RECOVERY_PASS", len(recovered.values["messages"]))
            for persisted_thread in (thread_id, f"user_{reminder_thread}", f"user_{profile_thread}"):
                await graph.checkpointer.adelete_thread(persisted_thread)

        async with generate_store() as fresh_store:
            assert (await fresh_store.aget(("profile", user_id), "user_profile")) is not None
            await fresh_store.adelete(("profile", user_id), "user_profile")
    finally:
        if user_id:
            async with AsyncSessionLocal() as db:
                # R4.3 may turn the synthetic reminder into a durable scheduled
                # occurrence during the lifespan. Remove only this test user's
                # dependent rows before deleting the synthetic user.
                occurrence_ids = (await db.scalars(select(ScheduledOccurrence.id).where(
                    ScheduledOccurrence.user_id == user_id))).all()
                if occurrence_ids:
                    delivery_ids = (await db.scalars(select(OutboundDelivery.id).where(
                        OutboundDelivery.occurrence_id.in_(occurrence_ids)))).all()
                    if delivery_ids:
                        await db.execute(delete(OutboundDeliveryAttempt).where(
                            OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDeliveryChunk).where(
                            OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
                        await db.execute(delete(OutboundDelivery).where(
                            OutboundDelivery.id.in_(delivery_ids)))
                    await db.execute(delete(ScheduledOccurrence).where(
                        ScheduledOccurrence.id.in_(occurrence_ids)))
                await db.execute(delete(Todo).where(Todo.user_id == user_id))
                await db.execute(delete(User).where(User.id == user_id))
                await db.commit()
        await async_engine.dispose()
