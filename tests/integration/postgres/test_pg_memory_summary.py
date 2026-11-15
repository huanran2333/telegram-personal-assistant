"""Real PostgreSQL cursor/restart/failure tests for non-destructive summary."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select, text

from app.agent.memory_summary import advance_running_summary, load_summary, SummarySourceDrift
from app.db.models import Base, ConversationSummary
from tests.integration.postgres.current_schema import create_current_tables

pytestmark = pytest.mark.integration


def _history(turns: int = 10):
    result = []
    for index in range(turns):
        result.extend([
            HumanMessage(id=f"u{index}", content=f"Synthetic project Aurora has {index + 47} pages. " * 12),
            AIMessage(id=f"a{index}", content=f"I heard {index + 47} pages." * 5),
        ])
    return result


@pytest.mark.asyncio
async def test_incremental_summary_persists_cursor_across_restart(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
        await connection.execute(text(
            "ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS "
            "covered_source_message_ids JSON NOT NULL DEFAULT '[]'"))
    user = await new_user()
    thread = f"r3-summary-{uuid4().hex}"
    calls = []

    def summarize(messages):
        calls.append(messages)
        return AIMessage(content=f"Synthetic running summary version {len(calls)}")

    model = RunnableLambda(summarize)
    async with factory() as db:
        first = await advance_running_summary(
            db, user_id=user.id, thread_id=thread, history=_history(10),
            model=model, provider="test", model_name="fake",
            max_tokens=260, trigger_tokens=100, max_summary_tokens=40,
            retained_groups=2)
        await db.commit()
        assert first is not None and first.version == 1
        assert first.last_summarized_message_id in {f"turn:a{i}" for i in range(8)}
        assert all(item.startswith("turn:a") for item in first.summarized_message_ids)
        assert all(item.startswith(("u", "a")) for item in first.covered_source_message_ids)

    async with factory() as restarted:
        loaded = await load_summary(restarted, user_id=user.id, thread_id=thread)
        assert loaded == first
        second = await advance_running_summary(
            restarted, user_id=user.id, thread_id=thread, history=_history(14),
            model=model, provider="test", model_name="fake",
            max_tokens=260, trigger_tokens=100, max_summary_tokens=40,
            retained_groups=2)
        await restarted.commit()
        assert second.version == 2
        assert set(first.summarized_message_ids).issubset(second.summarized_message_ids)
        assert second.last_summarized_message_id != first.last_summarized_message_id
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_failure_keeps_previous_summary_and_cursor(pg_harness):
    factory, new_user, engine = pg_harness
    async with engine.begin() as connection:
        await connection.run_sync(create_current_tables)
        await connection.execute(text(
            "ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS "
            "covered_source_message_ids JSON NOT NULL DEFAULT '[]'"))
    user = await new_user()
    thread = f"r3-summary-fail-{uuid4().hex}"
    good = RunnableLambda(lambda _: AIMessage(content="Valid summary"))
    empty = RunnableLambda(lambda _: AIMessage(content=""))
    async with factory() as db:
        before = await advance_running_summary(
            db, user_id=user.id, thread_id=thread, history=_history(10),
            model=good, provider="test", model_name="fake", max_tokens=260,
            trigger_tokens=100, max_summary_tokens=40, retained_groups=2)
        await db.commit()
    async with factory() as db:
        with pytest.raises(ValueError, match="empty summary"):
            await advance_running_summary(
                db, user_id=user.id, thread_id=thread, history=_history(14),
                model=empty, provider="test", model_name="fake", max_tokens=260,
                trigger_tokens=100, max_summary_tokens=40, retained_groups=2)
        await db.commit()
    async with factory() as db:
        after = await load_summary(db, user_id=user.id, thread_id=thread)
        assert after.version == before.version
        assert after.content == before.content
        assert after.last_summarized_message_id == before.last_summarized_message_id
        assert after.last_failure_category == "ValueError"
        with pytest.raises(SummarySourceDrift):
            await advance_running_summary(
                db, user_id=user.id, thread_id=thread, history=_history(3),
                model=good, provider="test", model_name="fake", max_tokens=260,
                trigger_tokens=100, max_summary_tokens=40, retained_groups=2)
        await db.commit()
        row = await db.scalar(select(ConversationSummary).where(
            ConversationSummary.user_id == user.id, ConversationSummary.thread_id == thread))
        assert row.version == before.version
        assert row.last_failure_category == "SummarySourceDrift"
