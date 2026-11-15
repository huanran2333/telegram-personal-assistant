"""Real PostgreSQL fixtures; never substitute SQLite for this suite."""
import asyncio
import sys
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import crud
from app.db.models import (Base, ActionExecution, CalendarActionEvidence,
                           TodoCalendarMirror,
                           CalendarDependentContinuation,
                           DailyBriefDelivery, Todo, User,
                           ScheduledOccurrence, OutboundDelivery, OutboundDeliveryChunk,
                           OutboundDeliveryAttempt,
                           TranscriptProjection, TranscriptSuppression, ConversationSummary,
                           StructuredMemory, MemorySuppression, MemoryConsolidationCursor,
                           TranscriptEmbedding, StructuredMemoryEmbedding)
from app.db.models import (MeetingDraft, WebCredential, WebSession, WebRequest, WebLoginThrottle,
                           WebLoginNonce, WebMutation, TelegramMiniAppLaunchTicket,
                           TelegramBusinessOperation)


@pytest_asyncio.fixture(autouse=True)
async def require_isolated_postgres_before_test(isolated_pg18_target):
    """All tests in this writable suite first prove the cluster identity."""
    yield


@pytest.fixture(scope="session")
def event_loop_policy():
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest_asyncio.fixture
async def pg_harness(isolated_pg18_target):
    engine = create_async_engine(isolated_pg18_target, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    user_ids = []

    # The isolated cluster can be restarted from an older snapshot. Integration
    # tests must not rely on another test module having created newer R3 tables.
    async with engine.begin() as connection:
        from app.db.models import Todo, User
        from app.db.migrations.r4_3_proactive_occurrences import create_occurrence_table
        await connection.run_sync(lambda sync: Base.metadata.create_all(
            sync, tables=[User.__table__], checkfirst=True))
        await connection.run_sync(create_occurrence_table)
        from app.db.migrations.r4_2_durable_ingress import apply as apply_r4_2
        from app.db.migrations.r4_2_outcome_proof import apply as apply_outcome_proof
        from app.db.migrations.r4_3_proactive_occurrences import apply as apply_r4_3
        await connection.run_sync(lambda sync: apply_r4_2(sync, include_legacy=False))
        await connection.run_sync(apply_outcome_proof)
        await connection.run_sync(apply_r4_3)
        from app.db.migrations.r4_4_terminal_payload_redaction import apply as apply_r4_4
        await connection.run_sync(apply_r4_4)
        from app.db.migrations.r5_1_calendar_action_evidence import apply as apply_r5_1
        await connection.run_sync(apply_r5_1)
        from app.db.migrations.r5_2_create_plan import apply as apply_r5_2
        await connection.run_sync(apply_r5_2)
        from app.db.migrations.r5_3_conditional_modification import apply as apply_r5_3
        await connection.run_sync(apply_r5_3)
        await connection.run_sync(lambda sync: Base.metadata.create_all(
            sync, tables=[ActionExecution.__table__], checkfirst=True))
        from app.db.migrations.r5_4_dependent_continuation import apply as apply_r5_4
        await connection.run_sync(apply_r5_4)
        from app.db.migrations.v3_1_web_boundary import apply as apply_v3_1
        await connection.run_sync(apply_v3_1)
        # Match production init_db: V3.2 alters the pre-existing Todo table.
        await connection.run_sync(lambda sync: Base.metadata.create_all(
            sync, tables=[Todo.__table__], checkfirst=True))
        from app.db.migrations.v3_2_workbench import apply as apply_v3_2
        await connection.run_sync(apply_v3_2)
        from app.db.migrations.v3_miniapp_g1 import apply as apply_miniapp_g1
        await connection.run_sync(apply_miniapp_g1)
        await connection.run_sync(lambda sync: Base.metadata.create_all(
            sync, tables=[table for table in Base.metadata.sorted_tables
                          if table is not MeetingDraft.__table__
                          and table is not TelegramBusinessOperation.__table__]))
        from app.db.migrations.g6_meeting_drafts import apply as apply_g6
        await connection.run_sync(apply_g6)
        from app.db.migrations.g5_todo_time_blocks import apply as apply_g5
        await connection.run_sync(apply_g5)
        from app.db.migrations.v3_1_cross_thread_lineage import apply as apply_web_lineage
        await connection.run_sync(apply_web_lineage)

    async def new_user():
        async with factory() as db:
            user = await crud.get_or_create_user(db, f"itest-{uuid4().hex}")
            await db.commit()
            user_ids.append(user.id)
            return user

    yield factory, new_user, engine

    if user_ids:
        async with factory() as db:
            # Disable opt-in before deleting test rows so the trigger cannot
            # re-enqueue a remote tombstone during fixture cleanup.
            for user_id in user_ids:
                user = await db.get(User, user_id)
                if user is not None:
                    user.calendar_time_block_sync = False
            await db.flush()
            await db.execute(delete(TodoCalendarMirror).where(
                TodoCalendarMirror.user_id.in_(user_ids)))
            telegram_ids = (await db.scalars(select(User.telegram_id).where(
                User.id.in_(user_ids)))).all()
            if telegram_ids:
                await db.execute(delete(TelegramMiniAppLaunchTicket).where(
                    TelegramMiniAppLaunchTicket.telegram_user_id.in_(telegram_ids)))
            await db.execute(delete(WebMutation).where(WebMutation.user_id.in_(user_ids)))
            await db.execute(delete(WebRequest).where(WebRequest.user_id.in_(user_ids)))
            await db.execute(delete(WebSession).where(WebSession.user_id.in_(user_ids)))
            await db.execute(delete(MeetingDraft).where(MeetingDraft.user_id.in_(user_ids)))
            await db.execute(delete(WebCredential).where(WebCredential.user_id.in_(user_ids)))
            delivery_ids = (await db.scalars(select(OutboundDelivery.id).join(
                ScheduledOccurrence,
                ScheduledOccurrence.id == OutboundDelivery.occurrence_id).where(
                ScheduledOccurrence.user_id.in_(user_ids)))).all()
            if delivery_ids:
                await db.execute(delete(OutboundDeliveryAttempt).where(
                    OutboundDeliveryAttempt.delivery_id.in_(delivery_ids)))
                await db.execute(delete(OutboundDeliveryChunk).where(
                    OutboundDeliveryChunk.delivery_id.in_(delivery_ids)))
                await db.execute(delete(OutboundDelivery).where(OutboundDelivery.id.in_(delivery_ids)))
            await db.execute(delete(ScheduledOccurrence).where(ScheduledOccurrence.user_id.in_(user_ids)))
            await db.execute(delete(TranscriptEmbedding).where(TranscriptEmbedding.user_id.in_(user_ids)))
            await db.execute(delete(StructuredMemoryEmbedding).where(
                StructuredMemoryEmbedding.user_id.in_(user_ids)))
            await db.execute(delete(MemoryConsolidationCursor).where(MemoryConsolidationCursor.user_id.in_(user_ids)))
            await db.execute(delete(MemorySuppression).where(MemorySuppression.user_id.in_(user_ids)))
            await db.execute(delete(StructuredMemory).where(StructuredMemory.user_id.in_(user_ids)))
            await db.execute(delete(ConversationSummary).where(ConversationSummary.user_id.in_(user_ids)))
            await db.execute(delete(TranscriptProjection).where(TranscriptProjection.user_id.in_(user_ids)))
            await db.execute(delete(TranscriptSuppression).where(TranscriptSuppression.user_id.in_(user_ids)))
            await db.execute(delete(CalendarDependentContinuation).where(
                CalendarDependentContinuation.parent_action_id.in_(select(ActionExecution.action_id).where(
                    ActionExecution.user_id.in_(user_ids)))))
            await db.execute(delete(ActionExecution).where(ActionExecution.user_id.in_(user_ids)))
            await db.execute(delete(CalendarActionEvidence).where(
                CalendarActionEvidence.user_id.in_(user_ids)))
            await db.execute(delete(DailyBriefDelivery).where(DailyBriefDelivery.user_id.in_(user_ids)))
            await db.execute(delete(Todo).where(Todo.user_id.in_(user_ids)))
            await db.execute(delete(User).where(User.id.in_(user_ids)))
            await db.commit()
    await engine.dispose()
