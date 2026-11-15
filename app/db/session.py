from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from app.config import get_settings
from app.db.models import Base, TelegramBusinessOperation

settings = get_settings()

# Async engine for application use
async_engine = create_async_engine(
    settings.database_url,
    # echo=settings.app_env == "development",
    echo=False,
    pool_pre_ping=True,
    hide_parameters=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for LangGraph checkpointer
sync_engine = create_engine(
    settings.database_url_sync,
    pool_pre_ping=True,
    hide_parameters=True,
)

async def init_db():
    """Create all tables."""
    async with async_engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            # Explicit retirement must happen before deploying this runtime.
            # Never silently ignore an old replay proof or drop user data.
            legacy = await conn.scalar(text(
                "SELECT to_regclass('telegram_business_operations')"))
            if legacy is not None:
                raise RuntimeError("TelegramExecutionLedgerRetirementRequired")
            # Current R4.2 metadata sees the later occurrence FK on a fresh
            # install. Satisfy that dependency with R4.3's frozen DDL, never
            # the future-mutable ScheduledOccurrence ORM table definition.
            from app.db.models import User
            from app.db.migrations.r4_3_proactive_occurrences import create_occurrence_table
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[User.__table__], checkfirst=True))
            await conn.run_sync(create_occurrence_table)
            from app.db.migrations.r4_2_durable_ingress import apply as apply_r4_2
            await conn.run_sync(lambda sync: apply_r4_2(sync, include_legacy=False))
            from app.db.migrations.r4_2_outcome_proof import apply as apply_outcome_proof
            await conn.run_sync(apply_outcome_proof)
            from app.db.migrations.r4_3_proactive_occurrences import apply as apply_r4_3
            await conn.run_sync(apply_r4_3)
            from app.db.migrations.r4_4_terminal_payload_redaction import apply as apply_r4_4
            await conn.run_sync(apply_r4_4)
            from app.db.migrations.r5_1_calendar_action_evidence import apply as apply_r5_1
            await conn.run_sync(apply_r5_1)
            from app.db.migrations.r5_2_create_plan import apply as apply_r5_2
            await conn.run_sync(apply_r5_2)
            from app.db.migrations.r5_3_conditional_modification import apply as apply_r5_3
            await conn.run_sync(apply_r5_3)
            from app.db.models import ActionExecution
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[ActionExecution.__table__], checkfirst=True))
            from app.db.migrations.r5_4_dependent_continuation import apply as apply_r5_4
            await conn.run_sync(apply_r5_4)
            from app.db.migrations.v3_1_web_boundary import apply as apply_v3_1
            await conn.run_sync(apply_v3_1)
            # V3.2's additive ALTER needs the legacy Todo authority on a fresh
            # database. checkfirst leaves an upgraded Todo table untouched.
            from app.db.models import Todo
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[Todo.__table__], checkfirst=True))
            from app.db.migrations.v3_2_workbench import apply as apply_v3_2
            await conn.run_sync(apply_v3_2)
            from app.db.migrations.v3_miniapp_g1 import apply as apply_miniapp_g1
            await conn.run_sync(apply_miniapp_g1)
        # Meeting drafts require an explicitly reviewed additive migration.
        # Startup must not silently create that business table in a formal DB.
        from app.db.models import MeetingDraft
        await conn.run_sync(lambda sync: Base.metadata.create_all(
            sync, tables=[table for table in Base.metadata.sorted_tables
                          if table is not MeetingDraft.__table__
                          and table is not TelegramBusinessOperation.__table__]))
        if conn.dialect.name == "postgresql":
            from app.db.migrations.g4_approval_markup_sync import apply as apply_markup_sync
            await conn.run_sync(apply_markup_sync)
            from app.db.migrations.g5_todo_time_blocks import apply as apply_todo_time_blocks
            await conn.run_sync(apply_todo_time_blocks)
            from app.db.migrations.v3_1_cross_thread_lineage import apply as apply_web_lineage
            await conn.run_sync(apply_web_lineage)
        # The project predates Alembic migrations.  Keep existing deployments
        # usable while introducing reminder metadata; PostgreSQL accepts this
        # idempotent additive migration and fresh installs use metadata above.
        if conn.dialect.name == "postgresql":
            for statement in (
                "ALTER TABLE todos ADD COLUMN IF NOT EXISTS reminder_at TIMESTAMPTZ",
                "ALTER TABLE todos ADD COLUMN IF NOT EXISTS reminder_claimed_at TIMESTAMPTZ",
                "ALTER TABLE todos ADD COLUMN IF NOT EXISTS reminder_sent_at TIMESTAMPTZ",
                "CREATE INDEX IF NOT EXISTS ix_transcript_fts ON transcript_projection USING GIN (to_tsvector('simple', search_text))",
                "ALTER TABLE transcript_suppressions ADD COLUMN IF NOT EXISTS supersedes_message_id VARCHAR(160)",
                "ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS covered_source_message_ids JSON NOT NULL DEFAULT '[]'",
                "ALTER TABLE memory_consolidation_cursors ADD COLUMN IF NOT EXISTS last_source_occurred_at TIMESTAMPTZ",
            ):
                await conn.execute(text(statement))

async def get_db() -> AsyncSession:
    """Dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
