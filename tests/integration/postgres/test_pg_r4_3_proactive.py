"""E4: real isolated PostgreSQL 18 occurrence/lease/outbox recovery."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db import crud
from app.config import get_settings
from app.db.models import (OutboundDelivery, OutboundDeliveryChunk,
                           ScheduledOccurrence, Todo, Base, User)
from app.scheduler.proactive import (ProactiveRuntime, claim_occurrence,
                                     ready_occurrence, upsert_daily,
                                     upsert_due_reminders, utc_now)
from app.telegram.durable import dispatch_delivery, claim_next_chunk, finish_chunk

pytestmark = pytest.mark.integration


class FakeBot:
    def __init__(self):
        self.calls = []

    async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
        self.calls.append((chat_id, text))
        return SimpleNamespace(chat_id=chat_id, message_id=len(self.calls) + 700)


@pytest.mark.asyncio
async def test_attended_chat_scope_excludes_other_users_and_old_occurrences(pg_harness):
    factory, new_user, _ = pg_harness
    alice, bob, charlie = await new_user(), await new_user(), await new_user()
    now = utc_now()
    due = now - timedelta(minutes=1)
    async with factory() as db:
        for user, title in ((alice, "Scoped reminder"), (bob, "Other reminder"),
                            (charlie, "Unqueued other reminder")):
            todo = await crud.create_todo(db, user.id, title)
            await crud.update_todo(db, todo.id, user.id, reminder_at=due)
        await db.commit()

    # Simulate an occurrence already queued for another user before scoped start.
    await upsert_due_reminders(
        factory, bot_identity=771, timezone_name="UTC",
        allowed_chat_ids=frozenset({alice.telegram_id, bob.telegram_id}))
    await upsert_daily(factory, bot_identity=771, timezone_name="UTC",
                       hour=now.hour, local_date=now.date(), now=now,
                       target_user_id=bob.id)

    async def render(row):
        return f"test {row.kind}", "en"

    bot = FakeBot()
    runtime = ProactiveRuntime(
        factory, bot, bot_identity=771, timezone_name="UTC", daily_hour=now.hour,
        daily_renderer=render, reminder_renderer=render,
        reminder_allowed_chat_ids=frozenset({alice.telegram_id}),
        daily_allowed_chat_ids=frozenset({alice.telegram_id}),
    )
    await runtime.tick()
    assert len(bot.calls) == 2
    assert all(chat_id == alice.telegram_id for chat_id, _ in bot.calls)
    async with factory() as db:
        rows = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id.in_([alice.id, bob.id, charlie.id])))).all()
        assert sum(row.state == "DELIVERED" for row in rows
                   if row.user_id == alice.id) == 2
        assert all(row.state in {"DELIVERED", "SKIPPED"} for row in rows
                   if row.user_id == alice.id)
        assert all(row.state == "PENDING" for row in rows if row.user_id == bob.id)
        assert not any(row.user_id == charlie.id for row in rows)
        todos = (await db.scalars(select(Todo).where(Todo.user_id.in_(
            [alice.id, bob.id, charlie.id])))).all()
        assert next(row for row in todos if row.user_id == alice.id).reminder_sent_at is not None
        assert next(row for row in todos if row.user_id == bob.id).reminder_sent_at is None
        assert next(row for row in todos if row.user_id == charlie.id).reminder_sent_at is None


@pytest.mark.asyncio
async def test_two_worker_upsert_claim_fence_restart_and_duplicate_fire(pg_harness):
    factory, new_user, engine = pg_harness
    user = await new_user()
    now = utc_now()
    local = now.astimezone(ZoneInfo("UTC"))

    async def fire():
        return await upsert_daily(factory, bot_identity=771, timezone_name="UTC",
                                  hour=local.hour, local_date=local.date(), now=now,
                                  target_user_id=user.id)

    keys = await asyncio.gather(fire(), fire())
    assert keys[0] == keys[1]
    async with factory() as db:
        rows = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))).all()
        assert len(rows) == 1
        row_id = rows[0].id

    tokens = await asyncio.gather(claim_occurrence(factory, row_id, "worker-A"),
                                  claim_occurrence(factory, row_id, "worker-B"))
    assert sum(token is not None for token in tokens) == 1
    old = next(token for token in tokens if token)
    async with factory() as db:
        row = await db.get(ScheduledOccurrence, row_id)
        row.lease_until = utc_now() - timedelta(seconds=1)
        await db.commit()
    fresh = await claim_occurrence(factory, row_id, "worker-C")
    assert fresh and fresh != old
    assert await ready_occurrence(factory, row_id, old, text="stale", locale="en") is None
    delivery_id = await ready_occurrence(factory, row_id, fresh, text="immutable", locale="en")
    assert delivery_id

    # A reconstructed runtime uses the committed payload, never calls render.
    async def forbidden_render(_):
        raise AssertionError("must not regenerate a READY occurrence")

    bot = FakeBot()
    fresh_engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    fresh_factory = async_sessionmaker(fresh_engine, expire_on_commit=False)
    try:
        restarted = ProactiveRuntime(fresh_factory, bot, bot_identity=771,
                                     timezone_name="UTC", daily_hour=local.hour,
                                     daily_renderer=forbidden_render,
                                     reminder_renderer=forbidden_render)
        await restarted.process(row_id)
        await fire()
        await restarted.process(row_id)
    finally:
        await fresh_engine.dispose()
    assert bot.calls == [(user.telegram_id, "immutable")]
    async with factory() as db:
        row = await db.get(ScheduledOccurrence, row_id)
        deliveries = (await db.scalars(select(OutboundDelivery).where(
            OutboundDelivery.occurrence_id == row_id))).all()
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 0})
        assert row.state == "DELIVERED" and len(deliveries) == 1
        assert chunk.telegram_chat_id == user.telegram_id and chunk.telegram_message_id == 701


@pytest.mark.asyncio
async def test_reminder_multiuser_discovery_remote_success_reconciliation(pg_harness):
    factory, new_user, engine = pg_harness
    alice, bob = await new_user(), await new_user()
    due = utc_now() - timedelta(minutes=1)
    async with factory() as db:
        first = await crud.create_todo(db, alice.id, "Alice reminder")
        second = await crud.create_todo(db, bob.id, "Bob reminder")
        await crud.update_todo(db, first.id, alice.id, reminder_at=due)
        await crud.update_todo(db, second.id, bob.id, reminder_at=due)
        await db.commit()

    await asyncio.gather(
        upsert_due_reminders(factory, bot_identity=771, timezone_name="UTC"),
        upsert_due_reminders(factory, bot_identity=771, timezone_name="UTC"))
    async with factory() as db:
        rows = (await db.scalars(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id.in_([alice.id, bob.id])))).all()
        assert len(rows) == 2 and {row.user_id for row in rows} == {alice.id, bob.id}
        first_row = next(row for row in rows if row.user_id == alice.id)
    token = await claim_occurrence(factory, first_row.id, "renderer")
    delivery_id = await ready_occurrence(factory, first_row.id, token,
                                          text="Alice reminder rendered", locale="en")
    first_bot = FakeBot()
    await dispatch_delivery(factory, delivery_id, "sender", first_bot)
    async with factory() as db:
        assert (await db.get(ScheduledOccurrence, first_row.id)).state == "READY"
    restarted_bot = FakeBot()
    restarted = ProactiveRuntime(factory, restarted_bot, bot_identity=771,
                                 timezone_name="UTC", daily_hour=utc_now().hour,
                                 daily_renderer=None, reminder_renderer=None)
    await restarted.process(first_row.id)
    assert restarted_bot.calls == [] and len(first_bot.calls) == 1
    async with factory() as db:
        assert (await db.get(ScheduledOccurrence, first_row.id)).state == "DELIVERED"
        assert (await db.get(Todo, first.id)).reminder_sent_at is not None
        assert (await db.get(Todo, second.id)).reminder_sent_at is None


@pytest.mark.asyncio
async def test_fresh_pg18_migration_references_one_shared_delivery(isolated_pg18_target):
    """A fresh isolated database must migrate, not just an upgraded one."""
    name = "r43_fixture_" + uuid4().hex[:12]
    assert name.startswith("r43_fixture_") and name.replace("r43_fixture_", "").isalnum()
    admin = create_async_engine(isolated_pg18_target, isolation_level="AUTOCOMMIT")
    target = None
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        target = create_async_engine(isolated_pg18_target.set(database=name))
        async with target.begin() as conn:
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[User.__table__]))
            from app.db.migrations.r4_2_durable_ingress import apply as r42
            from app.db.migrations.r4_2_outcome_proof import apply as outcome
            from app.db.migrations.r4_3_proactive_occurrences import (
                apply as r43, create_occurrence_table)
            await conn.run_sync(create_occurrence_table)
            await conn.run_sync(r42)
            await conn.run_sync(outcome)
            await conn.run_sync(r43)
            await conn.run_sync(r43)  # repeated startup is idempotent
            await conn.run_sync(Base.metadata.create_all)
            columns = set((await conn.scalars(text("""
                SELECT column_name FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = 'scheduled_occurrences'
            """))).all())
            assert columns == {
                "id", "occurrence_key", "kind", "user_id", "bot_identity",
                "chat_id", "local_date", "timezone_name", "planned_at",
                "todo_id", "reminder_at_snapshot", "state", "attempts",
                "lease_owner", "fence_token", "lease_until", "next_attempt_at",
                "failure_category", "created_at", "completed_at",
            }
            column = await conn.scalar(text("""
                SELECT count(*) FROM information_schema.columns
                 WHERE table_name='outbound_deliveries' AND column_name='occurrence_id'
            """))
            assert column == 1
            await _assert_r43_constraints(conn)
    finally:
        if target is not None:
            await target.dispose()
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await admin.dispose()


async def _assert_r43_constraints(conn):
    """The actual target relation, not a same-named constraint elsewhere."""
    rows = (await conn.execute(text("""
        SELECT conname, contype::text
          FROM pg_constraint
         WHERE conrelid = 'outbound_deliveries'::regclass
           AND conname IN ('fk_outbound_occurrence', 'ck_outbound_one_origin')
    """))).all()
    assert set(rows) == {('fk_outbound_occurrence', 'f'),
                         ('ck_outbound_one_origin', 'c')}
    unique = await conn.scalar(text("""
        SELECT count(*)
          FROM pg_index i
          JOIN pg_class t ON t.oid = i.indrelid
          JOIN pg_class idx ON idx.oid = i.indexrelid
         WHERE t.oid = 'outbound_deliveries'::regclass
           AND idx.relname = 'uq_outbound_occurrence'
           AND i.indisunique
    """))
    assert unique == 1


@pytest.mark.asyncio
async def test_existing_r42_schema_upgrade_preserves_inbound_and_enforces_r43(isolated_pg18_target):
    """Upgrade an R4.2-shaped outbound table without a future occurrence FK."""
    name = "r43_upgrade_" + uuid4().hex[:12]
    assert name.startswith("r43_upgrade_") and name.replace("r43_upgrade_", "").isalnum()
    admin = create_async_engine(isolated_pg18_target, isolation_level="AUTOCOMMIT")
    target = None
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        target = create_async_engine(isolated_pg18_target.set(database=name))
        receipt_id, old_delivery_id = uuid4().hex, uuid4().hex
        async with target.begin() as conn:
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[User.__table__]))
            from app.db.models import TelegramInboundReceipt
            await conn.run_sync(lambda sync: Base.metadata.create_all(
                sync, tables=[TelegramInboundReceipt.__table__]))
            # The R4.2 shape: no occurrence column/FK and receipt is required.
            await conn.execute(text("""
                CREATE TABLE outbound_deliveries (
                    id VARCHAR(36) PRIMARY KEY,
                    receipt_id VARCHAR(36) NOT NULL REFERENCES telegram_inbound_receipts(id),
                    reply_ordinal INTEGER NOT NULL,
                    kind VARCHAR(32) NOT NULL,
                    chat_id VARCHAR(64) NOT NULL,
                    locale VARCHAR(16) NOT NULL,
                    text TEXT NOT NULL,
                    parse_mode VARCHAR(24),
                    keyboard JSON,
                    payload_sha256 VARCHAR(64) NOT NULL,
                    state VARCHAR(20) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    sent_at TIMESTAMPTZ,
                    CONSTRAINT uq_outbound_receipt_ordinal UNIQUE (receipt_id, reply_ordinal)
                )
            """))
            from app.db.migrations.r4_2_durable_ingress import apply as r42
            from app.db.migrations.r4_2_outcome_proof import apply as outcome
            await conn.run_sync(r42)
            await conn.run_sync(outcome)
            await conn.execute(text("""
                INSERT INTO telegram_inbound_receipts
                    (id, bot_identity, update_id, envelope, envelope_sha256,
                     state, attempts, received_at)
                VALUES (:id, 77, 1, CAST(:envelope AS json), :sha,
                        'PROCESSED', 1, now())
            """), {"id": receipt_id, "envelope": "{}", "sha": "a" * 64})
            await conn.execute(text("""
                INSERT INTO outbound_deliveries
                    (id, receipt_id, reply_ordinal, kind, chat_id, locale,
                     text, payload_sha256, state, created_at)
                VALUES (:id, :receipt, 0, 'reply', 'chat', 'en',
                        'historical inbound', :sha, 'SENT', now())
            """), {"id": old_delivery_id, "receipt": receipt_id, "sha": "b" * 64})
            # Same constraint names on another relation must not suppress the
            # R4.3 constraints on outbound_deliveries.
            await conn.execute(text("CREATE TABLE r43_constraint_decoy (n INTEGER)"))
            await conn.execute(text("""
                ALTER TABLE r43_constraint_decoy
                ADD CONSTRAINT fk_outbound_occurrence CHECK (n >= 0)
            """))
            await conn.execute(text("""
                ALTER TABLE r43_constraint_decoy
                ADD CONSTRAINT ck_outbound_one_origin CHECK (n <= 100)
            """))

            from app.db.migrations.r4_3_proactive_occurrences import apply as r43
            await conn.run_sync(r43)
            await conn.run_sync(r43)
            await _assert_r43_constraints(conn)
            old = (await conn.execute(text("""
                SELECT receipt_id, occurrence_id FROM outbound_deliveries WHERE id = :id
            """), {"id": old_delivery_id})).one()
            assert old == (receipt_id, None)

            user_id, occurrence_id = uuid4().hex, uuid4().hex
            await conn.execute(text("""
                INSERT INTO users (id, telegram_id, created_at, updated_at)
                VALUES (:id, 'r43-upgrade-user', now(), now())
            """), {"id": user_id})
            await conn.execute(text("""
                INSERT INTO scheduled_occurrences
                    (id, occurrence_key, kind, user_id, bot_identity, chat_id,
                     timezone_name, planned_at, state, attempts, created_at)
                VALUES (:id, :key, 'daily_rundown', :user_id, 77, 'chat',
                        'UTC', now(), 'READY', 1, now())
            """), {"id": occurrence_id, "key": uuid4().hex, "user_id": user_id})

            async def insert_delivery(*, receipt=None, occurrence=None):
                await conn.execute(text("""
                    INSERT INTO outbound_deliveries
                        (id, receipt_id, occurrence_id, reply_ordinal, kind,
                         chat_id, locale, text, payload_sha256, state, created_at)
                    VALUES (:id, :receipt, :occurrence, 0, 'daily_rundown',
                            'chat', 'en', 'test', :sha, 'PENDING', now())
                """), {"id": uuid4().hex, "receipt": receipt,
                       "occurrence": occurrence, "sha": "c" * 64})

            await insert_delivery(occurrence=occurrence_id)
            for invalid in (
                {"receipt": None, "occurrence": None},
                {"receipt": receipt_id, "occurrence": occurrence_id},
                {"receipt": None, "occurrence": uuid4().hex},
                {"receipt": None, "occurrence": occurrence_id},
            ):
                with pytest.raises(IntegrityError):
                    async with conn.begin_nested():
                        await insert_delivery(**invalid)
            assert await conn.scalar(text("""
                SELECT count(*) FROM outbound_deliveries
                 WHERE occurrence_id = :id
            """), {"id": occurrence_id}) == 1
    finally:
        if target is not None:
            await target.dispose()
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await admin.dispose()


@pytest.mark.asyncio
async def test_pg_chunk_crash_recovery_preserves_confirmed_remote_id(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    now = utc_now()
    await upsert_daily(factory, bot_identity=771, timezone_name="UTC",
                       hour=now.hour, local_date=now.date(), now=now,
                       target_user_id=user.id)
    async with factory() as db:
        row = await db.scalar(select(ScheduledOccurrence).where(
            ScheduledOccurrence.user_id == user.id))
    token = await claim_occurrence(factory, row.id, "renderer")
    delivery_id = await ready_occurrence(factory, row.id, token,
                                          text="a" * 4096 + "b", locale="en")
    first = await claim_next_chunk(factory, delivery_id, "before-crash")
    assert first[0] == 0
    assert await finish_chunk(factory, delivery_id, 0, first[2],
                              message_id=811, chat_id=user.telegram_id)
    second = await claim_next_chunk(factory, delivery_id, "before-crash")
    assert second[0] == 1
    async with factory() as db:
        chunk = await db.get(OutboundDeliveryChunk,
                             {"delivery_id": delivery_id, "chunk_index": 1})
        chunk.lease_until = utc_now() - timedelta(seconds=1)
        await db.commit()
    fresh_engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    fresh_factory = async_sessionmaker(fresh_engine, expire_on_commit=False)
    try:
        bot = FakeBot()
        restarted = ProactiveRuntime(fresh_factory, bot, bot_identity=771,
                                     timezone_name="UTC", daily_hour=now.hour,
                                     daily_renderer=None, reminder_renderer=None)
        await restarted.process(row.id)
        async with fresh_factory() as db:
            chunk = await db.get(OutboundDeliveryChunk,
                                 {"delivery_id": delivery_id, "chunk_index": 1})
            assert chunk.state == "UNCERTAIN"
            chunk.next_attempt_at = utc_now() - timedelta(seconds=1)
            await db.commit()
        await restarted.process(row.id)
        assert bot.calls == [(user.telegram_id, "b")]
        async with fresh_factory() as db:
            first_chunk = await db.get(OutboundDeliveryChunk,
                                       {"delivery_id": delivery_id, "chunk_index": 0})
            assert first_chunk.telegram_message_id == 811
            assert (await db.get(ScheduledOccurrence, row.id)).state == "DELIVERED"
    finally:
        await fresh_engine.dispose()
