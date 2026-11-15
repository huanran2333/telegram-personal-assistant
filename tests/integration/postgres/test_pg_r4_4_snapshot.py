"""E4: isolated PG18 consistent operational reads, not a Telegram E2E."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import delete, text, update

from app import main
from app.db.models import TelegramInboundReceipt
from app.runtime import RuntimeCapabilities
from app.telegram import operational_snapshot as observed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_backlog_over_100_and_restart_snapshot(pg_harness):
    factory, _, engine = pg_harness
    bot_identity = 10**15 + int(uuid4().hex[:10], 16)
    now = datetime.now(timezone.utc)
    baseline = await observed.messaging_snapshot(factory, now=now)
    try:
        async with factory() as db:
            db.add_all(TelegramInboundReceipt(
                bot_identity=bot_identity, update_id=index, envelope={"text": "PG_PRIVATE"},
                envelope_sha256="a" * 64, state="RECEIVED",
                received_at=now - timedelta(minutes=4)) for index in range(121))
            await db.commit()
        result = await observed.messaging_snapshot(factory, now=now)
        assert result["counts"]["receipts"]["RECEIVED"] >= (
            baseline["counts"]["receipts"]["RECEIVED"] + 121)
        assert result["oldest_due_age_seconds"] >= 240
        assert "PG_PRIVATE" not in json.dumps(result)

        # Recreate sessionmaker/engine: the status is derived from durable PG,
        # not a process-local counter or a recovery batch capped at 100.
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        fresh_engine = create_async_engine(engine.url, pool_pre_ping=True)
        try:
            fresh = async_sessionmaker(fresh_engine, expire_on_commit=False)
            recovered = await observed.messaging_snapshot(fresh, now=now)
            assert recovered["counts"]["receipts"]["RECEIVED"] >= (
                baseline["counts"]["receipts"]["RECEIVED"] + 121)
        finally:
            await fresh_engine.dispose()
    finally:
        async with factory() as db:
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.bot_identity == bot_identity))
            await db.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_concurrent_update_preserves_one_consistent_snapshot(pg_harness, monkeypatch):
    factory, _, _ = pg_harness
    bot_identity = 10**15 + int(uuid4().hex[:10], 16)
    receipt = TelegramInboundReceipt(
        bot_identity=bot_identity, update_id=1, envelope={}, envelope_sha256="b" * 64,
        state="RECEIVED", received_at=datetime(2000, 1, 1, tzinfo=timezone.utc))
    async with factory() as db:
        db.add(receipt)
        await db.commit()
    first_read = asyncio.Event()
    update_done = asyncio.Event()
    original = observed._state_counts

    async def pause_after_first_read(db, name, model):
        result = await original(db, name, model)
        if name == "receipts":
            first_read.set()
            await asyncio.wait_for(update_done.wait(), timeout=10)
        return result

    async def writer():
        await asyncio.wait_for(first_read.wait(), timeout=10)
        async with factory() as db:
            await db.execute(update(TelegramInboundReceipt).where(
                TelegramInboundReceipt.id == receipt.id).values(state="PROCESSED"))
            await db.commit()
        update_done.set()

    try:
        monkeypatch.setattr(observed, "_state_counts", pause_after_first_read)
        writer_task = asyncio.create_task(writer())
        result = await observed.messaging_snapshot(factory)
        await writer_task
        assert result["oldest_backlog_age_seconds"] > 20 * 365 * 24 * 3600
        assert result["counts"]["receipts"]["RECEIVED"] >= 1
    finally:
        monkeypatch.setattr(observed, "_state_counts", original)
        async with factory() as db:
            await db.execute(delete(TelegramInboundReceipt).where(
                TelegramInboundReceipt.bot_identity == bot_identity))
            await db.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pg_status_query_failure_does_not_return_sql_or_private_content(
    pg_harness, monkeypatch,
):
    factory, _, _ = pg_harness
    original = observed._state_counts

    async def invalid_pg_query(db, name, model):
        await db.execute(text("SELECT count(*) FROM r44_missing_snapshot_fault_table"))
        return await original(db, name, model)

    monkeypatch.setattr(observed, "_state_counts", invalid_pg_query)
    runtime = RuntimeCapabilities()
    for component in runtime.components:
        runtime.mark_ready(component)
    monkeypatch.setattr(main.app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(main, "AsyncSessionLocal", factory)
    response = await main.messaging_health()
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "dependency_not_ready", "runtime_ready": False,
        "error_category": "StatusQueryUnavailable"}
    assert "r44_missing_snapshot_fault_table" not in response.body.decode()
    monkeypatch.setattr(observed, "_state_counts", original)
    assert (await observed.messaging_snapshot(factory))["status"] in {
        "healthy", "delivery_backlogged", "delivery_degraded"}
