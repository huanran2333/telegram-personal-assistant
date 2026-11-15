"""E3 read-only custody observation; SQLite here is not PG concurrency proof."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import (Base, OutboundDelivery, OutboundDeliveryChunk,
                           ScheduledOccurrence, TelegramInboundReceipt, User)
from app.main import app, messaging_health
from app.runtime import RuntimeCapabilities
from app.telegram.operational_snapshot import messaging_snapshot


@pytest.fixture
async def snapshot_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync: Base.metadata.create_all(
            sync, tables=[User.__table__, TelegramInboundReceipt.__table__,
                          ScheduledOccurrence.__table__, OutboundDelivery.__table__,
                          OutboundDeliveryChunk.__table__]))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_more_than_one_recovery_batch_is_counted_without_content(snapshot_db, monkeypatch):
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    async with snapshot_db() as db:
        db.add_all(TelegramInboundReceipt(
            bot_identity=99, update_id=1000 + index,
            envelope={"text": "PRIVATE_SYNTHETIC_BODY"}, envelope_sha256="a" * 64,
            state="RECEIVED", received_at=now - timedelta(minutes=10))
            for index in range(121))
        db.add(TelegramInboundReceipt(
            bot_identity=99, update_id=2000, envelope={}, envelope_sha256="b" * 64,
            state="DEAD", failure_category="PRIVATE_PERSON_NAME", received_at=now))
        db.add(TelegramInboundReceipt(
            bot_identity=99, update_id=3000, envelope={}, envelope_sha256="c" * 64,
            state="PROCESSING", lease_until=now + timedelta(minutes=1), received_at=now))
        await db.commit()

    result = await messaging_snapshot(snapshot_db, now=now)
    assert result["counts"]["receipts"]["RECEIVED"] == 121
    assert result["counts"]["receipts"]["DEAD"] == 1
    assert result["status_counts"]["pending"]["receipts"] == 121
    assert result["status_counts"]["processing"]["receipts"] == 1
    assert result["status_counts"]["dead"]["receipts"] == 1
    assert result["safe_failure_categories"]["receipts"] == {"Other": 1}
    assert result["oldest_due_age_seconds"] == 600
    assert result["status"] == "delivery_degraded"
    assert "PRIVATE" not in json.dumps(result)

    runtime = RuntimeCapabilities()
    for component in runtime.components:
        runtime.mark_ready(component)
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr("app.main.AsyncSessionLocal", snapshot_db)
    response = await messaging_health()
    assert response.status_code == 200
    assert b"PRIVATE" not in response.body
    assert json.loads(response.body)["messaging"]["counts"]["receipts"]["RECEIVED"] == 121
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://localhost") as client:
        public_response = await client.get("/health/messaging")
    assert public_response.status_code == 200
    assert "PRIVATE" not in public_response.text


@pytest.mark.asyncio
async def test_backlog_and_uncertainty_are_distinct_from_dependency_failure(snapshot_db):
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    async with snapshot_db() as db:
        receipt = TelegramInboundReceipt(bot_identity=1, update_id=1, envelope={},
                                         envelope_sha256="a" * 64, state="PROCESSED")
        db.add(receipt)
        await db.flush()
        delivery = OutboundDelivery(receipt_id=receipt.id, reply_ordinal=0,
                                    kind="interactive_reply", chat_id="PRIVATE_CHAT_ID",
                                    locale="zh-CN", text="PRIVATE_REPLY", payload_sha256="b" * 64,
                                    state="UNCERTAIN", created_at=now - timedelta(seconds=30))
        db.add(delivery)
        await db.flush()
        db.add(OutboundDeliveryChunk(delivery_id=delivery.id, chunk_index=0,
                                     text="PRIVATE_REPLY", content_sha256="b" * 64,
                                     state="UNCERTAIN", attempts=1,
                                     next_attempt_at=now - timedelta(seconds=1),
                                     failure_category="TimeoutError"))
        exhausted = OutboundDelivery(receipt_id=receipt.id, reply_ordinal=1,
                                     kind="interactive_reply", chat_id="PRIVATE_CHAT_ID",
                                     locale="zh-CN", text="PRIVATE_OTHER_REPLY",
                                     payload_sha256="c" * 64, state="EXHAUSTED")
        db.add(exhausted)
        await db.flush()
        db.add(OutboundDeliveryChunk(delivery_id=exhausted.id, chunk_index=0,
                                     text="PRIVATE_OTHER_REPLY", content_sha256="c" * 64,
                                     state="EXHAUSTED", attempts=5))
        await db.commit()
    result = await messaging_snapshot(snapshot_db, now=now)
    assert result["status"] == "delivery_degraded"
    assert result["counts"]["deliveries"]["UNCERTAIN"] == 1
    assert result["counts"]["chunks"]["UNCERTAIN"] == 1
    assert result["status_counts"]["uncertain"] == {"deliveries": 1, "chunks": 1}
    assert result["status_counts"]["exhausted"] == {
        "occurrences": 0, "deliveries": 1, "chunks": 1}
    assert result["oldest_due_age_seconds"] == 30
    assert result["safe_failure_categories"]["chunks"] == {"TimeoutError": 1}
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.asyncio
async def test_status_query_failure_is_503_without_private_exception(monkeypatch, caplog):
    runtime = RuntimeCapabilities()
    for component in runtime.components:
        runtime.mark_ready(component)
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)

    def unavailable_factory():
        raise RuntimeError("PRIVATE_DATABASE_URL_AND_PASSWORD")

    monkeypatch.setattr("app.main.AsyncSessionLocal", unavailable_factory)
    response = await messaging_health()
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "dependency_not_ready", "runtime_ready": False,
        "error_category": "StatusQueryUnavailable"}
    assert "PRIVATE" not in caplog.text + response.body.decode()


@pytest.mark.asyncio
async def test_static_r1_failure_skips_status_query(monkeypatch):
    monkeypatch.setattr(app.state, "runtime", RuntimeCapabilities(), raising=False)

    def must_not_open_db():
        raise AssertionError("status query should not run while dependency not ready")

    monkeypatch.setattr("app.main.AsyncSessionLocal", must_not_open_db)
    response = await messaging_health()
    assert response.status_code == 503
    assert json.loads(response.body)["status"] == "dependency_not_ready"
