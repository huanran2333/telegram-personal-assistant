"""G1 real isolated PG18 launch/session concurrency and migration checks."""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import urlencode
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text

from app.db.models import TelegramMiniAppLaunchTicket, User, WebSession, WebCredential
from app.db.migrations.v3_miniapp_g1 import apply as migrate
from app.web.auth import digest, principal
from app.web.telegram_miniapp import exchange_ticket, issue_ticket

pytestmark = pytest.mark.integration
BOT_ID = 123456
TOKEN = "123456:synthetic-test-token"


def signed(telegram_id: int) -> str:
    fields = {"auth_date": str(int(datetime.now(timezone.utc).timestamp())),
              "query_id": "AAE-synthetic",
              "user": json.dumps({"id": telegram_id}, separators=(",", ":"))}
    secret = hmac.new(b"WebAppData", TOKEN.encode(), hashlib.sha256).digest()
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


@pytest.mark.asyncio
async def test_ticket_two_workers_restart_and_password_coexist(pg_harness):
    factory, new_user, engine = pg_harness
    user = await new_user()
    telegram_id = str(800_000_000_000 + int(uuid4().hex[:6], 16))
    async with factory() as db:
        row = await db.get(User, user.id)
        row.telegram_id = telegram_id
        await db.commit()
        db.add(WebCredential(user_id=user.id, password_hash="synthetic-unusable", version=1))
        await db.commit()
    async with factory() as db:
        ticket = await issue_ticket(db, bot_identity=BOT_ID, telegram_user_id=telegram_id)
        stored = await db.get(TelegramMiniAppLaunchTicket, hashlib.sha256(ticket.encode()).hexdigest())
        assert stored is not None and ticket not in stored.token_hash
        issued_expiry = stored.expires_at
        assert timedelta(minutes=9, seconds=59) <= stored.expires_at - stored.created_at <= timedelta(minutes=10, seconds=1)
    raw = signed(int(telegram_id))

    async def claim():
        async with factory() as db:
            try:
                return await exchange_ticket(db, ticket=ticket, raw_init_data=raw,
                                             bot_token=TOKEN, bot_identity=BOT_ID,
                                             session_hours=24)
            except HTTPException:
                await db.rollback()
                return None

    first, second = await asyncio.gather(claim(), claim())
    assert first is not None and second is not None
    assert first[0] != second[0]
    token, session = first
    async with factory() as db:
        rows = (await db.scalars(select(WebSession).where(
            WebSession.user_id == user.id, WebSession.auth_source == "telegram"))).all()
        assert len(rows) == 2  # Two same-owner launches, serialized by the Ticket lock.
        unchanged = await db.get(TelegramMiniAppLaunchTicket, hashlib.sha256(ticket.encode()).hexdigest())
        assert unchanged.expires_at == issued_expiry  # Re-entry never slides the original deadline.
        owner, _ = await principal(db, token, bot_identity=BOT_ID)
        assert owner.id == user.id
        password_token = uuid4().hex
        db.add(WebSession(token_hash=digest(password_token), user_id=user.id,
                          credential_version=1, auth_source="password",
                          csrf_token=uuid4().hex,
                          expires_at=datetime.now(timezone.utc).replace(year=2027)))
        await db.commit()
        browser_owner, _ = await principal(db, password_token)
        assert browser_owner.id == user.id
        next_ticket = await issue_ticket(db, bot_identity=BOT_ID,
                                         telegram_user_id=telegram_id)
    async with factory() as db:
        replacement_token, _ = await exchange_ticket(
            db, ticket=next_ticket, raw_init_data=raw,
            bot_token=TOKEN, bot_identity=BOT_ID, session_hours=24,
            prior_session_token=password_token)
    async with factory() as db:
        prior = await db.get(WebSession, digest(password_token))
        assert prior.revoked_at is not None
        with pytest.raises(HTTPException):
            await principal(db, password_token)
        replacement_owner, _ = await principal(db, replacement_token,
                                                bot_identity=BOT_ID)
        assert replacement_owner.id == user.id
    # A fresh factory/transaction may re-open the same button while valid.
    async with factory() as db:
        third, _ = await exchange_ticket(db, ticket=ticket, raw_init_data=raw,
                                         bot_token=TOKEN, bot_identity=BOT_ID,
                                         session_hours=24)
        assert (await principal(db, third, bot_identity=BOT_ID))[0].id == user.id


@pytest.mark.asyncio
async def test_upgrade_migration_idempotent_and_existing_rows_preserved(pg_harness):
    factory, new_user, engine = pg_harness
    user = await new_user()
    async with engine.begin() as connection:
        await connection.run_sync(migrate)
        await connection.run_sync(migrate)
        columns = (await connection.execute(text("""SELECT column_name FROM information_schema.columns
            WHERE table_name = 'web_sessions' AND column_name IN
            ('auth_source','telegram_bot_identity','telegram_user_id')"""))).scalars().all()
        assert set(columns) == {"auth_source", "telegram_bot_identity", "telegram_user_id"}
    async with factory() as db:
        assert await db.get(User, user.id) is not None


@pytest.mark.asyncio
async def test_ticket_rollback_expiry_and_wrong_bot_do_not_consume(pg_harness):
    factory, new_user, _ = pg_harness
    user = await new_user()
    telegram_id = str(900_000_000_000 + int(uuid4().hex[:6], 16))
    async with factory() as db:
        row = await db.get(User, user.id)
        row.telegram_id = telegram_id
        await db.commit()
        ticket = await issue_ticket(db, bot_identity=BOT_ID, telegram_user_id=telegram_id)
    key = hashlib.sha256(ticket.encode()).hexdigest()
    async with factory() as db:
        row = await db.get(TelegramMiniAppLaunchTicket, key, with_for_update=True)
        row.consumed_at = datetime.now(timezone.utc)
        await db.flush()
        await db.rollback()  # interrupted transaction cannot spend a ticket
    raw = signed(int(telegram_id))
    async with factory() as db:
        with pytest.raises(HTTPException):
            await exchange_ticket(db, ticket=ticket, raw_init_data=raw,
                                  bot_token=TOKEN, bot_identity=BOT_ID + 1,
                                  session_hours=8)
        await db.rollback()
        token, _ = await exchange_ticket(db, ticket=ticket, raw_init_data=raw,
                                         bot_token=TOKEN, bot_identity=BOT_ID,
                                         session_hours=8)
        assert (await principal(db, token, bot_identity=BOT_ID))[0].id == user.id
        expired = await issue_ticket(db, bot_identity=BOT_ID, telegram_user_id=telegram_id)
        row = await db.get(TelegramMiniAppLaunchTicket,
                           hashlib.sha256(expired.encode()).hexdigest())
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        await db.commit()
        with pytest.raises(HTTPException):
            await exchange_ticket(db, ticket=expired, raw_init_data=raw,
                                  bot_token=TOKEN, bot_identity=BOT_ID,
                                  session_hours=8)


@pytest.mark.asyncio
async def test_miniapp_additive_upgrade_from_v33_session_shape(pg_harness):
    _, _, engine = pg_harness
    schema = "g1_upgrade_" + uuid4().hex[:16]
    async with engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'SET LOCAL search_path = "{schema}"'))
            await connection.execute(text(
                "CREATE TABLE users (id VARCHAR(36) PRIMARY KEY, telegram_id VARCHAR(64) UNIQUE NOT NULL)"))
            await connection.execute(text(
                "CREATE TABLE app_schema_migrations (revision VARCHAR(128) PRIMARY KEY)"))
            await connection.execute(text(
                "CREATE TABLE web_sessions (token_hash VARCHAR(64) PRIMARY KEY, "
                "user_id VARCHAR(36) NOT NULL REFERENCES users(id), credential_version INTEGER NOT NULL, "
                "csrf_token VARCHAR(64) NOT NULL, expires_at TIMESTAMPTZ NOT NULL, "
                "revoked_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"))
            await connection.execute(text(
                "INSERT INTO users (id, telegram_id) VALUES ('g1-old-user', 'g1-old-telegram')"))
            await connection.execute(text(
                "INSERT INTO web_sessions (token_hash,user_id,credential_version,csrf_token,expires_at) "
                "VALUES ('g1-old-token','g1-old-user',1,'g1-old-csrf',CURRENT_TIMESTAMP + INTERVAL '1 day')"))
            await connection.run_sync(migrate)
            await connection.run_sync(migrate)
            assert await connection.scalar(text(
                "SELECT auth_source FROM web_sessions WHERE token_hash='g1-old-token'")) == "password"
            assert await connection.scalar(text(
                "SELECT count(*) FROM telegram_miniapp_launch_tickets")) == 0
            assert await connection.scalar(text(
                "SELECT count(*) FROM users WHERE id='g1-old-user'")) == 1
            assert await connection.scalar(text(
                "SELECT count(*) FROM pg_constraint WHERE conrelid='web_sessions'::regclass "
                "AND conname='ck_web_session_auth_source'")) == 1
        finally:
            await transaction.rollback()
