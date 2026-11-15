"""G1 deterministic protocol, session, and reply-surface checks; no real Bot."""

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from fastapi import HTTPException
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from telegram import Bot, InlineKeyboardMarkup

from app.db.models import TelegramInboundReceipt, TelegramMiniAppLaunchTicket, User, WebSession
from app.telegram import sender
from app.web.auth import SESSION_COOKIE, digest, principal
from app.web.telegram_miniapp import (exchange_ticket, issue_ticket,
                                      validate_init_data)

BOT_TOKEN = "123456:synthetic-test-token"
BOT_ID = 123456


def signed_data(user_id: int, when: datetime, *, token: str = BOT_TOKEN,
                extra: dict | None = None) -> str:
    fields = {"auth_date": str(int(when.timestamp())), "query_id": "AAE-synthetic",
              "user": json.dumps({"id": user_id, "first_name": "Fixture"}, separators=(",", ":")),
              **(extra or {})}
    # Independent test construction following the official sorted data-check-string.
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def test_telegram_spec_bot_token_hmac_and_malformed_input():
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    raw = signed_data(987654, now)
    assert validate_init_data(raw, BOT_TOKEN, BOT_ID, at=now) == "987654"
    assert validate_init_data("&".join(reversed(raw.split("&"))), BOT_TOKEN, BOT_ID,
                              at=now) == "987654"
    assert validate_init_data(signed_data(987654, now, extra={"start_param": "a b&c"}),
                              BOT_TOKEN, BOT_ID, at=now) == "987654"
    assert validate_init_data(signed_data(987654, now - timedelta(minutes=9)),
                              BOT_TOKEN, BOT_ID, at=now) == "987654"
    for invalid in (
        raw.replace("Fixture", "Mallory"), raw + "&auth_date=1",
        raw.replace("hash=", "hash=bad"),
        signed_data(987654, now, token="123456:other-token"),
        signed_data(987654, now - timedelta(minutes=11)),
        signed_data(987654, now + timedelta(minutes=1)),
        signed_data(987654, now, extra={"user": "not-json"}),
        "hash=x&auth_date=1&user=%7B%7D",
    ):
        with pytest.raises(HTTPException):
            validate_init_data(invalid, BOT_TOKEN, BOT_ID, at=now)
    with pytest.raises(HTTPException):
        validate_init_data(raw, BOT_TOKEN, BOT_ID + 1, at=now)


@pytest.mark.asyncio
async def test_ticket_session_owner_logout_and_replay(session_factory):
    factory = session_factory
    async with factory() as db:
        a = User(telegram_id="987654")
        b = User(telegram_id="111111")
        db.add_all([a, b])
        await db.commit()
        a_id, b_id = a.id, b.id
    async with factory() as db:
        ticket = await issue_ticket(db, bot_identity=BOT_ID, telegram_user_id="987654")
        row = await db.scalar(select(WebSession).limit(1))
        assert row is None
    assert ticket not in repr(row)
    raw = signed_data(987654, datetime.now(timezone.utc))
    async with factory() as db:
        with pytest.raises(HTTPException):
            await exchange_ticket(db, ticket=ticket,
                                  raw_init_data=signed_data(111111, datetime.now(timezone.utc)),
                                  bot_token=BOT_TOKEN, bot_identity=BOT_ID, session_hours=24)
        await db.rollback()
    async with factory() as db:
        token, session = await exchange_ticket(db, ticket=ticket, raw_init_data=raw,
                                               bot_token=BOT_TOKEN, bot_identity=BOT_ID,
                                               session_hours=24)
        assert session.user_id == a_id and session.auth_source == "telegram"
        assert session.credential_version == 0
        assert session.expires_at <= datetime.now(timezone.utc) + timedelta(hours=8)
    async with factory() as db:
        user, _ = await principal(db, token, bot_identity=BOT_ID)
        assert user.id == a_id and user.id != b_id
        for wrong in (None, BOT_ID + 1):
            with pytest.raises(HTTPException):
                await principal(db, token, bot_identity=wrong)
        reopened_token, reopened = await exchange_ticket(
            db, ticket=ticket, raw_init_data=raw,
            bot_token=BOT_TOKEN, bot_identity=BOT_ID, session_hours=24)
        assert reopened.user_id == a_id and reopened.token_hash != session.token_hash
        assert (await principal(db, reopened_token, bot_identity=BOT_ID))[0].id == a_id
    async with factory() as db:
        user = await db.get(User, a_id)
        user.telegram_id = "222222"
        await db.commit()
        with pytest.raises(HTTPException):
            await principal(db, token, bot_identity=BOT_ID)
        row = await db.get(WebSession, session.token_hash)
        row.revoked_at = datetime.now(timezone.utc)
        await db.commit()
        with pytest.raises(HTTPException):
            await principal(db, token, bot_identity=BOT_ID)


@pytest.mark.asyncio
async def test_workbench_keyboard_is_real_web_app_and_uses_r4_queue(monkeypatch):
    seen = {}

    async def queue(chat_id, text, *, kind, keyboard):
        seen.update(chat_id=chat_id, kind=kind, keyboard=keyboard.to_dict())
        return True

    monkeypatch.setattr(sender, "_queue_interactive", queue)
    await sender.send_workbench_launch(987654, "https://workbench.example/workbench/#launchTicket=synthetic")
    button = seen["keyboard"]["inline_keyboard"][0][0]
    assert seen["kind"] == "workbench_launch"
    assert button["web_app"]["url"].startswith("https://workbench.example/")
    assert "url" not in button and "callback_data" not in button
    # R4 persists JSON and later reconstructs PTB markup at dispatch time.
    restored = InlineKeyboardMarkup.de_json(seen["keyboard"], Bot(token=BOT_TOKEN))
    assert restored.inline_keyboard[0][0].web_app.url == button["web_app"]["url"]


@pytest.mark.asyncio
async def test_workbench_launch_never_falls_back_to_direct_send(monkeypatch):
    from app.telegram.durable import DurableSafetyError

    async def no_receipt(*_args, **_kwargs):
        return False

    monkeypatch.setattr(sender, "_queue_interactive", no_receipt)
    monkeypatch.setattr(sender, "get_bot", lambda: (_ for _ in ()).throw(
        AssertionError("Direct Bot API send is forbidden for launch")))
    with pytest.raises(DurableSafetyError, match="WorkbenchLaunchRequiresDurableReceipt"):
        await sender.send_workbench_launch(
            987654, "https://workbench.example/workbench/#launchTicket=synthetic")


@pytest.mark.asyncio
async def test_private_workbench_command_issues_bound_ticket_under_receipt(session_factory, monkeypatch):
    from telegram import Bot
    from app.telegram import webhook
    from app.telegram.processing_context import receipt_scope
    from app.localization import ResponsePreferences

    factory = session_factory
    async with factory() as db:
        db.add(User(telegram_id="987654"))
        receipt = TelegramInboundReceipt(bot_identity=BOT_ID, update_id=881,
                                         state="PROCESSING", attempts=1)
        db.add(receipt)
        await db.commit()
        receipt_id = receipt.id
    monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
    monkeypatch.setattr(webhook, "get_bot", lambda: Bot(token=BOT_TOKEN))

    async def preferences(*_args, **_kwargs):
        return ResponsePreferences()

    monkeypatch.setattr(webhook, "load_preferences", preferences)
    monkeypatch.setattr(webhook, "get_settings", lambda: SimpleNamespace(
        web_enabled=True, web_public_origin="https://workbench.example",
        secret_key="synthetic-non-default"))
    delivered = []

    async def send_launch(chat_id, url):
        delivered.append((chat_id, url))

    monkeypatch.setattr(webhook, "send_workbench_launch", send_launch)
    envelope = {"update_id": 881, "message": {
        "message_id": 4, "date": 0,
        "from": {"id": 987654, "is_bot": False, "first_name": "Fixture"},
        "chat": {"id": 987654, "type": "private"}, "text": "/workbench"}}
    with receipt_scope(receipt_id, "synthetic-fence") as run:
        assert await webhook.process_admitted_update(envelope) == {"ok": True}
        assert run.outcome_kind == "no_business"
    assert len(delivered) == 1 and delivered[0][0] == 987654
    assert "/workbench/?view=" in delivered[0][1]
    assert "#launchTicket=" in delivered[0][1]
    ticket = delivered[0][1].split("launchTicket=", 1)[1]
    async with factory() as db:
        row = await db.get(TelegramMiniAppLaunchTicket, hashlib.sha256(ticket.encode()).hexdigest())
        assert row is not None and row.bot_identity == BOT_ID
        assert row.telegram_user_id == "987654" and row.consumed_at is None
    denied = []

    async def denial(chat_id, text, *_args, **_kwargs):
        denied.append((chat_id, text))

    monkeypatch.setattr(webhook, "send_message", denial)
    group_envelope = {"update_id": 882, "message": {
        "message_id": 5, "date": 0,
        "from": {"id": 987654, "is_bot": False, "first_name": "Fixture"},
        "chat": {"id": -100222, "type": "group"}, "text": "/workbench"}}
    assert await webhook.process_admitted_update(group_envelope) == {"ok": True}
    assert len(delivered) == 1 and len(denied) == 1


@pytest.mark.asyncio
async def test_exchange_api_keeps_origin_csrf_cookie_and_owner(session_factory, monkeypatch):
    from app.web import api as web_api

    factory = session_factory
    async with factory() as db:
        user = User(telegram_id="987654", first_name="Fixture")
        db.add(user)
        await db.commit()
        prior_token = "synthetic-prior-session"
        prior = WebSession(token_hash=digest(prior_token), user_id=user.id,
                           credential_version=0, auth_source="telegram",
                           telegram_bot_identity=BOT_ID, telegram_user_id="987654",
                           csrf_token="synthetic-prior-csrf",
                           expires_at=datetime.now(timezone.utc) + timedelta(hours=1))
        db.add(prior)
        await db.commit()
        ticket = await issue_ticket(db, bot_identity=BOT_ID, telegram_user_id="987654")
    settings = SimpleNamespace(web_public_origin="https://workbench.example",
                               telegram_bot_token=BOT_TOKEN, web_session_hours=24,
                               default_timezone="UTC", google_calendar_enabled=False,
                               gmail_enabled=False, memory_enabled=False)
    monkeypatch.setattr(web_api, "AsyncSessionLocal", factory)
    monkeypatch.setattr(web_api, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(web_api.router)
    app.state.web_ready = True
    app.state.web_worker = SimpleNamespace(accepting=True)
    app.state.telegram_durable = SimpleNamespace(bot_identity=BOT_ID, is_running=True)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="https://workbench.example") as client:
        client.cookies.set(SESSION_COOKIE, prior_token)
        bootstrap = await client.get("/api/v3/auth/bootstrap",
                                     headers={"sec-fetch-site": "same-origin"})
        csrf = bootstrap.json()["csrf_token"]
        body = {"init_data": signed_data(987654, datetime.now(timezone.utc)),
                "launch_ticket": ticket}
        wrong_origin = await client.post("/api/v3/auth/telegram/exchange", json=body,
                                         headers={"origin": "https://other.example",
                                                  "x-csrf-token": csrf})
        assert wrong_origin.status_code == 403
        wrong_csrf = await client.post("/api/v3/auth/telegram/exchange", json=body,
                                      headers={"origin": settings.web_public_origin,
                                               "x-csrf-token": "wrong"})
        assert wrong_csrf.status_code == 403
        accepted = await client.post("/api/v3/auth/telegram/exchange", json=body,
                                     headers={"origin": settings.web_public_origin,
                                              "x-csrf-token": csrf})
        assert accepted.status_code == 204
        assert "Secure" in accepted.headers["set-cookie"]
        assert "HttpOnly" in accepted.headers["set-cookie"]
        assert "SameSite=strict" in accepted.headers["set-cookie"]
        owner = await client.get("/api/v3/me")
        assert owner.status_code == 200
        assert owner.json()["display_label"] == "Fixture"
        assert owner.json()["auth_source"] == "telegram"
        # A live session reopens without rotation, but only under this same
        # unexpired launch and a freshly signed matching Telegram identity.
        session_body = {"init_data": signed_data(987654, datetime.now(timezone.utc)),
                        "launch_ticket": ticket}
        session_headers = {"origin": settings.web_public_origin,
                           "x-csrf-token": owner.json()["csrf_token"]}
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json=session_body, headers=session_headers)).status_code == 204
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json={**session_body,
                                        "init_data": signed_data(987655, datetime.now(timezone.utc))},
                                  headers=session_headers)).status_code == 401
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json={**session_body, "launch_ticket": "invalid-ticket"},
                                  headers=session_headers)).status_code == 401
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json=session_body, headers={**session_headers,
                                                              "origin": "https://other.example"})).status_code == 403
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json=session_body, headers={**session_headers,
                                                              "x-csrf-token": "wrong"})).status_code == 403
        # A lost WebView cookie can obtain another session during the same
        # ten-minute window; the Ticket alone still never authenticates.
        client.cookies.delete(SESSION_COOKIE)
        nonce = await client.get("/api/v3/auth/bootstrap",
                                 headers={"sec-fetch-site": "same-origin"})
        reopened = await client.post("/api/v3/auth/telegram/exchange", json=body,
                                     headers={"origin": settings.web_public_origin,
                                              "x-csrf-token": nonce.json()["csrf_token"]})
        assert reopened.status_code == 204
        reopened_owner = (await client.get("/api/v3/me")).json()
        assert reopened_owner["auth_source"] == "telegram"
        async with factory() as db:
            old = await db.get(WebSession, digest(prior_token))
            assert old.revoked_at is not None
            with pytest.raises(HTTPException):
                await principal(db, prior_token, bot_identity=BOT_ID)
            row = await db.get(TelegramMiniAppLaunchTicket, digest(ticket))
            row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await db.commit()
        # Expiry closes this launch, not the independently issued WebSession.
        assert (await client.get("/api/v3/me")).status_code == 200
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json=session_body,
                                  headers={"origin": settings.web_public_origin,
                                           "x-csrf-token": reopened_owner["csrf_token"]})).status_code == 401
        async with factory() as db:
            another_ticket = await issue_ticket(db, bot_identity=BOT_ID,
                                                telegram_user_id="987654")
        logout = await client.post("/api/v3/auth/logout",
                                   headers={"origin": settings.web_public_origin,
                                            "x-csrf-token": reopened_owner["csrf_token"]})
        assert logout.status_code == 204
        async with factory() as db:
            revoked_launch = await db.get(TelegramMiniAppLaunchTicket, digest(another_ticket))
            assert revoked_launch.expires_at.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc)
        assert (await client.get("/api/v3/me")).status_code == 401
        assert (await client.post("/api/v3/auth/telegram/revalidate",
                                  json=session_body, headers=session_headers)).status_code == 401
        nonce = await client.get("/api/v3/auth/bootstrap",
                                 headers={"sec-fetch-site": "same-origin"})
        assert (await client.post("/api/v3/auth/telegram/exchange", json=body,
                                  headers={"origin": settings.web_public_origin,
                                           "x-csrf-token": nonce.json()["csrf_token"]})).status_code == 401
