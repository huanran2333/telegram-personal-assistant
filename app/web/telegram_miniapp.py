"""Bot-scoped Mini App bootstrap; tickets bind launch, signed initData proves identity."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl

from fastapi import HTTPException
from sqlalchemy import delete, select

from app.db.models import TelegramMiniAppLaunchTicket, User, WebSession
from app.web.auth import _utc, digest, now

TICKET_TTL = timedelta(minutes=10)
INIT_DATA_MAX_AGE = timedelta(minutes=10)
FUTURE_SKEW = timedelta(seconds=30)


def _reject() -> HTTPException:
    # Identical public error for bad signature, unknown user, expired ticket, etc.
    return HTTPException(401, "MiniAppAuthenticationRequired")


def validate_init_data(raw: str, bot_token: str, bot_identity: int,
                       *, at: datetime | None = None) -> str:
    """Validate Telegram's bot-token HMAC over decoded, sorted raw initData fields."""
    if not raw or len(raw) > 16384 or not bot_token or bot_identity <= 0:
        raise _reject()
    # A configured token for a different bot must never validate a ticket
    # issued under this runtime's authenticated getMe() identity.
    if not bot_token.startswith(f"{bot_identity}:"):
        raise _reject()
    try:
        fields = parse_qsl(raw, keep_blank_values=True, strict_parsing=True,
                           encoding="utf-8", errors="strict", max_num_fields=64)
        if len({key for key, _ in fields}) != len(fields):
            raise ValueError("duplicate key")
        params = dict(fields)
        claimed = params.pop("hash")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", claimed):
            raise ValueError("bad hash")
        check = "\n".join(f"{key}={value}" for key, value in sorted(params.items()))
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        expected = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, claimed.lower()):
            raise ValueError("signature mismatch")
        timestamp = params["auth_date"]
        if not re.fullmatch(r"[0-9]{1,12}", timestamp):
            raise ValueError("invalid time")
        issued = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        current = at or now()
        if issued < current - INIT_DATA_MAX_AGE or issued > current + FUTURE_SKEW:
            raise ValueError("stale time")
        user = json.loads(params["user"])
        user_id = user["id"]
        if (not isinstance(user_id, int) or isinstance(user_id, bool)
                or not 0 < user_id <= (1 << 52) - 1):
            raise ValueError("invalid user")
        return str(user_id)
    except (ValueError, KeyError, TypeError, OverflowError, UnicodeError, json.JSONDecodeError):
        raise _reject() from None


async def issue_ticket(db, *, bot_identity: int, telegram_user_id: str) -> str:
    if bot_identity <= 0 or not re.fullmatch(r"[1-9][0-9]{0,15}", telegram_user_id):
        raise ValueError("InvalidMiniAppOwner")
    token = secrets.token_urlsafe(32)
    db.add(TelegramMiniAppLaunchTicket(
        token_hash=digest(token), bot_identity=bot_identity,
        telegram_user_id=telegram_user_id, expires_at=now() + TICKET_TTL))
    # Bounded opportunistic cleanup; never delete an unexpired authority row.
    old = (await db.scalars(select(TelegramMiniAppLaunchTicket.token_hash).where(
        TelegramMiniAppLaunchTicket.expires_at < now() - timedelta(days=1)
    ).limit(100).with_for_update(skip_locked=True))).all()
    if old:
        await db.execute(delete(TelegramMiniAppLaunchTicket).where(
            TelegramMiniAppLaunchTicket.token_hash.in_(old)))
    await db.commit()
    return token


async def exchange_ticket(db, *, ticket: str, raw_init_data: str,
                          bot_token: str, bot_identity: int,
                          session_hours: int,
                          prior_session_token: str = "") -> tuple[str, WebSession]:
    # The launch is deliberately reusable for ten minutes, but never proves
    # identity by itself: every exchange verifies fresh, Bot-bound Telegram
    # initData and the exact ticket owner under one row lock. Response loss or
    # a WebView cookie reset can safely establish a new Session in this window.
    telegram_user_id = validate_init_data(raw_init_data, bot_token, bot_identity)
    if not ticket or len(ticket) > 128:
        raise _reject()
    row = await db.scalar(select(TelegramMiniAppLaunchTicket).where(
        TelegramMiniAppLaunchTicket.token_hash == digest(ticket)).with_for_update())
    if (row is None or _utc(row.expires_at) <= now()
            or row.bot_identity != bot_identity
            or row.telegram_user_id != telegram_user_id):
        raise _reject()
    user = await db.scalar(select(User).where(User.telegram_id == telegram_user_id))
    if user is None:
        raise _reject()
    if prior_session_token:
        prior = await db.get(WebSession, digest(prior_session_token), with_for_update=True)
        if prior is not None and prior.revoked_at is None:
            prior.revoked_at = now()
    token = secrets.token_urlsafe(48)
    session = WebSession(token_hash=digest(token), user_id=user.id,
                         credential_version=0, auth_source="telegram",
                         telegram_bot_identity=bot_identity,
                         telegram_user_id=telegram_user_id,
                         csrf_token=secrets.token_urlsafe(32),
                         expires_at=now() + timedelta(hours=min(session_hours, 8)))
    if row.consumed_at is None:
        row.consumed_at = now()  # First redemption audit, not a one-use fence.
    db.add(session)
    await db.commit()
    return token, session


async def verify_session_launch(db, *, ticket: str, telegram_user_id: str,
                                bot_identity: int) -> None:
    """A live Session may re-enter only while this exact launch is valid."""
    if not ticket or len(ticket) > 128:
        raise _reject()
    row = await db.get(TelegramMiniAppLaunchTicket, digest(ticket))
    if (row is None or _utc(row.expires_at) <= now()
            or row.telegram_user_id != telegram_user_id
            or row.bot_identity != bot_identity):
        raise _reject()
