"""Operator-bound Argon2id credentials and PostgreSQL-backed browser sessions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from cryptography.exceptions import InvalidKey
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import User, WebCredential, WebLoginNonce, WebLoginThrottle, WebSession

SESSION_COOKIE = "__Host-pa_session"
NONCE_COOKIE = "__Host-pa_login"


def now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def password_hash(password: str) -> str:
    if not 12 <= len(password) <= 1024:
        raise ValueError("Password length must be 12–1024 characters")
    return Argon2id(salt=secrets.token_bytes(16), length=32, iterations=3,
                    lanes=4, memory_cost=65536).derive_phc_encoded(password.encode("utf-8"))


def verify_password(password: str, encoded: str) -> bool:
    try:
        Argon2id.verify_phc_encoded(password.encode("utf-8"), encoded)
        return True
    except (InvalidKey, ValueError):
        return False


def validate_web_configuration(settings) -> None:
    origin = urlparse(settings.web_public_origin)
    if (not settings.web_enabled or origin.scheme != "https" or not origin.netloc
            or origin.path not in {"", "/"} or origin.query or origin.fragment
            or settings.secret_key == "change_me_in_production"):
        raise ValueError("Web HTTPS origin or application secret is not configured")


def require_origin(request: Request, settings) -> None:
    origin = request.headers.get("origin", "")
    if not hmac.compare_digest(origin, settings.web_public_origin.rstrip("/")):
        raise HTTPException(403, "OriginRejected")


def require_bootstrap_origin(request: Request, settings) -> None:
    """Same-origin GET fetches may omit Origin; reject cross-site metadata."""
    origin = request.headers.get("origin")
    if origin:
        require_origin(request, settings)
        return
    fetch_site = request.headers.get("sec-fetch-site", "")
    referer = request.headers.get("referer", "")
    if fetch_site != "same-origin" or (referer and not referer.startswith(
            settings.web_public_origin.rstrip("/") + "/")):
        raise HTTPException(403, "OriginRejected")


def no_store(response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _client_key(request: Request, settings) -> str:
    # Trust the connection peer, never a browser-controlled X-Forwarded-For.
    peer = request.client.host if request.client else "unknown"
    return hmac.new(settings.secret_key.encode(), peer.encode(), hashlib.sha256).hexdigest()


async def issue_login_nonce(db) -> tuple[str, str]:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    db.add(WebLoginNonce(token_hash=digest(token), csrf_token=csrf,
                         expires_at=now() + timedelta(minutes=10)))
    await db.commit()
    return token, csrf


async def consume_login_nonce(db, token: str, csrf: str) -> None:
    row = await db.scalar(select(WebLoginNonce).where(
        WebLoginNonce.token_hash == digest(token)).with_for_update()) if token else None
    if (row is None or _utc(row.expires_at) <= now() or
            not hmac.compare_digest(row.csrf_token, csrf)):
        raise HTTPException(403, "LoginCsrfRejected")
    await db.delete(row)
    await db.flush()


async def login(db, request: Request, settings, password: str) -> tuple[str, WebSession]:
    key = _client_key(request, settings)
    await db.execute(pg_insert(WebLoginThrottle).values(
        client_hash=key, attempts=0, window_started_at=now()).on_conflict_do_nothing())
    throttle = await db.scalar(select(WebLoginThrottle).where(
        WebLoginThrottle.client_hash == key).with_for_update())
    if throttle.blocked_until and _utc(throttle.blocked_until) > now():
        raise HTTPException(429, "LoginRateLimited")
    if _utc(throttle.window_started_at) < now() - timedelta(minutes=15):
        throttle.window_started_at, throttle.attempts, throttle.blocked_until = now(), 0, None
    # The personal workbench has a bounded local credential set. The browser
    # supplies a password, never a user_id or Telegram identity.
    credentials = (await db.scalars(select(WebCredential).limit(11))).all()
    if len(credentials) > 10:
        raise HTTPException(503, "CredentialSetUnavailable")
    match = None
    for row in credentials:
        if await asyncio.to_thread(verify_password, password, row.password_hash):
            match = row
            break
    if match is None:
        throttle.attempts += 1
        if throttle.attempts >= 5:
            throttle.blocked_until = now() + timedelta(minutes=15)
        await db.commit()
        raise HTTPException(401, "InvalidCredentials")
    throttle.attempts, throttle.blocked_until = 0, None
    token = secrets.token_urlsafe(48)
    session = WebSession(token_hash=digest(token), user_id=match.user_id,
                         credential_version=match.version,
                         csrf_token=secrets.token_urlsafe(32),
                         expires_at=now() + timedelta(hours=settings.web_session_hours))
    db.add(session)
    await db.commit()
    return token, session


async def principal(db, token: str, *, bot_identity: int | None = None) -> tuple[User, WebSession]:
    session = await db.get(WebSession, digest(token)) if token else None
    if session is None or session.revoked_at or _utc(session.expires_at) <= now():
        raise HTTPException(401, "AuthenticationRequired")
    user = await db.get(User, session.user_id)
    if user is None:
        raise HTTPException(401, "AuthenticationRequired")
    if session.auth_source == "password":
        credential = await db.get(WebCredential, session.user_id)
        if (credential is None or credential.version != session.credential_version
                or session.telegram_bot_identity is not None or session.telegram_user_id is not None):
            raise HTTPException(401, "AuthenticationRequired")
    elif session.auth_source == "telegram":
        if (bot_identity is None or session.telegram_bot_identity != bot_identity
                or session.telegram_user_id != user.telegram_id
                or session.credential_version != 0):
            raise HTTPException(401, "AuthenticationRequired")
    else:
        raise HTTPException(401, "AuthenticationRequired")
    return user, session


def require_csrf(request: Request, session: WebSession, settings) -> None:
    require_origin(request, settings)
    token = request.headers.get("x-csrf-token", "")
    if not token or not hmac.compare_digest(token, session.csrf_token):
        raise HTTPException(403, "CsrfRejected")


async def bind_existing_user(db, user_id: str, password: str, *, reset: bool = False) -> None:
    user = await db.get(User, user_id)
    if user is None or not user.telegram_id:
        raise ValueError("An existing Telegram-backed User.id is required")
    credential = await db.get(WebCredential, user_id, with_for_update=True)
    if credential and not reset:
        raise ValueError("Web credential already exists; use explicit reset")
    # Password-only login is intentionally a single-user UX, but tests and
    # future operator bindings may create a second principal. Never let one
    # password ambiguously authenticate two existing User IDs.
    others = (await db.scalars(select(WebCredential).where(
        WebCredential.user_id != user_id).limit(11))).all()
    if len(others) > 10 or any(verify_password(password, item.password_hash) for item in others):
        raise ValueError("Web password must uniquely identify one bound user")
    encoded = password_hash(password)
    if credential is None:
        db.add(WebCredential(user_id=user_id, password_hash=encoded, version=1))
    else:
        credential.password_hash = encoded
        credential.version += 1
        credential.updated_at = now()
        await db.execute(WebSession.__table__.update().where(
            WebSession.user_id == user_id, WebSession.revoked_at.is_(None)).values(revoked_at=now()))
    await db.commit()
