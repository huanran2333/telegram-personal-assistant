"""Web authentication for direct business APIs; no Web Agent ingress."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import update

from app.config import get_settings
from app.db.models import TelegramMiniAppLaunchTicket, WebSession
from app.db.session import AsyncSessionLocal
from app.web.auth import (NONCE_COOKIE, SESSION_COOKIE, consume_login_nonce,
                          issue_login_nonce, login, no_store, now, principal,
                          require_csrf, require_origin, require_bootstrap_origin)

router = APIRouter(prefix="/api/v3", tags=["web-v3"])


def _ready(request: Request) -> None:
    if not getattr(request.app.state, "web_ready", False):
        raise HTTPException(503, "WebRuntimeUnavailable")


def _key(request: Request) -> str:
    value = request.headers.get("idempotency-key", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,96}", value):
        raise HTTPException(422, "IdempotencyKeyRequired")
    return value


async def _authenticated(request: Request, *, write: bool = False):
    _ready(request)
    async with AsyncSessionLocal() as db:
        runtime = getattr(request.app.state, "telegram_durable", None)
        user, session = await principal(db, request.cookies.get(SESSION_COOKIE, ""),
                                        bot_identity=getattr(runtime, "bot_identity", None))
        if write:
            require_csrf(request, session, get_settings())
        return user, session


class LoginBody(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


class MiniAppExchangeBody(BaseModel):
    init_data: str = Field(min_length=1, max_length=16384)
    launch_ticket: str = Field(min_length=1, max_length=128)


class MiniAppSessionBody(BaseModel):
    init_data: str = Field(min_length=1, max_length=16384)
    launch_ticket: str = Field(min_length=1, max_length=128)


@router.get("/auth/bootstrap")
async def bootstrap(request: Request, response: Response):
    _ready(request)
    require_bootstrap_origin(request, get_settings())
    async with AsyncSessionLocal() as db:
        token, csrf = await issue_login_nonce(db)
    response.set_cookie(NONCE_COOKIE, token, secure=True, httponly=True,
                        samesite="strict", path="/", max_age=600)
    no_store(response)
    return {"csrf_token": csrf}


@router.post("/auth/login", status_code=204)
async def authenticate(request: Request, response: Response, body: LoginBody):
    _ready(request)
    settings = get_settings()
    require_origin(request, settings)
    async with AsyncSessionLocal() as db:
        await consume_login_nonce(db, request.cookies.get(NONCE_COOKIE, ""),
                                  request.headers.get("x-csrf-token", ""))
        token, _ = await login(db, request, settings, body.password)
    response.set_cookie(SESSION_COOKIE, token, secure=True, httponly=True,
                        samesite="strict", path="/",
                        max_age=settings.web_session_hours * 3600)
    response.delete_cookie(NONCE_COOKIE, path="/", secure=True, httponly=True, samesite="strict")
    no_store(response)


@router.post("/auth/telegram/exchange", status_code=204)
async def telegram_exchange(request: Request, response: Response, body: MiniAppExchangeBody):
    _ready(request)
    settings = get_settings()
    require_origin(request, settings)
    runtime = getattr(request.app.state, "telegram_durable", None)
    bot_identity = getattr(runtime, "bot_identity", None)
    if bot_identity is None or not getattr(runtime, "is_running", False):
        raise HTTPException(503, "TelegramIdentityUnavailable")
    from app.web.telegram_miniapp import exchange_ticket
    async with AsyncSessionLocal() as db:
        await consume_login_nonce(db, request.cookies.get(NONCE_COOKIE, ""),
                                  request.headers.get("x-csrf-token", ""))
        token, session = await exchange_ticket(
            db, ticket=body.launch_ticket, raw_init_data=body.init_data,
            bot_token=settings.telegram_bot_token, bot_identity=bot_identity,
            session_hours=settings.web_session_hours,
            prior_session_token=request.cookies.get(SESSION_COOKIE, ""))
    response.set_cookie(SESSION_COOKIE, token, secure=True, httponly=True,
                        samesite="strict", path="/",
                        max_age=min(settings.web_session_hours, 8) * 3600)
    response.delete_cookie(NONCE_COOKIE, path="/", secure=True,
                           httponly=True, samesite="strict")
    no_store(response)


@router.post("/auth/telegram/revalidate", status_code=204)
async def telegram_revalidate(request: Request, response: Response, body: MiniAppSessionBody):
    """Reuse an existing Session only within its Bot-bound launch window."""
    user, session = await _authenticated(request, write=True)
    if session.auth_source != "telegram":
        raise HTTPException(401, "MiniAppAuthenticationRequired")
    settings = get_settings()
    runtime = getattr(request.app.state, "telegram_durable", None)
    bot_identity = getattr(runtime, "bot_identity", None)
    if bot_identity is None or not getattr(runtime, "is_running", False):
        raise HTTPException(503, "TelegramIdentityUnavailable")
    from app.web.telegram_miniapp import validate_init_data, verify_session_launch
    if validate_init_data(body.init_data, settings.telegram_bot_token, bot_identity) != user.telegram_id:
        raise HTTPException(401, "MiniAppAuthenticationRequired")
    async with AsyncSessionLocal() as db:
        await verify_session_launch(db, ticket=body.launch_ticket,
                                    telegram_user_id=user.telegram_id,
                                    bot_identity=bot_identity)
    no_store(response)


@router.post("/auth/logout", status_code=204)
async def logout(request: Request, response: Response):
    _, session = await _authenticated(request, write=True)
    async with AsyncSessionLocal() as db:
        row = await db.get(WebSession, session.token_hash, with_for_update=True)
        if row is not None:
            row.revoked_at = now()
            if row.auth_source == "telegram" and row.telegram_user_id and row.telegram_bot_identity:
                # Explicit logout is stronger than a lost WebView Cookie:
                # invalidate this user's still-live launch links as well.
                await db.execute(update(TelegramMiniAppLaunchTicket).where(
                    TelegramMiniAppLaunchTicket.bot_identity == row.telegram_bot_identity,
                    TelegramMiniAppLaunchTicket.telegram_user_id == row.telegram_user_id,
                    TelegramMiniAppLaunchTicket.expires_at > now()).values(expires_at=now()))
            await db.commit()
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="strict")
    no_store(response)


@router.get("/me")
async def me(request: Request, response: Response):
    user, session = await _authenticated(request)
    settings = get_settings()
    no_store(response)
    return {"display_label": user.first_name or user.username or "Personal Assistant",
            "timezone": settings.default_timezone, "csrf_token": session.csrf_token,
            "auth_source": session.auth_source,
            "calendar_time_block_sync": bool(user.calendar_time_block_sync),
            "capabilities": {"calendar": bool(settings.google_calendar_enabled and
                                               user.telegram_id == settings.google_calendar_owner_telegram_id),
                             "gmail": bool(settings.gmail_enabled and
                                            user.telegram_id == settings.gmail_owner_telegram_id),
                             "memory": bool(settings.memory_enabled)}}
