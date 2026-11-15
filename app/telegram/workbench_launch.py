"""Identify only this Bot's durable Mini App launch messages and expiry."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlsplit

from app.db.models import OutboundDelivery, TelegramMiniAppLaunchTicket
from app.web.auth import _utc
from app.web.telegram_miniapp import TICKET_TTL

WORKBENCH_LEGACY_TEXT = "点击下方按钮打开个人工作台。"
WORKBENCH_LAUNCH_TEXT = "此工作台按钮可在 10 分钟内重复打开；过期后请重新发送 /workbench。"
LEGACY_TICKET_TTL = timedelta(minutes=5)


def launch_ticket_hash(delivery: OutboundDelivery) -> str | None:
    """Reject unrelated inline keyboards, URLs and malformed historical rows."""
    if delivery.kind not in {"workbench_launch", "interactive_reply"}:
        return None
    if delivery.kind == "interactive_reply" and delivery.text != WORKBENCH_LEGACY_TEXT:
        return None
    try:
        rows = delivery.keyboard["inline_keyboard"]
        if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0]["text"] != "打开工作台":
            return None
        url = urlsplit(rows[0][0]["web_app"]["url"])
        if (url.scheme != "https" or not url.netloc or url.username or url.password
                or url.path not in {"/workbench", "/workbench/"}):
            return None
        fields = parse_qsl(url.fragment, keep_blank_values=True, strict_parsing=True)
        if len(fields) != 1 or fields[0][0] != "launchTicket":
            return None
        ticket = fields[0][1]
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", ticket):
            return None
        return hashlib.sha256(ticket.encode()).hexdigest()
    except (TypeError, KeyError, IndexError, AttributeError, ValueError):
        return None


async def launch_expired(db, delivery: OutboundDelivery, bot_identity: int,
                         *, at: datetime | None = None) -> bool:
    """Remove a button only after its actual first-issued expiry.

    Older purged five-minute tickets retain their original age-based deadline.
    """
    token_hash = launch_ticket_hash(delivery)
    if token_hash is None or not delivery.chat_id.isdigit():
        return False
    current = at or datetime.now(timezone.utc)
    ticket = await db.get(TelegramMiniAppLaunchTicket, token_hash)
    if ticket is None:
        lifetime = LEGACY_TICKET_TTL if delivery.kind == "interactive_reply" else TICKET_TTL
        return _utc(delivery.created_at) + lifetime <= current
    return (ticket.bot_identity == bot_identity
            and ticket.telegram_user_id == delivery.chat_id
            and _utc(ticket.expires_at) <= current)


async def launch_sendable(db, delivery: OutboundDelivery, bot_identity: int,
                          *, at: datetime | None = None) -> bool:
    """R4 must not send a button whose original login binding cannot work."""
    token_hash = launch_ticket_hash(delivery)
    if token_hash is None or not delivery.chat_id.isdigit():
        return False
    ticket = await db.get(TelegramMiniAppLaunchTicket, token_hash)
    current = at or datetime.now(timezone.utc)
    return bool(ticket and ticket.bot_identity == bot_identity
                and ticket.telegram_user_id == delivery.chat_id
                and _utc(ticket.expires_at) > current)
