"""Only the exact durable Mini App button is eligible for expiry edits."""

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.telegram.workbench_launch import (launch_expired, launch_sendable,
                                           launch_ticket_hash, WORKBENCH_LEGACY_TEXT)


def _delivery(kind, text, url, *, button="打开工作台"):
    return SimpleNamespace(kind=kind, text=text, keyboard={"inline_keyboard": [[{
        "text": button, "web_app": {"url": url}}]]})


def test_launch_parser_accepts_new_and_legacy_but_not_other_buttons():
    ticket = "synthetic_ticket_0123456789_abcdef0123456789"
    url = "https://example.test/workbench/?view=public#launchTicket=" + ticket
    expected = hashlib.sha256(ticket.encode()).hexdigest()
    assert launch_ticket_hash(_delivery("workbench_launch", "new", url)) == expected
    assert launch_ticket_hash(_delivery("interactive_reply", WORKBENCH_LEGACY_TEXT,
                                        url)) == expected
    assert launch_ticket_hash(_delivery("interactive_reply", "Other reply", url)) is None
    assert launch_ticket_hash(_delivery("hitl_proposal", WORKBENCH_LEGACY_TEXT,
                                        url)) is None
    assert launch_ticket_hash(_delivery("workbench_launch", "new",
                                        "http://example.test/workbench/#launchTicket=" + ticket)) is None
    assert launch_ticket_hash(_delivery("workbench_launch", "new",
                                        "https://example.test/workbench/#launchTicket=" + ticket,
                                        button="Other")) is None


@pytest.mark.asyncio
async def test_launch_stays_usable_until_actual_expiry_and_never_crosses_owner():
    now = datetime.now(timezone.utc)
    token = "synthetic_ticket_0123456789_abcdef0123456789"
    delivery = _delivery("workbench_launch", "new",
                         "https://example.test/workbench/#launchTicket=" + token)
    delivery.chat_id = "987654"
    delivery.created_at = now - timedelta(minutes=1)
    ticket = SimpleNamespace(bot_identity=42, telegram_user_id="987654",
                             consumed_at=now, expires_at=now + timedelta(minutes=9))

    class DB:
        async def get(self, _model, _key):
            return ticket

    db = DB()
    assert not await launch_expired(db, delivery, 42, at=now)
    # First redemption does not shorten the ten-minute button window.
    assert await launch_sendable(db, delivery, 42, at=now)
    ticket.telegram_user_id = "other"
    assert not await launch_sendable(db, delivery, 42, at=now)
    assert not await launch_expired(db, delivery, 42, at=now + timedelta(minutes=6))
    ticket.telegram_user_id = "987654"
    assert not await launch_expired(db, delivery, 42, at=now + timedelta(minutes=6))
    assert await launch_expired(db, delivery, 42, at=now + timedelta(minutes=11))

    class PrunedDB:
        async def get(self, _model, _key):
            return None

    delivery.created_at = now - timedelta(minutes=6)
    assert not await launch_expired(PrunedDB(), delivery, 42, at=now)
    delivery.kind, delivery.text = "interactive_reply", WORKBENCH_LEGACY_TEXT
    assert await launch_expired(PrunedDB(), delivery, 42, at=now)
