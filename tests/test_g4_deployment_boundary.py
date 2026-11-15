"""Public deployment guardrails that need no real Bot or public tunnel."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_webhook_diagnostics_reject_anonymous_before_bot_io(monkeypatch):
    from app.main import app
    from app.telegram import webhook

    def forbidden_bot():
        raise AssertionError("anonymous request reached Telegram Bot API")

    monkeypatch.setattr(webhook, "get_bot", forbidden_bot)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://example.test") as client:
        response = await client.get("/webhook-info")
    assert response.status_code == 401


def test_windows_entrypoint_never_trusts_forwarded_client_identity(monkeypatch):
    from app import run_windows

    captured = {}
    monkeypatch.setattr(run_windows.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    run_windows.main()
    assert captured["host"] == "127.0.0.1"
    assert captured["proxy_headers"] is False
