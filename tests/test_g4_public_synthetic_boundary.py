"""The public acceptance helper must fail before DB/provider I/O if unsafe."""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("name,value", [
    ("telegram_bot_token", "123456:real-looking"),
    ("telegram_webhook_url", "https://example.test/webhook"),
    ("scheduler_enabled", True),
    ("google_calendar_enabled", True),
    ("gmail_enabled", True),
    ("memory_enabled", True),
    ("openai_base_url", "https://model.example/v1"),
    ("web_public_origin", "http://example.test"),
    ("secret_key", "change_me_in_production"),
])
async def test_public_synthetic_server_rejects_external_authority_before_db(
        monkeypatch, tmp_path, name, value):
    from tests.browser import g4_public_synthetic_server as helper

    cluster = tmp_path / "g4_public_pg18_data"
    cluster.mkdir()
    monkeypatch.setenv("PPA_ISOLATED_PG18_DATA_DIR", str(cluster))
    monkeypatch.setenv("PPA_G4_SYNTHETIC_PUBLIC", "YES")
    settings = SimpleNamespace(
        web_public_origin="https://example-node-1.ts.net",
        web_enabled=True,
        secret_key="synthetic-test-secret-of-at-least-32-characters",
        telegram_bot_token="", telegram_webhook_url="", scheduler_enabled=False,
        google_calendar_enabled=False, gmail_enabled=False, memory_enabled=False,
        llm_provider="openai", openai_base_url="http://127.0.0.1:1/v1",
        database_url="postgresql+asyncpg://postgres@127.0.0.1:55588/assistant",
        database_url_sync="postgresql://postgres@127.0.0.1:55588/assistant",
    )
    setattr(settings, name, value)
    monkeypatch.setattr(helper, "get_settings", lambda: settings)
    with pytest.raises(RuntimeError, match="G4SyntheticBoundaryRejected"):
        await helper.verify_synthetic_boundary()
