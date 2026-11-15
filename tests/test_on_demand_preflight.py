"""The launcher must reject unsafe config without contacting external services."""

from pathlib import Path

from app.config import Settings
from app.on_demand_preflight import check_settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "openai",
        "llm_model": "test-model",
        "openai_api_key": "fake-test-key",
        "telegram_bot_token": "fake-test-token",
        "telegram_webhook_url": "https://example.test/webhook",
        "database_url": (
            "postgresql+asyncpg://personal_assistant_app:fake@"
            "127.0.0.1:5432/personal_assistant"
        ),
        "database_url_sync": (
            "postgresql://personal_assistant_app:fake@"
            "127.0.0.1:5432/personal_assistant"
        ),
        "app_env": "production",
        "secret_key": "x" * 32,
        "web_enabled": True,
        "web_public_origin": "https://example.test",
        "scheduler_enabled": False,
        "daily_rundown_enabled": False,
        "gmail_enabled": False,
        "telegram_live_start_preflight_ack": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_preflight_accepts_reviewed_config_without_network(tmp_path: Path) -> None:
    dist = tmp_path / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("test", encoding="utf-8")

    errors, _notices = check_settings(_settings(), repo=tmp_path)

    assert errors == []


def test_preflight_rejects_missing_webhook_and_frontend(tmp_path: Path) -> None:
    errors, notices = check_settings(
        _settings(telegram_webhook_url="", web_public_origin="http://example.test"),
        repo=tmp_path,
    )

    assert any("WEB_PUBLIC_ORIGIN" in error for error in errors)
    assert any("TELEGRAM_WEBHOOK_URL" in error for error in errors)
    assert any("web/dist/index.html" in error for error in errors)
    assert any("SCHEDULER_ENABLED=false" in notice for notice in notices)


def test_preflight_reuses_live_database_guard(tmp_path: Path) -> None:
    errors, _notices = check_settings(
        _settings(database_url="postgresql+asyncpg://postgres:fake@127.0.0.1:5432/assistant"),
        repo=tmp_path,
    )

    assert any("Live-start config guard failed" in error for error in errors)


def test_preflight_rejects_missing_bot_token(tmp_path: Path) -> None:
    errors, _notices = check_settings(_settings(telegram_bot_token=""), repo=tmp_path)

    assert any("TELEGRAM_BOT_TOKEN" in error for error in errors)


def test_preflight_rejects_public_http_model_endpoint(tmp_path: Path) -> None:
    errors, _ = check_settings(
        _settings(openai_base_url="http://inference.example.test/model/v1"), repo=tmp_path,
    )
    assert any("OPENAI_BASE_URL" in error for error in errors)

    private_errors, _ = check_settings(
        _settings(openai_base_url="http://10.0.0.2:3027/v1"), repo=tmp_path,
    )
    assert not any("OPENAI_BASE_URL" in error for error in private_errors)
