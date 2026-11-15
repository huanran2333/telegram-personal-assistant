"""Offline contract tests; never contact Tailscale or Telegram."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
import telegram

from app import on_demand_runner as runner


def _status(host: str = "example-node-2.ts.net.") -> str:
    return json.dumps({"BackendState": "Running", "Self": {"DNSName": host}})


def _funnel_status(host: str = "example-node-2.ts.net",
                   proxy: str = "http://127.0.0.1:8000") -> str:
    key = f"{host}:443"
    return json.dumps({"Foreground": {"test-id": {
        "Web": {key: {"Handlers": {"/": {"Proxy": proxy}}}},
        "AllowFunnel": {key: True},
    }}})


def test_origin_comes_from_running_tailscale_self() -> None:
    assert runner.origin_from_status(_status()) == "https://example-node-2.ts.net"
    with pytest.raises(runner.LaunchError):
        runner.origin_from_status(_status("example.invalid."))
    with pytest.raises(runner.LaunchError):
        runner.origin_from_status(json.dumps({"BackendState": "NeedsLogin"}))


def test_attended_health_accepts_disabled_scheduler_but_requires_webhook() -> None:
    payload = {
        "status": "degraded", "ready": True,
        "components": {"scheduler": False, "telegram_webhook": True,
                       "web_auth": True},
    }
    assert runner.app_health_ready(payload)
    assert not runner.app_health_ready({**payload, "ready": False})
    assert not runner.app_health_ready({**payload, "components": {"telegram_webhook": False}})
    assert not runner.app_health_ready({**payload, "components": {
        "telegram_webhook": True, "web_auth": False,
    }})


def test_mapping_must_match_hostname_and_local_port() -> None:
    origin = "https://example-node-2.ts.net"
    assert runner.no_serve_config("{}")
    assert not runner.no_serve_config(_funnel_status())
    assert runner.owns_funnel(_funnel_status(), origin)
    assert not runner.owns_funnel(_funnel_status("example-node-3.ts.net"), origin)
    assert not runner.owns_funnel(_funnel_status("example-node-4.ts.net"), origin)
    assert not runner.owns_funnel(_funnel_status(proxy="http://127.0.0.1:9000"), origin)
    assert not runner.owns_funnel("No serve config", origin)
    mixed = json.loads(_funnel_status())
    mixed["Foreground"]["other-id"] = {"Web": {}}
    assert not runner.owns_funnel(json.dumps(mixed), origin)


def test_cleanup_never_disables_another_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_tailscale(*args: str, timeout: float = 10) -> str:
        calls.append(args)
        return _funnel_status("example-node-5.ts.net")

    monkeypatch.setattr(runner, "tailscale", fake_tailscale)
    assert not runner.cleanup_funnel("https://example-node-2.ts.net")
    assert calls == [("funnel", "status", "--json")]


def test_cleanup_disables_only_matching_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []
    statuses = iter([
        _funnel_status(),
        "{}",
    ])

    def fake_tailscale(*args: str, timeout: float = 10) -> str:
        calls.append(args)
        return next(statuses) if args == ("funnel", "status", "--json") else ""

    monkeypatch.setattr(runner, "tailscale", fake_tailscale)
    assert runner.cleanup_funnel("https://example-node-2.ts.net")
    assert ("funnel", "--https=443", "off") in calls


def test_webhook_cleanup_never_drops_updates_or_other_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBot:
        def __init__(self, token: str) -> None:
            assert token == "fake-token"
            self.url = "https://example-node-5.ts.net/webhook"
            self.deletes: list[bool | None] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get_webhook_info(self, **_kwargs):
            return SimpleNamespace(url=self.url)

        async def delete_webhook(self, *, drop_pending_updates=None, **_kwargs):
            self.deletes.append(drop_pending_updates)
            self.url = ""
            return True

    bot = FakeBot("fake-token")
    monkeypatch.setattr(telegram, "Bot", lambda token: bot)
    expected = "https://example-node-2.ts.net/webhook"
    assert asyncio.run(runner.webhook_state("fake-token", expected, clear=True)) == "different"
    assert bot.deletes == []
    bot.url = expected
    assert asyncio.run(runner.webhook_state("fake-token", expected, clear=True)) == "cleared"
    assert bot.deletes == [False]


def test_check_only_never_starts_tunnel_or_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_tailscale(*args: str, timeout: float = 10) -> str:
        return _status() if args == ("status", "--json") else "{}"

    monkeypatch.setattr(runner, "tailscale", fake_tailscale)
    monkeypatch.setattr(
        runner, "session_settings", lambda origin: SimpleNamespace(shutdown_drain_seconds=30),
    )
    monkeypatch.setattr(runner, "check_settings", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("started child"))
    monkeypatch.setattr(runner, "webhook_state", lambda *_args, **_kwargs: pytest.fail("called Bot"))
    monkeypatch.setattr(sys, "argv", ["on_demand_runner", "--check-only"])

    assert runner.main() == 0


def test_existing_serve_mapping_blocks_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_tailscale(*args: str, timeout: float = 10) -> str:
        return _status() if args == ("status", "--json") else _funnel_status()

    monkeypatch.setattr(runner, "tailscale", fake_tailscale)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("started child"))
    monkeypatch.setattr(sys, "argv", ["on_demand_runner", "--check-only"])

    assert runner.main() == 1


def test_attended_session_injects_origin_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children: list[tuple[list[str], dict | None]] = []
    cleared: list[tuple[str, bool]] = []
    cleaned_funnel: list[str] = []
    origin = "https://example-node-2.ts.net"

    class FakeChild:
        def wait(self, timeout=None):
            return 0

    def fake_popen(args, *, cwd, env=None):
        children.append((args, env))
        return FakeChild()

    async def fake_webhook_state(token, expected_url, *, clear):
        assert token == "fake-token"
        cleared.append((expected_url, clear))
        return "cleared" if clear else "empty"

    def fake_tailscale(*args: str, timeout: float = 10) -> str:
        return _status() if args == ("status", "--json") else "{}"

    monkeypatch.setattr(runner, "tailscale", fake_tailscale)
    monkeypatch.setattr(runner, "session_settings", lambda _origin: SimpleNamespace(
        telegram_bot_token="fake-token", telegram_webhook_url=f"{origin}/webhook",
        shutdown_drain_seconds=30,
    ))
    monkeypatch.setattr(runner, "check_settings", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(runner, "port_in_use", lambda: False)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "webhook_state", fake_webhook_state)
    monkeypatch.setattr(runner, "wait_for_funnel", lambda *_args: None)
    monkeypatch.setattr(runner, "wait_for_app", lambda *_args: None)
    monkeypatch.setattr(runner, "stop_child", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "cleanup_funnel", lambda value: cleaned_funnel.append(value) or True)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "START")
    monkeypatch.setattr(sys, "argv", ["on_demand_runner"])

    assert runner.main() == 0
    assert children[0][0] == ["tailscale", "funnel", "8000"]
    app_environment = children[1][1]
    assert app_environment is not None
    assert app_environment["WEB_PUBLIC_ORIGIN"] == origin
    assert app_environment["TELEGRAM_WEBHOOK_URL"] == f"{origin}/webhook"
    assert app_environment["TELEGRAM_LIVE_START_PREFLIGHT_ACK"] == "true"
    assert app_environment["SCHEDULER_ENABLED"] == "true"
    assert app_environment["DAILY_RUNDOWN_ENABLED"] == "true"
    assert cleared == [(f"{origin}/webhook", False), (f"{origin}/webhook", True)]
    assert cleaned_funnel == [origin]
