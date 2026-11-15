"""No external service is contacted by these release-staging safety tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops import local_staging as staging


def test_reserved_ports_and_paths_are_rejected():
    for port in (5432, 55432, 55632, 0, 65536):
        with pytest.raises(RuntimeError):
            staging.locations(port)
    data, private = staging.locations(56520)
    assert data.is_relative_to(staging.DATA_ROOT)
    assert private.is_relative_to(staging.PRIVATE_ROOT)


def test_safe_environment_forces_external_services_off(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "old-test-value")
    monkeypatch.setenv("GOOGLE_CALENDAR_ENABLED", "true")
    monkeypatch.setenv("SCHEDULER_ENABLED", "true")
    result = staging.safe_environment(56520, "unused", "synthetic-password")
    assert result["TELEGRAM_BOT_TOKEN"] == ""
    assert result["GOOGLE_CALENDAR_ENABLED"] == "false"
    assert result["GMAIL_ENABLED"] == "false"
    assert result["SCHEDULER_ENABLED"] == "false"
    assert result["TELEGRAM_LIVE_START_PREFLIGHT_ACK"] == "false"
    assert "127.0.0.1:56520/personal_assistant_stage" in result["DATABASE_URL"]


def test_initialize_refuses_existing_data_before_running_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(staging, "PRIVATE_ROOT", tmp_path / "private")
    data, _ = staging.locations(56520)
    data.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        staging.initialize(56520, tmp_path / "not-installed")


def test_private_config_cannot_redirect_existing_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(staging, "PRIVATE_ROOT", tmp_path / "private")
    data, private = staging.locations(56520)
    data.mkdir(parents=True)
    private.mkdir(parents=True)
    (data / "PG_VERSION").write_text("18", encoding="utf-8")
    (private / "config.json").write_text(json.dumps({
        "port": 56520, "data": str(tmp_path / "different"), "pg_bin": str(tmp_path)
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        staging.read_private(56520)


def test_failed_command_does_not_echo_arguments(monkeypatch):
    monkeypatch.setattr(staging.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError) as error:
        staging.run(["initdb.exe", "--pwfile", "private-password-file"])
    assert "private-password-file" not in str(error.value)
