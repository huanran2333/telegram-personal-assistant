"""Supervise one attended Telegram/Funnel session without editing .env."""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from app.config import Settings
from app.on_demand_preflight import check_settings

REPO = Path(__file__).resolve().parent.parent
LOCAL_TARGET = "127.0.0.1:8000"


class LaunchError(RuntimeError):
    pass


def tailscale(*args: str, timeout: float = 10) -> str:
    try:
        result = subprocess.run(
            ["tailscale", *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError("Tailscale is unavailable; start/sign in to its Windows app") from exc
    if result.returncode:
        raise LaunchError(
            f"Tailscale {' '.join(args)} failed (exit {result.returncode}); "
            "ensure the Windows app is running and signed in"
        )
    return result.stdout.strip()


def origin_from_status(raw: str) -> str:
    try:
        status = json.loads(raw)
        if status.get("BackendState") != "Running":
            raise ValueError("not running")
        host = status["Self"]["DNSName"].rstrip(".").lower()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchError("Tailscale is not signed in and running with a DNS name") from exc
    if not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.ts\.net", host):
        raise LaunchError("Tailscale DNS name is not a trusted ts.net hostname")
    return f"https://{host}"


def no_serve_config(raw: str) -> bool:
    try:
        status = json.loads(raw)
    except (TypeError, ValueError):
        return False
    return isinstance(status, dict) and not any(bool(value) for value in status.values())


def owns_funnel(raw: str, origin: str) -> bool:
    host = urlsplit(origin).hostname or ""
    try:
        status = json.loads(raw)
    except (TypeError, ValueError):
        return False
    if not isinstance(status, dict):
        return False
    foreground = status.get("Foreground") or {}
    if status.get("Background") or len(foreground) != 1:
        return False
    config = next(iter(foreground.values()))
    if not isinstance(config, dict):
        return False
    key = f"{host}:443"
    web = config.get("Web") or {}
    allowed = config.get("AllowFunnel") or {}
    if set(web) != {key} or set(allowed) != {key} or allowed[key] is not True:
        return False
    handlers = web[key].get("Handlers") or {}
    return (set(handlers) == {"/"}
            and handlers["/"].get("Proxy") in {
                f"http://{LOCAL_TARGET}", "http://localhost:8000",
            })


def port_in_use() -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", 8000)) == 0


async def webhook_state(token: str, expected_url: str, *, clear: bool) -> str:
    """Never modify a Webhook belonging to another URL or drop updates."""
    from telegram import Bot

    async with Bot(token=token) as bot:
        info = await bot.get_webhook_info(connect_timeout=10, read_timeout=10)
        if not clear:
            return "empty" if not info.url else "occupied"
        if not info.url:
            return "empty"
        if info.url != expected_url:
            return "different"
        deleted = await bot.delete_webhook(
            drop_pending_updates=False, connect_timeout=10, read_timeout=10,
        )
        if not deleted:
            return "failed"
        verified = await bot.get_webhook_info(connect_timeout=10, read_timeout=10)
        return "cleared" if not verified.url else "failed"


def session_settings(origin: str) -> Settings:
    # Only the current process and child receive these values. The ignored
    # .env and its secrets are never rewritten by this launcher.
    return Settings(
        telegram_live_start_preflight_ack=False,
        web_public_origin=origin,
        telegram_webhook_url=f"{origin}/webhook",
        web_enabled=True,
        scheduler_enabled=True,
        daily_rundown_enabled=True,
    )


def wait_for_funnel(child: subprocess.Popen[bytes], origin: str) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise LaunchError("Foreground Funnel exited before its mapping became active")
        try:
            status = tailscale("funnel", "status", "--json", timeout=5)
        except LaunchError:
            time.sleep(1)
            continue
        if owns_funnel(status, origin):
            return
        time.sleep(1)
    raise LaunchError("Funnel did not publish the expected hostname and local port")


def wait_for_app(child: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise LaunchError(f"Application exited during startup (exit {child.returncode})")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=2)
            try:
                connection.request("GET", "/health")
                payload = json.loads(connection.getresponse().read())
            finally:
                connection.close()
            if app_health_ready(payload):
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise LaunchError("Application, Telegram Webhook and Mini App did not become ready within 120 seconds; inspect /health")


def app_health_ready(payload: dict) -> bool:
    # Proactive schedules are intentionally disabled for an attended session,
    # so /health may be "degraded" even when the required runtime is ready.
    components = payload.get("components", {})
    return (payload.get("ready") is True
            and components.get("telegram_webhook") is True
            and components.get("web_auth") is True)


def cleanup_funnel(origin: str) -> bool:
    try:
        status = tailscale("funnel", "status", "--json")
        if no_serve_config(status):
            return True
        if not owns_funnel(status, origin):
            print("WARNING: Funnel changed ownership; leaving it untouched")
            return False
        tailscale("funnel", "--https=443", "off", timeout=20)
        return no_serve_config(tailscale("funnel", "status", "--json"))
    except LaunchError:
        return False


def stop_child(child: subprocess.Popen[bytes] | None, *, grace_seconds: float = 8) -> None:
    if child is None or child.poll() is not None:
        return
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()

    funnel_child: subprocess.Popen[bytes] | None = None
    app_child: subprocess.Popen[bytes] | None = None
    origin = ""
    may_have_started_funnel = False
    started_app = False
    app_grace_seconds = 45.0
    cleanup_ok = True
    result = 0
    try:
        origin = origin_from_status(tailscale("status", "--json"))
        if args.cleanup_only:
            if port_in_use():
                raise LaunchError("Port 8000 is in use; stop the application first")
            settings = session_settings(origin)
            if not settings.telegram_bot_token:
                raise LaunchError("TELEGRAM_BOT_TOKEN is missing")
            if not sys.stdin.isatty():
                raise LaunchError("Interactive CLEANUP confirmation is required")
            if input("Type CLEANUP to clear only this host's Webhook/Funnel: ") != "CLEANUP":
                return 0
            try:
                state = asyncio.run(webhook_state(
                    settings.telegram_bot_token, settings.telegram_webhook_url, clear=True,
                ))
            except Exception as exc:
                raise LaunchError(f"Telegram Webhook cleanup failed ({type(exc).__name__})") from exc
            if state == "different":
                raise LaunchError("Telegram points to another Webhook; refusing to change it")
            if state == "failed":
                raise LaunchError("Could not verify Webhook deletion")
            if not cleanup_funnel(origin):
                raise LaunchError("Could not verify Funnel shutdown")
            print("Cleanup verified; no Telegram updates were dropped.")
            return 0

        for kind in ("serve", "funnel"):
            if not no_serve_config(tailscale(kind, "status", "--json")):
                raise LaunchError(f"An existing Tailscale {kind} mapping is present; refusing to replace it")
        settings = session_settings(origin)
        app_grace_seconds = settings.shutdown_drain_seconds + 15
        errors, notices = check_settings(settings, repo=REPO)
        print(f"Detected HTTPS origin: {origin}")
        for notice in notices:
            print(f"NOTICE: {notice}")
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            return 2
        if args.check_only:
            print("Check-only passed; no Funnel, Bot or application was started.")
            return 0
        if port_in_use():
            raise LaunchError("127.0.0.1:8000 is already in use")
        print("Starting may send queued Telegram replies, scheduled reminders and daily briefs.")
        print("Daily briefs include each user's Todos; Calendar is available only to its configured owner.")
        if not sys.stdin.isatty():
            raise LaunchError("Interactive START confirmation is required")
        if input("Type START to open Funnel and run the real Bot: ") != "START":
            print("Cancelled; nothing was started.")
            return 0
        try:
            prior = asyncio.run(webhook_state(
                settings.telegram_bot_token, settings.telegram_webhook_url, clear=False,
            ))
        except Exception as exc:
            raise LaunchError(f"Telegram Webhook preflight failed ({type(exc).__name__})") from exc
        if prior != "empty":
            raise LaunchError("A Telegram Webhook already exists; refusing to replace it")

        print("Starting foreground Tailscale Funnel...")
        funnel_child = subprocess.Popen(["tailscale", "funnel", "8000"], cwd=REPO)
        may_have_started_funnel = True
        wait_for_funnel(funnel_child, origin)

        app_env = os.environ.copy()
        app_env.update({
            "WEB_PUBLIC_ORIGIN": origin,
            "TELEGRAM_WEBHOOK_URL": settings.telegram_webhook_url,
            "WEB_ENABLED": "true",
            "TELEGRAM_LIVE_START_PREFLIGHT_ACK": "true",
            "SCHEDULER_ENABLED": "true",
            "DAILY_RUNDOWN_ENABLED": "true",
        })
        print("Starting application; Ctrl+C stops this session and cleans up.")
        app_child = subprocess.Popen(
            [sys.executable, "-m", "app.run_windows"], cwd=REPO, env=app_env,
        )
        started_app = True
        wait_for_app(app_child)
        print(f"Ready: {origin}/workbench/ (health: http://127.0.0.1:8000/health)")
        result = app_child.wait()
    except KeyboardInterrupt:
        print("\nStopping on-demand session...")
    except ValidationError as exc:
        fields = sorted({".".join(map(str, error["loc"])) for error in exc.errors()})
        print("ERROR: configuration invalid (fields: " + ", ".join(fields) + ")")
        result = 1
    except LaunchError as exc:
        print(f"ERROR: {exc}")
        result = 1
    finally:
        if started_app:
            stop_child(app_child, grace_seconds=app_grace_seconds)
            try:
                settings = session_settings(origin)
                state = asyncio.run(webhook_state(
                    settings.telegram_bot_token, settings.telegram_webhook_url, clear=True,
                ))
                if state not in {"empty", "cleared"}:
                    print(f"WARNING: Webhook cleanup not verified ({state}); use --cleanup-only")
                    cleanup_ok = False
            except Exception as exc:
                print(f"WARNING: Webhook cleanup failed ({type(exc).__name__}); use --cleanup-only")
                cleanup_ok = False
        if funnel_child is not None:
            stop_child(funnel_child)
        if may_have_started_funnel and not cleanup_funnel(origin):
            print("WARNING: Funnel cleanup not verified; inspect `tailscale funnel status`")
            cleanup_ok = False
        if cleanup_ok and (started_app or may_have_started_funnel):
            print("Webhook/Funnel cleanup verified; pending Telegram updates were not dropped.")
    return result if cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
