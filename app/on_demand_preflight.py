"""Read-only checks for the operator's foreground Windows launcher.

This module never contacts Telegram, Google, the model provider, or PostgreSQL.
The application lifespan performs its own readiness checks after launch.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from app.config import Settings


def check_settings(settings: Settings, *, repo: Path) -> tuple[list[str], list[str]]:
    """Return actionable errors and non-blocking capability notices, sans secrets."""
    errors: list[str] = []
    notices: list[str] = []

    if not settings.telegram_bot_token:
        errors.append("TELEGRAM_BOT_TOKEN is missing")
    origin = settings.web_public_origin.rstrip("/")
    try:
        origin_parts = urlsplit(origin)
        valid_origin = (settings.web_public_origin == origin
                        and origin_parts.scheme == "https"
                        and bool(origin_parts.hostname)
                        and origin_parts.username is None
                        and origin_parts.password is None
                        and not origin_parts.path
                        and not origin_parts.query
                        and not origin_parts.fragment)
    except ValueError:
        valid_origin = False
    if not valid_origin:
        errors.append("WEB_PUBLIC_ORIGIN must be an HTTPS origin without a path")
    if settings.telegram_webhook_url != f"{origin}/webhook":
        errors.append("TELEGRAM_WEBHOOK_URL must equal WEB_PUBLIC_ORIGIN + /webhook")
    if settings.llm_provider == "openai" and settings.openai_base_url:
        model_url = urlsplit(settings.openai_base_url)
        host = model_url.hostname or ""
        try:
            private_ip = ipaddress.ip_address(host).is_private
        except ValueError:
            private_ip = False
        if model_url.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"} and not private_ip:
            errors.append("OPENAI_BASE_URL must use HTTPS for a non-local model host")
    if not settings.web_enabled:
        notices.append("WEB_ENABLED=false: Mini App and Web features will be unavailable")
    elif not (repo / "web" / "dist" / "index.html").is_file():
        errors.append("web/dist/index.html is missing; build the frontend first")

    if settings.scheduler_enabled:
        notices.append("SCHEDULER_ENABLED=true: scheduled notifications may be sent")
    else:
        notices.append("SCHEDULER_ENABLED=false: proactive reminders are disabled")
    if settings.scheduler_enabled and settings.daily_rundown_enabled:
        notices.append(
            "DAILY_RUNDOWN_ENABLED=true: eligible users may receive their own Todo brief; "
            "Calendar is read only for its configured owner"
        )
    for field in ("reminder_allowed_telegram_ids",
                  "daily_rundown_allowed_telegram_ids",
                  "telegram_allowed_ingress_ids"):
        value = getattr(settings, field)
        if value.strip():
            ids = [part.strip() for part in value.split(",")]
            if not all(part.isascii() and part.isdecimal() for part in ids):
                errors.append(f"{field.upper()} must be comma-separated Telegram IDs")
            else:
                boundary = ("inbound updates" if field == "telegram_allowed_ingress_ids"
                            else "proactive delivery")
                notices.append(f"{field.upper()} scopes {boundary} to {len(ids)} chat(s)")
    if settings.google_calendar_enabled and not settings.google_calendar_owner_telegram_id:
        notices.append("Google Calendar is enabled but its owner Telegram ID is unset")
    if settings.gmail_enabled and not settings.gmail_owner_telegram_id:
        notices.append("Gmail is enabled but its owner Telegram ID is unset")
    if not settings.gmail_enabled:
        notices.append("GMAIL_ENABLED=false: Gmail tools will be unavailable")

    # Reuse the exact live-start validator, including its dedicated database
    # target and application-secret guard. Do not duplicate those requirements.
    try:
        live_values = settings.model_dump()
        live_values["telegram_live_start_preflight_ack"] = True
        Settings(_env_file=None, **live_values)
    except ValidationError:
        errors.append(
            "Live-start config guard failed: check APP_ENV, SECRET_KEY, "
            "WEB_PUBLIC_ORIGIN and both dedicated PostgreSQL URLs"
        )
    return errors, notices


def main() -> int:
    try:
        # Explicitly keep live startup disabled while inspecting configuration.
        settings = Settings(telegram_live_start_preflight_ack=False)
    except ValidationError as exc:
        fields = sorted({".".join(map(str, error["loc"])) for error in exc.errors()})
        print("Invalid settings (field names only): " + ", ".join(fields))
        return 1

    repo = Path(__file__).resolve().parent.parent
    errors, notices = check_settings(settings, repo=repo)
    print(f"Model provider: {settings.llm_provider}; model: {settings.llm_model}")
    print(f"Web enabled: {settings.web_enabled}; Calendar enabled: "
          f"{settings.google_calendar_enabled}; Gmail enabled: {settings.gmail_enabled}")
    for notice in notices:
        print(f"NOTICE: {notice}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("Configuration check passed. Database, model, tunnel and Bot are NOT yet tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
