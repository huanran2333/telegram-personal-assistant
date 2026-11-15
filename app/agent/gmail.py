"""Read-only Gmail boundary. Message bodies are untrusted evidence, not instructions."""
from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header, make_header
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from pydantic import BaseModel, Field


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_BODY_CHARS = 4000


class GmailError(RuntimeError):
    """A Gmail failure must never be interpreted as an empty inbox."""


class GmailUnavailable(GmailError):
    pass


class GmailMessage(BaseModel):
    message_id: str
    thread_id: str
    sender: str = ""
    recipients: list[str] = Field(default_factory=list)
    subject: str = ""
    timestamp: datetime | None = None
    snippet: str = ""
    body: str = ""
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)

    def evidence(self) -> dict[str, Any]:
        """Bounded, source-attributed data suitable for a model input."""
        return self.model_dump(mode="json")


class GmailSource(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[GmailMessage]: ...
    async def unread(self, query: str = "", limit: int = 5) -> list[GmailMessage]: ...
    async def thread(self, thread_id: str) -> list[GmailMessage]: ...
    async def account_address(self) -> str: ...


def _decoded_header(value: str) -> str:
    try:
        return str(make_header(decode_header(value)))[:500]
    except (UnicodeError, ValueError):
        return value[:500]


def _body_text(payload: dict[str, Any]) -> str:
    """Prefer text/plain; never fetch or interpret attachments."""
    plain: list[str] = []
    fallback_html: list[str] = []

    def visit(part: dict[str, Any]) -> None:
        mime = str(part.get("mimeType") or "").lower()
        if part.get("filename"):
            return
        data = (part.get("body") or {}).get("data")
        if data and mime in {"text/plain", "text/html"}:
            try:
                raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
            except (ValueError, TypeError):
                raw = ""
            (plain if mime == "text/plain" else fallback_html).append(raw)
        for child in part.get("parts") or []:
            if isinstance(child, dict):
                visit(child)

    visit(payload)
    text = "\n".join(plain)
    if not text and fallback_html:
        text = re.sub(r"<[^>]+>", " ", "\n".join(fallback_html))
        text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text).strip()[:MAX_BODY_CHARS]


def parse_message(raw: dict[str, Any]) -> GmailMessage:
    if not isinstance(raw, dict):
        raise GmailError("Gmail returned an invalid message")
    message_id, thread_id = raw.get("id"), raw.get("threadId")
    if not isinstance(message_id, str) or not isinstance(thread_id, str):
        raise GmailError("Gmail returned a message without source identifiers")
    payload = raw.get("payload") or {}
    if not isinstance(payload, dict):
        raise GmailError("Gmail returned an invalid message payload")
    headers = {str(item.get("name", "")).lower(): _decoded_header(str(item.get("value", "")))
               for item in payload.get("headers") or [] if isinstance(item, dict)}
    attachments = []

    def collect(part: dict[str, Any]) -> None:
        filename = part.get("filename")
        if filename:
            body = part.get("body") or {}
            try:
                size = max(0, int(body.get("size") or 0))
            except (TypeError, ValueError):
                size = 0
            attachments.append({"filename": str(filename)[:255], "mime_type": str(part.get("mimeType") or ""),
                                "size": size, "attachment_id": body.get("attachmentId")})
        for child in part.get("parts") or []:
            if isinstance(child, dict):
                collect(child)

    collect(payload)
    try:
        stamp = datetime.fromtimestamp(int(raw["internalDate"]) / 1000, timezone.utc)
    except (KeyError, ValueError, TypeError, OverflowError):
        stamp = None
    return GmailMessage(
        message_id=message_id, thread_id=thread_id, sender=headers.get("from", ""),
        recipients=[headers[key] for key in ("to", "cc") if headers.get(key)],
        subject=headers.get("subject", ""), timestamp=stamp,
        snippet=str(raw.get("snippet") or "")[:300], body=_body_text(payload),
        attachments=attachments[:20], labels=[str(label) for label in raw.get("labelIds") or []],
    )


@dataclass
class OAuthGmailSource:
    token_file: Path
    timeout_seconds: float = 15.0
    transport: httpx.AsyncBaseTransport | None = None

    async def _token(self) -> str:
        if not self.token_file.is_file():
            raise GmailUnavailable("Gmail OAuth token is not configured")
        try:
            token_data = json.loads(self.token_file.read_text(encoding="utf-8"))
            if READONLY_SCOPE not in (token_data.get("scopes") or []):
                raise GmailUnavailable("Gmail OAuth token lacks gmail.readonly scope")
            credentials = Credentials.from_authorized_user_file(str(self.token_file), [READONLY_SCOPE])
            if not credentials.valid:
                if not credentials.refresh_token:
                    raise GmailUnavailable("Gmail OAuth token needs reauthorization")
                await asyncio.to_thread(credentials.refresh, Request())
                temporary = self.token_file.with_name(self.token_file.name + ".tmp")
                temporary.write_text(credentials.to_json(), encoding="utf-8")
                os.replace(temporary, self.token_file)
            return credentials.token
        except GmailError:
            raise
        except Exception as exc:
            raise GmailUnavailable("Gmail OAuth authorization failed") from exc

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        token = await self._token()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.get(f"{GMAIL_API}/{path}", params=params,
                                            headers={"Authorization": f"Bearer {token}"})
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                error = exc.response.json().get("error", {})
                reasons = {item.get("reason") for item in error.get("errors", [])
                           if isinstance(item, dict)}
            except (ValueError, AttributeError, TypeError):
                reasons = set()
            safe_reasons = reasons & {"accessNotConfigured", "insufficientPermissions",
                                      "rateLimitExceeded", "userRateLimitExceeded"}
            reason = f" ({sorted(safe_reasons)[0]})" if safe_reasons else ""
            raise GmailUnavailable(f"Gmail API returned HTTP {status}{reason}") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise GmailUnavailable("Gmail API read failed") from exc
        if not isinstance(data, dict):
            raise GmailError("Gmail API returned an invalid response")
        return data

    async def search(self, query: str, limit: int = 5) -> list[GmailMessage]:
        if not 1 <= limit <= 10:
            raise ValueError("Gmail search limit must be between 1 and 10")
        listing = await self._get("messages", {"q": query[:500], "maxResults": limit})
        refs = listing.get("messages") or []
        if not isinstance(refs, list):
            raise GmailError("Gmail API returned an invalid message list")
        results = []
        for ref in refs[:limit]:
            if not isinstance(ref, dict) or not ref.get("id"):
                raise GmailError("Gmail API returned a message without an ID")
            results.append(parse_message(await self._get(
                "messages/" + quote(str(ref["id"]), safe=""), {"format": "full"})))
        return results

    async def unread(self, query: str = "", limit: int = 5) -> list[GmailMessage]:
        return await self.search("is:unread " + query.strip(), limit)

    async def thread(self, thread_id: str) -> list[GmailMessage]:
        if not thread_id or len(thread_id) > 200:
            raise ValueError("A Gmail thread ID is required")
        data = await self._get("threads/" + quote(thread_id, safe=""), {"format": "full"})
        if data.get("id") != thread_id or not isinstance(data.get("messages"), list):
            raise GmailError("Gmail API returned an invalid thread")
        return [parse_message(raw) for raw in data["messages"][:10]]

    async def account_address(self) -> str:
        data = await self._get("profile")
        address = data.get("emailAddress")
        if not isinstance(address, str) or "@" not in address:
            raise GmailError("Gmail API returned no account address")
        return address.lower()
