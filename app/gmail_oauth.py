"""Explicit desktop authorization for the separate Gmail read-only token.

Run manually once after enabling Gmail API; application startup never opens an
OAuth browser. The token file is ignored by Git and must be kept private.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

from app.agent.gmail import READONLY_SCOPE


def main() -> None:
    load_dotenv()
    credentials_file = Path(os.environ.get("GMAIL_CREDENTIALS_FILE", ""))
    token_file = Path(os.environ.get("GMAIL_TOKEN_FILE", ""))
    if not credentials_file.is_file() or not os.environ.get("GMAIL_TOKEN_FILE"):
        raise SystemExit("Set GMAIL_CREDENTIALS_FILE and GMAIL_TOKEN_FILE in .env first")
    if token_file.exists():
        raise SystemExit("Gmail token already exists; move it aside deliberately before reauthorizing")
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), [READONLY_SCOPE])
    credentials = flow.run_local_server(port=0)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    with token_file.open("x", encoding="utf-8") as output:
        output.write(credentials.to_json())
    print("Gmail read-only authorization saved to the configured local token file")


if __name__ == "__main__":
    main()
