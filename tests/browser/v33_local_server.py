"""Isolated HTTPS browser-acceptance server; operator starts/stops PG18 separately.

Never use this helper for public deployment. Calendar, Gmail, scheduler and
Telegram sends are disabled by the invoking test process, not by this file.
"""

import asyncio
from pathlib import Path
import site

import uvicorn


def main() -> None:
    local_tools = Path(__file__).resolve().parents[2] / ".local-tools"
    # The ignored Windows test dependency bundle contains PyWin32 .pth hooks
    # used by the real Calendar MCP subprocess. PYTHONPATH alone skips them.
    local_site = local_tools / "v32site"
    if local_site.is_dir():
        site.addsitedir(str(local_site))
    certificate_dir = local_tools / "v32-browser"
    config = uvicorn.Config(
        "app.main:app", host="127.0.0.1", port=18443,
        ssl_certfile=str(certificate_dir / "cert.pem"),
        ssl_keyfile=str(certificate_dir / "key.pem"),
        lifespan="on", log_level="warning",
    )
    asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
