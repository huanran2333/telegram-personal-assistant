"""Run the application with the event loop required by async psycopg on Windows."""
import asyncio
import sys

import uvicorn


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # The public TLS tunnel terminates on this same host. Do not let forwarded
    # headers rewrite request.client: login throttling uses that peer identity
    # and must not trust client-controlled X-Forwarded-For values.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000,
                loop="none", proxy_headers=False)


if __name__ == "__main__":
    main()
